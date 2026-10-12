"""Sol's zero-order compensation as two Triton passes, next to FA4.

`veda.attention.zero_order_compensation` computes the same term in
torch, and its own docstring says why that is expensive: the term is per
query *row*, so it needs an `[H, rows, n_tiles]` score matrix that
cannot be materialised whole. It chunks the rows and writes that matrix
to memory one chunk at a time.

Fusing it is what the ComfyUI Sol+Veda kernel does, by folding the
compensation into the same online softmax as the exact blocks. But that
kernel also has to *be* the exact attention, in Triton, and measurement
says that is the expensive half: upstream's sibling INT8 kernel, the
same block-sparse walk with no compensation at all, runs at 18.4 ms
where FA4's CuTe kernel does the identical work in 10.5 ms (37376
slots, 56 heads, density 0.1, RTX PRO 6000). Triton is not going to win
that loop.

So this module splits the difference the other way: FA4 keeps the exact
blocks, and the compensation -- which is 313 GFLOP, 1/128 of the dense
matmul, and never wants a tensor core -- is one Triton kernel that
keeps its score matrix in registers. The two are joined by the identity
`zero_order_compensation` is built on: with `ref = lse_kept` the kept
denominator is exactly 1, so

    out = (o_kept + sum_dropped p_t v_sum_t) / (1 + sum_dropped p_t B_t)

and p_t = exp(q . mean_j k_j * scale - lse_kept). Nothing here needs the
dense LSE.

Measured end to end against the fused kernel, same mask and geometry:
see docs/features/veda2.md section 12.
"""

from __future__ import annotations

import functools

import torch

from miowtion.veda import tiling

TILE = tiling.TILE_SIZE
# Key tiles per step of the summary walk, and the launch. Swept by
# scripts/bench_block_size.py on an RTX PRO 6000 at 37376 slots, 56
# heads, density 0.1: the walk wants a small step and deep pipelining
# (C16 w4 s3, 2.25 ms) rather than wide steps, because the [128, C]
# score block competes with the [128, 128] accumulator for registers --
# C128 w4 costs 5.5 ms, 2.4x as much.
COLUMNS = 16
LAUNCH = {'num_warps': 4, 'num_stages': 3}


@functools.cache
def _triton():
    """Triton, or None when it is not installed.

    Imported lazily and kernels built on first use, so that this module
    imports on a CPU box -- `tests/unit` has to be able to reach the
    argument checks in `compensate`.
    """
    try:
        import triton                                     # noqa: PLC0415
        import triton.language as tl                      # noqa: PLC0415
    except ImportError:
        return None
    return triton, tl


@functools.cache
def _kernels():
    """(summary, compensate) as compiled Triton entry points.

    Raises:
        RuntimeError: If Triton is not installed.
    """
    if _triton() is None:
        raise RuntimeError('the fused compensation needs Triton')
    triton, tl = _triton()

    @triton.jit
    def _summary_kernel(K, V, Counts, KC, VC, H: tl.constexpr, D: tl.constexpr,
                        B: tl.constexpr):
        """One streaming pass: mean(K) and sum(V) over each tile's real rows.

        The torch form of this (`masked_fill(...).sum(1)`) materialises a
        second copy of K and of V, which on SM120 costs 3.9 ms where the
        pass itself costs 0.7.
        """
        tile, head = tl.program_id(0), tl.program_id(1)
        r, d = tl.arange(0, B), tl.arange(0, D)
        count = tl.load(Counts + tile)
        valid = r[:, None] < count
        offset = ((tile * B + r[:, None]) * H + head) * D + d[None, :]
        keys = tl.load(K + offset, mask=valid, other=0.).to(tl.float32)
        values = tl.load(V + offset, mask=valid, other=0.).to(tl.float32)
        out = (tile * H + head) * D + d
        divisor = tl.maximum(count, 1).to(tl.float32)
        tl.store(KC + out, (tl.sum(keys, 0) / divisor).to(KC.dtype.element_ty))
        tl.store(VC + out, tl.sum(values, 0).to(VC.dtype.element_ty))


    @triton.jit
    def _compensate_kernel(O, Lse, Q, KC, VC, Counts, Kept, Out,
                           N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                           B: tl.constexpr, C: tl.constexpr,
                           SCALE: tl.constexpr):
        """Add the dropped tiles' rank-1 terms to one query tile's output.

        One program owns one (query tile, head), so the mask row it needs is
        a single `[N]` vector and the `[B, N]` scores never leave registers.
        """
        row, head = tl.program_id(0), tl.program_id(1)
        r, d, j = tl.arange(0, B), tl.arange(0, D), tl.arange(0, C)
        rows = row * B + r
        qoff = (rows[:, None] * H + head) * D + d[None, :]
        query = tl.load(Q + qoff)
        reference = tl.load(Lse + head * (N * B) + rows)
        # A row that kept nothing has lse -inf and carries no mass either
        # way; neutralise it rather than computing inf - inf. Matches the
        # torch form's `safe`.
        finite = (reference == reference) & (reference != float('inf')) & (
            reference != -float('inf'))
        reference = tl.where(finite, reference, 0.)
        numerator = tl.zeros((B, D), tl.float32)
        denominator = tl.zeros((B,), tl.float32)
        for first in range(0, N, C):
            cols = first + j
            ok = cols < N
            lengths = tl.load(Counts + cols, mask=ok, other=0)
            kept = tl.load(Kept + (head * N + row) * N + cols, mask=ok,
                           other=True)
            # An all-padding tile has no centroid and no mass. The torch
            # form clamps its row count to 1 instead and so lets exp(-lse)
            # into the denominator; see docs/pitfalls.md.
            dropped = ok & ~kept & (lengths > 0)
            summary_off = (cols[None, :] * H + head) * D + d[:, None]
            kc = tl.load(KC + summary_off, mask=ok[None, :], other=0)
            score = tl.dot(query, kc) * SCALE
            probability = tl.where(dropped[None, :],
                                   tl.exp(score - reference[:, None]), 0.)
            vc = tl.load(VC + (cols[:, None] * H + head) * D + d[None, :],
                         mask=ok[:, None], other=0)
            numerator += tl.dot(probability.to(tl.bfloat16), vc)
            weights = lengths[None, :].to(tl.float32)
            denominator += tl.sum(probability * weights, 1)
        kept_out = tl.load(O + qoff).to(tl.float32)
        answer = (kept_out + numerator) / (1. + denominator)[:, None]
        tl.store(Out + qoff, answer.to(Out.dtype.element_ty))

    return _summary_kernel, _compensate_kernel


def summaries(k_tiles: torch.Tensor, v_tiles: torch.Tensor,
              counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile mean(K) and sum(V).

    Args:
        k_tiles: [N * 128, H, D] bf16 contiguous, tile ordered.
        v_tiles: [N * 128, H, D] bf16 contiguous, tile ordered.
        counts: [N] int32 real rows per tile (a prefix of the tile).

    Returns:
        (kc, vc), both [N, H, D] bf16 contiguous.
    """
    tiles, heads, dim = k_tiles.shape[0] // TILE, k_tiles.shape[1], \
        k_tiles.shape[2]
    kc = torch.empty((tiles, heads, dim), dtype=torch.bfloat16,
                     device=k_tiles.device)
    vc = torch.empty_like(kc)
    summary_kernel, _ = _kernels()
    summary_kernel[(tiles, heads)](k_tiles, v_tiles, counts, kc, vc,
                                   H=heads, D=dim, B=TILE, num_warps=4)
    return kc, vc


def compensate(o_kept: torch.Tensor, lse_kept: torch.Tensor,
               q_tiles: torch.Tensor, k_tiles: torch.Tensor,
               v_tiles: torch.Tensor, block_mask: torch.Tensor,
               tile_layout: tiling.TileLayout) -> torch.Tensor:
    """Adds Sol's zero-order term for the tiles the mask dropped.

    Same contract and the same arithmetic as
    `veda.attention.zero_order_compensation`, with the per-row score
    matrix kept in registers. The summaries are bf16 rather than fp32,
    which is the only deliberate numerical difference; the measured gap
    is in docs/features/veda2.md section 12.

    Args:
        o_kept: [R * 128, H, D] block-sparse kernel output (num/den over
            the kept keys).
        lse_kept: [H, R * 128] fp32 natural log-sum-exp over the kept
            keys, as `fa4.block_sparse_attention(..., return_lse=True)`
            returns it.
        q_tiles: [R * 128, H, D] bf16 tile-ordered queries.
        k_tiles: [N * 128, H, D] bf16 tile-ordered keys.
        v_tiles: [N * 128, H, D] bf16 tile-ordered values.
        block_mask: [H, R, N] bool, True where the tile was kept exactly.
        tile_layout: Tile layout, for each key tile's real row count.

    Returns:
        [R * 128, H, D] in o_kept.dtype.

    Raises:
        ValueError: On a shape mismatch, or if the query tiles are a
            subset: one program reads mask row `r` for query tile `r`,
            so a partial mask would mis-pair them.
    """
    rows, heads, dim = o_kept.shape
    tiles = tile_layout.n_tiles
    if q_tiles.shape != o_kept.shape:
        raise ValueError(f'q_tiles is {tuple(q_tiles.shape)}, expected '
                         f'{tuple(o_kept.shape)}')
    if lse_kept.shape != (heads, rows):
        raise ValueError(f'lse_kept is {tuple(lse_kept.shape)}, expected '
                         f'{(heads, rows)}')
    if block_mask.shape != (heads, rows // TILE, tiles):
        raise ValueError(f'block_mask is {tuple(block_mask.shape)}, '
                         f'expected {(heads, rows // TILE, tiles)}')
    if rows // TILE != tiles:
        raise ValueError(
            f'{rows // TILE} query tiles but {tiles} key tiles; this '
            f'kernel indexes both with the same id')
    counts = tile_layout.valid_count.to(torch.int32)
    kc, vc = summaries(k_tiles.contiguous(), v_tiles.contiguous(), counts)
    out = torch.empty_like(o_kept)
    _, compensate_kernel = _kernels()
    compensate_kernel[(rows // TILE, heads)](
        o_kept, lse_kept.contiguous(), q_tiles, kc, vc, counts,
        block_mask.contiguous(), out, N=tiles, H=heads, D=dim, B=TILE,
        C=COLUMNS, SCALE=dim ** -0.5, num_warps=LAUNCH['num_warps'],
        num_stages=LAUNCH['num_stages'])
    return out


def attention(q_tiles: torch.Tensor, k_tiles: torch.Tensor,
              v_tiles: torch.Tensor, block_mask: torch.Tensor,
              tile_layout: tiling.TileLayout) -> torch.Tensor:
    """FA4 block-sparse attention plus the compensation, end to end.

    The drop-in alternative to `sol_veda.block_sparse_attention`: same
    arguments, same correction, BF16 throughout instead of INT8 for the
    kept blocks.

    Args:
        q_tiles, k_tiles, v_tiles: [N * 128, H, D] bf16, tile ordered.
        block_mask: [H, N, N] bool, True where the tile is kept exactly.
        tile_layout: Tile layout.

    Returns:
        [N * 128, H, D] bf16.
    """
    from miowtion.kernels import fa4                       # noqa: PLC0415
    kept = block_mask & tile_layout.kv_ok[None, None, :]
    o_kept, lse = fa4.block_sparse_attention(q_tiles, k_tiles, v_tiles,
                                             kept, tile_layout,
                                             return_lse=True)
    return compensate(o_kept, lse, q_tiles, k_tiles, v_tiles, kept,
                      tile_layout)
