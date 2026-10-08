"""Fused Triton kernel for block attention heat (teacher heat / oracle mask).

`reduce='max'` gives
heat[i, j] = max over the 128x128 block of exp(q.k / sqrt(D) - lse): the
largest true attention probability inside a block. It supervises the
predictor (teacher heat) and defines the oracle mask of the tile search.

The torch reference materializes [H', rows, N] bf16 scores (tens of GB per
layer on long clips) only to reduce them to one value per 128x128 block.
This kernel computes each block's QK^T in registers and writes a single
fp32 heat value, so the cost is the QK^T FLOPs alone.

Numerics mirror the reference: scores are rounded to bf16 before the max
(done once on each row max, which is identical because rounding is
monotonic), the scale is applied after the max, invalid keys are -inf and
invalid query rows contribute nothing. Accumulation order differs from
cuBLAS, so parity with the reference is to tolerance, not bitwise (see
tests/gpu/test_kernels_gpu.py).

Launch parameters were tuned on RTX 4090 (d=128, 56 heads, 38k tokens):
4 warps and 16 key tiles per program run QK^T at ~169 TFLOPS, 0.47x the time
of FA4's dense forward (which does twice the FLOPs); 8 warps, software
pipelining (num_stages > 1) and 64-wide key sub-tiles were all slower.
"""

from __future__ import annotations

import functools
import math

import torch

from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE
# Key tiles handled by one program; each program loads its query tile once.
_KEY_TILES_PER_PROGRAM = 16
_NUM_WARPS = 4
# No software pipelining of the key-tile loop (Triton's default of 3 stages
# is ~20% slower here).
_NUM_STAGES = 1


@functools.cache
def _triton():
    try:
        import triton  # pylint: disable=import-outside-toplevel
        import triton.language as tl  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return triton, tl


def available() -> bool:
    return torch.cuda.is_available() and _triton() is not None


@functools.cache
def _kernel():
    triton, tl = _triton()

    @triton.jit
    def heat_kernel(q_ptr, k_ptr, lse_ptr, valid_ptr, q_tiles_ptr, out_ptr,
                    stride_qn, stride_qh, stride_kn, stride_kh, stride_ln,
                    stride_lh, stride_oh, stride_or, n_tiles, scale,
                    HEAD_DIM: tl.constexpr, TILE: tl.constexpr,
                    TILES_PER_PROGRAM: tl.constexpr,
                    SUM: tl.constexpr):
        r = tl.program_id(0)
        group = tl.program_id(1)
        h = tl.program_id(2)
        q_tile = tl.load(q_tiles_ptr + r)
        rows = q_tile * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + rows[:, None] * stride_qn + h * stride_qh
                    + dims[None, :])
        row_ok = tl.load(valid_ptr + rows) != 0
        lse = tl.load(lse_ptr + rows * stride_ln + h * stride_lh)
        # An infinite lse turns an invalid row's term into exp(-inf) = 0.
        lse = tl.where(row_ok, lse, float('inf'))
        start = group * TILES_PER_PROGRAM
        for j in range(start, tl.minimum(start + TILES_PER_PROGRAM, n_tiles)):
            cols = j * TILE + tl.arange(0, TILE)
            k = tl.load(k_ptr + cols[:, None] * stride_kn + h * stride_kh
                        + dims[None, :])
            col_ok = tl.load(valid_ptr + cols) != 0
            s = tl.dot(q, tl.trans(k))
            s = tl.where(col_ok[None, :], s, float('-inf'))
            if SUM:
                # Every masked term is exp(-inf) = 0: invalid columns are
                # -inf above, and invalid rows carry lse = +inf.
                p = tl.exp(s * scale - lse[:, None])
                agg = tl.sum(tl.sum(p, axis=1), axis=0)
            else:
                row_max = tl.max(s, axis=1).to(tl.bfloat16).to(tl.float32)
                agg = tl.exp(tl.max(row_max * scale - lse, axis=0))
            tl.store(out_ptr + h * stride_oh + r * stride_or + j, agg)

    return heat_kernel


@torch.no_grad()
def teacher_heat(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor,
                 layout: tiling.TileLayout, q_tiles: torch.Tensor,
                 reduce: str = 'max') -> torch.Tensor:
    """Same contract as heatmap.teacher_heat_reference.

    Raises:
        ValueError: On an unknown reduction or an unsupported head_dim.
    """
    if reduce not in ('max', 'sum'):
        raise ValueError(f"reduce must be 'max' or 'sum': {reduce!r}")
    if q.shape[-1] not in (64, 128) or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError('heat kernel needs contiguous head_dim 64 or 128')
    heads, n_tiles = q.shape[1], layout.n_tiles
    out = torch.empty(heads, q_tiles.numel(), n_tiles, dtype=torch.float32,
                      device=q.device)
    grid = (q_tiles.numel(), -(-n_tiles // _KEY_TILES_PER_PROGRAM), heads)
    _kernel()[grid](
        q, k, lse, layout.slot_valid, q_tiles.contiguous(), out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), lse.stride(0),
        lse.stride(1), out.stride(0), out.stride(1), n_tiles,
        1.0 / math.sqrt(q.shape[-1]), HEAD_DIM=q.shape[-1], TILE=_TILE,
        TILES_PER_PROGRAM=_KEY_TILES_PER_PROGRAM, SUM=reduce == 'sum',
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES)
    return out
