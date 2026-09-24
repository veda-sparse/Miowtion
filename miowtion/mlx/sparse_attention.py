"""Veda block-sparse attention on MLX, by gathering the selected key tiles.

MLX's fused SDPA (`mx.fast.scaled_dot_product_attention`) takes a boolean
mask but applies it *after* the Q@K.T tile matmul: only the built-in
`"causal"` mask shrinks the key loop, an arbitrary block mask does not. So a
mask buys correctness and O(S) memory but no FLOPs (measured: 0.97x at 90 %
block sparsity, see docs/features/mlx_inference.md).

This module gets the FLOPs back without a custom Metal kernel. Veda gives
every query tile the *same* number of key tiles (a fixed budget), so the
selected key tiles can be gathered into a dense, rectangular
`[n_query_tiles, heads, budget * k_block, head_dim]` tensor and the whole
thing handed to one *batched* dense SDPA call. The kernel then runs on a
problem that is genuinely `budget * k_block` keys wide instead of `S`.

Cost model: the gather rewrites `density * S^2 / q_block` rows per head, so
the query tile must be reasonably large (Veda's tiles are) for the gather to
stay cheap next to the attention it saves.

Invariant: the result must equal dense attention under the equivalent
boolean block mask (up to the reduction order of the two kernels).
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx


@dataclasses.dataclass(frozen=True)
class SparsePlan:
    """The key tiles every query tile attends to.

    Attributes:
        index: [seq_len // q_block, budget] int32, selected key-tile ids.
            Every query tile carries the same budget, which is what lets the
            gathered problem stay rectangular.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
    """

    index: mx.array
    q_block: int
    k_block: int

    @property
    def budget(self) -> int:
        return self.index.shape[1]

    def density(self, seq_len: int) -> float:
        """Fraction of the full attention matrix that is kept."""
        return self.budget / (seq_len / self.k_block)


def block_mask_from_index(index: mx.array, q_block: int, k_block: int,
                          seq_len: int) -> mx.array:
    """Dense boolean mask of a block-sparse selection (reference path).

    Args:
        index: [n_query_tiles, budget] int32, selected key-tile ids.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
        seq_len: Padded sequence length (multiple of both tile sizes).

    Returns:
        [seq_len, seq_len] bool; True where attention is kept.
    """
    n_q, budget = index.shape
    n_k = seq_len // k_block
    if n_q * q_block != seq_len or n_k * k_block != seq_len:
        raise ValueError(f'index has {n_q} query tiles of {q_block} rows, '
                         f'which does not tile seq_len {seq_len} (k_block '
                         f'{k_block})')
    tiles = mx.zeros((n_q, n_k), dtype=mx.bool_)
    rows = mx.repeat(mx.arange(n_q), budget)
    tiles[rows, index.reshape(-1)] = True
    return mx.repeat(mx.repeat(tiles, q_block, axis=0), k_block, axis=1)


def block_sparse_attention(q: mx.array, k: mx.array, v: mx.array,
                           index: mx.array, *, q_block: int, k_block: int,
                           scale: float | None = None,
                           head_chunk: int | None = None) -> mx.array:
    """Attention restricted to the selected key tiles.

    Args:
        q: [heads, seq_len, head_dim], seq_len a multiple of q_block.
        k: [heads, seq_len, head_dim], seq_len a multiple of k_block.
        v: Same shape as k.
        index: [seq_len // q_block, budget] int32, the key tiles each query
            tile attends to. Every query tile must have the same budget.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
        scale: Query scale; defaults to head_dim ** -0.5.
        head_chunk: Heads processed at once (bounds the gather buffer).
            None processes every head in one call.

    Returns:
        [heads, seq_len, head_dim] in q's dtype.

    Raises:
        ValueError: On shape or tiling mismatches.
    """
    heads, seq_len, head_dim = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f'q, k, v must agree, got {q.shape}, {k.shape}, '
                         f'{v.shape}')
    if seq_len % q_block or seq_len % k_block:
        raise ValueError(f'seq_len {seq_len} must be a multiple of q_block '
                         f'{q_block} and k_block {k_block}')
    n_q, budget = index.shape
    if n_q != seq_len // q_block:
        raise ValueError(f'index has {n_q} query tiles, expected '
                         f'{seq_len // q_block}')
    if budget > seq_len // k_block:
        raise ValueError(f'budget {budget} exceeds {seq_len // k_block} key '
                         'tiles')
    if scale is None:
        scale = head_dim ** -0.5
    if head_chunk is not None and head_chunk < 1:
        raise ValueError(f'head_chunk must be >= 1, got {head_chunk}')
    step = head_chunk or heads

    out = []
    for start in range(0, heads, step):
        stop = min(start + step, heads)
        out.append(_chunk(q[start:stop], k[start:stop], v[start:stop], index,
                          q_block, k_block, scale))
    return mx.concatenate(out, axis=0) if len(out) > 1 else out[0]


def _chunk(q: mx.array, k: mx.array, v: mx.array, index: mx.array,
           q_block: int, k_block: int, scale: float) -> mx.array:
    """block_sparse_attention for one group of heads."""
    heads, seq_len, head_dim = q.shape
    n_q, budget = index.shape

    # [n_q, heads, q_block, head_dim]: batch over query tiles.
    queries = q.reshape(heads, n_q, q_block, head_dim).transpose(1, 0, 2, 3)

    gathered = []
    for source in (k, v):
        tiles = source.reshape(heads, seq_len // k_block, k_block, head_dim)
        # [heads, n_q, budget, k_block, head_dim] -> one flat key axis.
        picked = mx.take(tiles, index, axis=1)
        gathered.append(picked.reshape(heads, n_q, budget * k_block, head_dim)
                        .transpose(1, 0, 2, 3))

    out = mx.fast.scaled_dot_product_attention(
        queries, gathered[0], gathered[1], scale=scale, mask=None)
    return out.transpose(1, 0, 2, 3).reshape(heads, seq_len, head_dim)


def dense_reference(q: mx.array, k: mx.array, v: mx.array,
                    mask: mx.array | None = None,
                    scale: float | None = None) -> mx.array:
    """Dense attention under an optional boolean mask (the reference)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return mx.fast.scaled_dot_product_attention(
        q[None], k[None], v[None], scale=scale,
        mask=None if mask is None else mask[None, None])[0]


def random_index(n_query_tiles: int, n_key_tiles: int, budget: int,
                 seed: int = 0) -> mx.array:
    """A random selection with a fixed budget per query tile (for tests).

    Returns:
        [n_query_tiles, budget] int32, sorted and distinct per row.
    """
    if budget > n_key_tiles:
        raise ValueError(f'budget {budget} exceeds {n_key_tiles} key tiles')
    mx.random.seed(seed)
    scores = mx.random.uniform(shape=(n_query_tiles, n_key_tiles))
    return mx.sort(mx.argsort(scores, axis=1)[:, :budget],
                   axis=1).astype(mx.int32)


def attention_flops(seq_len: int, heads: int, head_dim: int,
                    density: float) -> float:
    """FLOPs of block-sparse attention (Q@K.T and P@V)."""
    return 4.0 * heads * seq_len * seq_len * head_dim * density


def gathered_bytes(seq_len: int, heads: int, head_dim: int, q_block: int,
                   density: float, itemsize: int = 2) -> float:
    """Bytes the K/V gather writes (the price of reusing the dense kernel)."""
    return (2.0 * heads * (seq_len / q_block) * density * seq_len * head_dim
            * itemsize)


def suggest_q_block(seq_len: int, density: float, k_block: int) -> int:
    """Largest power-of-two query tile whose gather stays under the QK cost.

    Heuristic only; the benchmark is what decides.
    """
    target = max(k_block, int(math.sqrt(seq_len * density * k_block)))
    return 1 << max(int(target).bit_length() - 1, int(math.log2(k_block)))
