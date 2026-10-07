"""Veda block-sparse attention on MLX, by gathering the selected key tiles.

MLX's fused SDPA (`mx.fast.scaled_dot_product_attention`) takes a boolean
mask but applies it *after* the Q@K.T tile matmul: only the built-in
`"causal"` mask shrinks the key loop, an arbitrary block mask does not. So a
mask buys correctness and O(S) memory but no FLOPs (measured: 0.97x at 90 %
block sparsity).

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
from collections.abc import Sequence

import mlx.core as mx


@dataclasses.dataclass(frozen=True)
class SparsePlan:
    """The key tiles every query tile attends to.

    Attributes:
        index: [n_query_tiles, budget] int32 selected key-tile ids, shared
            by every head, or [heads, n_query_tiles, budget] with one
            selection per head (Veda runs its top-k per head). Every query
            tile carries the same budget, which is what lets the gathered
            problem stay rectangular.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
        keep: Bool of index's shape, or None. False marks a slot
            that only pads the budget out to a common width, so that a
            selection with a per-row budget (Veda spreads the fractional
            part of the budget with a Bresenham pattern) still fits the
            rectangular gather. Padding slots must be masked, not repeated:
            a repeated key tile would enter the softmax twice.
        key_valid: [seq_len] bool or None. False marks a key row that is
            padding inside its tile. Veda's permuted layout leaves partial
            tiles, so validity is per row, not per tile.
        dense_rows: Trailing query rows that attend to every key. Veda keeps
            the global (text / audio) query tiles dense, and they sit at the
            end of the permuted sequence; they cannot share a uniform budget
            with the video rows, so they get their own dense call. They are
            few, so it stays cheap.
    """

    index: mx.array
    q_block: int
    k_block: int
    keep: mx.array | None = None
    key_valid: mx.array | None = None
    dense_rows: int = 0

    @property
    def budget(self) -> int:
        return self.index.shape[-1]

    @property
    def n_query_tiles(self) -> int:
        return self.index.shape[-2]

    def density(self, seq_len: int) -> float:
        """Fraction of the full attention matrix that is kept.

        Counts padding slots, i.e. it is the density the kernel pays for,
        not the density of the selection.
        """
        return self.budget / (seq_len / self.k_block)

    def select_heads(self, positions: Sequence[int]) -> SparsePlan:
        """The same plan restricted to some of its heads.

        A block that chunks its heads hands the kernel a q of
        `head_chunk` heads, so a per-head selection has to be narrowed
        the same way; a shared selection is returned unchanged.

        Args:
            positions: Ascending head positions of this plan to keep.

        Returns:
            A plan whose per-head arrays have `len(positions)` heads.
        """
        if self.index.ndim == 2 or len(positions) == self.index.shape[0]:
            return self
        pick = mx.array(list(positions), dtype=mx.int32)
        return dataclasses.replace(
            self, index=mx.take(self.index, pick, axis=0),
            keep=None if self.keep is None
            else mx.take(self.keep, pick, axis=0))


def block_mask_from_index(index: mx.array, q_block: int, k_block: int,
                          seq_len: int, keep: mx.array | None = None,
                          key_valid: mx.array | None = None,
                          dense_rows: int = 0) -> mx.array:
    """Dense boolean mask of a block-sparse selection (reference path).

    Args:
        index: [n_query_tiles, budget] int32, selected key-tile ids.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
        seq_len: Padded sequence length (multiple of both tile sizes).
        keep: Bool of index's shape, or None; see SparsePlan.
        key_valid: [seq_len] bool or None; see SparsePlan.
        dense_rows: Trailing query rows that are kept everywhere.

    Returns:
        [seq_len, seq_len] bool, or [heads, seq_len, seq_len] for a per-head
        index; True where attention is kept.
    """
    n_q, budget = index.shape[-2:]
    n_k = seq_len // k_block
    sparse_rows = seq_len - dense_rows
    if n_q * q_block != sparse_rows or n_k * k_block != seq_len:
        raise ValueError(f'index has {n_q} query tiles of {q_block} rows, '
                         f'which does not tile the {sparse_rows} sparse rows '
                         f'of seq_len {seq_len} (k_block {k_block})')
    if index.ndim == 3:
        return mx.stack([block_mask_from_index(
            index[h], q_block, k_block, seq_len,
            None if keep is None else keep[h], key_valid, dense_rows)
            for h in range(index.shape[0])])
    # One spare column absorbs the padding slots, so that they cannot switch
    # on a tile; it is dropped again before the mask is expanded.
    tiles = mx.zeros((n_q, n_k + 1), dtype=mx.bool_)
    rows = mx.repeat(mx.arange(n_q), budget)
    columns = index.reshape(-1)
    if keep is not None:
        columns = mx.where(keep.reshape(-1), columns, n_k)
    tiles[rows, columns] = True
    mask = mx.repeat(mx.repeat(tiles[:, :n_k], q_block, axis=0), k_block,
                     axis=1)
    if dense_rows:
        mask = mx.concatenate(
            [mask, mx.ones((dense_rows, seq_len), dtype=mx.bool_)], axis=0)
    if key_valid is not None:
        mask = mask & key_valid[None, :]
    return mask


def block_sparse_attention(q: mx.array, k: mx.array, v: mx.array,
                           index: mx.array, *, q_block: int, k_block: int,
                           scale: float | None = None,
                           head_chunk: int | None = None,
                           keep: mx.array | None = None,
                           key_valid: mx.array | None = None,
                           dense_rows: int = 0) -> mx.array:
    """Attention restricted to the selected key tiles.

    Args:
        q: [heads, seq_len, head_dim], seq_len a multiple of q_block.
        k: [heads, seq_len, head_dim], seq_len a multiple of k_block.
        v: Same shape as k.
        index: [n_query_tiles, budget] int32 shared by all heads, or
            [heads, n_query_tiles, budget] with one selection per head
            (Veda's top-k is per head). n_query_tiles is
            (seq_len - dense_rows) // q_block, and every query tile must
            have the same budget.
        q_block: Rows per query tile.
        k_block: Rows per key tile.
        scale: Query scale; defaults to head_dim ** -0.5.
        head_chunk: Heads processed at once (bounds the gather buffer).
            None processes every head in one call.
        keep: Bool of index's shape, or None; see SparsePlan.
        key_valid: [seq_len] bool or None; see SparsePlan.
        dense_rows: Trailing query rows that attend to every key; see
            SparsePlan.

    Returns:
        [heads, seq_len, head_dim] in q's dtype.

    Raises:
        ValueError: On shape or tiling mismatches, or if a query tile keeps
            no key row at all (its softmax would be NaN).
    """
    heads, seq_len, head_dim = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f'q, k, v must agree, got {q.shape}, {k.shape}, '
                         f'{v.shape}')
    sparse_rows = seq_len - dense_rows
    if dense_rows < 0 or sparse_rows <= 0:
        raise ValueError(f'dense_rows {dense_rows} must be in [0, {seq_len})')
    if sparse_rows % q_block or seq_len % k_block:
        raise ValueError(f'{sparse_rows} sparse rows must be a multiple of '
                         f'q_block {q_block}, and seq_len {seq_len} of '
                         f'k_block {k_block}')
    if index.ndim not in (2, 3):
        raise ValueError(f'index must be 2-D or 3-D, got {index.shape}')
    if index.ndim == 3 and index.shape[0] != heads:
        raise ValueError(f'per-head index has {index.shape[0]} heads, '
                         f'expected {heads}')
    n_q, budget = index.shape[-2:]
    if n_q != sparse_rows // q_block:
        raise ValueError(f'index has {n_q} query tiles, expected '
                         f'{sparse_rows // q_block}')
    if budget > seq_len // k_block:
        raise ValueError(f'budget {budget} exceeds {seq_len // k_block} key '
                         'tiles')
    if scale is None:
        scale = head_dim ** -0.5
    if head_chunk is not None and head_chunk < 1:
        raise ValueError(f'head_chunk must be >= 1, got {head_chunk}')
    step = head_chunk or heads
    if keep is not None and keep.shape != index.shape:
        raise ValueError(f'keep {keep.shape} must match index {index.shape}')
    if key_valid is not None and key_valid.shape != (seq_len,):
        raise ValueError(f'key_valid must be [{seq_len}], got '
                         f'{key_valid.shape}')
    dense_mask = None if key_valid is None else key_valid[None, None, None]

    out = []
    for start in range(0, heads, step):
        stop = min(start + step, heads)
        kc, vc = k[start:stop], v[start:stop]
        idx = index if index.ndim == 2 else index[start:stop]
        sub_keep = None if keep is None else (
            keep if keep.ndim == 2 else keep[start:stop])
        mask = _gathered_mask(idx, k_block, seq_len, sub_keep, key_valid)
        chunk = _chunk(q[start:stop, :sparse_rows], kc, vc, idx, q_block,
                       k_block, scale, mask)
        if dense_rows:
            tail = mx.fast.scaled_dot_product_attention(
                q[start:stop, sparse_rows:][None], kc[None], vc[None],
                scale=scale, mask=dense_mask)[0]
            chunk = mx.concatenate([chunk, tail], axis=1)
        out.append(chunk)
    return mx.concatenate(out, axis=0) if len(out) > 1 else out[0]


def _gathered_mask(index: mx.array, k_block: int, seq_len: int,
                   keep: mx.array | None,
                   key_valid: mx.array | None) -> mx.array | None:
    """Boolean mask over the gathered key axis, or None if nothing is masked.

    The mask is the same for every query row of a tile, so it stays a
    [n_q, heads_or_1, 1, budget * k_block] broadcast operand: it costs memory
    proportional to the gathered *tile* count, not to the gathered rows. It
    is applied by the kernel after the Q@K.T tile matmul, which is why it
    buys correctness and never FLOPs -- the gather already bought those.

    Args:
        index: [n_q, budget] or [heads, n_q, budget] int32.
        k_block: Rows per key tile.
        seq_len: Key rows.
        keep: Bool of index's shape, or None.
        key_valid: [seq_len] bool, or None.
    """
    n_q, budget = index.shape[-2:]
    parts = []
    if keep is not None:
        parts.append(mx.repeat(keep, k_block, axis=-1))
    if key_valid is not None:
        tiles = key_valid.reshape(seq_len // k_block, k_block)
        parts.append(mx.take(tiles, index, axis=0)
                     .reshape(*index.shape[:-2], n_q, budget * k_block))
    if not parts:
        return None
    mask = parts[0] if len(parts) == 1 else parts[0] & parts[1]
    if not mx.all(mx.any(mask, axis=-1)).item():
        raise ValueError('every query tile must keep at least one key row')
    # [n_q, 1, 1, L] shared, or [n_q, heads, 1, L] per head.
    if mask.ndim == 2:
        return mask[:, None, None, :]
    return mask.transpose(1, 0, 2)[:, :, None, :]


def _chunk(q: mx.array, k: mx.array, v: mx.array, index: mx.array,
           q_block: int, k_block: int, scale: float,
           mask: mx.array | None = None) -> mx.array:
    """block_sparse_attention for one group of heads (sparse rows only)."""
    heads, seq_len, head_dim = k.shape
    n_q, budget = index.shape[-2:]
    n_k = seq_len // k_block
    if index.ndim == 3:
        # Flatten (head, tile) into one axis so that a per-head selection is
        # still a single gather.
        index = (index + (mx.arange(heads) * n_k).reshape(heads, 1, 1))

    gathered = []
    for source in (k, v):
        if index.ndim == 2:
            tiles = source.reshape(heads, n_k, k_block, head_dim)
            picked = mx.take(tiles, index, axis=1)
        else:
            tiles = source.reshape(heads * n_k, k_block, head_dim)
            picked = mx.take(tiles, index.reshape(-1), axis=0)
        # [heads, n_q, budget, k_block, head_dim] -> one flat key axis.
        gathered.append(picked.reshape(heads, n_q, budget * k_block, head_dim)
                        .transpose(1, 0, 2, 3))

    # [n_q, heads, q_block, head_dim]: batch over query tiles.
    queries = q.reshape(heads, n_q, q_block, head_dim).transpose(1, 0, 2, 3)

    out = mx.fast.scaled_dot_product_attention(
        queries, gathered[0], gathered[1], scale=scale, mask=mask)
    return out.transpose(1, 0, 2, 3).reshape(heads, n_q * q_block, head_dim)


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


@dataclasses.dataclass(frozen=True)
class HeadGroupPlan:
    """One head group's permutation and selection.

    Veda's tile plan gives a layer up to two tile shapes, so the heads of a
    layer split into groups that each see a *different* permutation of the
    sequence. The permutation is part of the plan, not of the block: the
    block keeps q / k / v in packed order and every group permutes its own
    heads in and out.

    Attributes:
        heads: Ascending global head ids of the group.
        gather: [num_slots] int32 packed row of every permuted slot. Padding
            slots point at row 0; `plan.key_valid` keeps them out of the
            softmax, so their value never matters.
        scatter: [rows] int32 permuted slot of every packed row. Rows that
            no tile covers point at `num_slots`, a zero row appended to the
            output.
        plan: The selection, on the permuted sequence.
    """

    heads: tuple[int, ...]
    gather: mx.array
    scatter: mx.array
    plan: SparsePlan

    def select_heads(self, positions: Sequence[int]) -> HeadGroupPlan:
        """The group restricted to some of its heads.

        The permutation is shared by the whole group, so only the
        selection narrows.

        Args:
            positions: Ascending positions into `heads` to keep.

        Returns:
            A group of `len(positions)` heads.
        """
        if len(positions) == len(self.heads):
            return self
        return dataclasses.replace(
            self, heads=tuple(self.heads[i] for i in positions),
            plan=self.plan.select_heads(positions))


@dataclasses.dataclass(frozen=True)
class LayerPlan:
    """The head groups of one layer; the groups must partition the heads.

    Raises:
        ValueError: If the groups do not partition [0, num_heads).
    """

    groups: tuple[HeadGroupPlan, ...]
    num_heads: int

    def __post_init__(self):
        seen = sorted(h for g in self.groups for h in g.heads)
        if seen != list(range(self.num_heads)):
            raise ValueError(f'head groups {seen} do not partition '
                             f'[0, {self.num_heads})')

    def density(self, seq_len: int | None = None) -> float:
        """Kept fraction of the attention matrix, averaged over the heads.

        Args:
            seq_len: Ignored; every group knows how many rows its own
                permutation has. The argument only keeps the signature
                interchangeable with SparsePlan.density.
        """
        del seq_len
        kept = sum(len(g.heads) * g.plan.density(g.gather.shape[0])
                   for g in self.groups)
        return kept / self.num_heads


def group_attention(group: HeadGroupPlan, q: mx.array, k: mx.array,
                    v: mx.array, scale: float | None = None,
                    head_chunk: int | None = None) -> mx.array:
    """Block-sparse attention of one head group, in packed row order.

    Args:
        q: [heads, rows, head_dim] packed rows of this group's heads only.
        k: Same shape as q.
        v: Same shape as q.
        scale: Query scale; defaults to head_dim ** -0.5.
        head_chunk: Heads per gather inside the group.

    Returns:
        [heads, rows, head_dim] in q's dtype.
    """
    plan = group.plan
    permuted = [mx.take(x, group.gather, axis=1) for x in (q, k, v)]
    out = block_sparse_attention(*permuted, plan.index, q_block=plan.q_block,
                                 k_block=plan.k_block, scale=scale,
                                 head_chunk=head_chunk, keep=plan.keep,
                                 key_valid=plan.key_valid,
                                 dense_rows=plan.dense_rows)
    del permuted
    zero = mx.zeros((out.shape[0], 1, out.shape[2]), dtype=out.dtype)
    return mx.take(mx.concatenate([out, zero], axis=1), group.scatter, axis=1)


def layer_attention(layer: LayerPlan, q: mx.array, k: mx.array, v: mx.array,
                    head_start: int = 0, scale: float | None = None,
                    head_chunk: int | None = None) -> mx.array:
    """Block-sparse attention of a contiguous range of a layer's heads.

    Args:
        layer: The layer's head groups.
        q: [heads, rows, head_dim] of heads
            [head_start, head_start + heads).
        k: Same shape as q.
        v: Same shape as q.
        head_start: Global id of q's first head.
        scale: Query scale; defaults to head_dim ** -0.5.
        head_chunk: Heads per gather inside a group.

    Returns:
        [heads, rows, head_dim], the heads back in their input order.

    Raises:
        ValueError: If the range runs past the layer's heads.
    """
    heads = q.shape[0]
    stop = head_start + heads
    if head_start < 0 or stop > layer.num_heads:
        raise ValueError(f'heads [{head_start}, {stop}) outside '
                         f'[0, {layer.num_heads})')
    parts, order = [], []
    for group in layer.groups:
        positions = [position for position, head in enumerate(group.heads)
                     if head_start <= head < stop]
        if not positions:
            continue
        local = [group.heads[position] - head_start for position in positions]
        if len(local) == heads:
            sub = (q, k, v)
        else:
            pick = mx.array(local, dtype=mx.int32)
            sub = tuple(mx.take(x, pick, axis=0) for x in (q, k, v))
        # The group's selection is per head, so it has to be narrowed to
        # the same heads q was narrowed to.
        parts.append(group_attention(group.select_heads(positions), *sub,
                                     scale=scale, head_chunk=head_chunk))
        order += local
    if len(parts) == 1:
        return parts[0]
    # The groups took the heads out of order; put them back.
    out = mx.concatenate(parts, axis=0)
    inverse = [0] * heads
    for position, head in enumerate(order):
        inverse[head] = position
    return mx.take(out, mx.array(inverse, dtype=mx.int32), axis=0)
