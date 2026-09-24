"""Veda selections as MLX sparse-attention plans.

`miowtion.veda.mask.select_video_blocks` produces the selection the CUDA
kernels consume; this module turns the same selection into a
`sparse_attention.SparsePlan`, so that MLX runs exactly the mask that
`mask.dense_block_mask` describes. Three properties of a Veda selection do
not fit the plain "fixed budget per query tile" gather, and each maps onto
one field of the plan:

  1. The budget varies by +-1 per query tile (the fractional part is spread
     by a Bresenham pattern) and is split over the reference and target
     column blocks. `Selection.keep` already marks which of the K slots are
     real, so it becomes `SparsePlan.keep` and the short rows are masked
     rather than trimmed.
  2. Global (text / audio) rows and columns are dense. The columns are
     appended to every video row's budget; the rows sit at the end of the
     permuted sequence and become `SparsePlan.dense_rows`.
  3. Tiles are only partly filled. `TileLayout.slot_valid` becomes
     `SparsePlan.key_valid`, which keeps the padding slots out of the
     softmax.

The plan applies to the *permuted* sequence (`layout.num_slots` rows, i.e.
`used = layout.num_slots` for the MLX block), not to the packed one: the
tiles are contiguous only after `layout.gather_index` has been applied.

Because Veda runs its top-k per head, the index is per head, and because
`TilePlan` allows two tile shapes (hence two permutations) per layer, a
layer needs one plan per head group -- see docs/features/mlx_inference.md.
"""

from __future__ import annotations

import torch

from miowtion.mlx import interop
from miowtion.mlx import sparse_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import tiling


def plan_from_selection(selection: veda_mask.Selection,
                        layout: tiling.TileLayout,
                        share_heads: bool = True
                        ) -> sparse_attention.SparsePlan:
    """Builds the MLX plan of one head group's selection.

    Args:
        selection: Selection of all `layout.n_video_tiles` video query tiles
            for the H' heads of one group.
        layout: The tile layout the selection was made on.
        share_heads: Collapse the head axis when every head selected the
            same tiles, which lets one gather serve the whole group.

    Returns:
        The equivalent plan, with q_block = k_block = the Veda tile size.

    Raises:
        ValueError: If the selection does not cover every video query tile.
    """
    heads, num_rows, _ = selection.index.shape
    n_video, n_tiles = layout.n_video_tiles, layout.n_tiles
    if num_rows != n_video:
        raise ValueError(f'selection has {num_rows} query tiles, expected '
                         f'all {n_video} video tiles')
    device = selection.index.device
    # Rule 1 of mask.py: every query tile sees every global key tile. They
    # are appended to the budget instead of being selected, so the gather
    # stays one rectangular problem.
    globals_ = torch.arange(n_video, n_tiles, device=device)
    index = torch.cat(
        [selection.index,
         globals_.expand(heads, num_rows, n_tiles - n_video)], dim=-1)
    keep = torch.cat(
        [selection.keep,
         layout.kv_ok[n_video:].expand(heads, num_rows, n_tiles - n_video)],
        dim=-1)
    # Veda's top-k returns the tiles by score. Sorting them by tile id makes
    # the gathered key order the same as the dense one, which is what keeps
    # the sparse result bitwise equal to dense attention under the mask.
    order = torch.argsort(index, dim=-1, stable=True)
    index = index.gather(-1, order)
    keep = keep.gather(-1, order)
    if share_heads and heads > 1 and bool((index == index[:1]).all()
                                          & (keep == keep[:1]).all()):
        index, keep = index[0], keep[0]
    return sparse_attention.SparsePlan(
        index=interop.from_torch(index.to(torch.int32)),
        q_block=tiling.TILE_SIZE, k_block=tiling.TILE_SIZE,
        keep=interop.from_torch(keep),
        key_valid=interop.from_torch(layout.slot_valid.to(torch.bool)),
        dense_rows=layout.n_global_tiles * tiling.TILE_SIZE)
