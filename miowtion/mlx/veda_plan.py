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

from collections.abc import Callable, Sequence

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.mlx import interop
from miowtion.mlx import sparse_attention
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_tile_plan
from miowtion.veda import tiling
from miowtion.utils import progress


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


def head_group_plan(heads: Sequence[int], selection: veda_mask.Selection,
                    layout: tiling.TileLayout, share_heads: bool = True
                    ) -> sparse_attention.HeadGroupPlan:
    """Builds one head group's permutation and plan.

    Args:
        heads: Global head ids of the group, in the order the selection's
            head axis uses.
        selection: The group's selection (H' = len(heads) heads).
        layout: The tile layout of the group's tile shape.
        share_heads: See `plan_from_selection`.

    Returns:
        The head group plan, which permutes packed rows into tile order and
        back around the sparse attention.

    Raises:
        ValueError: If the selection's head count does not match `heads`, or
            if some real row is not covered by a tile.
    """
    if selection.index.shape[0] != len(heads):
        raise ValueError(f'selection has {selection.index.shape[0]} heads, '
                         f'expected {len(heads)}')
    plan = plan_from_selection(selection, layout, share_heads)
    num_slots = layout.num_slots
    # Rows that no tile covers would read from the zero row appended to the
    # output; with build_tile_layout every real row is tiled, so treat a gap
    # as a bug in the caller's layout rather than silently zeroing it.
    scatter = torch.full((layout.used,), num_slots, dtype=torch.int32)
    slots = torch.nonzero(layout.slot_valid.to(torch.bool)).view(-1)
    scatter[layout.perm[slots]] = slots.to(torch.int32)
    if bool((scatter == num_slots).any()):
        raise ValueError('the tile layout leaves real rows untiled')
    return sparse_attention.HeadGroupPlan(
        heads=tuple(int(h) for h in heads),
        gather=interop.from_torch(layout.gather_index.to(torch.int32)),
        scatter=interop.from_torch(scatter),
        plan=plan)


def layer_plan(groups: Sequence[sparse_attention.HeadGroupPlan],
               num_heads: int) -> sparse_attention.LayerPlan:
    """The plan of one layer; `groups` must partition the layer's heads."""
    return sparse_attention.LayerPlan(tuple(groups), num_heads)


def layer_plan_from_scores(tile_plan: veda_tile_plan.TilePlan,
                           clip: veda_attention.ClipTiling, layer: int,
                           scores: Callable[[tiling.TileLayout, torch.Tensor],
                                            torch.Tensor],
                           share_heads: bool = True
                           ) -> sparse_attention.LayerPlan:
    """Builds one layer's plan from a tile plan and a tile scorer.

    This is the same sequence `veda.attention.SparseStudent` runs -- one
    tile layout per head group, the predictor's logits, then the top-k per
    column block -- but it stops at the selection and hands it to MLX
    instead of to the FA4 kernel. The scoring itself stays on the torch
    side, so the predictor does not have to be ported to run the block.

    Args:
        tile_plan: The geometry's tile plan.
        clip: Tile layouts of the clip (one per tile shape).
        layer: Trunk layer index.
        scores: Called with a head group's tile layout and its head ids;
            returns [H', rows, n_tiles] logits with at least
            `n_video_tiles` rows (extra rows are ignored, as in
            SparseStudent).
        share_heads: See `plan_from_selection`.

    Returns:
        The layer's plan, one head group at a time.
    """
    groups = []
    for group in tile_plan.head_groups(layer, clip.device):
        layout = clip.get(group.shape)
        logits = scores(layout, group.heads)
        selection = veda_mask.select_video_blocks(
            logits[:, :layout.n_video_tiles], layout, clip.blocks(layout))
        groups.append(head_group_plan(group.heads.tolist(), selection,
                                      layout, share_heads))
    return layer_plan(groups, len(tile_plan.head_shape[layer]))


def uniform_tile_plan(geometry: h3_geometry.Geometry, num_layers: int,
                      num_heads: int,
                      shape: tiling.TileShape | None = None
                      ) -> veda_tile_plan.TilePlan:
    """The bootstrap plan: one least-padding tile shape for every head.

    A searched plan (`plans/`) gives each layer up to two shapes; until one
    exists for a geometry, the least-padding shape is what the search itself
    starts from, so it is the honest baseline for a speed measurement.

    Args:
        geometry: The clip's geometry.
        num_layers: Trunk layers.
        num_heads: Attention heads per layer.
        shape: Tile shape; None picks the least-padding one for the grid.

    Returns:
        The uniform tile plan.
    """
    if shape is None:
        shape = tiling.least_padding_shape(geometry.video_grid)
    return veda_tile_plan.TilePlan.uniform(geometry, shape, num_layers,
                                           num_heads)


def random_scores(seed: int = 0
                  ) -> Callable[[tiling.TileLayout, torch.Tensor],
                                torch.Tensor]:
    """A stand-in tile scorer drawing uniform logits.

    **Speed only.** The selection it produces has the shape, budget and
    per-head spread of a real Veda selection, so the kernel does exactly the
    work the trained predictor would cause, but the tiles it keeps are
    arbitrary: a clip generated with it says nothing about quality. Use it
    to measure the sparse path where no predictor checkpoint is at hand, and
    label the result as such.

    Args:
        seed: Seed of the generator; each call advances it, so different
            layers and head groups get different scores.

    Returns:
        A scorer for `layer_plan_from_scores`.
    """
    generator = torch.Generator().manual_seed(seed)

    def scores(layout: tiling.TileLayout, heads: torch.Tensor) -> torch.Tensor:
        return torch.rand(len(heads), layout.n_video_tiles, layout.n_tiles,
                          generator=generator)

    return scores


def clip_plans(tile_plan: veda_tile_plan.TilePlan,
               clip: veda_attention.ClipTiling, num_layers: int,
               scores: Callable[[tiling.TileLayout, torch.Tensor],
                                torch.Tensor],
               dense_layers: frozenset[int] = frozenset(),
               share_heads: bool = True
               ) -> list[sparse_attention.LayerPlan | None]:
    """Every trunk layer's plan, or None where the layer stays dense.

    Built once per clip: the selection depends on the tile layout and the
    scorer, not on the denoise step, so the 50 plans are reused by every
    step (the trunk weights are not -- they are streamed again).

    Args:
        tile_plan: The geometry's tile plan; must cover `num_layers`.
        clip: Tile layouts of the clip.
        num_layers: Trunk layers.
        scores: Tile scorer, see `layer_plan_from_scores`.
        dense_layers: Layers that keep dense attention.
        share_heads: See `plan_from_selection`.

    Returns:
        A list of `num_layers` plans, None for the dense layers.

    Raises:
        ValueError: If the tile plan covers fewer layers than the trunk.
    """
    if tile_plan.num_layers < num_layers:
        raise ValueError(f'tile plan covers {tile_plan.num_layers} layers, '
                         f'expected at least {num_layers}')
    bar = progress.Progress('veda plans', num_layers, every=10)
    plans: list[sparse_attention.LayerPlan | None] = []
    for layer in range(num_layers):
        if layer in dense_layers:
            plans.append(None)
        else:
            plans.append(layer_plan_from_scores(tile_plan, clip, layer,
                                                scores, share_heads))
        bar.update()
    return plans
