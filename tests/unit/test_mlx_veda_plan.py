"""Tests for miowtion.mlx.veda_plan (Veda selections -> MLX plans)."""

import math

import pytest
import torch

from miowtion.veda import mask as veda_mask
from miowtion.veda import tiling

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import interop  # noqa: E402
from miowtion.mlx import sparse_attention as sa  # noqa: E402
from miowtion.mlx import veda_plan  # noqa: E402

_HEADS = 3


def _layout(cond=(2, 4, 4), text=70):
    spans, start = [], text
    if cond is not None:
        spans.append(tiling.TiledSpan(start, cond,
                                      tiling.least_padding_shape(cond)))
        start += math.prod(cond)
    target = tiling.TiledSpan(start + 30, (5, 6, 8),
                              tiling.TileShape.parse('8x4x4'))
    used = target.start + target.num_rows
    return tiling.build_tile_layout(spans + [target], used, used)


def _selection(layout, ratio=0.4, seed=0):
    torch.manual_seed(seed)
    scores = torch.randn(_HEADS, layout.n_video_tiles, layout.n_tiles)
    blocks = veda_mask.column_blocks(layout, veda_mask.Budget(ratio=ratio),
                                     veda_mask.Budget(ratio=ratio))
    return veda_mask.select_video_blocks(scores, layout, blocks)


def test_plan_reproduces_the_veda_block_mask():
    # The plan is only useful if MLX runs exactly the mask the CUDA kernels
    # get, including the dense global rows and columns and the partial
    # tiles, so compare against mask.dense_block_mask expanded to rows.
    layout = _layout()
    selection = _selection(layout)
    plan = veda_plan.plan_from_selection(selection, layout)
    assert plan.q_block == plan.k_block == tiling.TILE_SIZE
    assert plan.dense_rows == layout.n_global_tiles * tiling.TILE_SIZE

    slots = layout.num_slots
    got = sa.block_mask_from_index(plan.index, plan.q_block, plan.k_block,
                                   slots, plan.keep, plan.key_valid,
                                   plan.dense_rows)
    tiles = veda_mask.dense_block_mask(selection, layout)
    valid = layout.slot_valid.to(torch.bool)
    want = tiles.repeat_interleave(tiling.TILE_SIZE, 1).repeat_interleave(
        tiling.TILE_SIZE, 2) & valid[None, None, :]
    assert torch.equal(interop.to_torch(got), want)


def test_plan_runs_and_matches_dense_attention_under_that_mask():
    layout = _layout()
    selection = _selection(layout, seed=1)
    plan = veda_plan.plan_from_selection(selection, layout)
    slots = layout.num_slots
    mx.random.seed(0)
    # H3's head_dim; below 64 MLX picks a different SDPA kernel for the
    # narrow gathered problem and the two paths are only close, not equal.
    q, k, v = (mx.random.normal((_HEADS, slots, 128)) for _ in range(3))
    mx.eval(q, k, v)
    got = sa.block_sparse_attention(q, k, v, plan.index, q_block=plan.q_block,
                                    k_block=plan.k_block, keep=plan.keep,
                                    key_valid=plan.key_valid,
                                    dense_rows=plan.dense_rows)
    mask = sa.block_mask_from_index(plan.index, plan.q_block, plan.k_block,
                                    slots, plan.keep, plan.key_valid,
                                    plan.dense_rows)
    want = mx.stack([sa.dense_reference(q[h:h + 1], k[h:h + 1], v[h:h + 1],
                                        mask=mask[h])[0]
                     for h in range(_HEADS)])
    mx.eval(got, want)
    assert mx.array_equal(got, want).item()


def test_full_budget_collapses_the_head_axis():
    # With ratio >= 1 every head keeps every tile, so one gather can serve
    # the whole group.
    layout = _layout()
    plan = veda_plan.plan_from_selection(_selection(layout, ratio=1.0),
                                         layout)
    assert plan.index.ndim == 2
    assert plan.budget == layout.n_tiles
    plan3 = veda_plan.plan_from_selection(_selection(layout, ratio=1.0),
                                          layout, share_heads=False)
    assert plan3.index.shape == (_HEADS, layout.n_video_tiles,
                                 layout.n_tiles)


def test_plan_needs_every_video_query_tile():
    layout = _layout()
    selection = _selection(layout)
    selection.index = selection.index[:, :-1]
    selection.keep = selection.keep[:, :-1]
    with pytest.raises(ValueError):
        veda_plan.plan_from_selection(selection, layout)
