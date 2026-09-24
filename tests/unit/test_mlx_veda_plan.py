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


def _layout(cond=(2, 4, 4), text=70, shape='8x4x4'):
    spans, start = [], text
    if cond is not None:
        spans.append(tiling.TiledSpan(start, cond,
                                      tiling.least_padding_shape(cond)))
        start += math.prod(cond)
    target = tiling.TiledSpan(start + 30, (5, 6, 8),
                              tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout(spans + [target], used, used)


def _selection(layout, ratio=0.4, seed=0, heads=_HEADS):
    torch.manual_seed(seed)
    scores = torch.randn(heads, layout.n_video_tiles, layout.n_tiles)
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


def _reference(group, q, k, v):
    """Dense attention under the group's mask, in packed row order."""
    plan = group.plan
    slots = group.gather.shape[0]
    mask = sa.block_mask_from_index(plan.index, plan.q_block, plan.k_block,
                                    slots, plan.keep, plan.key_valid,
                                    plan.dense_rows)
    tiled = [mx.take(x, group.gather, axis=1) for x in (q, k, v)]
    heads = q.shape[0]
    out = mx.stack([sa.dense_reference(
        tiled[0][h:h + 1], tiled[1][h:h + 1], tiled[2][h:h + 1],
        mask=mask if mask.ndim == 2 else mask[h])[0] for h in range(heads)])
    zero = mx.zeros((heads, 1, q.shape[2]), dtype=out.dtype)
    return mx.take(mx.concatenate([out, zero], axis=1), group.scatter, axis=1)


def test_layer_plan_gives_every_head_group_its_own_permutation():
    # Two tile shapes in one layer: the heads split into groups that see
    # different permutations of the same packed sequence.
    layout_a, layout_b = _layout(), _layout(shape='4x8x4')
    assert layout_a.num_slots != layout_b.num_slots or not torch.equal(
        layout_a.perm, layout_b.perm)
    group_a = veda_plan.head_group_plan(
        [0, 2], _selection(layout_a, seed=2, heads=2), layout_a)
    group_b = veda_plan.head_group_plan(
        [1], _selection(layout_b, seed=3, heads=1), layout_b)
    layer = veda_plan.layer_plan([group_a, group_b], _HEADS)
    used = layout_a.used
    mx.random.seed(0)
    q, k, v = (mx.random.normal((_HEADS, used, 128)) for _ in range(3))
    mx.eval(q, k, v)

    got = sa.layer_attention(layer, q, k, v)
    pick = {group_a: mx.array([0, 2]), group_b: mx.array([1])}
    for group, heads in pick.items():
        want = _reference(group, *(mx.take(x, heads, axis=0)
                                   for x in (q, k, v)))
        mx.eval(want)
        assert mx.array_equal(mx.take(got, heads, axis=0), want).item()


def test_layer_plan_needs_a_partition_of_the_heads():
    layout = _layout()
    group = veda_plan.head_group_plan([0, 1], _selection(layout, heads=2),
                                      layout)
    with pytest.raises(ValueError):
        veda_plan.layer_plan([group], _HEADS)
    with pytest.raises(ValueError):
        veda_plan.head_group_plan([0], _selection(layout, heads=2), layout)


def test_layer_plan_from_scores_follows_the_tile_plan():
    # The same sequence SparseStudent runs, stopping at the selection: two
    # tile shapes in one layer must give two head groups, each with its own
    # layout, and the heads must come back in the plan's order.
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import plan as veda_tile_plan

    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    packed = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    clip = veda_attention.ClipTiling(
        packed, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=0.25)), torch.device('cpu'))
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    tile_plan = veda_tile_plan.TilePlan(geo.name, geo.video_grid, shapes,
                                        [[0, 1, 1, 0]], {})

    def scores(layout, heads):
        torch.manual_seed(int(heads.sum()))
        return torch.randn(len(heads), layout.n_video_tiles, layout.n_tiles)

    layer = veda_plan.layer_plan_from_scores(tile_plan, clip, 0, scores)
    assert layer.num_heads == 4
    assert [g.heads for g in layer.groups] == [(0, 3), (1, 2)]
    for group, shape in zip(layer.groups, shapes):
        assert group.gather.shape == (clip.get(shape).num_slots,)
        assert group.scatter.shape == (packed.used,)
    assert 0.0 < layer.density() < 1.0

    mx.random.seed(0)
    q, k, v = (mx.random.normal((4, packed.used, 128)) for _ in range(3))
    mx.eval(q, k, v)
    got = sa.layer_attention(layer, q, k, v)
    for group in layer.groups:
        heads = mx.array(list(group.heads))
        want = _reference(group, *(mx.take(x, heads, axis=0)
                                   for x in (q, k, v)))
        mx.eval(want)
        assert mx.array_equal(mx.take(got, heads, axis=0), want).item()
