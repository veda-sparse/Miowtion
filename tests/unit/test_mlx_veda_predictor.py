"""Tests for miowtion.mlx.veda_predictor (Veda's tile scorer on MLX)."""

import math

import pytest
import torch

from miowtion.veda import mask as veda_mask
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import interop  # noqa: E402
from miowtion.mlx import veda_plan  # noqa: E402
from miowtion.mlx import veda_predictor as mlx_predictor  # noqa: E402

_HEADS = 3
_HEAD_DIM = 8


def _layout(cond=(2, 4, 4), text=70, shape='8x4x4'):
    """A layout with conditions, global rows and partial tiles."""
    spans, start = [], text
    if cond is not None:
        spans.append(tiling.TiledSpan(start, cond,
                                      tiling.least_padding_shape(cond)))
        start += math.prod(cond)
    target = tiling.TiledSpan(start + 30, (5, 6, 8),
                              tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout(spans + [target], used, used)


def _rows(seq_len, heads=_HEADS, dim=_HEAD_DIM, seed=0):
    """[S, H, D] bf16 packed activations, as torch and as MLX."""
    torch.manual_seed(seed)
    x = torch.randn(seq_len, heads, dim).to(torch.bfloat16)
    return x, interop.from_torch(x)


def _geometry(layout):
    return veda_plan.tile_geometry(layout)


def test_gather_tiles_matches_the_torch_permutation():
    # Pure data movement, so it has to be bitwise equal.
    layout = _layout()
    x, xm = _rows(layout.seq_len)
    heads = torch.arange(_HEADS)
    want = tiling.gather_tiles(x, layout, heads)
    got = mlx_predictor.gather_tiles(xm, _geometry(layout))
    assert got.shape == (layout.n_tiles, tiling.TILE_SIZE, _HEADS, _HEAD_DIM)
    assert torch.equal(interop.to_torch(got).reshape(want.shape), want)


def test_pool_tiles_matches_the_torch_predictor():
    # max / min are order-free and must be bitwise equal; the mean is a
    # reduction whose accumulation order differs between the backends.
    layout = _layout()
    x, xm = _rows(layout.seq_len, seed=1)
    heads = torch.arange(_HEADS)
    want = veda_predictor.pool_tiles(tiling.gather_tiles(x, layout, heads),
                                     layout)
    got = interop.to_torch(mlx_predictor.pool_tiles(
        mlx_predictor.gather_tiles(xm, _geometry(layout)), _geometry(layout)))
    assert got.shape == want.shape
    assert torch.equal(got[..., _HEAD_DIM:], want[..., _HEAD_DIM:])
    assert torch.allclose(got[..., :_HEAD_DIM], want[..., :_HEAD_DIM],
                          rtol=1e-6, atol=1e-6)
    # Empty tiles stay exactly zero (an unmasked max would leave -inf).
    empty = ~layout.kv_ok
    if empty.any():
        assert torch.equal(got[:, empty], torch.zeros_like(got[:, empty]))


def test_logits_match_the_torch_predictor_with_and_without_projections():
    layout = _layout(shape='4x8x4')
    q, qm = _rows(layout.seq_len, seed=2)
    k, km = _rows(layout.seq_len, seed=3)
    heads = torch.arange(_HEADS)
    geom = _geometry(layout)
    feats_q = veda_predictor.pool_tiles(tiling.gather_tiles(q, layout, heads),
                                        layout)
    feats_k = veda_predictor.pool_tiles(tiling.gather_tiles(k, layout, heads),
                                        layout)

    torch.manual_seed(4)
    layer = veda_predictor.LayerPredictor(_HEADS, _HEAD_DIM)
    for proj_q, proj_k in ((None, None),
                           (layer.proj_q.detach(), layer.proj_k.detach())):
        if proj_q is None:
            mean_q = feats_q[..., :_HEAD_DIM]
            mean_k = feats_k[..., :_HEAD_DIM]
            want = mean_q @ mean_k.transpose(1, 2) / math.sqrt(_HEAD_DIM)
            projections = (None, None)
        else:
            want = layer(feats_q, feats_k, heads)
            projections = (interop.from_torch(proj_q.contiguous()),
                           interop.from_torch(proj_k.contiguous()))
        got = interop.to_torch(mlx_predictor.score_tiles(qm, km, geom,
                                                         *projections))
        assert got.shape == (_HEADS, layout.n_video_tiles, layout.n_tiles)
        assert torch.allclose(got, want[:, :layout.n_video_tiles], rtol=1e-5,
                              atol=1e-5)


def _clip(ratio=0.25):
    """A small real clip: geometry, packed layout and tiling cache."""
    from miowtion.h3 import geometry as h3_geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import attention as veda_attention

    geo = h3_geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    packed = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    clip = veda_attention.ClipTiling(
        packed, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=ratio)), torch.device('cpu'))
    return geo, packed, clip


def test_planner_selects_what_the_torch_scorer_selects():
    # The end of the chain: pooled q / k -> logits -> top-k -> plan. The
    # selection is what the kernel runs, so it must match the torch path
    # tile for tile, not just approximately.
    from miowtion.veda import plan as veda_tile_plan

    geo, packed, clip = _clip()
    shape = tiling.TileShape(4, 4, 8)
    tile_plan = veda_tile_plan.TilePlan(geo.name, geo.video_grid, [shape],
                                        [[0, 0]], {})
    layout = clip.get(shape)
    q, qm = _rows(packed.seq_len, heads=2, dim=128, seed=5)
    k, km = _rows(packed.seq_len, heads=2, dim=128, seed=6)

    planner = veda_plan.ActivationPlanner(tile_plan, clip, 0)
    layer = planner(qm, km, 0)
    assert layer.num_heads == 2
    assert [g.heads for g in layer.groups] == [(0, 1)]

    heads = torch.arange(2)
    feats_q = veda_predictor.pool_tiles(tiling.gather_tiles(q, layout, heads),
                                        layout)
    feats_k = veda_predictor.pool_tiles(tiling.gather_tiles(k, layout, heads),
                                        layout)
    scores = (feats_q[:, :layout.n_video_tiles, :128]
              @ feats_k[..., :128].transpose(1, 2)) / math.sqrt(128)
    selection = veda_mask.select_video_blocks(scores, layout,
                                              clip.blocks(layout))
    want = veda_plan.plan_from_selection(selection, layout)
    assert torch.equal(interop.to_torch(layer.groups[0].plan.index),
                       interop.to_torch(want.index))
    assert torch.equal(interop.to_torch(layer.groups[0].plan.keep),
                       interop.to_torch(want.keep))


def test_planner_covers_only_the_heads_it_is_given():
    # The trunk plans one head chunk at a time; the returned plan must
    # describe exactly those heads, renumbered from 0, and every head must
    # keep the selection it gets when the whole layer is planned at once.
    from miowtion.veda import plan as veda_tile_plan

    geo, packed, clip = _clip()
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    tile_plan = veda_tile_plan.TilePlan(geo.name, geo.video_grid, shapes,
                                        [[0, 1, 1, 0]], {})
    _, qm = _rows(packed.seq_len, heads=4, dim=128, seed=7)
    _, km = _rows(packed.seq_len, heads=4, dim=128, seed=8)
    planner = veda_plan.ActivationPlanner(tile_plan, clip, 0,
                                          share_heads=False)
    whole = planner(qm, km, 0)
    assert [g.heads for g in whole.groups] == [(0, 3), (1, 2)]
    want = {head: (group.plan, position)
            for group in whole.groups
            for position, head in enumerate(group.heads)}

    for start in range(0, 4, 2):
        chunk = planner(qm[:, start:start + 2], km[:, start:start + 2], start)
        assert chunk.num_heads == 2
        assert sorted(h for g in chunk.groups for h in g.heads) == [0, 1]
        for group in chunk.groups:
            for position, head in enumerate(group.heads):
                plan, source = want[head + start]
                assert mx.array_equal(
                    group.plan.index[position], plan.index[source]).item()
                assert mx.array_equal(
                    group.plan.keep[position], plan.keep[source]).item()
