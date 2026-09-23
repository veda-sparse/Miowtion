"""Tests for miowtion.veda.tiling and miowtion.veda.plan."""

import pytest
import torch

from miowtion.h3 import geometry
from miowtion.veda import plan as veda_plan
from miowtion.veda import tiling

_GRID = (37, 24, 42)


def test_shape_menu_and_padding():
    assert len(tiling.all_shapes()) == 36
    assert tiling.TileShape(1, 8, 16).num_tiles(_GRID) == 333
    assert tiling.TileShape(8, 4, 4).num_tiles(_GRID) == 330
    assert tiling.TileShape(8, 4, 4).padded_grid(_GRID) == (40, 24, 44)
    assert tiling.least_padding_shape((1, 64, 112)).t == 1
    with pytest.raises(ValueError):
        tiling.TileShape(2, 2, 2)


def _layout(shape='4x8x4', text=100, cond=None):
    spans = []
    start = text
    if cond is not None:
        spans.append(tiling.TiledSpan(start, cond,
                                      tiling.least_padding_shape(cond)))
        start += cond[0] * cond[1] * cond[2]
    audio = 50
    target = tiling.TiledSpan(start + audio, (7, 6, 10),
                              tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout(spans + [target], used, used + 30)


def test_permutation_covers_every_real_row_once():
    lay = _layout()
    real = lay.perm[lay.perm >= 0]
    used = real.max().item() + 1
    assert torch.equal(real.sort().values, torch.arange(used))


def test_valid_rows_form_a_prefix():
    lay = _layout('8x2x8')
    tiles = lay.perm.view(-1, tiling.TILE_SIZE)
    for tile, count in zip(tiles, lay.valid_count):
        assert (tile[:count] >= 0).all() and (tile[count:] < 0).all()
    assert (lay.valid_count[lay.partial_tiles] < 128).all()


def test_tile_contents_are_3d_boxes():
    shape = tiling.TileShape(2, 4, 16)
    span = tiling.TiledSpan(0, (4, 8, 16), shape)
    tiles = tiling.span_tiles(span)
    # Tile order is (h-block, w-block, t-block); first tile = t 0-1, h 0-3.
    t, h, w = torch.unravel_index(tiles[0], (4, 8, 16))
    assert set(t.tolist()) == {0, 1} and set(h.tolist()) == {0, 1, 2, 3}
    t, h, w = torch.unravel_index(tiles[1], (4, 8, 16))
    assert set(t.tolist()) == {2, 3} and set(h.tolist()) == {0, 1, 2, 3}
    del w


def test_global_tiles_follow_video_and_references_come_first():
    lay = _layout(cond=(1, 6, 10))
    assert lay.n_ref_tiles == tiling.least_padding_shape(
        (1, 6, 10)).num_tiles((1, 6, 10))
    video = lay.perm[:lay.n_video_tiles * 128]
    glob = lay.perm[lay.n_video_tiles * 128:]
    assert (glob[glob >= 0] < 100 + 60 + 50).all()
    assert (video[video >= 0] >= 100).all()
    assert lay.ref_tokens == 60 and lay.target_tokens == 420


def test_gather_scatter_roundtrip():
    lay = _layout('2x8x8')
    seq_len = lay.seq_len
    x = torch.randn(seq_len, 4, 8)
    heads = torch.tensor([1, 3])
    tiled = tiling.gather_tiles(x, lay, heads)
    assert tiled.shape == (lay.num_slots, 2, 8)
    assert (tiled[lay.pad_slots] == 0).all()
    out = torch.zeros(seq_len + 1, 4, 8)
    tiling.scatter_tiles_(out, tiled, lay, heads)
    real = lay.perm[lay.perm >= 0]
    assert torch.equal(out[real][:, heads], x[real][:, heads])
    assert (out[:seq_len, 0] == 0).all()


def test_plan_limits_and_groups(tmp_path):
    geo = geometry.resolve_geometry('16:9', 5.0)
    shapes = [tiling.TileShape(4, 8, 4), tiling.TileShape(8, 8, 2),
              tiling.TileShape(2, 8, 8)]
    with pytest.raises(ValueError):
        veda_plan.TilePlan(geo.name, geo.video_grid, shapes, [[0, 1, 2, 0]])
    plan = veda_plan.TilePlan(geo.name, geo.video_grid, shapes,
                              [[0, 1, 1, 0], [2, 2, 2, 2]])
    groups = plan.head_groups(0, 'cpu')
    assert [g.heads.tolist() for g in groups] == [[0, 3], [1, 2]]
    path = tmp_path / 'p.json'
    plan.save(str(path))
    assert veda_plan.TilePlan.load(str(path)).to_json() == plan.to_json()


def test_plan_table_selection():
    def make(aspect, latent_t):
        geo = geometry.geometry_from_latent_t(aspect, latent_t)
        return veda_plan.TilePlan.uniform(geo, tiling.TileShape(4, 8, 4), 1, 2)

    table = veda_plan.PlanTable([make('16:9', 37), make('16:9', 102),
                                 make('9:16', 37)])
    pick = lambda a, t: table.select(geometry.geometry_from_latent_t(a, t))
    assert pick('16:9', 37).geometry == '16x9_t37'
    assert pick('16:9', 67).geometry == '16x9_t37'   # 9.458 s -> short
    assert pick('16:9', 72).geometry == '16x9_t102'  # 10.167 s -> long
    assert pick('9:16', 102).geometry == '9x16_t37'  # never re-transposed
    with pytest.raises(KeyError):
        pick('4:3', 37)
