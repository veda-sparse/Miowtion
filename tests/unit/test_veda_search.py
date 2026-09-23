"""Tests for miowtion.veda.search."""

import pathlib

import pytest
import torch
import yaml

from miowtion.h3 import geometry
from miowtion.h3 import layout as h3_layout
from miowtion.train import data
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import search
from miowtion.veda import tiling


def _clip(ratio):
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)  # (12,8,16)
    lay = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=ratio))
    return lay, veda_attention.ClipTiling(lay, config, torch.device('cpu'))


def _qkv(seq_len, heads=2, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, heads, dim, generator=g).to(torch.bfloat16)
            for _ in range(3)]


def test_keep_all_has_zero_error():
    lay, clip = _clip(1.0)
    q, k, v = _qkv(lay.seq_len)
    tl = clip.get(tiling.TileShape(4, 4, 8))
    rows = torch.arange(tl.n_video_tiles)
    err = search.oracle_rel_mse(q, k, v, tl, clip.blocks(tl), rows)
    assert torch.all(err < 1e-10)


def test_error_grows_with_sparsity():
    q = None
    errs = []
    for ratio in (0.8, 0.3, 0.1):
        lay, clip = _clip(ratio)
        if q is None:
            q, k, v = _qkv(lay.seq_len)
        tl = clip.get(tiling.TileShape(4, 4, 8))
        rows = torch.arange(tl.n_video_tiles)
        errs.append(search.oracle_rel_mse(q, k, v, tl, clip.blocks(tl),
                                          rows).mean().item())
    assert 0 < errs[0] < errs[1] < errs[2]


def test_scorer_is_dense_and_fills_tables():
    lay, _ = _clip(0.3)
    q, k, v = _qkv(lay.seq_len)
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    scorer = search.OracleScorer(
        lay, shapes, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=0.3)),
        query_tiles=8, seed=0, device=torch.device('cpu'),
        dense_backend='math')
    out = scorer(q, k, v, layer_index=3)
    from miowtion.h3 import attention as h3_attention
    ref = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    assert torch.equal(out, ref)
    assert scorer.scores[3].shape == (2, 2)
    assert torch.isfinite(scorer.scores[3]).all()


def _entries(tables_by_layer):
    return [search.ScoreEntry('c0', step, layer, table)
            for layer, tables in enumerate(tables_by_layer)
            for step, table in enumerate(tables)]


def test_build_plan_votes_and_limits_shapes():
    grid = (12, 8, 16)
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8),
              tiling.TileShape(8, 4, 4), tiling.TileShape(1, 8, 16)]
    # Layer 0: heads prefer shapes 0,0,1,2 -> shape 2 is dropped and head 3
    # moves to whichever of {0, 1} it scores better on (1).
    t0 = torch.tensor([[0.1, 0.1, 0.5, 0.4],
                       [0.3, 0.3, 0.1, 0.3],
                       [0.4, 0.4, 0.4, 0.2],
                       [0.9, 0.9, 0.9, 0.9]])
    plan = search.build_plan('g', grid, shapes, _entries([[t0, t0]]),
                             num_layers=1)
    picked = [str(plan.shapes[i]) for i in plan.head_shape[0]]
    assert picked == ['4x4x8', '4x4x8', '2x8x8', '2x8x8']
    assert plan.provenance['entries'] == 2
    assert plan.provenance['plan_mse'] == pytest.approx(
        (0.1 + 0.1 + 0.1 + 0.3) / 4)


def test_build_plan_padding_filter():
    grid = (12, 8, 16)
    shapes = [tiling.TileShape(8, 8, 2), tiling.TileShape(4, 4, 8)]
    assert shapes[0].padding_ratio(grid) > 0.2
    table = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    plan = search.build_plan('g', grid, shapes, _entries([[table]]), 1)
    assert [str(s) for s in plan.shapes] == ['4x4x8']


def test_entries_require_completion_marker(tmp_path):
    shapes = [tiling.TileShape(4, 4, 8)]
    path = str(tmp_path / 'scores.json')
    entries = [search.ScoreEntry('c', 0, 0, torch.ones(1, 2))]
    search.save_entries(path, entries, shapes, {'keep_ratio': 0.1})
    cands, loaded, meta = search.load_entries([path])
    assert cands == shapes and meta['keep_ratio'] == 0.1
    assert torch.equal(loaded[0].table, entries[0].table)
    (tmp_path / 'scores.json.done').unlink()
    with pytest.raises(FileNotFoundError):
        search.load_entries([path])


def test_search_configs_parse():
    root = pathlib.Path(__file__).resolve().parents[2] / 'configs'
    paths = sorted(root.glob('search_*.yaml'))
    assert paths
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        config = search.SearchConfig(**raw)
        assert config.geometries, path
        for spec in config.geometries:
            data.parse_geometry(spec)
        for shape in config.candidates:
            tiling.TileShape.parse(shape)
