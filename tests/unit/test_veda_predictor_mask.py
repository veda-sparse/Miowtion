"""Tests for miowtion.veda.predictor and miowtion.veda.mask."""

import math

import pytest
import torch

from miowtion.veda import mask as veda_mask
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling


def _layout(target_grid=(7, 6, 10), shape='8x4x4', cond=None, text=90):
    spans, start = [], text
    if cond is not None:
        spans.append(tiling.TiledSpan(start, cond,
                                      tiling.least_padding_shape(cond)))
        start += math.prod(cond)
    target = tiling.TiledSpan(start + 40, target_grid,
                              tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout(spans + [target], used, used)


def test_pooling_matches_naive_masked_stats():
    lay = _layout()
    x = torch.randn(lay.seq_len, 3, 16).to(torch.bfloat16)
    tiles = tiling.gather_tiles(x, lay, None)
    feats = veda_predictor.pool_tiles(tiles, lay)
    assert feats.shape == (3, lay.n_tiles, 48)
    view = tiles.view(lay.n_tiles, 128, 3, 16).float()
    for j in range(lay.n_tiles):
        n = lay.valid_count[j].item()
        rows = view[j, :n]
        expected = torch.cat([rows.mean(0), rows.amax(0), rows.amin(0)], -1)
        torch.testing.assert_close(feats[:, j], expected, rtol=1e-5,
                                   atol=1e-5)


def test_pooling_empty_tile_is_zero_not_nan():
    lay = _layout()
    lay.valid_count[0] = 0
    lay.kv_ok[0] = False
    lay.partial_tiles = torch.cat([torch.tensor([0]), lay.partial_tiles])
    tiles = torch.randn(lay.num_slots, 2, 8).to(torch.bfloat16)
    feats = veda_predictor.pool_tiles(tiles, lay)
    assert torch.isfinite(feats).all() and (feats[:, 0] == 0).all()


def test_untrained_predictor_is_mean_pooled_qk():
    lay = _layout()
    pred = veda_predictor.TileScorePredictor(2, 4, 16)
    q = torch.randn(lay.seq_len, 4, 16).to(torch.bfloat16)
    k = torch.randn(lay.seq_len, 4, 16).to(torch.bfloat16)
    heads = torch.tensor([0, 2])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    logits = pred.scores(1, qt, kt, lay, heads)
    mq = veda_predictor.pool_tiles(qt, lay)[..., :16]
    mk = veda_predictor.pool_tiles(kt, lay)[..., :16]
    torch.testing.assert_close(logits, mq @ mk.transpose(1, 2) / 4.0,
                               rtol=1e-2, atol=1e-2)
    assert sum(p.numel() for p in pred.parameters()) == 2 * 2 * 4 * 48 * 16


@pytest.mark.parametrize('budget', [1.0, 2.3, 3.75, 5.5])
def test_bresenham_mean_equals_budget(budget):
    n_rows = 400
    k_lo, _, frac = veda_mask.split_budget(budget, 64)
    extra = veda_mask.bresenham_extra(n_rows, frac, 'cpu')
    mean = (k_lo + extra.double()).mean().item()
    # Bresenham keeps floor(n * frac) extras: the mean is within 1/n below.
    assert 0 <= budget - mean < 1.0 / n_rows


def test_equal_cost_budget():
    budget = veda_mask.Budget(ratio=0.1)
    # 37296 tokens -> 292 ideal tiles; a 330-tile plan keeps 0.1*292^2/330.
    assert budget.per_row(37296, 330) == pytest.approx(0.1 * 292**2 / 330)
    assert veda_mask.Budget(tiles=7).per_row(37296, 330) == 7.0
    assert veda_mask.Budget(ratio=1.0).keeps_all


def test_selection_rules():
    lay = _layout(target_grid=(16, 8, 16), shape='4x8x4')
    n_video = lay.n_video_tiles
    heads = 2
    scores = torch.randn(heads, lay.n_tiles, lay.n_tiles)
    ratio = 0.3
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=ratio))
    sel = veda_mask.select_video_blocks(scores[:, :n_video], lay, blocks)
    kept = sel.keep.sum(-1).float()
    budget = blocks[0].budget.per_row(lay.target_tokens, n_video)
    assert 0 <= budget - kept.double().mean().item() < 1.0 / n_video
    mask = veda_mask.dense_block_mask(sel, lay)
    diag = torch.arange(n_video)
    assert mask[:, diag, diag].all()  # the diagonal is always kept ...
    # ... and consumes budget: the video quadrant holds exactly `kept`.
    assert torch.equal(mask[:, :n_video, :n_video].sum(-1).float(), kept)
    assert mask[:, :, n_video:].all() and mask[:, n_video:, :].all()
    # The kept set is the top-k of the scores (diagonal aside).
    s = scores[:, :n_video, :n_video].clone()
    s[:, diag, diag] = float('inf')
    top1 = s.topk(2, -1).indices[..., 1]
    assert torch.gather(mask[:, :n_video], 2, top1[..., None]).all()


def test_segmented_blocks_have_independent_budgets():
    lay = _layout(target_grid=(16, 8, 16), cond=(1, 16, 32))
    n_ref, n_video = lay.n_ref_tiles, lay.n_video_tiles
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.25),
                                     veda_mask.Budget(tiles=1))
    assert [(b.start, b.stop) for b in blocks] == [(0, n_ref),
                                                    (n_ref, n_video)]
    scores = torch.randn(1, n_video, lay.n_tiles)
    mask = veda_mask.dense_block_mask(
        veda_mask.select_video_blocks(scores, lay, blocks), lay)[0]
    ref_kept = mask[:n_video, :n_ref].sum(-1)
    assert (ref_kept == 1).all()
    # Diagonal forced only inside its own block.
    assert mask[torch.arange(n_ref), torch.arange(n_ref)].all()
    target_rows = torch.arange(n_ref, n_video)
    assert mask[target_rows, target_rows].all()
    with pytest.raises(ValueError):
        veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.25))


def test_row_subset_follows_tile_ids():
    lay = _layout(target_grid=(16, 8, 16))
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    scores = torch.randn(1, lay.n_video_tiles, lay.n_tiles)
    full = veda_mask.select_video_blocks(scores, lay, blocks)
    rows = torch.tensor([3, 7, 11])
    sub = veda_mask.select_video_blocks(scores[:, rows], lay, blocks, rows)
    assert torch.equal(sub.index, full.index[:, rows])
    assert torch.equal(sub.keep, full.keep[:, rows])


def test_dense_block_mask_for_a_row_subset():
    lay = _layout(target_grid=(16, 8, 16))
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    scores = torch.randn(2, lay.n_tiles, lay.n_tiles)
    full = veda_mask.select_video_blocks(scores[:, :lay.n_video_tiles], lay,
                                         blocks)
    rows = torch.tensor([1, 5, lay.n_video_tiles - 1])
    sub = veda_mask.select_video_blocks(scores[:, rows], lay, blocks, rows)
    sub_mask = veda_mask.dense_block_mask(sub, lay, global_rows=False)
    assert sub_mask.shape == (2, 3, lay.n_tiles)
    assert torch.equal(sub_mask, veda_mask.dense_block_mask(full, lay)[:, rows])
    with pytest.raises(ValueError):
        veda_mask.dense_block_mask(sub, lay)
