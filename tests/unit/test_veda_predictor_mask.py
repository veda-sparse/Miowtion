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


def _select_with_nonzero(scores, layout, blocks, rows):
    """The selection as first written: diagonal via torch.nonzero indexing."""
    heads, num_rows = scores.shape[0], rows.numel()
    indices, keeps = [], []
    for block in blocks:
        n_cols = block.stop - block.start
        s = scores[:, :, block.start:block.stop].float().clone()
        s.masked_fill_(~layout.kv_ok[None, None, block.start:block.stop],
                       float('-inf'))
        own = torch.nonzero((rows >= block.start)
                            & (rows < block.stop)).view(-1)
        s[:, own, rows[own] - block.start] = float('inf')
        budget = block.budget.per_row(block.real_tokens, n_cols)
        k_lo, k_hi, frac = veda_mask.split_budget(budget, n_cols)
        vals, idx = torch.topk(s, k_hi, dim=-1, sorted=True)
        extra = veda_mask.bresenham_extra(layout.n_video_tiles, frac,
                                          'cpu').index_select(0, rows)
        allowed = k_lo + extra.to(torch.long)
        keep = (torch.arange(k_hi)[None, None, :]
                < allowed[None, :, None]) & (vals > float('-inf'))
        indices.append(idx + block.start)
        keeps.append(keep.expand(heads, num_rows, k_hi))
    return torch.cat(indices, -1), torch.cat(keeps, -1)


def test_diagonal_without_nonzero_is_bitwise_the_original():
    lay = _layout(target_grid=(16, 8, 16), cond=(1, 16, 32))
    n_video = lay.n_video_tiles
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.25),
                                     veda_mask.Budget(tiles=2))
    torch.manual_seed(0)
    # Coarse scores force ties, so topk order is part of the comparison.
    scores = torch.randint(0, 4, (3, n_video, lay.n_tiles)).float()
    for rows in (torch.arange(n_video),
                 torch.tensor([0, lay.n_ref_tiles - 1, lay.n_ref_tiles,
                               n_video - 1])):
        sel = veda_mask.select_video_blocks(scores[:, rows], lay, blocks,
                                            rows)
        index, keep = _select_with_nonzero(scores[:, rows], lay, blocks,
                                           rows)
        assert torch.equal(sel.index, index)
        assert torch.equal(sel.keep, keep)


# --- second-order head ---------------------------------------------------


def test_second_moment_pooling_matches_naive_masked_stats():
    lay = _layout()
    x = torch.randn(lay.seq_len, 3, 16).to(torch.bfloat16)
    tiles = tiling.gather_tiles(x, lay, None)
    raw = veda_predictor.pool_tiles(tiles, lay, veda_predictor.SECOND_RAW)
    central = veda_predictor.pool_tiles(tiles, lay,
                                        veda_predictor.SECOND_CENTRAL)
    assert raw.shape == central.shape == (3, lay.n_tiles, 16)
    view = tiles.view(lay.n_tiles, 128, 3, 16).float()
    for j in range(lay.n_tiles):
        rows = view[j, :lay.valid_count[j].item()]
        torch.testing.assert_close(raw[:, j], rows.square().mean(0),
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(central[:, j], rows.var(0, unbiased=False),
                                   rtol=1e-5, atol=1e-5)


def test_pooling_rejects_an_unknown_second_moment():
    lay = _layout()
    tiles = torch.zeros(lay.num_slots, 1, 8).to(torch.bfloat16)
    with pytest.raises(ValueError, match='unknown second moment'):
        veda_predictor.pool_tiles(tiles, lay, 'stddev')


def test_rank_zero_second_order_is_the_old_predictor():
    """The default must stay bit-for-bit what it was."""
    lay = _layout()
    torch.manual_seed(3)
    old = veda_predictor.TileScorePredictor(2, 4, 16)
    torch.manual_seed(3)
    new = veda_predictor.TileScorePredictor(2, 4, 16, second_order_rank=0)
    q = torch.randn(lay.seq_len, 4, 16).to(torch.bfloat16)
    k = torch.randn(lay.seq_len, 4, 16).to(torch.bfloat16)
    heads = torch.tensor([0, 2])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    assert torch.equal(old.scores(1, qt, kt, lay, heads),
                       new.scores(1, qt, kt, lay, heads))
    assert not any(n.startswith('so_') for n, _ in old.named_parameters())


def test_exact_warm_start_equals_the_diagonal_cumulant():
    """At full rank the warm start reproduces the term in closed form.

    log mass has the expansion
        log E = log B + Qbar . Kbar / sqrt(D) + E[q^2] . Var(k) / (2 D),
    and the predictor must land on its last two terms exactly, with the
    base projections zeroed so only the pooled means survive.
    """
    dim = 16
    lay = _layout()
    pred = veda_predictor.TileScorePredictor(2, 4, dim,
                                             second_order_rank=dim)
    pred.init_exact_second_order_()
    with torch.no_grad():
        for layer in pred.layers:
            layer.proj_q.zero_()
            layer.proj_k.zero_()
    q = torch.randn(lay.seq_len, 4, dim).to(torch.bfloat16)
    k = torch.randn(lay.seq_len, 4, dim).to(torch.bfloat16)
    heads = torch.tensor([1, 3])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    logits = pred.scores(0, qt, kt, lay, heads)

    mean_q = veda_predictor.pool_tiles(qt, lay)[..., :dim]
    mean_k = veda_predictor.pool_tiles(kt, lay)[..., :dim]
    sq_q = veda_predictor.pool_tiles(qt, lay, veda_predictor.SECOND_RAW)
    var_k = veda_predictor.pool_tiles(kt, lay, veda_predictor.SECOND_CENTRAL)
    want = (mean_q @ mean_k.transpose(1, 2) / math.sqrt(dim)
            + sq_q @ var_k.transpose(1, 2) / (2 * dim))
    torch.testing.assert_close(logits, want, rtol=1e-4, atol=1e-5)


def test_low_rank_head_costs_only_its_rank():
    dim, rank = 16, 4
    pred = veda_predictor.TileScorePredictor(2, 4, dim,
                                             second_order_rank=rank)
    base = 2 * 2 * 4 * 3 * dim * dim
    assert sum(p.numel() for p in pred.parameters()) == \
        base + 2 * 2 * 4 * dim * rank


def test_second_order_rank_is_range_checked():
    with pytest.raises(ValueError, match='must be'):
        veda_predictor.LayerPredictor(2, 16, second_order_rank=17)
    with pytest.raises(ValueError, match='must be'):
        veda_predictor.LayerPredictor(2, 16, second_order_rank=-1)


def test_exact_warm_start_needs_full_rank():
    pred = veda_predictor.LayerPredictor(2, 16, second_order_rank=8)
    with pytest.raises(ValueError, match='second_order_rank == '):
        pred.init_exact_second_order_()


def test_features_and_rank_must_agree():
    lay = _layout()
    layer = veda_predictor.LayerPredictor(4, 16, second_order_rank=4)
    feats = torch.zeros(1, lay.n_tiles, 48)
    with pytest.raises(ValueError, match='must be set together'):
        layer(feats, feats, torch.tensor([0]))


def test_variance_feature_survives_a_tight_tile():
    """Nearly aligned rows are the case the one-pass identity cannot do.

    `E[x^2] - E[x]^2` on rows this close loses almost every digit, so the
    two-pass form is not a nicety. The variance must stay non-negative and
    still match a direct computation.
    """
    lay = _layout()
    base = torch.randn(1, 2, 16)
    x = (base + 1e-4 * torch.randn(lay.seq_len, 2, 16)).to(torch.bfloat16)
    tiles = tiling.gather_tiles(x, lay, None)
    central = veda_predictor.pool_tiles(tiles, lay,
                                        veda_predictor.SECOND_CENTRAL)
    assert torch.all(central >= 0.0)
    assert torch.isfinite(central).all()
    view = tiles.view(lay.n_tiles, 128, 2, 16).float()
    for j in range(lay.n_tiles):
        rows = view[j, :lay.valid_count[j].item()]
        torch.testing.assert_close(central[:, j],
                                   rows.var(0, unbiased=False),
                                   rtol=1e-4, atol=1e-7)


def test_count_term_adds_exactly_log_b_at_init():
    """A per-key-tile constant with a gain that starts at 1."""
    dim = 16
    lay = _layout()
    plain = veda_predictor.TileScorePredictor(2, 4, dim)
    counted = veda_predictor.TileScorePredictor(2, 4, dim, count_term=True)
    with torch.no_grad():
        for src, dst in zip(plain.layers, counted.layers):
            dst.proj_q.copy_(src.proj_q)
            dst.proj_k.copy_(src.proj_k)
    q = torch.randn(lay.seq_len, 4, dim).to(torch.bfloat16)
    k = torch.randn(lay.seq_len, 4, dim).to(torch.bfloat16)
    heads = torch.tensor([0, 2])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    delta = (counted.scores(0, qt, kt, lay, heads)
             - plain.scores(0, qt, kt, lay, heads))
    want = torch.log(lay.valid_count.clamp(min=1).float())
    torch.testing.assert_close(delta, want.expand_as(delta), rtol=1e-5,
                               atol=1e-5)


def test_count_term_ranks_a_fuller_key_tile_higher():
    """The behaviour it exists for: a padded tile carries less mass."""
    lay = _layout()
    partial = int(lay.partial_tiles[0]) if lay.partial_tiles.numel() else None
    assert partial is not None, 'this layout should have a partial tile'
    full = next(j for j in range(lay.n_tiles)
                if bool(lay.full_tile[j]) and bool(lay.kv_ok[j]))
    dim = 16
    pred = veda_predictor.TileScorePredictor(1, 2, dim, count_term=True)
    with torch.no_grad():
        pred.layers[0].proj_q.zero_()
        pred.layers[0].proj_k.zero_()
    # Give every key row the same vector, so the pooled means tie and only
    # the row count can separate the two tiles.
    q = torch.randn(lay.seq_len, 2, dim).to(torch.bfloat16)
    k = torch.ones(lay.seq_len, 2, dim).to(torch.bfloat16)
    heads = torch.tensor([0])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    logits = pred.scores(0, qt, kt, lay, heads)[0]
    assert int(lay.valid_count[full]) > int(lay.valid_count[partial])
    assert torch.all(logits[:, full] > logits[:, partial])


def test_count_term_parameter_count_and_pairing():
    pred = veda_predictor.TileScorePredictor(2, 4, 16, count_term=True)
    base = 2 * 2 * 4 * 3 * 16 * 16
    assert sum(p.numel() for p in pred.parameters()) == base + 2 * 4
    layer = veda_predictor.LayerPredictor(4, 16, count_term=True)
    feats = torch.zeros(1, 3, 48)
    with pytest.raises(ValueError, match='count_term must be set'):
        layer(feats, feats, torch.tensor([0]))
    plain = veda_predictor.LayerPredictor(4, 16)
    with pytest.raises(ValueError, match='count_term must be set'):
        plain(feats, feats, torch.tensor([0]),
              log_count=torch.zeros(3))
