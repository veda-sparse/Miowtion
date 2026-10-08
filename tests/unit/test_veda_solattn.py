"""Tests for miowtion.veda.solattn (Sol-Attn selection / budget ablation)."""

import math

import pytest
import torch

from miowtion.h3 import geometry
from miowtion.h3 import layout as h3_layout
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import solattn
from miowtion.veda import tiling

TILE = tiling.TILE_SIZE


def _layout(shape=tiling.TileShape(4, 4, 8)):
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=1.0))
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    return lay, clip.get(shape)


def _qkv(seq_len, heads=2, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, heads, dim, generator=g,
                        dtype=torch.float32) for _ in range(3)]


def _lse(q, k, used):
    """[S, H] fp32 dense log-sum-exp over the real packed rows."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.zeros(q.shape[0], q.shape[1])
    for h in range(q.shape[1]):
        s = (q[:used, h] @ k[:used, h].transpose(0, 1)) * scale
        out[:used, h] = torch.logsumexp(s, dim=-1)
    return out


def _out_proj(heads, dim, hidden=96, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(hidden, heads * dim, generator=g)


def _tables(head=0, seed=0, rows=None):
    lay, tl = _layout()
    q, k, v = _qkv(lay.seq_len, seed=seed)
    lse = _lse(q, k, lay.used)
    w_o = _out_proj(q.shape[1], q.shape[2])
    whiten = solattn.whitening(w_o, head, q.shape[2])
    if rows is None:
        rows = torch.arange(tl.n_video_tiles)
    tab = solattn.head_tables(q, k, v, lse, tl, rows, head, whiten)
    return tab, (q, k, v, lse, tl, rows, head, whiten, w_o)


# --- whitening -----------------------------------------------------------


def test_whitening_preserves_row_norms():
    w_o = _out_proj(2, 32)
    whiten = solattn.whitening(w_o, 1, 32)
    x = torch.randn(7, 32, generator=torch.Generator().manual_seed(1))
    direct = (x @ w_o[:, 32:64].transpose(0, 1)).norm(dim=-1)
    assert torch.allclose((x @ whiten).norm(dim=-1), direct, rtol=1e-4,
                          atol=1e-4)


def test_whitening_rejects_out_of_range_head():
    with pytest.raises(ValueError, match='exceeds'):
        solattn.whitening(_out_proj(2, 32), 2, 32)


def test_whitening_handles_rank_deficient_slice():
    w_o = torch.zeros(96, 64)
    w_o[:, :32] = torch.randn(96, 32,
                              generator=torch.Generator().manual_seed(2))
    whiten = solattn.whitening(w_o, 1, 32)  # all-zero slice
    assert torch.all(torch.isfinite(whiten))
    assert float(whiten.abs().max()) == pytest.approx(0.0, abs=1e-5)


# --- score tables --------------------------------------------------------


def test_attention_mass_sums_to_the_real_rows():
    tab, _ = _tables()
    assert torch.allclose(tab.attn_mass.sum(-1),
                          tab.valid_rows.float(), rtol=1e-4, atol=1e-3)


def test_proxy_is_nan_on_global_columns_and_finite_on_video():
    tab, _ = _tables()
    video = tab.proxy[:, :tab.n_video_tiles]
    assert torch.all(torch.isfinite(video))
    assert torch.all(torch.isnan(tab.proxy[:, tab.n_video_tiles:]))


def test_omega_matches_its_definition():
    head, rows = 1, torch.tensor([0, 5])
    tab, (q, k, v, lse, tl, _, _, whiten, w_o) = _tables(head=head,
                                                         rows=rows)
    dim = q.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    n = tl.n_tiles
    k_t = solattn._tile_order(k, tl, head)
    v_t = solattn._tile_order(v, tl, head)
    q_t = solattn._tile_order(q, tl, head)
    lse_t = solattn._tile_order(lse[:, head, None], tl, 0).reshape(-1)
    valid = tl.slot_valid.float()
    w_head = w_o[:, head * dim:(head + 1) * dim].transpose(0, 1)
    want = torch.zeros(rows.numel(), n)
    for r, tile in enumerate(rows.tolist()):
        for u in range(int(tl.valid_count[tile])):
            slot = tile * TILE + u
            p = torch.exp((q_t[slot] @ k_t.transpose(0, 1)) * scale
                          - lse_t[slot]) * valid
            pb = p.view(n, TILE)
            a = pb.sum(-1)
            n_uj = torch.einsum('nb,nbd->nd', pb, v_t.view(n, TILE, dim))
            o_u = n_uj.sum(0)
            xi = n_uj - a[:, None] * o_u[None, :]
            want[r] += (xi @ w_head).norm(dim=-1)
    assert torch.allclose(tab.omega0, want, rtol=1e-3, atol=1e-4)


def test_omega1_differs_from_omega0():
    tab, _ = _tables()
    assert not torch.allclose(tab.omega0, tab.omega1, rtol=1e-2)


def test_head_tables_rejects_global_query_rows():
    _, tl = _layout()
    with pytest.raises(ValueError, match='video query tiles'):
        solattn.head_tables(
            torch.zeros(tl.seq_len, 1, 32), torch.zeros(tl.seq_len, 1, 32),
            torch.zeros(tl.seq_len, 1, 32), torch.zeros(tl.seq_len, 1),
            tl, torch.tensor([tl.n_video_tiles]), 0, torch.eye(32))


# --- relative error ------------------------------------------------------


def _keep_all(tab):
    return tab.kv_ok[None, :].expand(tab.rows.numel(), -1).clone()


def test_keep_everything_has_zero_error():
    tab, args = _tables()
    q, k, v, lse, tl, rows, head, whiten, _ = args
    err = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  {'all': _keep_all(tab)})
    assert err[('all', 0)] < 1e-5
    assert err[('all', 1)] < 1e-5


def test_error_matches_renormalized_attention_without_compensation():
    head = 0
    tab, args = _tables(head=head, rows=torch.tensor([1, 4]))
    q, k, v, lse, tl, rows, _, whiten, w_o = args
    dim = q.shape[-1]
    budget = solattn.video_budget(0.5, tab.n_tiles,
                                  tab.n_tiles - tab.n_video_tiles)
    mask = solattn.fixed_topk_mask(tab.proxy, budget, tab.kv_ok,
                                   tab.n_video_tiles, rows)
    got = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  {'m': mask}, compensate=(0,))[('m', 0)]

    scale = 1.0 / math.sqrt(dim)
    q_t = solattn._tile_order(q, tl, head)
    k_t = solattn._tile_order(k, tl, head)
    v_t = solattn._tile_order(v, tl, head)
    w_head = w_o[:, head * dim:(head + 1) * dim].transpose(0, 1)
    num = den = 0.0
    for r, tile in enumerate(rows.tolist()):
        keep_tok = mask[r].repeat_interleave(TILE) & tl.slot_valid.bool()
        for u in range(int(tl.valid_count[tile])):
            slot = tile * TILE + u
            s = (q_t[slot] @ k_t.transpose(0, 1)) * scale
            dense = torch.softmax(
                s.masked_fill(~tl.slot_valid.bool(), -torch.inf), -1) @ v_t
            sparse = torch.softmax(
                s.masked_fill(~keep_tok, -torch.inf), -1) @ v_t
            num += float(((sparse - dense) @ w_head).square().sum())
            den += float((dense @ w_head).square().sum())
    assert got == pytest.approx(math.sqrt(num / den), rel=2e-3)


def test_error_matches_zero_order_compensation():
    head = 1
    rows = torch.tensor([2, 7])
    tab, args = _tables(head=head, rows=rows)
    q, k, v, lse, tl, _, _, whiten, w_o = args
    dim = q.shape[-1]
    budget = solattn.video_budget(0.4, tab.n_tiles,
                                  tab.n_tiles - tab.n_video_tiles)
    mask = solattn.fixed_topk_mask(tab.proxy, budget, tab.kv_ok,
                                   tab.n_video_tiles, rows)
    got = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  {'m': mask}, compensate=(1,))[('m', 1)]

    scale = 1.0 / math.sqrt(dim)
    n = tl.n_tiles
    q_t = solattn._tile_order(q, tl, head)
    k_t = solattn._tile_order(k, tl, head)
    v_t = solattn._tile_order(v, tl, head)
    counts = tl.valid_count.float().clamp(min=1.0)
    k_bar = k_t.view(n, TILE, dim).sum(1) / counts[:, None]
    v_hat = v_t.view(n, TILE, dim).sum(1)
    w_head = w_o[:, head * dim:(head + 1) * dim].transpose(0, 1)
    num = den = 0.0
    for r, tile in enumerate(rows.tolist()):
        for u in range(int(tl.valid_count[tile])):
            slot = tile * TILE + u
            s = (q_t[slot] @ k_t.transpose(0, 1)) * scale
            dense = torch.softmax(
                s.masked_fill(~tl.slot_valid.bool(), -torch.inf), -1) @ v_t
            # Shift by the row max for a stable reference sum.
            shift = float(s.masked_fill(~tl.slot_valid.bool(),
                                        -torch.inf).max())
            w = torch.exp(s - shift) * tl.slot_valid.float()
            s_hat = (q_t[slot] @ k_bar.transpose(0, 1)) * scale
            w_hat = torch.exp(s_hat - shift)
            top = torch.zeros(dim)
            bot = 0.0
            for j in range(n):
                block = slice(j * TILE, (j + 1) * TILE)
                if bool(mask[r, j]):
                    top += w[block] @ v_t[block]
                    bot += float(w[block].sum())
                elif bool(tl.kv_ok[j]):
                    top += w_hat[j] * v_hat[j]
                    bot += float(w_hat[j] * counts[j])
            sparse = top / bot
            num += float(((sparse - dense) @ w_head).square().sum())
            den += float((dense @ w_head).square().sum())
    assert got == pytest.approx(math.sqrt(num / den), rel=2e-3)


def test_error_falls_as_density_grows():
    tab, args = _tables()
    q, k, v, lse, tl, rows, head, whiten, _ = args
    errs = []
    for density in (0.2, 0.5, 0.9):
        budget = solattn.video_budget(density, tab.n_tiles,
                                      tab.n_tiles - tab.n_video_tiles)
        mask = solattn.fixed_topk_mask(tab.proxy, budget, tab.kv_ok,
                                       tab.n_video_tiles, rows)
        errs.append(solattn.relative_errors(
            q, k, v, lse, tl, rows, head, whiten, {'m': mask},
            compensate=(0,))[('m', 0)])
    assert errs[0] > errs[1] > errs[2]


def test_relative_errors_rejects_mask_of_wrong_shape():
    tab, args = _tables()
    q, k, v, lse, tl, rows, head, whiten, _ = args
    with pytest.raises(ValueError, match='expected'):
        solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                {'bad': torch.ones(2, 3, dtype=torch.bool)})


# --- allocation rules ----------------------------------------------------


def test_video_budget_accounts_for_global_tiles():
    assert solattn.video_budget(0.5, 13, 1) == pytest.approx(5.5)
    with pytest.raises(ValueError, match='density'):
        solattn.video_budget(0.1, 13, 1)


def test_fixed_topk_hits_the_average_budget_exactly():
    tab, _ = _tables()
    n_global = tab.n_tiles - tab.n_video_tiles
    budget = solattn.video_budget(0.5, tab.n_tiles, n_global)
    mask = solattn.fixed_topk_mask(tab.proxy, budget, tab.kv_ok,
                                   tab.n_video_tiles, tab.rows)
    kept = solattn.kept_per_row(mask, tab.n_video_tiles).float()
    assert float(kept.mean()) == pytest.approx(budget, abs=1e-6)
    assert torch.all(mask[:, tab.n_video_tiles:])


def test_fixed_topk_matches_a_reference_loop():
    tab, _ = _tables()
    budget = 4.0
    mask = solattn.fixed_topk_mask(tab.omega0, budget, tab.kv_ok,
                                   tab.n_video_tiles, tab.rows)
    want = torch.zeros_like(mask)
    want[:, tab.n_video_tiles:] = tab.kv_ok[None, tab.n_video_tiles:]
    for r in range(tab.rows.numel()):
        scores = tab.omega0[r, :tab.n_video_tiles].clone()
        scores[~tab.kv_ok[:tab.n_video_tiles]] = -torch.inf
        want[r, scores.topk(int(budget)).indices] = True
    assert torch.equal(mask, want)


def test_threshold_mask_respects_the_row_clamps():
    tab, _ = _tables()
    stat = solattn.zscore_statistic(tab)
    mask = solattn.threshold_mask(stat, tab.kv_ok, tab.n_video_tiles,
                                  threshold=10.0, k_min=3, k_max=5)
    kept = solattn.kept_per_row(mask, tab.n_video_tiles)
    assert torch.equal(kept, torch.full_like(kept, 3))
    mask = solattn.threshold_mask(stat, tab.kv_ok, tab.n_video_tiles,
                                  threshold=-10.0, k_min=3, k_max=5)
    kept = solattn.kept_per_row(mask, tab.n_video_tiles)
    assert torch.equal(kept, torch.full_like(kept, 5))


def test_calibrate_threshold_reaches_the_target_density():
    tab, _ = _tables()
    stat = solattn.zscore_statistic(tab)
    for budget in (3.0, 6.0):
        thr = solattn.calibrate_threshold(stat, tab.kv_ok,
                                          tab.n_video_tiles, budget)
        mask = solattn.threshold_mask(stat, tab.kv_ok, tab.n_video_tiles,
                                      thr)
        kept = solattn.kept_per_row(mask, tab.n_video_tiles).float()
        assert float(kept.mean()) == pytest.approx(budget, abs=1e-6)


def test_zscore_rows_are_standardized():
    tab, _ = _tables()
    z = solattn.zscore_statistic(tab)[:, :tab.n_video_tiles]
    z = z[:, tab.kv_ok[:tab.n_video_tiles]]
    assert torch.allclose(z.mean(-1), torch.zeros(z.shape[0]), atol=1e-4)
    assert torch.allclose(z.std(-1, unbiased=False),
                          torch.ones(z.shape[0]), atol=1e-3)


def test_allocation_masks_share_one_average_density():
    tab, _ = _tables()
    masks = solattn.allocation_masks(tab, density=0.5)
    budget = solattn.video_budget(0.5, tab.n_tiles,
                                  tab.n_tiles - tab.n_video_tiles)
    calibrated = [n for n in masks if not n.startswith('R2g')]
    for name in calibrated:
        kept = solattn.kept_per_row(masks[name],
                                    tab.n_video_tiles).float()
        assert float(kept.mean()) == pytest.approx(budget, abs=1e-6), name
    assert 'R2g_zscore_global' in masks


def test_selection_masks_cover_the_four_oracle_scores():
    tab, _ = _tables()
    masks = solattn.selection_masks(tab, density=0.5)
    assert sorted(masks) == ['A', 'Mx', 'Omega0', 'Omega1']


def test_oracle_topk_minimizes_the_dropped_omega():
    tab, _ = _tables()
    masks = solattn.selection_masks(tab, density=0.5)
    dropped = {name: float((tab.omega0 * (~m).float()).sum())
               for name, m in masks.items()}
    assert dropped['Omega0'] == min(dropped.values())


def test_omega_selection_finds_the_blocks_that_carry_the_output():
    """Flat attention, block-dependent value norms.

    Scores are tiny, so every block holds nearly the same attention mass
    and `A` / `Mx` rank blocks almost arbitrarily. The output is carried by
    the blocks with large V, which is exactly what Omega measures. This
    checks that the machinery ranks by output contribution; whether real
    layers look like this is what the ablation is for.
    """
    head = 0
    lay, tl = _layout()
    g = torch.Generator().manual_seed(11)
    dim = 32
    q = torch.randn(lay.seq_len, 1, dim, generator=g) * 0.02
    k = torch.randn(lay.seq_len, 1, dim, generator=g) * 0.02
    v = torch.zeros(lay.seq_len, 1, dim)
    scale = torch.rand(tl.n_tiles, generator=g) * 10.0 + 0.05
    for slot in range(tl.num_slots):
        row = int(tl.perm[slot])
        if row >= 0:
            v[row, 0] = torch.randn(dim, generator=g) * scale[slot // TILE]
    lse = _lse(q, k, lay.used)
    w_o = _out_proj(1, dim)
    whiten = solattn.whitening(w_o, head, dim)
    rows = torch.arange(tl.n_video_tiles)
    tab = solattn.head_tables(q, k, v, lse, tl, rows, head, whiten)
    masks = solattn.selection_masks(tab, density=0.5)
    err = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  masks, compensate=(0,))
    assert err[('Omega0', 0)] < err[('Mx', 0)]
    assert err[('Omega0', 0)] < err[('A', 0)]


# --- diagnostics ---------------------------------------------------------


def test_load_imbalance_is_one_for_a_fixed_budget():
    tab, _ = _tables()
    mask = solattn.fixed_topk_mask(tab.proxy, 4.0, tab.kv_ok,
                                   tab.n_video_tiles, tab.rows)
    assert solattn.load_imbalance(mask, tab.n_video_tiles) == \
        pytest.approx(1.0)


def test_row_moments_are_finite_per_row():
    tab, _ = _tables()
    moments = solattn.row_moments(tab)
    assert moments['skew'].shape == (tab.rows.numel(),)
    assert torch.all(torch.isfinite(moments['excess_kurtosis']))


def test_spearman_is_exact_on_a_monotone_pair():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert solattn._spearman(a, 2 * a) == pytest.approx(1.0, abs=1e-6)
    assert solattn._spearman(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_proxy_omega_spearman_reports_every_zone():
    tab, _ = _tables()
    out = solattn.proxy_omega_spearman(tab)
    assert len(out) == 2
    assert all(math.isnan(x) or -1.0 <= x <= 1.0 for x in out.values())


def test_dropped_mass_is_zero_when_nothing_is_dropped():
    tab, _ = _tables()
    r = solattn.dropped_mass(tab, _keep_all(tab))
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-4)


# --- reports, gates and io -----------------------------------------------


def _records(tab, args, densities=(0.3, 0.6)):
    q, k, v, lse, tl, rows, head, whiten, _ = args
    config = solattn.AblationConfig(densities=densities, query_tiles=4,
                                    alphas=(1.0,))
    entries = solattn.head_report(q, k, v, lse, tl, rows, head, whiten,
                                  config)
    return [solattn.Record(clip='c', step=0, layer=0, head=head,
                           shape='4x4x8', **e) for e in entries]


def test_head_report_covers_every_density_and_mask():
    tab, args = _tables()
    entries = solattn.head_report(*args[:4], args[4], args[5], args[6],
                                  args[7],
                                  solattn.AblationConfig(densities=(0.5,),
                                                         query_tiles=4,
                                                         alphas=(1.0,)))
    assert len(entries) == 1
    names = {key.split('|')[0] for key in entries[0]['errors']}
    assert {'A', 'Mx', 'Omega0', 'Omega1', 'R1_topk', 'R2_zscore',
            'R3_sigma_a1', 'oracle_topk', 'oracle_threshold',
            'R2g_zscore_global'} == names
    assert all(key.endswith('|0') or key.endswith('|1')
               for key in entries[0]['errors'])


def test_head_report_error_falls_as_density_grows():
    tab, args = _tables()
    entries = solattn.head_report(*args[:4], args[4], args[5], args[6],
                                  args[7],
                                  solattn.AblationConfig(
                                      densities=(0.3, 0.7), query_tiles=4,
                                      alphas=(1.0,)))
    assert (entries[0]['errors']['R1_topk|0']
            > entries[1]['errors']['R1_topk|0'])


def test_summarize_reports_one_row_per_density():
    tab, args = _tables()
    summary = solattn.summarize(_records(tab, args))
    assert [row['density'] for row in summary] == [0.3, 0.6]
    assert all(row['heads'] == 1 for row in summary)
    assert all('Omega0|0' in row['median_error'] for row in summary)


def test_g1_verdict_follows_the_required_margin():
    base = dict(clip='c', step=0, layer=0, head=0, density=0.1,
                shape='4x4x8', kept_mean={}, kept_max_over_mean={},
                spearman={}, skew=0.0, excess_kurtosis=0.0,
                dropped_mass_r1=0.0)
    better = solattn.Record(errors={'Omega0|0': 0.5, 'Mx|0': 1.0}, **base)
    assert solattn.summarize([better], eta=0.15)[0]['g1_pass']
    assert solattn.summarize([better], eta=0.6)[0]['g1_pass'] is False
    tie = solattn.Record(errors={'Omega0|0': 1.0, 'Mx|0': 1.0}, **base)
    assert solattn.summarize([tie])[0]['g1_pass'] is False
    assert solattn.summarize([better, tie])[0]['g1_pass_fraction'] == 0.5


def test_g2_verdict_follows_the_oracle_gap():
    base = dict(clip='c', step=0, layer=0, head=0, density=0.1,
                shape='4x4x8', kept_mean={}, kept_max_over_mean={},
                spearman={}, skew=0.0, excess_kurtosis=0.0,
                dropped_mass_r1=0.0)
    wide = solattn.Record(errors={'oracle_topk|0': 1.0,
                                  'oracle_threshold|0': 0.8}, **base)
    narrow = solattn.Record(errors={'oracle_topk|0': 1.0,
                                    'oracle_threshold|0': 0.99}, **base)
    assert solattn.summarize([wide])[0]['g2_pass']
    assert solattn.summarize([narrow])[0]['g2_pass'] is False
    assert solattn.summarize([wide])[0]['g2_oracle_gap'] == \
        pytest.approx(0.2)


def test_records_survive_a_round_trip(tmp_path):
    tab, args = _tables()
    records = _records(tab, args)
    path = str(tmp_path / 'ablation.json')
    solattn.save_records(path, records, {'note': 'unit'})
    back, meta = solattn.load_records(path)
    assert meta == {'note': 'unit'}
    assert [r.to_json() for r in back] == [r.to_json() for r in records]


def test_dense_weight_refuses_a_sharded_parameter():
    class DTensor(torch.Tensor):  # name is what the check looks at
        pass

    with pytest.raises(ValueError, match='single process'):
        solattn._dense_weight(DTensor(torch.zeros(4, 4)))
    plain = torch.zeros(4, 4)
    assert solattn._dense_weight(plain) is plain


# --- driver --------------------------------------------------------------


def _scorer(num_heads=2, densities=(0.5,), heads=None):
    from miowtion.veda import plan as veda_plan
    lay, tl = _layout()
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    plan = veda_plan.TilePlan.uniform(geo, tiling.TileShape(4, 4, 8),
                                      num_layers=2, num_heads=num_heads)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=max(densities)))
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    ablation = solattn.AblationConfig(densities=densities, query_tiles=2,
                                      alphas=(1.0,), heads=heads)
    w_o = _out_proj(num_heads, 32)
    return lay, solattn.AblationScorer(
        lay, plan, clip, ablation, out_proj=lambda i: w_o, clip_id='c',
        step=3, seed=1, dense_backend='math')


def test_ablation_scorer_returns_dense_attention_and_records():
    lay, scorer = _scorer()
    q, k, v = _qkv(lay.seq_len)
    out = scorer(q, k, v, layer_index=1)
    assert out.shape == q.shape
    assert len(scorer.records) == 2  # 2 heads x 1 density
    assert {r.head for r in scorer.records} == {0, 1}
    assert all(r.step == 3 and r.layer == 1 and r.clip == 'c'
               for r in scorer.records)
    assert all(r.shape == '4x4x8' for r in scorer.records)


def test_ablation_scorer_matches_the_dense_reference():
    from miowtion.h3 import attention as h3_attention
    lay, scorer = _scorer()
    q, k, v = _qkv(lay.seq_len)
    want, _ = h3_attention.dense_attention(q, k, v, lay.used,
                                           backend='math')
    assert torch.equal(scorer(q, k, v, layer_index=0), want)


def test_ablation_scorer_honours_the_head_filter():
    lay, scorer = _scorer(num_heads=2, heads=[1])
    q, k, v = _qkv(lay.seq_len)
    scorer(q, k, v, layer_index=0)
    assert {r.head for r in scorer.records} == {1}


def test_run_config_rejects_unknown_oracle_through_the_report():
    tab, args = _tables()
    config = solattn.AblationConfig(densities=(0.5,), query_tiles=4,
                                    oracle='nope')
    with pytest.raises(ValueError, match='unknown score'):
        solattn.head_report(*args[:4], args[4], args[5], args[6], args[7],
                            config)


def test_ablation_scorer_drives_a_full_tiny_forward():
    """The whole glue: out_proj lookup, plan groups, model forward."""
    from miowtion.h3 import config as h3_config
    from miowtion.h3 import layout as h3_layout
    from miowtion.h3 import model as h3_model
    from miowtion.h3 import noise
    from miowtion.h3 import schedule as h3_schedule
    from miowtion.veda import plan as veda_plan

    cfg = h3_config.H3Config.tiny()
    torch.manual_seed(0)
    model = h3_model.H3DiT(cfg)
    for param in model.parameters():
        with torch.no_grad():
            param.normal_(0.0, 0.02)
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = h3_layout.pack(torch.ones(12, dtype=torch.long), geo)
    shape = tiling.least_padding_shape(geo.video_grid)
    plan = veda_plan.TilePlan.uniform(geo, shape, cfg.num_layers,
                                      cfg.num_heads)
    veda_config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.5))
    clip_tiling = veda_attention.ClipTiling(lay, veda_config,
                                            torch.device('cpu'))
    scorer = solattn.AblationScorer(
        lay, plan, clip_tiling,
        solattn.AblationConfig(densities=(0.9,), query_tiles=2,
                               alphas=(1.0,), heads=[0]),
        out_proj=lambda i: model.blocks[i].attn.out_proj.weight,
        clip_id='tiny', step=0, seed=0, dense_backend='math')
    text = model.refine_text(torch.randn(12, cfg.text_dim))
    clip = model.clip_inputs(lay, text, torch.device('cpu'))
    video, audio = noise.initial_noise(geo, 0)
    state = h3_schedule.build_timestep_state(lay, 0.3, 0.1)
    with torch.no_grad():
        video_v, _ = model(clip, video, audio, state, scorer)
    assert video_v.shape == (geo.num_video_tokens, 96)
    assert len(scorer.records) == cfg.num_layers  # 1 head x 1 density
    assert {r.layer for r in scorer.records} == set(range(cfg.num_layers))
    assert all(0.0 < r.errors['R1_topk|0'] < 10.0 for r in scorer.records)
    summary = solattn.summarize(scorer.records)
    assert len(summary) == 1 and summary[0]['heads'] == cfg.num_layers


def test_dropped_mass_in_the_report_is_a_per_token_fraction():
    tab, args = _tables()
    entry = solattn.head_report(*args[:4], args[4], args[5], args[6],
                                args[7],
                                solattn.AblationConfig(densities=(0.3,),
                                                       query_tiles=4,
                                                       alphas=(1.0,)))[0]
    assert 0.0 < entry['dropped_mass_r1'] < 1.0
