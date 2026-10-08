"""Tests for miowtion.veda.solattn (Sol-Attn selection / budget ablation)."""

import json
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
            'R2g_zscore_global', 'proxy_topk', 'proxy_logb_topk',
            'proxy_diag_topk', 'proxy_full_topk'} == names
    # proxy_topk is R1_topk by another name: same score, same budget.
    assert entries[0]['errors']['proxy_topk|0'] == \
        entries[0]['errors']['R1_topk|0']
    assert entries[0]['recall_vs_omega']['Omega0'] == 1.0
    assert all(key.rsplit('|', 1)[1] in ('0', '1', '2')
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
                recall_vs_omega={}, recall_vs_mass={},
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
                recall_vs_omega={}, recall_vs_mass={},
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


def test_oracle_mass_compensation_is_exact_when_v_is_block_constant():
    """c=2 only approximates the direction inside a dropped block.

    Give every key tile a single value vector and that approximation is no
    approximation at all: the block mean *is* every member, and c=2's
    denominator is exact by construction, so the error must vanish at any
    mask. c=1 still misses, because its mass is a zero-order guess. This
    pins what the equal-cost ceiling can and cannot reach.
    """
    head = 0
    lay, tl = _layout()
    g = torch.Generator().manual_seed(7)
    dim = 32
    q = torch.randn(lay.seq_len, 1, dim, generator=g)
    k = torch.randn(lay.seq_len, 1, dim, generator=g)
    v = torch.zeros(lay.seq_len, 1, dim)
    per_tile = torch.randn(tl.n_tiles, dim, generator=g)
    for slot in range(tl.num_slots):
        row = int(tl.perm[slot])
        if row >= 0:
            v[row, 0] = per_tile[slot // TILE]
    lse = _lse(q, k, lay.used)
    whiten = solattn.whitening(_out_proj(1, dim), head, dim)
    rows = torch.arange(tl.n_video_tiles)
    tab = solattn.head_tables(q, k, v, lse, tl, rows, head, whiten)
    budget = solattn.video_budget(0.4, tab.n_tiles,
                                  tab.n_tiles - tab.n_video_tiles)
    mask = solattn.fixed_topk_mask(tab.proxy, budget, tab.kv_ok,
                                   tab.n_video_tiles, rows)
    err = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  {'m': mask}, compensate=(0, 1, 2))
    assert err[('m', 2)] < 1e-5
    assert err[('m', 1)] > 1e-3
    assert err[('m', 0)] > 1e-3


def test_oracle_mass_compensation_has_an_exact_denominator():
    """Dropping everything leaves sum_j a_uj == 1, so c=2 cannot blow up."""
    tab, args = _tables()
    q, k, v, lse, tl, rows, head, whiten, _ = args
    # Keep only the global columns: every video block is dropped.
    mask = torch.zeros(rows.numel(), tab.n_tiles, dtype=torch.bool)
    mask[:, tab.n_video_tiles:] = tab.kv_ok[None, tab.n_video_tiles:]
    err = solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                  {'m': mask}, compensate=(0, 1, 2))
    assert err[('m', 2)] < err[('m', 0)]
    assert all(math.isfinite(e) for e in err.values())


def test_relative_errors_rejects_an_unknown_compensation():
    tab, args = _tables()
    q, k, v, lse, tl, rows, head, whiten, _ = args
    with pytest.raises(ValueError, match='compensate must be'):
        solattn.relative_errors(q, k, v, lse, tl, rows, head, whiten,
                                {'m': _keep_all(tab)}, compensate=(3,))


# --- second-order proxy variants -----------------------------------------


def _block_constant_keys(seed=5, dim=32):
    """q random, but every key tile holds one repeated key vector."""
    lay, tl = _layout()
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(lay.seq_len, 1, dim, generator=g)
    k = torch.zeros(lay.seq_len, 1, dim)
    v = torch.randn(lay.seq_len, 1, dim, generator=g)
    per_tile = torch.randn(tl.n_tiles, dim, generator=g)
    for slot in range(tl.num_slots):
        row = int(tl.perm[slot])
        if row >= 0:
            k[row, 0] = per_tile[slot // TILE]
    return lay, tl, q, k, v


def test_second_order_terms_vanish_when_keys_are_block_constant():
    """No within-block spread means no cumulant beyond the mean."""
    lay, tl, q, k, v = _block_constant_keys()
    lse = _lse(q, k, lay.used)
    whiten = solattn.whitening(_out_proj(1, 32), 0, 32)
    rows = torch.arange(tl.n_video_tiles)
    tab = solattn.head_tables(q, k, v, lse, tl, rows, 0, whiten)
    nv = tab.n_video_tiles
    base = tab.proxy_logb[:, :nv]
    torch.testing.assert_close(tab.proxy_diag[:, :nv], base,
                               rtol=0, atol=2e-3)
    torch.testing.assert_close(tab.proxy_full[:, :nv], base,
                               rtol=0, atol=2e-3)


def test_log_count_term_only_moves_partial_tiles():
    tab, _ = _tables()
    nv = tab.n_video_tiles
    delta = (tab.proxy_logb - tab.proxy)[:, :nv]
    # Every row sees the same per-column offset: log of the row count.
    assert torch.allclose(delta, delta[0:1].expand_as(delta), atol=1e-5)
    full = tab.valid_rows.new_tensor(TILE)
    assert torch.all(delta <= math.log(float(full)) + 1e-5)


def test_diagonal_second_order_ranks_the_block_mass_better():
    """The term the predictor structurally cannot express is worth a lot.

    Veda's untrained predictor is exactly `proxy`. Adding the diagonal
    cumulant term costs one more pooled feature per side and leaves the
    n^2 bilinear at rank head_dim, so this gap is the headroom of a
    feature change rather than a capacity change.
    """
    tab, _ = _tables()
    nv = tab.n_video_tiles
    support = tab.kv_ok[:nv]
    log_mass = torch.log(tab.attn_mass[:, :nv][:, support].clamp(min=1e-30))
    rho = {}
    for name in ('proxy', 'proxy_diag', 'proxy_full'):
        t = tab.score(name)[:, :nv][:, support]
        rho[name] = sum(solattn._spearman(t[r], log_mass[r])
                        for r in range(t.shape[0])) / t.shape[0]
    assert rho['proxy_diag'] > rho['proxy'] + 0.2
    assert rho['proxy_full'] >= rho['proxy_diag'] - 0.05


def test_scorer_masks_share_the_budget_with_the_oracle():
    tab, _ = _tables()
    masks = solattn.scorer_masks(tab, density=0.5)
    assert sorted(masks) == ['proxy_diag_topk', 'proxy_full_topk',
                             'proxy_logb_topk', 'proxy_topk']
    kept = {n: float(solattn.kept_per_row(m, tab.n_video_tiles)
                     .float().mean()) for n, m in masks.items()}
    assert len(set(round(x, 6) for x in kept.values())) == 1


def test_recall_against_is_a_fraction_and_exact_on_itself():
    tab, _ = _tables()
    masks = solattn.selection_masks(tab, density=0.3)
    nv = tab.n_video_tiles
    assert solattn.recall_against(masks['A'], masks['A'], nv) == 1.0
    r = solattn.recall_against(masks['Mx'], masks['A'], nv)
    assert 0.0 < r < 1.0


def test_budget_rules_can_be_switched_off():
    tab, args = _tables()
    cfg = solattn.AblationConfig(densities=(0.5,), query_tiles=4,
                                 alphas=(1.0,), budget_rules=False)
    entry = solattn.head_report(*args[:4], args[4], args[5], args[6],
                                args[7], cfg)[0]
    names = {k.split('|')[0] for k in entry['errors']}
    assert names == {'A', 'Mx', 'Omega0', 'Omega1', 'proxy_topk',
                     'proxy_logb_topk', 'proxy_diag_topk',
                     'proxy_full_topk'}
    # G2 has no oracle threshold to compare, so it must simply abstain.
    rec = solattn.Record(clip='c', step=0, layer=0, head=0, shape='s',
                         **entry)
    row = solattn.summarize([rec])[0]
    assert math.isnan(row['g2_oracle_gap'])
    assert row['g2_pass'] is False


def test_predictor_can_express_the_measured_diagonal_proxy():
    """The ablation's proxy_diag and the predictor's head are one formula.

    solattn measures the diagonal cumulant offline; predictor serves it as
    a low-rank head. Two independent implementations, so pin them against
    each other: at full rank with the exact warm start and the base
    projections zeroed, the predictor's logits must equal
    `proxy_diag - log B` on the sampled rows.
    """
    from miowtion.veda import predictor as veda_predictor

    dim = 32
    lay, tl = _layout()
    g = torch.Generator().manual_seed(4)
    q = torch.randn(lay.seq_len, 1, dim, generator=g).to(torch.bfloat16)
    k = torch.randn(lay.seq_len, 1, dim, generator=g).to(torch.bfloat16)
    v = torch.randn(lay.seq_len, 1, dim, generator=g).to(torch.bfloat16)
    lse = _lse(q.float(), k.float(), lay.used)
    whiten = solattn.whitening(_out_proj(1, dim), 0, dim)
    rows = torch.arange(tl.n_video_tiles)
    tab = solattn.head_tables(q, k, v, lse, tl, rows, 0, whiten)

    pred = veda_predictor.TileScorePredictor(1, 1, dim,
                                             second_order_rank=dim)
    pred.init_exact_second_order_()
    with torch.no_grad():
        pred.layers[0].proj_q.zero_()
        pred.layers[0].proj_k.zero_()
    heads = torch.tensor([0])
    qt, kt = (tiling.gather_tiles(t, tl, heads) for t in (q, k))
    logits = pred.scores(0, qt, kt, tl, heads)[0]

    nv = tab.n_video_tiles
    want = (tab.proxy_diag - tab.proxy_logb + tab.proxy)[:, :nv]
    got = logits.index_select(0, rows)[:, :nv]
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


# --- static per-head budget allocation -----------------------------------


def _curve_record(layer, head, density, err):
    return solattn.Record(
        clip='c', step=0, layer=layer, head=head, density=density,
        shape='s', errors={'proxy_topk|0': err}, recall_vs_omega={},
        recall_vs_mass={}, kept_mean={}, kept_max_over_mean={},
        spearman={}, skew=0.0, excess_kurtosis=0.0, dropped_mass_r1=0.0)


def test_error_curve_medians_over_clips_and_steps():
    records = [_curve_record(0, 0, 0.1, 0.4), _curve_record(0, 0, 0.1, 0.6),
               _curve_record(0, 0, 0.2, 0.2), _curve_record(1, 0, 0.1, 0.9)]
    curves = solattn.error_curve(records)
    assert curves[(0, 0)] == [(0.1, 0.5), (0.2, 0.2)]
    assert curves[(1, 0)] == [(0.1, 0.9)]
    with pytest.raises(ValueError, match='no record carries'):
        solattn.error_curve(records, mask='nope')


def test_log_log_interpolation_is_exact_on_a_power_law():
    curve = [(0.05, 0.4), (0.20, 0.1)]      # eps proportional to rho^-1
    mid = solattn._interpolate_log_log(curve, 0.10)
    assert mid == pytest.approx(0.4 * (0.10 / 0.05) ** -1.0, rel=1e-9)
    # Outside the range it extrapolates along the same slope.
    assert solattn._interpolate_log_log(curve, 0.40) == pytest.approx(
        0.05, rel=1e-9)


def test_allocation_hits_the_mean_density_and_beats_uniform():
    """One sensitive head and one flat head: the budget should move."""
    records = []
    for density in (0.05, 0.1, 0.2):
        # Head A halves its error when the budget doubles; head B barely
        # moves, so every block is worth more to A than to B.
        records.append(_curve_record(0, 0, density, 0.5 * (density / 0.05)
                                     ** -1.0))
        records.append(_curve_record(0, 1, density, 0.5 * (density / 0.05)
                                     ** -0.05))
    curves = solattn.error_curve(records)
    report = solattn.allocation_report(curves, mean_density=0.1)
    # Never overspend; with two heads on a grid the target is not exactly
    # attainable, so only the ceiling is guaranteed.
    assert report['achieved_mean_density'] <= 0.1 * (1 + 1e-9) + 1e-12
    assert report['achieved_mean_density'] > 0.09
    assert report['relative_saving'] > 0.05
    chosen = solattn.allocate_budget(curves, mean_density=0.1)
    assert chosen[(0, 0)] > chosen[(0, 1)]


def test_allocation_is_uniform_when_every_head_is_identical():
    records = [_curve_record(0, h, d, 0.5 * (d / 0.05) ** -0.6)
               for h in range(4) for d in (0.05, 0.1, 0.2)]
    curves = solattn.error_curve(records)
    chosen = solattn.allocate_budget(curves, mean_density=0.1)
    assert len(set(chosen.values())) == 1
    assert sum(chosen.values()) / len(chosen) <= 0.1 * (1 + 1e-9) + 1e-12
    report = solattn.allocation_report(curves, mean_density=0.1)
    assert abs(report['relative_saving']) < 0.02


def test_allocation_rejects_an_out_of_range_target():
    records = [_curve_record(0, 0, d, 0.5 / d) for d in (0.05, 0.2)]
    curves = solattn.error_curve(records)
    with pytest.raises(ValueError, match='outside the grid'):
        solattn.allocate_budget(curves, mean_density=0.9)
    with pytest.raises(ValueError, match='no error curves'):
        solattn.allocate_budget({}, mean_density=0.1)


def test_records_load_from_a_file_without_the_recall_fields(tmp_path):
    """Runs written before the recall maps existed must still load."""
    tab, args = _tables()
    records = _records(tab, args, densities=(0.5,))
    path = str(tmp_path / 'old.json')
    solattn.save_records(path, records, {})
    raw = json.loads(open(path).read())
    for row in raw['records']:
        row.pop('recall_vs_omega')
        row.pop('recall_vs_mass')
        row['a_future_field'] = 1
    open(path, 'w').write(json.dumps(raw))
    back, _ = solattn.load_records(path)
    assert len(back) == len(records)
    assert back[0].recall_vs_omega == {}
    assert back[0].errors == records[0].errors


def test_allocation_accepts_a_target_at_the_measured_boundary():
    """exp(log(x)) drift must not push the grid past its own endpoints."""
    records = [_curve_record(0, h, d, 0.5 * (d / 0.05) ** -0.6)
               for h in range(3) for d in (0.05, 0.1, 0.2)]
    curves = solattn.error_curve(records)
    for target in (0.05, 0.2):
        chosen = solattn.allocate_budget(curves, mean_density=target)
        mean = sum(chosen.values()) / len(chosen)
        assert mean == pytest.approx(target, rel=1e-9)


def test_scorer_report_measures_the_gap_closed():
    base = dict(clip='c', step=0, layer=0, head=0, density=0.1, shape='s',
                kept_mean={}, kept_max_over_mean={}, spearman={},
                skew=0.0, excess_kurtosis=0.0, dropped_mass_r1=0.0)
    rec = solattn.Record(
        errors={'proxy_topk|0': 1.0, 'proxy_diag_topk|0': 0.7,
                'Omega0|0': 0.5},
        recall_vs_omega={'proxy_topk': 0.4, 'proxy_diag_topk': 0.6,
                         'Omega0': 1.0},
        recall_vs_mass={}, **base)
    rows = {r['mask']: r for r in solattn.scorer_report([rec], 0.1)}
    assert rows['proxy_topk']['gap_closed'] == pytest.approx(0.0)
    assert rows['Omega0']['gap_closed'] == pytest.approx(1.0)
    assert rows['proxy_diag_topk']['gap_closed'] == pytest.approx(0.6)
    assert rows['proxy_diag_topk']['recall_vs_omega'] == pytest.approx(0.6)
    assert math.isnan(rows['proxy_topk']['recall_vs_mass'])
    with pytest.raises(ValueError, match='no record at density'):
        solattn.scorer_report([rec], 0.99)
    with pytest.raises(ValueError, match='is missing'):
        solattn.scorer_report([rec], 0.1, oracle='nope')


# --- predictor initialization probe --------------------------------------


def test_init_variants_differ_only_in_the_second_order_head():
    variants = solattn.init_variants(16)
    assert sorted(variants) == ['veda1', 'veda2']
    assert variants['veda1'].second_order_rank == 0
    assert variants['veda2'].second_order_rank == 16
    names = {n for n, _ in variants['veda2'].named_parameters()}
    assert any(n.endswith('so_q') for n in names)
    # The shared base projections are drawn from the same seed, so the two
    # differ only by the head under test.
    a = dict(variants['veda1'].named_parameters())
    b = dict(variants['veda2'].named_parameters())
    for name, param in a.items():
        assert torch.equal(param, b[name]), name


def test_init_probe_scores_both_variants_on_a_tiny_forward():
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
    clip_tiling = veda_attention.ClipTiling(
        lay, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=0.9)), torch.device('cpu'))
    probe = solattn.PredictorInitProbe(
        lay, plan, clip_tiling, solattn.init_variants(cfg.head_dim),
        query_tiles=2, seed=0, step=3, dense_backend='math')
    text = model.refine_text(torch.randn(12, cfg.text_dim))
    clip = model.clip_inputs(lay, text, torch.device('cpu'))
    video, audio = noise.initial_noise(geo, 0)
    state = h3_schedule.build_timestep_state(lay, 0.3, 0.1)
    with torch.no_grad():
        model(clip, video, audio, state, probe)
    assert len(probe.rows) == cfg.num_layers * 2 * 2  # variants x targets
    assert {r['variant'] for r in probe.rows} == {'veda1', 'veda2'}
    assert {r['target'] for r in probe.rows} == {'max', 'sum'}
    summary = solattn.summarize_init_probe(probe.rows)
    assert len(summary) == 4
    for row in summary:
        assert 0.0 <= row['heat_kept'] <= 1.0 + 1e-6
        assert row['heat_kept'] <= row['heat_ceiling'] + 1e-6
        assert row['layers'] == cfg.num_layers


def test_init_probe_summary_handles_an_all_nan_recall():
    rows = [{'step': 0, 'layer': 0, 'variant': 'veda1', 'target': 'max',
             'recall': float('nan'), 'heat_kept': 0.5,
             'heat_ceiling': 0.9}]
    summary = solattn.summarize_init_probe(rows)
    assert math.isnan(summary[0]['recall'])
    assert summary[0]['kept_over_ceiling'] == pytest.approx(0.5 / 0.9)


def test_init_variants_land_on_the_requested_device():
    variants = solattn.init_variants(16, device='cpu')
    for pred in variants.values():
        for param in pred.parameters():
            assert param.device.type == 'cpu'
