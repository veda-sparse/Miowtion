"""Tests for miowtion.veda.heatmap and miowtion.veda.attention."""

import dataclasses
import math

import pytest
import torch

from miowtion.h3 import attention as h3_attention
from miowtion.kernels import reference
from miowtion.veda import attention as veda_attention
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling


def _layout():
    target = tiling.TiledSpan(70, (6, 6, 12), tiling.TileShape(2, 4, 16))
    used = target.start + target.num_rows
    return tiling.build_tile_layout([target], used, used + 20)


def _qkv(seq_len, heads=2, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, heads, dim, generator=g).to(torch.bfloat16)
            for _ in range(3)]


def _tiny_oracle_case(ratio=0.9, heads=2, seed=1):
    """A layout plus teacher heat whose oracle set is a proper subset.

    The layout only holds six video tiles, so the ratio has to be high for
    the budget to reach past the forced diagonal: the balanced BCE needs
    both classes to be non-empty, and below ~0.7 the oracle keeps the
    diagonal and nothing else.
    """
    lay = _layout()
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=ratio))
    rows = torch.arange(3)
    g = torch.Generator().manual_seed(seed)
    heat = torch.rand(heads, rows.numel(), lay.n_tiles, generator=g) + 0.01
    sel = veda_mask.select_video_blocks(heat, lay, blocks, rows)
    # Guard against a degenerate case: the assertions below are about the
    # two classes, and several of them pass vacuously on an empty one.
    assert int(sel.keep.sum(-1).min()) > 1, 'oracle kept only the diagonal'
    return lay, blocks, rows, heat


def test_heat_matches_brute_force_block_max():
    lay = _layout()
    q, k, v = _qkv(lay.seq_len)
    _, lse = h3_attention.dense_attention(q, k, v, lay.used, return_lse=True,
                                          backend='math')
    heads = torch.tensor([0, 1])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = lse[lay.gather_index][:, heads]
    lse_t[lay.pad_slots] = 0
    rows = torch.tensor([0, 2, lay.n_tiles - 1])
    heat = heatmap.teacher_heat_reference(qt, kt, lse_t, lay, rows,
                                          q_chunk=128, k_chunk=256)
    # Brute force from the probabilities of the dense teacher.
    scores = torch.einsum('qhd,khd->hqk', q[:lay.used].float(),
                          k[:lay.used].float()) / 32**0.5
    probs = torch.softmax(scores, -1)
    perm = lay.perm
    for r_i, r in enumerate(rows.tolist()):
        for j in range(lay.n_tiles):
            qi = perm[r * 128:(r + 1) * 128]
            kj = perm[j * 128:(j + 1) * 128]
            qi, kj = qi[qi >= 0], kj[kj >= 0]
            expected = probs[:, qi][:, :, kj].amax(dim=(1, 2))
            torch.testing.assert_close(heat[:, r_i, j], expected, rtol=3e-2,
                                       atol=1e-6)


def test_kl_zero_at_optimum_and_recall_one():
    lay = _layout()
    heat = torch.rand(2, 3, lay.n_tiles) + 0.01
    logits = torch.log(heat).requires_grad_()
    kl = heatmap.seer_kl(logits, heat, lay)
    assert abs(kl.item()) < 1e-6
    kl.backward()
    assert torch.isfinite(logits.grad).all()
    rows = torch.tensor([0, 1, 2])
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.9))
    diag = heatmap.mask_diagnostics(logits.detach(), heat, lay, blocks, rows)
    assert diag['recall'].item() == 1.0
    # A perfect predictor keeps exactly what the oracle keeps.
    assert diag['heat_kept'].item() == diag['heat_ceiling'].item()


def test_reference_kernel_keep_all_equals_dense():
    lay = _layout()
    q, k, v = _qkv(lay.seq_len)
    heads = torch.tensor([0, 1])
    qt, kt, vt = (tiling.gather_tiles(t, lay, heads) for t in (q, k, v))
    mask = torch.ones(2, lay.n_tiles, lay.n_tiles, dtype=torch.bool)
    out = reference.block_sparse_attention(qt, kt, vt, mask, lay.valid_count)
    buf = torch.zeros(lay.seq_len + 1, 2, 32, dtype=torch.bfloat16)
    tiling.scatter_tiles_(buf, out, lay, heads)
    dense = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    torch.testing.assert_close(buf[:lay.used].float(),
                               dense[:lay.used].float(), rtol=1e-2,
                               atol=1e-2)


def test_teacher_collector_trains_only_the_predictor():
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    q, k, v = _qkv(lay.seq_len, heads=4)
    for t in (q, k, v):
        t.requires_grad_(False)
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8),
                               tiling.TileShape(2, 8, 8)],
                              [[0, 1, 1, 0]])
    pred = veda_predictor.TileScorePredictor(1, 4, 32)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.3), teacher_q_tiles=0.5,
        recall_every=1)
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    collector = veda_attention.TeacherCollector(
        clip, plan, pred, torch.Generator().manual_seed(0), grad_scale=1.0,
        dense_backend='math')
    with torch.no_grad():
        out = collector(q, k, v, 0)
    dense = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    assert torch.equal(out, dense)
    assert len(collector.stats.kl) == 1 and len(collector.stats.recall) == 2
    resolved = collector.stats.resolve()
    assert 0.0 <= resolved['heat_kept'][0] <= resolved['heat_ceiling'][0] <= 1.0
    assert pred.layers[0].proj_q.grad is not None
    assert pred.layers[0].proj_q.grad.abs().sum() > 0


def test_sparse_student_with_global_tiles():
    """Text rows give global tiles; keep-all budgets must equal dense."""
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    q, k, v = (t.float() for t in _qkv(lay.seq_len, heads=4))
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8),
                               tiling.TileShape(2, 8, 8)],
                              [[0, 1, 1, 0]])
    pred = veda_predictor.TileScorePredictor(1, 4, 32)
    dense = h3_attention.dense_attention(q, k, v, lay.used,
                                         backend='math')[0]
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=1.0))
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    group_layout = clip.get(tiling.TileShape(4, 4, 8))
    assert group_layout.n_tiles > group_layout.n_video_tiles  # global rows
    student = veda_attention.SparseStudent(clip, plan, pred,
                                           allow_reference_kernel=True)
    with torch.no_grad():
        out = student(q, k, v, 0)
    torch.testing.assert_close(out, dense, rtol=1e-5, atol=1e-5)
    # A real budget drops blocks and changes the output.
    sparse_clip = veda_attention.ClipTiling(
        lay, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=0.2)), torch.device('cpu'))
    with torch.no_grad():
        sparse = veda_attention.SparseStudent(
            sparse_clip, plan, pred, allow_reference_kernel=True)(q, k, v, 0)
    assert (sparse - dense).abs().max() > 1e-3


def test_teacher_collector_head_chunks_match_whole_groups():
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    q, k, v = _qkv(lay.seq_len, heads=4)
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8),
                               tiling.TileShape(2, 8, 8)],
                              [[0, 1, 1, 0]])
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.3), teacher_q_tiles=0.5,
        recall_every=1)

    def run(config):
        torch.manual_seed(0)
        pred = veda_predictor.TileScorePredictor(1, 4, 32)
        clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
        collector = veda_attention.TeacherCollector(
            clip, plan, pred, torch.Generator().manual_seed(0),
            grad_scale=1.0, dense_backend='math')
        with torch.no_grad():
            out = collector(q, k, v, 0)
        return out, collector.stats.kl[0], [p.grad.clone()
                                            for p in pred.parameters()]

    whole = run(config)
    chunked = run(dataclasses.replace(config, collect_bytes=1))  # 1 head
    assert torch.equal(chunked[0], whole[0])
    assert chunked[1] == pytest.approx(whole[1], rel=1e-6)
    for a, b in zip(chunked[2], whole[2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-12)


def test_sparse_student_head_chunks_are_exact():
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    q, k, v = (t.float() for t in _qkv(lay.seq_len, heads=4))
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8),
                               tiling.TileShape(2, 8, 8)],
                              [[0, 1, 1, 0]])
    torch.manual_seed(0)
    pred = veda_predictor.TileScorePredictor(1, 4, 32)
    clip = veda_attention.ClipTiling(lay, veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.2)), torch.device('cpu'))
    student = veda_attention.SparseStudent(clip, plan, pred,
                                           allow_reference_kernel=True)
    one_head = veda_attention.ClipTiling(lay, veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.2), collect_bytes=1),
        torch.device('cpu'))
    chunked_student = veda_attention.SparseStudent(
        one_head, plan, pred, allow_reference_kernel=True)
    with torch.no_grad():
        whole = student(q, k, v, 0)
        chunked = chunked_student(q, k, v, 0)
    assert torch.equal(chunked, whole)


def _unequal_plan():
    """A layer whose two head groups pad to different lengths.

    The existing mixed-shape tests run on a (12, 8, 16) grid, where both
    shapes happen to tile it exactly, so they never exercise the case the
    layouts are built for: latent_t = 11 makes 4x4x8 pad a whole t-block
    (12 tiles, partial tiles at the end) while 1x8x16 fits exactly (11).
    """
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 11, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8),
                               tiling.TileShape(1, 8, 16)],
                              [[0, 1, 1, 0]])
    return lay, plan


def test_head_groups_of_unequal_padded_length():
    """Different N per group, and a budget that equalizes kernel cost."""
    lay, plan = _unequal_plan()
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.3))
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    padded, exact = (clip.get(s) for s in plan.shapes)
    assert padded.n_tiles == exact.n_tiles + 1
    assert padded.num_slots == padded.n_tiles * tiling.TILE_SIZE
    assert exact.num_slots == exact.n_tiles * tiling.TILE_SIZE
    # partial_tiles holds tile indices, not a mask.
    assert padded.partial_tiles.numel() > exact.partial_tiles.numel()
    assert int(padded.valid_count.min()) < tiling.TILE_SIZE
    # Real rows are a prefix of every tile, and no row is lost or doubled.
    for layout in (padded, exact):
        rows = layout.perm.view(-1, tiling.TILE_SIZE)
        assert torch.equal((rows >= 0).sum(1).to(torch.int32),
                           layout.valid_count)
        pad = (rows < 0).to(torch.int8)  # pads are a suffix of every tile
        assert torch.equal(pad, pad.cummax(dim=1).values)
        assert torch.equal(torch.sort(layout.perm[layout.perm >= 0]).values,
                           torch.arange(layout.used))
    # A more padded group gets a smaller per-row keep count, so both groups
    # cost the same: n_tiles * per_row == ratio * n_ideal^2.
    costs = []
    for layout in (padded, exact):
        block = veda_mask.column_blocks(layout,
                                        veda_mask.Budget(ratio=0.3))[0]
        n_cols = block.stop - block.start
        costs.append(n_cols * block.budget.per_row(block.real_tokens, n_cols))
    assert costs[0] == pytest.approx(costs[1], rel=1e-12)


def test_sparse_student_unequal_groups_keep_all_equals_dense():
    lay, plan = _unequal_plan()
    q, k, v = (t.float() for t in _qkv(lay.seq_len, heads=4))
    pred = veda_predictor.TileScorePredictor(1, 4, 32)
    dense = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    clip = veda_attention.ClipTiling(
        lay, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=1.0)), torch.device('cpu'))
    with torch.no_grad():
        out = veda_attention.SparseStudent(
            clip, plan, pred, allow_reference_kernel=True)(q, k, v, 0)
    torch.testing.assert_close(out, dense, rtol=1e-5, atol=1e-5)


def test_teacher_collector_unequal_groups():
    lay, plan = _unequal_plan()
    q, k, v = _qkv(lay.seq_len, heads=4)
    pred = veda_predictor.TileScorePredictor(1, 4, 32)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.3), teacher_q_tiles=0.5,
        recall_every=1)
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    collector = veda_attention.TeacherCollector(
        clip, plan, pred, torch.Generator().manual_seed(0), grad_scale=1.0,
        dense_backend='math')
    with torch.no_grad():
        out = collector(q, k, v, 0)
    dense = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    assert torch.equal(out, dense)  # the teacher pass stays dense
    assert len(collector.stats.recall) == 2  # one per head group
    assert pred.layers[0].proj_q.grad.abs().sum() > 0


def test_oracle_bce_is_zero_when_logits_match_the_oracle():
    """A predictor that reproduces the oracle's set drives the term down."""
    torch.manual_seed(0)
    lay, blocks, rows, heat = _tiny_oracle_case()
    n_video = lay.n_video_tiles
    oracle = veda_mask.select_video_blocks(heat, lay, blocks, rows)
    target = torch.zeros(*oracle.index.shape[:2], n_video, dtype=torch.bool)
    target.scatter_(2, oracle.index, oracle.keep)
    # Large logits of the right sign: BCE -> 0 on both classes.
    confident = torch.where(target, 30.0, -30.0)
    loss = heatmap.oracle_bce(confident, heat, lay, blocks, rows)
    assert loss.item() < 1e-6
    # The opposite assignment is the worst case, and an indifferent
    # predictor sits at ln 2 whatever the keep ratio (the term is balanced).
    assert heatmap.oracle_bce(-confident, heat, lay, blocks, rows).item() > 10
    flat = torch.zeros_like(confident)
    assert heatmap.oracle_bce(flat, heat, lay, blocks, rows).item() == \
        pytest.approx(math.log(2.0), abs=1e-6)


def test_oracle_bce_gradient_pushes_towards_the_oracle_set():
    lay, blocks, rows, heat = _tiny_oracle_case()
    logits = torch.zeros(heat.shape, requires_grad=True)
    heatmap.oracle_bce(logits, heat, lay, blocks, rows).backward()
    n_video = lay.n_video_tiles
    oracle = veda_mask.select_video_blocks(heat, lay, blocks, rows)
    target = torch.zeros(*oracle.index.shape[:2], n_video, dtype=torch.bool)
    target.scatter_(2, oracle.index, oracle.keep)
    diag = torch.nn.functional.one_hot(rows, n_video).bool()[None]
    grad = logits.grad[:, :, :n_video]
    # Descent raises the kept blocks' logits and lowers the dropped ones.
    kept = target & ~diag & lay.kv_ok[None, None, :n_video]
    dropped = ~target & ~diag & lay.kv_ok[None, None, :n_video]
    assert (grad[kept] < 0).all()
    assert (grad[dropped] > 0).all()
    # The diagonal is forced by the kernel, so it gets no gradient at all.
    assert torch.equal(grad[diag.expand_as(grad)],
                       torch.zeros(int(diag.expand_as(grad).sum())))


def _heat_case(reduce):
    lay = _layout()
    q, k, v = _qkv(lay.seq_len)
    _, lse = h3_attention.dense_attention(q, k, v, lay.used, return_lse=True,
                                          backend='math')
    heads = torch.tensor([0, 1])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = lse[lay.gather_index][:, heads]
    lse_t[lay.pad_slots] = 0
    rows = torch.tensor([0, 2, lay.n_tiles - 1])
    heat = heatmap.teacher_heat_reference(qt, kt, lse_t, lay, rows,
                                          q_chunk=128, k_chunk=256,
                                          reduce=reduce)
    return lay, q, k, rows, heat


def test_mass_heat_matches_brute_force_block_sum():
    lay, q, k, rows, heat = _heat_case('sum')
    scores = torch.einsum('qhd,khd->hqk', q[:lay.used].float(),
                          k[:lay.used].float()) / 32**0.5
    probs = torch.softmax(scores, -1)
    perm = lay.perm
    for r_i, r in enumerate(rows.tolist()):
        for j in range(lay.n_tiles):
            qi = perm[r * 128:(r + 1) * 128]
            kj = perm[j * 128:(j + 1) * 128]
            qi, kj = qi[qi >= 0], kj[kj >= 0]
            if not qi.numel() or not kj.numel():
                assert torch.all(heat[:, r_i, j] == 0)
                continue
            want = probs[:, qi][:, :, kj].sum((1, 2))
            torch.testing.assert_close(heat[:, r_i, j], want, rtol=2e-2,
                                       atol=2e-3)


def test_mass_heat_rows_sum_to_the_real_query_rows():
    """Every row's probabilities sum to 1, so a tile's mass is its count."""
    lay, _, _, rows, heat = _heat_case('sum')
    want = lay.valid_count.index_select(0, rows).float()
    torch.testing.assert_close(heat.sum(-1), want.expand_as(heat.sum(-1)),
                               rtol=2e-2, atol=2e-2)


def test_mass_heat_is_not_max_heat():
    _, _, _, _, mass = _heat_case('sum')
    _, _, _, _, peak = _heat_case('max')
    assert mass.shape == peak.shape
    # Mass sums 128x128 terms, peak takes one: mass must dominate.
    assert torch.all(mass >= peak - 1e-6)
    assert float((mass - peak).abs().max()) > 1e-2


def test_teacher_heat_rejects_an_unknown_reduction():
    lay = _layout()
    q, k, _ = _qkv(lay.seq_len)
    heads = torch.tensor([0])
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = torch.zeros(lay.num_slots, 1)
    with pytest.raises(ValueError, match="reduce must be"):
        heatmap.teacher_heat_reference(qt, kt, lse_t, lay,
                                       torch.tensor([0]), reduce='mean')


def test_sparse_student_drives_a_second_order_predictor():
    """The student calls the layer directly, so it must supply the extras.

    `_gather_and_pool` only produces [mean|max|min], because that is what
    the fused gather kernel writes. A predictor with the Veda2 terms needs
    E[q^2], Var(k) and the row counts too, and the call site has to add
    them or the layer refuses.
    """
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    q, k, v = (t.float() for t in _qkv(lay.seq_len, heads=4))
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8)], [[0, 0, 0, 0]])
    plain = veda_predictor.TileScorePredictor(1, 4, 32)
    fancy = veda_predictor.TileScorePredictor(1, 4, 32,
                                              second_order_rank=32,
                                              count_term=True)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=1.0))
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    dense = h3_attention.dense_attention(q, k, v, lay.used,
                                         backend='math')[0]
    for pred in (plain, fancy):
        student = veda_attention.SparseStudent(clip, plan, pred,
                                               allow_reference_kernel=True)
        with torch.no_grad():
            out = student(q, k, v, 0)
        # Keeping everything equals dense whatever the score says, which
        # is the one thing a predictor can never change.
        torch.testing.assert_close(out, dense, rtol=1e-5, atol=1e-5)


def test_extra_features_are_empty_for_a_plain_predictor():
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    q, _, _ = (t.float() for t in _qkv(lay.seq_len, heads=4))
    clip = veda_attention.ClipTiling(
        lay, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=1.0)), torch.device('cpu'))
    tl = clip.get(tiling.TileShape(4, 4, 8))
    heads = torch.arange(4)
    plain = veda_predictor.TileScorePredictor(1, 4, 32)
    assert veda_attention._extra_features(plain, tl, None, None) == {}
    fancy = veda_predictor.TileScorePredictor(1, 4, 32,
                                              second_order_rank=8,
                                              count_term=True)
    _, _, sq = veda_attention._gather_and_pool(
        q, tl, heads, veda_predictor.SECOND_RAW)
    _, _, var = veda_attention._gather_and_pool(
        q, tl, heads, veda_predictor.SECOND_CENTRAL)
    extra = veda_attention._extra_features(fancy, tl, sq, var)
    assert sorted(extra) == ['log_count', 'sq_q', 'var_k']
    assert extra['log_count'].shape == (tl.n_tiles,)
    rows = torch.tensor([0, 2])
    sliced = veda_attention._extra_features(fancy, tl, sq, var, rows)
    assert sliced['sq_q'].shape[1] == 2
    assert sliced['var_k'].shape[1] == tl.n_tiles
    with pytest.raises(ValueError, match='needs the pooled'):
        veda_attention._extra_features(fancy, tl, None, None)


def test_gather_and_pool_second_moments_match_pool_tiles():
    """The fused path and the reference pooling must agree."""
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(300, dtype=torch.long), geo)
    q, _, _ = (t.float() for t in _qkv(lay.seq_len, heads=4))
    clip = veda_attention.ClipTiling(
        lay, veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=1.0)), torch.device('cpu'))
    tl = clip.get(tiling.TileShape(4, 4, 8))
    heads = torch.arange(4)
    tiles = tiling.gather_tiles(q, tl, heads)
    for which in (veda_predictor.SECOND_RAW, veda_predictor.SECOND_CENTRAL):
        rows, feats, moment = veda_attention._gather_and_pool(
            q, tl, heads, which)
        assert torch.equal(rows, tiles)
        torch.testing.assert_close(
            feats, veda_predictor.pool_tiles(tiles, tl), rtol=1e-5,
            atol=1e-6)
        torch.testing.assert_close(
            moment, veda_predictor.pool_tiles(tiles, tl, which), rtol=1e-5,
            atol=1e-6)
