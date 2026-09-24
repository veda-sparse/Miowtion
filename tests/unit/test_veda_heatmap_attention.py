"""Tests for miowtion.veda.heatmap and miowtion.veda.attention."""

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


def test_teacher_collector_head_chunks_match_whole_groups(monkeypatch):
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

    def run():
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

    whole = run()
    monkeypatch.setattr(veda_attention, '_COLLECT_BYTES', 1)  # 1 head/chunk
    chunked = run()
    assert torch.equal(chunked[0], whole[0])
    assert chunked[1] == pytest.approx(whole[1], rel=1e-6)
    for a, b in zip(chunked[2], whole[2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-12)


def test_sparse_student_head_chunks_are_exact(monkeypatch):
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
    with torch.no_grad():
        whole = student(q, k, v, 0)
        monkeypatch.setattr(veda_attention, '_COLLECT_BYTES', 1)
        chunked = student(q, k, v, 0)
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
