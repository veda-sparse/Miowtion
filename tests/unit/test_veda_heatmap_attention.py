"""Tests for miowtion.veda.heatmap and miowtion.veda.attention."""

import torch

from miowtion.h3 import attention as h3_attention
from miowtion.veda import attention as veda_attention
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling
from miowtion.veda.kernels import reference


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
    assert heatmap.mask_recall(logits.detach(), heat, lay, blocks,
                               rows).item() == 1.0


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
    assert pred.layers[0].proj_q.grad is not None
    assert pred.layers[0].proj_q.grad.abs().sum() > 0
