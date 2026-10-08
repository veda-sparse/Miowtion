"""GPU tests of the attention kernels against their references."""

import pytest
import torch

from miowtion.h3 import attention as h3_attention
from miowtion.kernels import block_heat_triton
from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.kernels import tile_gather_triton
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import search
from miowtion.veda import tiling

pytestmark = pytest.mark.gpu


def _layout(grid=(8, 12, 20), shape='4x8x4', text=300):
    target = tiling.TiledSpan(text + 60, grid, tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout([target], used, used + 32, 'cuda')


def _qkv(seq_len, heads=4, dim=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, heads, dim, generator=gen).to(
        'cuda', torch.bfloat16) for _ in range(3)]


def test_dense_lse_matches_math():
    q, k, v = _qkv(2048)
    out, lse = h3_attention.dense_attention(q, k, v, 2000, return_lse=True)
    ref_out, ref_lse = h3_attention.dense_attention(q, k, v, 2000,
                                                    return_lse=True,
                                                    backend='math')
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2,
                               atol=2e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3)
    assert (out[2000:] == 0).all() and (lse[2000:] == 0).all()


@pytest.mark.skipif(not block_heat_triton.available(), reason='needs triton')
def test_heat_triton_matches_reference():
    lay = _layout()
    q, k, v = _qkv(lay.seq_len)
    _, lse = h3_attention.dense_attention(q, k, v, lay.used, return_lse=True)
    heads = torch.arange(4, device='cuda')
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = lse[lay.gather_index][:, heads].contiguous()
    lse_t[lay.pad_slots] = 0
    rows = torch.arange(0, lay.n_tiles, 3, device='cuda')
    fused = block_heat_triton.teacher_heat(qt, kt, lse_t, lay, rows)
    ref = heatmap.teacher_heat_reference(qt, kt, lse_t, lay, rows)
    torch.testing.assert_close(fused, ref, rtol=3e-2, atol=1e-5)


@pytest.mark.skipif(not block_heat_triton.available(), reason='needs triton')
def test_mass_heat_triton_matches_reference():
    """The block-mass reduction, which Veda2 distils against."""
    lay = _layout()
    q, k, v = _qkv(lay.seq_len)
    _, lse = h3_attention.dense_attention(q, k, v, lay.used, return_lse=True)
    heads = torch.arange(4, device='cuda')
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = lse[lay.gather_index][:, heads].contiguous()
    lse_t[lay.pad_slots] = 0
    rows = torch.arange(0, lay.n_tiles, 3, device='cuda')
    fused = block_heat_triton.teacher_heat(qt, kt, lse_t, lay, rows,
                                           reduce='sum')
    ref = heatmap.teacher_heat_reference(qt, kt, lse_t, lay, rows,
                                         reduce='sum')
    torch.testing.assert_close(fused, ref, rtol=3e-2, atol=1e-4)
    # Every row's probabilities sum to 1, so a tile's mass is its row count.
    want = lay.valid_count.index_select(0, rows).float()
    torch.testing.assert_close(fused.sum(-1), want.expand(4, rows.numel()),
                               rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not block_heat_triton.available(), reason='needs triton')
def test_heat_kernel_rejects_an_unknown_reduction():
    lay = _layout()
    q, k, _ = _qkv(lay.seq_len)
    heads = torch.arange(1, device='cuda')
    qt, kt = (tiling.gather_tiles(t, lay, heads) for t in (q, k))
    lse_t = torch.zeros(lay.num_slots, 1, device='cuda')
    with pytest.raises(ValueError, match='reduce must be'):
        block_heat_triton.teacher_heat(qt, kt, lse_t, lay,
                                       torch.zeros(1, dtype=torch.long,
                                                   device='cuda'),
                                       reduce='mean')


@pytest.mark.skipif(not fa4.available(), reason='FA4 block sparsity '
                    'unsupported on this architecture')
def test_fa4_block_sparse_matches_reference():
    lay = _layout(grid=(8, 12, 20))  # padded grid -> partial tiles
    q, k, v = _qkv(lay.seq_len)
    heads = torch.arange(4, device='cuda')
    qt, kt, vt = (tiling.gather_tiles(t, lay, heads) for t in (q, k, v))
    scores = torch.randn(4, lay.n_tiles, lay.n_tiles, device='cuda')
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    sel = veda_mask.select_video_blocks(scores[:, :lay.n_video_tiles], lay,
                                        blocks)
    block_mask = veda_mask.dense_block_mask(sel, lay)
    out = fa4.block_sparse_attention(qt, kt, vt, block_mask, lay)
    ref = reference.block_sparse_attention(qt, kt, vt, block_mask,
                                           lay.valid_count)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(out[real].float(), ref[real].float(),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not fa4.available()
    or torch.cuda.get_device_capability()[0] not in fa4.PATCHED_MAJOR_ARCHS,
    reason='DenseBlockMask path is SM8x / SM120 only')
def test_dense_block_mask_all_blocks_match_dense_with_mask_mod():
    """Every block selected: the sparse walk matches the dense kernel.

    Bitwise equality needs the dense kernel on the sparse path's tile config;
    FA4's default dense config differs, which costs up to ~1 bf16 ulp.
    """
    lay = _layout(grid=(8, 12, 20))
    q, k, v = _qkv(lay.seq_len)
    qt, kt, vt = (tiling.gather_tiles(t, lay, None) for t in (q, k, v))
    block_mask = torch.ones(4, lay.n_tiles, lay.n_tiles, dtype=torch.bool,
                            device='cuda')
    out = fa4.block_sparse_attention(qt, kt, vt, block_mask, lay)
    dense = fa4.interface().flash_attn_func(
        qt[None], kt[None], vt[None], softmax_scale=128**-0.5,
        mask_mod=fa4._valid_key_mask_mod(),  # pylint: disable=protected-access
        aux_tensors=[lay.slot_valid])
    dense = (dense[0] if isinstance(dense, tuple) else dense)[0]
    real = lay.slot_valid.bool()
    torch.testing.assert_close(out[real].float(), dense[real].float(),
                               rtol=0, atol=2e-3)


@pytest.mark.skipif(not (fa4.available() and block_heat_triton.available()),
                    reason='needs FA4 block sparsity and triton')
def test_oracle_kernel_path_matches_reference():
    lay = _layout(grid=(8, 12, 20))
    q, k, v = _qkv(lay.seq_len)
    out, lse = h3_attention.dense_attention(q, k, v, lay.used,
                                            return_lse=True)
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    rows = torch.tensor([0, 3, 7, lay.n_video_tiles - 1], device='cuda')
    fused = search.oracle_rel_mse(q, k, v, lay, blocks, rows,
                                  dense_out=out, lse=lse)
    ref = search.oracle_rel_mse_reference(q, k, v, lay, blocks, rows)
    assert (ref > 1e-4).all()  # the sparse mask actually drops mass
    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(not (fa4.available() and block_heat_triton.available()),
                    reason='needs FA4 block sparsity and triton')
def test_oracle_head_chunks_do_not_change_the_result(monkeypatch):
    """Heat, masks and sparse outputs are bitwise equal per head; only the
    final fp32 sum over rows may accumulate in another order."""
    lay = _layout(grid=(8, 12, 20))
    q, k, v = _qkv(lay.seq_len)
    out, lse = h3_attention.dense_attention(q, k, v, lay.used,
                                            return_lse=True)
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    rows = torch.tensor([0, 3, 7, lay.n_video_tiles - 1], device='cuda')
    whole = search.oracle_rel_mse(q, k, v, lay, blocks, rows,
                                  dense_out=out, lse=lse)
    monkeypatch.setattr(search, '_ORACLE_GATHER_BYTES', 1)  # 1 head/chunk
    chunked = search.oracle_rel_mse(q, k, v, lay, blocks, rows,
                                    dense_out=out, lse=lse)
    torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=0)


@pytest.mark.skipif(not tile_gather_triton.available(), reason='needs triton')
@pytest.mark.parametrize('heads', [None, [0, 2, 3]])
def test_tile_gather_pool_scatter_match_torch(heads):
    lay = _layout(grid=(8, 12, 20))  # padded grid -> partial tiles
    assert lay.partial_tiles.numel() > 0
    x, _, _ = _qkv(lay.seq_len)
    idx = None if heads is None else torch.tensor(heads, device='cuda')
    ref_tiles = tiling.gather_tiles(x, lay, idx)
    tiles, feats = tile_gather_triton.gather_and_pool(x, lay, idx)
    assert torch.equal(tiles, ref_tiles)
    assert torch.equal(tile_gather_triton.gather_tiles(x, lay, idx),
                       ref_tiles)
    ref_feats = veda_predictor.pool_tiles(ref_tiles, lay)
    d = x.shape[-1]
    assert torch.equal(feats[..., d:], ref_feats[..., d:])  # max, min
    torch.testing.assert_close(feats[..., :d], ref_feats[..., :d],
                               rtol=1e-5, atol=1e-6)  # mean: sum order
    out_ref = x.new_zeros(lay.seq_len + 1, *x.shape[1:])
    out = x.new_zeros(lay.seq_len + 1, *x.shape[1:])
    tiling.scatter_tiles_(out_ref, ref_tiles, lay, idx)
    tile_gather_triton.scatter_tiles_(out, tiles, lay, idx)
    assert torch.equal(out[:lay.seq_len], out_ref[:lay.seq_len])


@pytest.mark.skipif(not fa4.available(), reason='FA4 block sparsity '
                    'unsupported on this architecture')
def test_fa4_block_sparse_backward_matches_reference():
    """dQ/dK/dV of the sparse kernel match autograd through the reference.

    The kernels read the fp32 dQ/dK/dV accumulators with the main kernel's
    MMA thread partition, so a postprocess launched with a different thread
    count corrupts the gradients *silently* (see fa4_sm8x/patches/0005).
    Only the real query rows are compared: padded slots carry no gradient.
    """
    lay = _layout(grid=(8, 12, 20))  # padded grid -> partial tiles
    q, k, v = _qkv(lay.seq_len)
    heads = torch.arange(4, device='cuda')
    qt, kt, vt = (tiling.gather_tiles(t, lay, heads) for t in (q, k, v))
    scores = torch.randn(4, lay.n_tiles, lay.n_tiles, device='cuda')
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=0.3))
    sel = veda_mask.select_video_blocks(scores[:, :lay.n_video_tiles], lay,
                                        blocks)
    block_mask = veda_mask.dense_block_mask(sel, lay)
    real = lay.slot_valid.bool()
    grad_out = torch.randn_like(qt)

    def grads(fn):
        tensors = [t.detach().clone().requires_grad_(True)
                   for t in (qt, kt, vt)]
        out = fn(*tensors)
        out.backward(grad_out)
        return [t.grad for t in tensors]

    got = grads(lambda a, b, c: fa4.block_sparse_attention(
        a, b, c, block_mask, lay))
    want = grads(lambda a, b, c: reference.block_sparse_attention(
        a, b, c, block_mask, lay.valid_count))
    # FA4's backward computes *dense* gradients when it is not given the block
    # pattern (it does so silently on SM90/SM100 without the _bwd lists), so
    # matching the reference is not enough: the dense gradients must also be
    # visibly further away, otherwise the test would pass on a dense kernel.
    dense_mask = torch.ones_like(block_mask)
    dense = grads(lambda a, b, c: reference.block_sparse_attention(
        a, b, c, dense_mask, lay.valid_count))
    for name, g, w, d in zip('qkv', got, want, dense):
        torch.testing.assert_close(
            g[real].float(), w[real].float(), rtol=3e-2, atol=3e-2,
            msg=lambda m, name=name: f'd{name} mismatch\n{m}')
        assert not g[real].eq(0).all(), f'd{name} is all zeros'
        sparse_err = (g[real].float() - w[real].float()).abs().max()
        dense_err = (g[real].float() - d[real].float()).abs().max()
        assert sparse_err < dense_err, (
            f'd{name} is as close to the dense gradient ({dense_err:.3e}) as '
            f'to the sparse one ({sparse_err:.3e}): the block pattern was '
            'likely ignored')


@pytest.mark.skipif(not tile_gather_triton.available(), reason='needs triton')
def test_fused_second_moments_match_the_reference_pooling():
    """The fused kernel computes both second moments in registers.

    The centred variance is the one that matters: pooling it afterwards
    needs the mean first, so it reads the tile-ordered rows a second time,
    while inside the kernel the tile is already there. This pins the fused
    result against predictor.pool_tiles.
    """
    from miowtion.veda import predictor as veda_predictor
    lay = _layout()
    q, _, _ = _qkv(lay.seq_len)
    heads = torch.arange(4, device='cuda')
    rows, feats, sq, var = tile_gather_triton.gather_and_pool(
        q, lay, heads, second=True)
    tiles = tiling.gather_tiles(q, lay, heads)
    assert torch.equal(rows, tiles)
    torch.testing.assert_close(feats,
                               veda_predictor.pool_tiles(tiles, lay),
                               rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        sq, veda_predictor.pool_tiles(tiles, lay,
                                      veda_predictor.SECOND_RAW),
        rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        var, veda_predictor.pool_tiles(tiles, lay,
                                       veda_predictor.SECOND_CENTRAL),
        rtol=1e-3, atol=1e-5)
    # Empty tiles stay zero, as the three base features already do.
    assert torch.isfinite(sq).all() and torch.isfinite(var).all()
    assert torch.all(var >= 0.0)
