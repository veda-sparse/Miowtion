"""GPU tests of the attention kernels against their references."""

import pytest
import torch

from miowtion.h3 import attention as h3_attention
from miowtion.kernels import block_heat_triton
from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.veda import heatmap
from miowtion.veda import mask as veda_mask
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
    out = fa4.block_sparse_attention(qt, kt, vt,
                                     veda_mask.kernel_indices(sel, lay), lay)
    ref = reference.block_sparse_attention(qt, kt, vt, block_mask,
                                           lay.valid_count)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(out[real].float(), ref[real].float(),
                               rtol=2e-2, atol=2e-2)


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
