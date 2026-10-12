"""GPU tests of the fused Sol+Veda kernel and of the FA4 + Triton path.

The fused kernel's host side and its walk over kept tiles were rewritten
for speed (docs/features/veda2.md section 2.33), so the published kernel
stays in the vendored file as `_forward_reference` and these tests are
what makes that worth keeping: a rewrite of an online softmax has to be
checked against the thing it replaces, not only against dense attention,
because at the working density dense attention is not the answer either.
"""

import pytest
import torch

from miowtion.kernels import compensation_triton
from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.kernels import sol_veda
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
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


def _tiles(lay, heads=4):
    """Tile-ordered q, k, v with the padding slots zeroed."""
    q, k, v = _qkv(lay.seq_len, heads)
    return tuple(tiling.gather_tiles(t, lay, None) for t in (q, k, v))


def _masked_mask(lay, heads=4, ratio=0.3):
    """A learned-looking mask that always keeps a row's own tile."""
    scores = torch.randn(heads, lay.n_tiles, lay.n_tiles, device='cuda')
    blocks = veda_mask.column_blocks(lay, veda_mask.Budget(ratio=ratio))
    chosen = veda_mask.select_video_blocks(scores[:, :lay.n_video_tiles],
                                           lay, blocks)
    mask = veda_mask.dense_block_mask(chosen, lay)
    eye = torch.eye(lay.n_tiles, dtype=torch.bool, device='cuda')
    return ((mask | eye[None]) & lay.kv_ok[None, None, :]).contiguous()


def _fused(qt, kt, vt, mask, lay, reference_kernel=False):
    from miowtion.kernels.sol_veda_patched import attention as fused
    _attend, sage = sol_veda._modules()                    # noqa: SLF001
    was, fused.USE_REFERENCE = fused.USE_REFERENCE, reference_kernel
    try:
        out, _exact = fused.launch(qt, kt, vt,
                                   lay.valid_count.to(torch.int32), mask,
                                   sage)
    finally:
        fused.USE_REFERENCE = was
    return out


@pytest.mark.skipif(not sol_veda.available('cuda'),
                    reason='Sol + Veda unavailable: '
                           f'{sol_veda.unavailable_reason()}')
def test_prepare_matches_the_passes_it_replaces():
    """The fused host pass folds three torch passes into one.

    The INT8 half must be bit-equal, because the exact blocks' scores
    are built from it and a different scale is a different kernel. The
    two reductions are fp32 sums of the same values in a different
    order, so they are pinned to within one bf16 ulp instead -- measured
    at 1 element in 11776 on this layout.
    """
    from miowtion.kernels.sol_veda_patched import attention as fused
    _attend, sage = sol_veda._modules()                    # noqa: SLF001
    lay = _layout(grid=(8, 12, 20))
    _, kt, vt = _tiles(lay)
    counts = lay.valid_count.to(torch.int32)
    kc, vc, ki, ks = fused.prepare(kt, vt, counts)
    ki_want, ks_want = sage.quantize(kt, fused.QUANT_BLOCK)
    kc_want, vc_want = fused._summaries(kt, vt, counts)    # noqa: SLF001
    shape = (lay.n_tiles, kt.shape[1], kt.shape[2])
    assert torch.equal(ki, ki_want)
    assert torch.equal(ks, ks_want)
    for got, want in ((kc, kc_want.view(shape)), (vc, vc_want.view(shape))):
        differ = got != want
        assert int(differ.sum()) <= got.numel() // 1000
        gap = (got[differ].float() - want[differ].float()).abs()
        scale = want[differ].float().abs().clamp(min=1e-30)
        # bf16 keeps 8 mantissa bits, so one ulp is 2**-8 relative.
        assert gap.numel() == 0 or float((gap / scale).max()) <= 2. ** -7


@pytest.mark.skipif(not sol_veda.available('cuda'),
                    reason='Sol + Veda unavailable: '
                           f'{sol_veda.unavailable_reason()}')
def test_fast_walk_is_bit_equal_with_nothing_skipped():
    """With every tile kept there is no compensation, and the two walks
    visit the key tiles in the same order, so this is exact."""
    lay = _layout(grid=(8, 12, 20))
    qt, kt, vt = _tiles(lay)
    mask = torch.ones(4, lay.n_tiles, lay.n_tiles, dtype=torch.bool,
                      device='cuda') & lay.kv_ok[None, None, :]
    mask = mask.contiguous()
    assert torch.equal(_fused(qt, kt, vt, mask, lay),
                       _fused(qt, kt, vt, mask, lay, reference_kernel=True))


@pytest.mark.skipif(not sol_veda.available('cuda'),
                    reason='Sol + Veda unavailable: '
                           f'{sol_veda.unavailable_reason()}')
def test_fast_walk_matches_the_published_kernel_under_a_mask():
    """Under a real mask the two differ only by the order in which the
    online softmax sees the blocks, which is rounding."""
    lay = _layout(grid=(8, 12, 20))
    qt, kt, vt = _tiles(lay)
    mask = _masked_mask(lay)
    got = _fused(qt, kt, vt, mask, lay)
    want = _fused(qt, kt, vt, mask, lay, reference_kernel=True)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(got[real].float(), want[real].float(),
                               rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not sol_veda.available('cuda'),
                    reason='Sol + Veda unavailable: '
                           f'{sol_veda.unavailable_reason()}')
def test_fused_matches_dense_with_nothing_skipped():
    """The INT8 path against the bf16 reference, compensation empty."""
    lay = _layout(grid=(8, 12, 20))
    qt, kt, vt = _tiles(lay)
    mask = (torch.ones(4, lay.n_tiles, lay.n_tiles, dtype=torch.bool,
                       device='cuda') & lay.kv_ok[None, None, :]).contiguous()
    got = _fused(qt, kt, vt, mask, lay)
    want = reference.block_sparse_attention(qt, kt, vt, mask,
                                            lay.valid_count)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(got[real].float(), want[real].float(),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not fa4.available() or compensation_triton._triton()
                    is None, reason='needs FA4 block sparsity and Triton')
def test_triton_compensation_matches_the_torch_form():
    """Same term as `veda.attention.zero_order_compensation`.

    The only deliberate difference is bf16 summaries where the torch
    form uses fp32, so the gap is bounded by the summaries' rounding and
    not by the size of the correction.
    """
    lay = _layout(grid=(8, 12, 20))
    qt, kt, vt = _tiles(lay)
    mask = _masked_mask(lay)
    o_kept, lse = fa4.block_sparse_attention(qt, kt, vt, mask, lay,
                                             return_lse=True)
    got = compensation_triton.compensate(o_kept, lse, qt, kt, vt, mask, lay)
    want = veda_attention.zero_order_compensation(o_kept, lse, qt, kt, vt,
                                                   mask, lay)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(got[real].float(), want[real].float(),
                               rtol=4e-3, atol=4e-3)


@pytest.mark.skipif(not fa4.available() or compensation_triton._triton()
                    is None, reason='needs FA4 block sparsity and Triton')
def test_fa4_compensation_is_dense_with_nothing_skipped():
    """Nothing dropped means nothing to compensate, so this is FA4."""
    lay = _layout(grid=(8, 12, 20))
    qt, kt, vt = _tiles(lay)
    mask = torch.ones(4, lay.n_tiles, lay.n_tiles, dtype=torch.bool,
                      device='cuda')
    got = compensation_triton.attention(qt, kt, vt, mask, lay)
    want = fa4.block_sparse_attention(qt, kt, vt,
                                      mask & lay.kv_ok[None, None, :], lay)
    real = lay.slot_valid.bool()
    torch.testing.assert_close(got[real].float(), want[real].float(),
                               rtol=1e-3, atol=1e-3)
