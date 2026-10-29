"""Ordered sparse backward: reference accuracy and bitwise repeatability."""

import pytest
import torch

from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.veda import tiling

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not fa4.available() or torch.cuda.get_device_capability()[0] not in
    fa4.PATCHED_MAJOR_ARCHS, reason='requires patched SM8x/SM120 FA4')]


@pytest.mark.parametrize('seed', [11, 23])
def test_ordered_dq_matches_sparse_reference_and_repeats(seed):
    torch.manual_seed(seed)
    span = tiling.TiledSpan(93, (3, 9, 19), tiling.TileShape(1, 8, 16))
    layout = tiling.build_tile_layout(
        [span], span.start + span.num_rows, span.start + span.num_rows + 32,
        'cuda')
    tensors = [torch.randn(layout.num_slots, 3, 128, device='cuda',
                           dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3)]
    mask = torch.rand(3, layout.n_tiles, layout.n_tiles, device='cuda') < .3
    mask[:, :, -1] = True
    mask[:, 0] = False  # empty query blocks must give zero dQ
    do = torch.randn_like(tensors[0])
    native = fa4.block_sparse_attention(*tensors, mask, layout)
    native_gradients = torch.autograd.grad(native, tensors, do)
    ref = reference.block_sparse_attention(*tensors, mask, layout.valid_count)
    ref_gradients = torch.autograd.grad(ref, tensors, do)
    first = None
    for _ in range(3):
        out = fa4.block_sparse_attention(*tensors, mask, layout,
                                         ordered_dq=True)
        assert torch.equal(out, native)
        gradients = torch.autograd.grad(out, tensors, do)
        if first is None:
            first = gradients
        assert all(torch.equal(a, b) for a, b in zip(gradients, first))
        # Preserve FA4's sparse dK/dV path exactly; no dense substitution.
        assert all(torch.equal(a, b) for a, b in
                   zip(gradients[1:], native_gradients[1:]))
        for actual, expected in zip(gradients, ref_gradients):
            torch.testing.assert_close(actual.float(), expected.float(),
                                       rtol=2e-2, atol=2e-2)
        assert (gradients[0][:128] == 0).all()
