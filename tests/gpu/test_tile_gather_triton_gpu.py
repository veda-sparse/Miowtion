"""Exact fused row permutation gradients, including padding/head chunks."""

import pytest
import torch

from miowtion.kernels import tile_gather_triton as kernel
from miowtion.veda import tiling

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not kernel.available(), reason='CUDA and Triton required')]


@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('pool', [False, True])
@pytest.mark.parametrize('head_ids', [None, [3, 1]])
def test_fused_gather_matches_torch_and_inference(dim, pool, head_ids):
    span = tiling.TiledSpan(17, (3, 5, 7), tiling.TileShape(2, 8, 8))
    used = span.start + span.num_rows
    layout = tiling.build_tile_layout([span], used, used + 11, 'cuda')
    heads = (None if head_ids is None else
             torch.tensor(head_ids, device='cuda'))
    torch.manual_seed(71)
    # QKV splitting produces strided rows/heads in the real model.
    source = torch.randn(layout.seq_len, 8, dim,
                         dtype=torch.bfloat16, device='cuda')[:, ::2, :]
    expected_x = source.detach().requires_grad_()
    expected = tiling.gather_tiles(expected_x, layout, heads)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    x = source.detach().requires_grad_()
    actual, features = kernel.differentiable_gather(x, layout, heads, pool)
    assert torch.equal(actual, expected)
    if pool:
        inference_rows, inference_features = kernel.gather_and_pool(
            source, layout, heads)
        assert torch.equal(actual, inference_rows)
        assert torch.equal(features, inference_features)
        assert not features.requires_grad
    else:
        assert features is None
    actual.backward(upstream)
    assert torch.equal(x.grad, expected_x.grad)
    # Broadcast/zero-stride upstream must be materialized for the kernel.
    x = source.detach().requires_grad_()
    kernel.differentiable_gather(x, layout, heads, pool)[0].sum().backward()
    expected_x = source.detach().requires_grad_()
    tiling.gather_tiles(expected_x, layout, heads).sum().backward()
    assert torch.equal(x.grad, expected_x.grad)
