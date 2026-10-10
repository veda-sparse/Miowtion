"""The Wan processor swap, checked against the upstream one.

The rotary embedding here is transcribed from diffusers' own processor,
and a transcription that is subtly wrong would route every layer through
slightly wrong queries and keys without failing. So the test is that
routing a *dense* attention through the swap reproduces what upstream
produces, bit for bit where the kernels allow.
"""

import pytest
import torch

diffusers = pytest.importorskip('diffusers')

from miowtion.wan import attention as wan_attention          # noqa: E402


def _module(heads=2, dim=8):
    """Wan's own attention module, built the way its block builds attn1."""
    from diffusers.models.transformers.transformer_wan import (
        WanAttention, WanAttnProcessor)

    torch.manual_seed(0)
    attn = WanAttention(dim=heads * dim, heads=heads, dim_head=dim,
                        eps=1e-6, cross_attention_dim_head=None,
                        processor=WanAttnProcessor())
    return attn.eval()


def _rope(seq, heads, dim):
    torch.manual_seed(1)
    angle = torch.randn(1, seq, 1, dim)
    return torch.cos(angle), torch.sin(angle)


def test_the_swap_reproduces_upstream_self_attention():
    heads, dim, seq = 2, 8, 12
    attn = _module(heads, dim)
    x = torch.randn(1, seq, heads * dim)
    rope = _rope(seq, heads, dim)

    with torch.no_grad():
        want = attn.processor(attn, x, None, None, rope)

    def dense(q, k, v, layer_index):
        del layer_index
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None],
            v.transpose(0, 1)[None])
        return out[0].transpose(0, 1)

    attn.set_processor(wan_attention.make_processor(dense, layer_index=0))
    with torch.no_grad():
        got = attn.processor(attn, x, None, None, rope)

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_a_batch_raises_instead_of_silently_taking_the_first_clip():
    attn = _module()
    attn.set_processor(wan_attention.make_processor(
        lambda q, k, v, layer_index: v, layer_index=0))
    x = torch.randn(2, 6, 16)
    with pytest.raises(ValueError, match='batch 2'):
        attn.processor(attn, x, None, None, _rope(6, 2, 8))
