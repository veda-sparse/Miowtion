"""GPU tests of miowtion.h3.attention.dense_attention.

The packed sequence is long (38k tokens at 16:9@37, 104k at t102), so a
dense call that falls back to the math backend materializes [H, S, S] and
dies. These tests pin that the fused path is actually taken.
"""

import pytest
import torch

from miowtion.h3 import attention as h3_attention

pytestmark = pytest.mark.gpu

# The real failure was 16:9@37: 38010 packed tokens, where the math
# fallback asked for 303.50 GiB in one allocation at 56 heads. The math
# backend chunks, so at the 8 heads used here it peaks around 11 GiB
# instead -- still an order of magnitude above the fused kernel's O(S)
# working set (~78 MiB per tensor), which is what the budget below pins.
_LONG = 38010
_FUSED_BUDGET = 4 * 2**30


def _qkv(seq_len, heads=8, dim=128, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    return [torch.randn(seq_len, heads, dim, generator=g, device='cuda',
                        dtype=torch.bfloat16) for _ in range(3)]


@pytest.mark.parametrize('return_lse', (False, True))
def test_sdpa_handles_a_full_length_sequence(return_lse):
    """A 3-D SDPA call is served by the math backend; this must not regress."""
    q, k, v = _qkv(_LONG)
    free_before = torch.cuda.mem_get_info()[0]
    out, lse = h3_attention.dense_attention(q, k, v, _LONG,
                                            return_lse=return_lse,
                                            backend='sdpa')
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out.float()).all()
    if return_lse:
        assert lse.shape == q.shape[:2] and lse.dtype == torch.float32
    else:
        assert lse is None
    # A fused kernel is O(S) in memory; the math fallback is O(S^2).
    used = free_before - torch.cuda.mem_get_info()[0]
    assert used < _FUSED_BUDGET, (
        f'{used / 2**30:.1f} GiB used; the fused kernel needs well under '
        '1 GiB here, so this is the math fallback')


def test_sdpa_matches_the_math_reference_on_a_short_sequence():
    q, k, v = _qkv(512, heads=4)
    fused, _ = h3_attention.dense_attention(q, k, v, 512, backend='sdpa')
    exact, _ = h3_attention.dense_attention(q, k, v, 512, backend='math')
    torch.testing.assert_close(fused.float(), exact.float(), rtol=2e-2,
                               atol=2e-2)


def test_padding_rows_stay_zero():
    seq = 1024
    used = 700
    q, k, v = _qkv(seq, heads=4)
    out, _ = h3_attention.dense_attention(q, k, v, used, backend='sdpa')
    assert torch.equal(out[used:], torch.zeros_like(out[used:]))
