"""Dense self-attention over the real rows of a packed sequence.

Rows [0, used) form one attention document; padding rows [used, S) never
exchange information with real rows, so their output is defined as zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from miowtion.kernels import fa4


def _math_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention in fp32. q, k, v: [S, H, D]."""
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)  # [H, S]
    probs = torch.exp(scores - lse[..., None])
    out = torch.einsum('hqk,khd->qhd', probs, v.float())
    return out.to(q.dtype), lse.transpose(0, 1).contiguous()


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    used: int, return_lse: bool = False,
                    backend: str = 'auto'
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Full attention of rows [0, used) among themselves.

    Args:
        q: [S, H, D] bf16 (post QK-norm and RoPE).
        k: [S, H, D] bf16.
        v: [S, H, D] bf16.
        used: Number of real rows.
        return_lse: Also return the per-row log-sum-exp of the scaled scores.
        backend: 'auto' (FA4 on CUDA when installed, else SDPA), 'fa4',
            'sdpa' or 'math' (fp32 reference).

    Returns:
        out: [S, H, D], zero on padding rows.
        lse: [S, H] fp32 natural-log LSE of q.k / sqrt(D) (0 on padding
            rows), or None.
    """
    seq_len, heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    qr, kr, vr = q[:used], k[:used], v[:used]
    if backend == 'auto':
        backend = 'fa4' if fa4.dense_available(q.device) else (
            'sdpa' if not return_lse or q.is_cuda else 'math')
    if backend == 'fa4':
        out, lse = fa4.dense_attention(qr, kr, vr, scale, return_lse)
    elif backend == 'sdpa':
        if return_lse:
            # The flash kernel returns LSE [B, H, S]; no public API does.
            out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
                qr.transpose(0, 1)[None], kr.transpose(0, 1)[None],
                vr.transpose(0, 1)[None], scale=scale)[:2]
            out = out[0].transpose(0, 1)
            lse = lse[0].transpose(0, 1).float()
        else:
            # The batch axis is not optional: every fused SDPA kernel
            # requires 4-D q/k/v, and a 3-D call is silently served by the
            # math backend, which materializes [H, S, S]. At 38k tokens
            # that is 300+ GiB, so the packed sequence must be batched.
            out = F.scaled_dot_product_attention(
                qr.transpose(0, 1)[None], kr.transpose(0, 1)[None],
                vr.transpose(0, 1)[None], scale=scale)[0].transpose(0, 1)
            lse = None
    elif backend == 'math':
        out, lse = _math_attention(qr, kr, vr, scale)
        lse = lse if return_lse else None
    else:
        raise ValueError(f'unknown attention backend {backend!r}')
    if used == seq_len:
        full_out = out.contiguous()
        full_lse = lse
    else:
        full_out = q.new_zeros(seq_len, heads, head_dim)
        full_out[:used] = out
        full_lse = None
        if lse is not None:
            full_lse = lse.new_zeros(seq_len, heads)
            full_lse[:used] = lse
    return full_out, full_lse
