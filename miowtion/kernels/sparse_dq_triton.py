"""Ordered sparse dQ reduction for reproducible SM8x/SM120 training.

FA4's KV-parallel backward atomically adds into dQ. Here each CTA owns
one query subtile/head and visits kept key tiles in ascending order.
The derivative formula is unchanged: dQ = (P * (dO V^T - sum(dO O))) K / sqrt(D).
FA4 still supplies the forward, LSE, dK and dV. No dense attention matrix
is materialized and no atomic accumulation is used by this kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _query_gradient(
            Q, K, V, O, DO, LSE, COUNTS, INDICES, VALID, DQ,
            H: tl.constexpr, TILES: tl.constexpr, D: tl.constexpr,
            Q0: tl.constexpr, Q1: tl.constexpr,
            K0: tl.constexpr, K1: tl.constexpr,
            V0: tl.constexpr, V1: tl.constexpr,
            O0: tl.constexpr, O1: tl.constexpr,
            DO0: tl.constexpr, DO1: tl.constexpr,
            L0: tl.constexpr, L1: tl.constexpr,
            SCALE: tl.constexpr, SCALE_LOG2: tl.constexpr,
            BQ: tl.constexpr = 64, BK: tl.constexpr = 64):
        block = tl.program_id(0)
        head = tl.program_id(1)
        rows = block * BQ + tl.arange(0, BQ)
        dim = tl.arange(0, D)
        q = tl.load(Q + rows[:, None] * Q0 + head * Q1 + dim[None, :])
        do = tl.load(DO + rows[:, None] * DO0 + head * DO1 + dim[None, :])
        o = tl.load(O + rows[:, None] * O0 + head * O1 + dim[None, :])
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        lse = tl.load(LSE + head * L0 + rows * L1) * 1.4426950408889634
        row = head * TILES + block // 2
        count = tl.load(COUNTS + row)
        acc = tl.full((BQ, D), 0, tl.float32)
        for pos in range(count):
            tile = tl.load(INDICES + row * TILES + pos)
            for subtile in tl.static_range(2):
                cols = tile * 128 + subtile * BK + tl.arange(0, BK)
                valid = tl.load(VALID + cols) != 0
                k = tl.load(K + cols[:, None] * K0 + head * K1 + dim[None, :])
                v = tl.load(V + cols[:, None] * V0 + head * V1 + dim[None, :])
                scores = tl.dot(q, tl.trans(k)) * SCALE_LOG2
                p = tl.exp2(scores - lse[:, None])
                p = tl.where(valid[None, :], p, 0.)
                dp = tl.dot(do, tl.trans(v))
                ds = (p * (dp - delta[:, None])).to(q.dtype)
                acc += tl.dot(ds, k)
        offsets = rows[:, None] * H * D + head * D + dim[None, :]
        tl.store(DQ + offsets, acc * SCALE)


def query_gradient(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor,
                   block_mask: torch.Tensor, slot_valid: torch.Tensor
                   ) -> torch.Tensor:
    """Ordered dQ for BF16 MHA, padded sequences and 128x128 block masks.

    Args:
        q, k, v, out, dout: [N, H, 128] BF16, last dimension contiguous.
        lse: [H, N] FP32 natural log of the FA4 softmax denominator.
        block_mask: [H, N/128, N/128] Boolean kept blocks, including globals.
        slot_valid: [N] integer key validity, excluding tile padding.

    Returns:
        [N, H, 128] BF16 dQ, bitwise repeatable for identical inputs.
    """
    if triton is None:
        raise RuntimeError('ordered sparse dQ requires Triton')
    if (q.device.type != 'cuda' or q.dtype != torch.bfloat16
            or q.ndim != 3 or q.shape[-1] != 128 or q.shape[0] % 128
            or q.shape[0] == 0 or q.shape[1] == 0
            or any(t.shape != q.shape or t.dtype != q.dtype
                   or t.device != q.device or t.stride(-1) != 1
                   for t in (q, k, v, out, dout))):
        raise ValueError('ordered dQ requires matching CUDA BF16 MHA D128')
    n, heads, dim = q.shape
    tiles = n // 128
    if (block_mask.shape != (heads, tiles, tiles)
            or block_mask.dtype != torch.bool or lse.shape != (heads, n)
            or slot_valid.shape != (n,) or lse.dtype != torch.float32
            or slot_valid.stride(0) != 1
            or any(t.device != q.device for t in
                   (block_mask, lse, slot_valid))):
        raise ValueError('ordered dQ block mask/LSE/validity mismatch')
    # Stable packing preserves ascending key-tile order, including ties
    # among kept blocks. At most one row of counts is read per query CTA.
    counts = block_mask.sum(-1).to(torch.int32).contiguous()
    indices = torch.argsort((~block_mask).to(torch.int8), dim=-1,
                            stable=True).to(torch.int32).contiguous()
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    _query_gradient[(n // 64, heads)](
        q, k, v, out, dout, lse, counts, indices, slot_valid, dq,
        heads, tiles, dim, *q.stride()[:2], *k.stride()[:2],
        *v.stride()[:2], *out.stride()[:2], *dout.stride()[:2],
        *lse.stride(), dim**-0.5, dim**-0.5 * 1.4426950408889634,
        num_warps=4, num_stages=1)
    return dq
