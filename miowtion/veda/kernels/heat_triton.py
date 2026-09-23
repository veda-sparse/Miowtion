"""Fused Triton kernel for the teacher block heat.

The torch reference materializes [H', rows, N] bf16 scores (tens of GB per
layer on long clips) only to reduce them to one value per 128x128 block.
This kernel computes each block's QK^T in registers and writes a single
fp32 heat value, so the cost is the QK^T FLOPs alone.

Numerics mirror the reference: scores are rounded to bf16 before the max,
the scale is applied after the max, invalid keys are -inf and invalid query
rows contribute nothing. Accumulation order differs from cuBLAS, so parity
with the reference is to tolerance, not bitwise (see
tests/gpu/test_heat_triton.py).
"""

from __future__ import annotations

import functools
import math

import torch

from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE
# Key tiles handled by one program; each program loads its query tile once.
_KEY_TILES_PER_PROGRAM = 16


@functools.cache
def _triton():
    try:
        import triton  # pylint: disable=import-outside-toplevel
        import triton.language as tl  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return triton, tl


def available() -> bool:
    return torch.cuda.is_available() and _triton() is not None


@functools.cache
def _kernel():
    triton, tl = _triton()

    @triton.jit
    def heat_kernel(q_ptr, k_ptr, lse_ptr, valid_ptr, q_tiles_ptr, out_ptr,
                    stride_qn, stride_qh, stride_kn, stride_kh, stride_ln,
                    stride_lh, stride_oh, stride_or, n_tiles, scale,
                    HEAD_DIM: tl.constexpr, TILE: tl.constexpr,
                    TILES_PER_PROGRAM: tl.constexpr):
        r = tl.program_id(0)
        group = tl.program_id(1)
        h = tl.program_id(2)
        q_tile = tl.load(q_tiles_ptr + r)
        rows = q_tile * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + rows[:, None] * stride_qn + h * stride_qh
                    + dims[None, :])
        lse = tl.load(lse_ptr + rows * stride_ln + h * stride_lh)
        row_ok = tl.load(valid_ptr + rows) != 0
        for i in range(TILES_PER_PROGRAM):
            j = group * TILES_PER_PROGRAM + i
            if j < n_tiles:
                cols = j * TILE + tl.arange(0, TILE)
                k = tl.load(k_ptr + cols[:, None] * stride_kn + h * stride_kh
                            + dims[None, :])
                col_ok = tl.load(valid_ptr + cols) != 0
                s = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
                s = tl.where(col_ok[None, :], s, float('-inf'))
                row_best = tl.max(s, axis=1) * scale - lse
                row_best = tl.where(row_ok, row_best, float('-inf'))
                best = tl.max(row_best, axis=0)
                tl.store(out_ptr + h * stride_oh + r * stride_or + j,
                         tl.exp(best))

    return heat_kernel


@torch.no_grad()
def teacher_heat(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor,
                 layout: tiling.TileLayout,
                 q_tiles: torch.Tensor) -> torch.Tensor:
    """Same contract as heatmap.teacher_heat_reference."""
    if q.shape[-1] not in (64, 128) or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError('heat kernel needs contiguous head_dim 64 or 128')
    heads, n_tiles = q.shape[1], layout.n_tiles
    out = torch.empty(heads, q_tiles.numel(), n_tiles, dtype=torch.float32,
                      device=q.device)
    grid = (q_tiles.numel(), -(-n_tiles // _KEY_TILES_PER_PROGRAM), heads)
    _kernel()[grid](
        q, k, lse, layout.slot_valid, q_tiles.contiguous(), out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), lse.stride(0),
        lse.stride(1), out.stride(0), out.stride(1), n_tiles,
        1.0 / math.sqrt(q.shape[-1]), HEAD_DIM=q.shape[-1], TILE=_TILE,
        TILES_PER_PROGRAM=_KEY_TILES_PER_PROGRAM, num_warps=8)
    return out
