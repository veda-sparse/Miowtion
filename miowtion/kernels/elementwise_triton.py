"""Fused Triton kernels for the memory-bound elementwise chains of a block.

In eager mode every op of these chains is its own kernel that reads and
writes a full [S, hidden] (or [S, H, D]) bf16 tensor: RoPE is four kernels
(mul, cat, neg, mul, add plus the pass-through copy), the AdaLN modulation
and the gated residual are an index_select, a mul and an add each, SwiGLU a
silu and a mul. Together they were ~17% of a Veda step on an RTX PRO 6000
at latent_t 102 (docs/benchmark/performance.md §11.2), all of it memory
traffic. Each kernel here reads its inputs once and writes its output once.

Numerics: bitwise equal to the eager chains, not merely close. Every
intermediate that eager materializes as a bf16 tensor is rounded to bf16
here too (fp32 compute, round-to-nearest-even, exactly like PyTorch's
opmath), in the same order. silu uses libdevice `exp` and correctly
rounded division, which is what PyTorch's CUDA silu compiles to; Triton's
default `exp` / `/` are approximations and would differ in the last bit.
tests/gpu/test_elementwise_triton_gpu.py pins all of this with torch.equal
(and silu over every bf16 value).
"""

from __future__ import annotations

import functools

import torch

# Launch options of every kernel here: no fp32 mul + add contraction into
# fma, and libdevice without flush-to-zero (PyTorch keeps subnormals; with
# Triton's default ftz, silu of a subnormal bf16 comes out as -0.0).
_EXACT = {'enable_fp_fusion': False, 'enable_reflect_ftz': False}

# Columns per program of the row kernels. hidden = 5376 = 5.25 * 1024, so
# the last block is masked; ffn = 14336 = 14 * 1024 exactly.
_BLOCK_COLS = 1024


@functools.cache
def _triton():
    try:
        import triton  # pylint: disable=import-outside-toplevel
        import triton.language as tl  # pylint: disable=import-outside-toplevel
        from triton.language.extra import libdevice  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return triton, tl, libdevice


def available() -> bool:
    return torch.cuda.is_available() and _triton() is not None


def usable(*tensors: torch.Tensor) -> bool:
    """Whether the fused kernels apply: CUDA, Triton, all bf16."""
    return available() and all(
        t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)


@functools.cache
def _kernels():
    triton, tl, libdevice = _triton()

    @triton.jit
    def _bf16(x):
        # Materialize an eager intermediate: round to bf16 (nearest even),
        # keep computing in fp32. Done on the bits: the compiler folds a
        # `.to(tl.bfloat16).to(tl.float32)` round trip away and then
        # contracts the surrounding mul + add into one fp32 fma, which is
        # exactly the rounding eager does *not* skip.
        bits = x.to(tl.uint32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return tl.where(x != x, x, bits.to(tl.float32, bitcast=True))

    @triton.jit
    def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, heads,
                    stride_xs, stride_xh, stride_os, stride_oh,
                    stride_cs, stride_ss,
                    HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
                    BLOCK_H: tl.constexpr):
        row = tl.program_id(0)
        h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        d = tl.arange(0, HEAD_DIM)
        half: tl.constexpr = ROPE_DIM // 2
        hmask = h < heads
        x_row = x_ptr + row * stride_xs + h[:, None] * stride_xh
        x = tl.load(x_row + d[None, :], mask=hmask[:, None], other=0.0)
        rot = d < ROPE_DIM
        # rotate_half: [-x2, x1] over the rope dims.
        partner = tl.where(d < half, d + half, d - half)
        xp = tl.load(x_row + partner[None, :],
                     mask=hmask[:, None] & rot[None, :], other=0.0)
        xp = tl.where((d < half)[None, :], -xp.to(tl.float32),
                      xp.to(tl.float32))
        c = tl.load(cos_ptr + row * stride_cs + d, mask=rot,
                    other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + row * stride_ss + d, mask=rot,
                    other=0.0).to(tl.float32)
        xf = x.to(tl.float32)
        y = _bf16(_bf16(xf * c[None, :]) + _bf16(xp * s[None, :]))
        y = tl.where(rot[None, :], y, xf)
        tl.store(out_ptr + row * stride_os + h[:, None] * stride_oh
                 + d[None, :], y.to(tl.bfloat16), mask=hmask[:, None])

    @triton.jit
    def modulate_kernel(x_ptr, scale_ptr, shift_ptr, index_ptr, out_ptr,
                        cols, stride_x, stride_scale, stride_shift,
                        stride_out, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = c < cols
        i = tl.load(index_ptr + row)
        x = tl.load(x_ptr + row * stride_x + c, mask=m).to(tl.float32)
        a = tl.load(scale_ptr + i * stride_scale + c, mask=m).to(tl.float32)
        b = tl.load(shift_ptr + i * stride_shift + c, mask=m).to(tl.float32)
        y = _bf16(x * a) + b
        tl.store(out_ptr + row * stride_out + c, y.to(tl.bfloat16), mask=m)

    @triton.jit
    def gated_residual_kernel(x_ptr, gate_ptr, index_ptr, h_ptr, out_ptr,
                              cols, stride_x, stride_gate, stride_h,
                              stride_out, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = c < cols
        i = tl.load(index_ptr + row)
        x = tl.load(x_ptr + row * stride_x + c, mask=m).to(tl.float32)
        g = tl.load(gate_ptr + i * stride_gate + c, mask=m).to(tl.float32)
        h = tl.load(h_ptr + row * stride_h + c, mask=m).to(tl.float32)
        y = x + _bf16(g * h)
        tl.store(out_ptr + row * stride_out + c, y.to(tl.bfloat16), mask=m)

    @triton.jit
    def swiglu_kernel(gate_ptr, up_ptr, out_ptr, cols, stride_gate,
                      stride_up, stride_out, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = c < cols
        g = tl.load(gate_ptr + row * stride_gate + c, mask=m).to(tl.float32)
        u = tl.load(up_ptr + row * stride_up + c, mask=m).to(tl.float32)
        # PyTorch's CUDA silu: x / (1 + expf(-x)) in fp32, IEEE division.
        act = _bf16(libdevice.div_rn(g, 1.0 + libdevice.exp(-g)))
        tl.store(out_ptr + row * stride_out + c, (act * u).to(tl.bfloat16),
                 mask=m)

    return rope_kernel, modulate_kernel, gated_residual_kernel, swiglu_kernel


def _check_rows(name: str, *tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.stride(-1) != 1:
            raise ValueError(f'{name}: last dim must be contiguous')


def rope(x: torch.Tensor, cos: torch.Tensor,
         sin: torch.Tensor) -> torch.Tensor:
    """miowtion.h3.model.apply_rope, fused.

    Args:
        x: [S, H, D] bf16, last dim contiguous.
        cos: [S, 1, rope_dim] bf16, rope_dim even and <= D.
        sin: [S, 1, rope_dim] bf16.

    Returns:
        [S, H, D] bf16, contiguous.
    """
    seq, heads, head_dim = x.shape
    rope_dim = cos.shape[-1]
    if rope_dim % 2 or rope_dim > head_dim or cos.shape != sin.shape:
        raise ValueError(f'bad rope tables {tuple(cos.shape)} for {x.shape}')
    _check_rows('rope', x, cos, sin)
    out = torch.empty(seq, heads, head_dim, dtype=x.dtype, device=x.device)
    block_h = 8
    grid = (seq, (heads + block_h - 1) // block_h)
    _kernels()[0][grid](
        x, cos, sin, out, heads, x.stride(0), x.stride(1), out.stride(0),
        out.stride(1), cos.stride(0), sin.stride(0), HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim, BLOCK_H=block_h, **_EXACT)
    return out


def modulate(x: torch.Tensor, one_plus_scale: torch.Tensor,
             shift: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """x * one_plus_scale[index] + shift[index], fused.

    Args:
        x: [S, C] bf16.
        one_plus_scale: [M, C] bf16.
        shift: [M, C] bf16.
        index: [S] int64 rows of the tables.

    Returns:
        [S, C] bf16.
    """
    _check_rows('modulate', x, one_plus_scale, shift)
    rows, cols = x.shape
    out = torch.empty_like(x)
    grid = (rows, triton_cdiv(cols, _BLOCK_COLS))
    _kernels()[1][grid](
        x, one_plus_scale, shift, index, out, cols, x.stride(0),
        one_plus_scale.stride(0), shift.stride(0), out.stride(0),
        BLOCK=_BLOCK_COLS, **_EXACT)
    return out


def gated_residual(x: torch.Tensor, gate: torch.Tensor,
                   index: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x + gate[index] * h, fused.

    Args:
        x: [S, C] bf16 residual stream.
        gate: [M, C] bf16.
        index: [S] int64 rows of `gate`.
        h: [S, C] bf16 branch output.

    Returns:
        [S, C] bf16.
    """
    _check_rows('gated_residual', x, gate, h)
    rows, cols = x.shape
    out = torch.empty_like(x)
    grid = (rows, triton_cdiv(cols, _BLOCK_COLS))
    _kernels()[2][grid](
        x, gate, index, h, out, cols, x.stride(0), gate.stride(0),
        h.stride(0), out.stride(0), BLOCK=_BLOCK_COLS, **_EXACT)
    return out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, fused; gate / up may be strided halves of fc1.

    Args:
        gate: [S, F] bf16.
        up: [S, F] bf16.

    Returns:
        [S, F] bf16, contiguous.
    """
    _check_rows('swiglu', gate, up)
    rows, cols = gate.shape
    out = torch.empty(rows, cols, dtype=gate.dtype, device=gate.device)
    grid = (rows, triton_cdiv(cols, _BLOCK_COLS))
    _kernels()[3][grid](gate, up, out, cols, gate.stride(0), up.stride(0),
                        out.stride(0), BLOCK=_BLOCK_COLS, **_EXACT)
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
