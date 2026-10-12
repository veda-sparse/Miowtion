"""Compute-capability gating for the kernels, in one place.

Every kernel in this package has a floor on the GPU it can run on, and the
floor is not the same for every kernel or even for every dtype of the same
kernel. Without a single table the floors are implicit in whatever the
kernel happens to do, and an unsupported GPU fails deep inside a JIT
compile ('Unsupported dtype', 'mma16816 requires sm80') with no mention of
which of our kernels asked for what. This module makes the floor a
declaration the entry points check up front, so the error names the
architecture, the kernel and the specific operation that is missing.

Two different questions decide a floor, and conflating them is how the
first version of this table got two entries wrong (docs/pitfalls.md):

  1. *Can the toolchain build it?* Measured, not recalled:
     `scripts/probe_triton_floor.py` compiles each kernel shape ahead of
     time for a target capability and reads the lowering back out of the
     PTX. Triton 3.4 compiles fp16 / bf16 / fp32 `tl.dot` for sm_70 and
     up, and refuses an INT8 `tl.dot` below sm_80 outright (it fails
     inside TritonGPUAccelerateMatmul, even though Turing has IMMA in
     hardware).
  2. *Does it reach a tensor core?* Also from that probe's PTX: `mma`
     first appears at sm_80. On sm_70 and sm_75 every `tl.dot` lowers to
     CUDA-core `fma`, so a Triton kernel there is not "a bit slower", it
     is off the tensor-core path entirely -- which for the heat kernel
     makes it *slower than the torch reference it exists to replace*,
     because cuBLAS still reaches the tensor cores.

So a kernel's gate is the higher of the two, and `Requirement` records
the compile floor separately when they differ, so nobody re-derives the
distinction from the gate alone.

The hardware facts the rest of the table rests on:

  * fp16 tensor cores (HMMA) exist from sm_70 on, INT8 (IMMA) from
    sm_75 on -- but see above: Triton 3.4 emits neither below sm_80.
  * `cp.async` (and so FlashAttention's pipelined tile loads) is sm_80
    on, which is dtype-independent and is what keeps every FA4 path off
    sm_7x.
  * fp8 arithmetic is sm_89 on; TMA descriptors are sm_90 on.
  * bf16 tensor cores and the `cvt.f32.bf16` instruction are sm_80 on.
    Triton nevertheless compiles a bf16 load-widen-reduce kernel for
    sm_70, so it must emit the conversion some other way; the bf16
    floors below are therefore a deliberate refusal (nothing has ever
    run that emulation on 7.x silicon) and not a compiler limit.

`require()` raises rather than silently selecting another path
(AGENTS.md 1.4). Callers that *have* a fallback ask `supported()` first
and route explicitly; the fallbacks that exist are
`veda.heatmap.teacher_heat_reference`, `veda.tiling`'s torch gather /
scatter, the eager chains behind `elementwise_triton.usable` and
`kernels.reference.block_sparse_attention` (opted into with
`allow_reference_kernel`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import torch

# Capability floors of the hardware features the kernels here use. Named
# so the error messages can say which one is missing.
FP16_MMA = (7, 0)
INT8_MMA = (7, 5)
BF16 = (8, 0)
CP_ASYNC = (8, 0)
FP8 = (8, 9)
TMA = (9, 0)
# What Triton 3.4 does, from scripts/probe_triton_floor.py (AOT compile
# for a target capability, lowering read out of the PTX):
# a float `tl.dot` builds from sm_70 on, reaches an `mma` only from
# sm_80 on, and an INT8 `tl.dot` does not build below sm_80 at all.
TRITON_COMPILES = (7, 0)
TRITON_MMA = (8, 0)
TRITON_INT8_DOT = (8, 0)

_ARCH_NAMES = {
    7: 'Volta/Turing',
    8: 'Ampere/Ada',
    9: 'Hopper',
    10: 'Blackwell',
    11: 'Blackwell',
    12: 'Blackwell (consumer)',
}
_MICRO_ARCH_NAMES = {
    (7, 0): 'Volta',
    (7, 2): 'Volta (Xavier)',
    (7, 5): 'Turing',
    (8, 0): 'Ampere',
    (8, 6): 'Ampere (consumer)',
    (8, 7): 'Ampere (Orin)',
    (8, 9): 'Ada',
    (9, 0): 'Hopper',
}


def capability(device: torch.device | str | int) -> tuple[int, int]:
    """The (major, minor) compute capability of a CUDA device.

    Args:
        device: A CUDA device, index or string.

    Returns:
        (major, minor), e.g. (7, 5) for a T4 or an RTX 2080.

    Raises:
        ValueError: If `device` is not a CUDA device. A capability
            question about a CPU tensor is a routing bug in the caller,
            not a device without features.
    """
    if not isinstance(device, int) and torch.device(device).type != 'cuda':
        raise ValueError(f'{device} is not a CUDA device; it has no '
                         'compute capability')
    return tuple(torch.cuda.get_device_capability(device))  # type: ignore


def name(cap: tuple[int, int]) -> str:
    """'sm_75 (Turing)' for (7, 5): what an error message should print."""
    arch = _MICRO_ARCH_NAMES.get(cap) or _ARCH_NAMES.get(cap[0])
    tag = f'sm_{cap[0]}{cap[1]}'
    return f'{tag} ({arch})' if arch else tag


@dataclasses.dataclass(frozen=True)
class Requirement:
    """What one kernel needs of the GPU, and why.

    Attributes:
        kernel: The name used in error messages, 'module.function' form.
        floor: The gate: the capability below which this kernel is not
            used, for every dtype.
        floor_reason: What sets `floor`, phrased so the message says what
            is missing and not just that it is.
        dtype_floors: Per-dtype floor and reason, on top of `floor`. A
            dtype absent from this mapping is not implemented by the
            kernel at all.
        compile_floor: Where the toolchain can still *build* this kernel,
            when that is below the gate. Set it whenever the two differ,
            so a later reader can see that the gate is our policy (the
            kernel would lose to its own fallback) and not a compiler
            error they could go and fix.
    """

    kernel: str
    floor: tuple[int, int]
    floor_reason: str
    dtype_floors: Mapping[torch.dtype, tuple[tuple[int, int], str]] = (
        dataclasses.field(default_factory=dict))
    compile_floor: tuple[int, int] | None = None


# Every dtype of a Triton kernel that multiplies with tl.dot shares one
# floor: the dot compiles from sm_70 on but only reaches an `mma` from
# sm_80 on, and below that it is CUDA-core `fma`.
_MMA_DOT = 'tensor-core tl.dot (sm_7x lowers it to CUDA-core fma)'
_TRITON_DOT_DTYPES = {
    torch.bfloat16: (TRITON_MMA, _MMA_DOT),
    torch.float16: (TRITON_MMA, _MMA_DOT),
    # fp32 is listed so routing does not change for a caller that was
    # already passing it.
    torch.float32: (TRITON_MMA, _MMA_DOT),
}
# A Triton kernel that only moves and reduces elements: no tl.dot, so
# no tensor core is involved and sm_70 is as good at it as sm_80. bf16
# is held at sm_80 by choice, not by the compiler -- Triton does build
# the bf16 widening for sm_70 (without cvt.f32.bf16, so by some
# emulation), but no 7.x card has ever run it here and the fallback
# (`veda.tiling`'s torch gather) is known good.
_TRITON_MOVE_DTYPES = {
    torch.bfloat16: (BF16,
                     'bf16 hardware conversion; Triton compiles an '
                     'emulation for sm_7x that we have never run, so '
                     'this is a refusal, not a compiler limit'),
    torch.float16: (TRITON_COMPILES, 'fp16 load/convert'),
    torch.float32: (TRITON_COMPILES, 'fp32 load/convert'),
}

REQUIREMENTS: dict[str, Requirement] = {
    'fa4.dense_attention': Requirement(
        kernel='fa4.dense_attention',
        floor=CP_ASYNC,
        floor_reason=('FlashAttention-4 CuTe kernels are built on cp.async '
                      'pipelined tile loads and sm_80 MMA shapes'),
        dtype_floors={
            torch.bfloat16: (BF16, 'bf16 MMA'),
            torch.float16: (CP_ASYNC, 'the sm_80 fp16 MMA tiling'),
        }),
    'fa4.block_sparse_attention': Requirement(
        kernel='fa4.block_sparse_attention',
        floor=CP_ASYNC,
        floor_reason=('FlashAttention-4 CuTe kernels are built on cp.async '
                      'pipelined tile loads and sm_80 MMA shapes; the '
                      'vendored block-sparse patch (fa4_sm8x) patches the '
                      'SM80 kernels and inherits that floor'),
        dtype_floors={
            torch.bfloat16: (BF16, 'bf16 MMA'),
            torch.float16: (CP_ASYNC, 'the sm_80 fp16 MMA tiling'),
        }),
    'block_heat_triton.teacher_heat': Requirement(
        kernel='block_heat_triton.teacher_heat',
        floor=TRITON_MMA,
        floor_reason=(
            "the kernel's QK^T is a tl.dot, and below sm_80 Triton "
            'lowers that to CUDA-core fma: the fused kernel would then '
            'be about 2x slower than the chunked torch reference it '
            'exists to replace, because cuBLAS still reaches the '
            'tensor cores there. It compiles from sm_70; this floor is '
            'the point below which it stops being worth calling'),
        dtype_floors=_TRITON_DOT_DTYPES,
        compile_floor=TRITON_COMPILES),
    'tile_gather_triton.gather': Requirement(
        kernel='tile_gather_triton.gather',
        floor=TRITON_COMPILES,
        floor_reason=('Triton itself (the kernel has no tl.dot, so no '
                      'tensor core and no sm_80 floor)'),
        dtype_floors=_TRITON_MOVE_DTYPES),
    'tile_gather_triton.scatter': Requirement(
        kernel='tile_gather_triton.scatter',
        floor=TRITON_COMPILES,
        floor_reason=('Triton itself (the kernel has no tl.dot, so no '
                      'tensor core and no sm_80 floor)'),
        dtype_floors=_TRITON_MOVE_DTYPES),
    'elementwise_triton': Requirement(
        kernel='elementwise_triton',
        floor=BF16,
        floor_reason=('the fused chains round every intermediate to bf16 '
                      'to stay bitwise equal to the eager path'),
        dtype_floors={torch.bfloat16: (BF16, 'bf16 load/convert')}),
    'compensation_triton.compensate': Requirement(
        kernel='compensation_triton.compensate',
        floor=BF16,
        floor_reason=('the exact blocks of this path go through '
                      'fa4.block_sparse_attention, which is sm_80 by way '
                      'of cp.async; the bf16 cast of the probability '
                      "block before its tl.dot is only the next thing "
                      'in the way'),
        dtype_floors={torch.bfloat16: (BF16, 'bf16 tl.dot')}),
    'h3.dense_attention.sdpa': Requirement(
        kernel="h3.attention.dense_attention backend='sdpa'",
        floor=FP16_MMA,
        floor_reason="PyTorch's fused SDPA kernels",
        dtype_floors={
            torch.bfloat16: (BF16,
                             'bf16 fused SDPA kernel below sm_80 (the math '
                             'backend would expand [H, S, S], 300+ GiB at '
                             '38k tokens)'),
            torch.float16: (FP16_MMA, 'fp16 memory-efficient SDPA'),
            # fp32 never had a bf16-style gap; listed so that a caller
            # already passing fp32 keeps working.
            torch.float32: (FP16_MMA, 'fp32 SDPA'),
        }),
    'h3.dense_attention.sdpa_lse': Requirement(
        kernel="h3.attention.dense_attention backend='sdpa', return_lse",
        floor=CP_ASYNC,
        floor_reason=('the only SDPA op that returns the log-sum-exp is the '
                      'flash backend, and it requires sm_80'),
        dtype_floors={
            torch.bfloat16: (CP_ASYNC, 'flash SDPA below sm_80'),
            torch.float16: (CP_ASYNC, 'flash SDPA below sm_80'),
        }),
    'sol_veda.block_sparse_attention': Requirement(
        kernel='sol_veda.block_sparse_attention',
        floor=TRITON_INT8_DOT,
        floor_reason=('Triton 3.4 refuses an INT8 tl.dot below sm_80 '
                      'outright (it fails in TritonGPUAccelerateMatmul) '
                      'even though Turing has IMMA, so the quantised '
                      'exact-block arm cannot be built there at all; the '
                      'bf16 summaries and online softmax would be the '
                      'next obstacle'),
        dtype_floors={torch.bfloat16: (BF16, 'bf16 online softmax')}),
}


def _requirement(kernel: str) -> Requirement:
    try:
        return REQUIREMENTS[kernel]
    except KeyError:
        raise KeyError(f'no capability requirement declared for {kernel!r}; '
                       f'known: {sorted(REQUIREMENTS)}') from None


def unsupported_reason(kernel: str, device: torch.device | str | int,
                       dtype: torch.dtype | None = None) -> str | None:
    """Why `kernel` cannot run here, or None when it can.

    Args:
        kernel: Key of `REQUIREMENTS`.
        device: The CUDA device the tensors are on.
        dtype: The tensor dtype, or None to ask only about the
            dtype-independent floor.

    Returns:
        A message naming the architecture, the kernel and the missing
        operation, or None.

    Raises:
        KeyError: If `kernel` has no declared requirement.
        ValueError: If `device` is not CUDA.
    """
    need = _requirement(kernel)
    cap = capability(device)
    if cap < need.floor:
        return (f'{need.kernel} needs sm_{need.floor[0]}{need.floor[1]} or '
                f'newer, this is {name(cap)}: {need.floor_reason}')
    if dtype is None:
        return None
    if dtype not in need.dtype_floors:
        supported = ', '.join(str(d).removeprefix('torch.')
                              for d in need.dtype_floors)
        return (f'{need.kernel} does not implement '
                f'{str(dtype).removeprefix("torch.")}; it implements '
                f'{supported}')
    floor, reason = need.dtype_floors[dtype]
    if cap < floor:
        return (f'{need.kernel} in '
                f'{str(dtype).removeprefix("torch.")} needs '
                f'sm_{floor[0]}{floor[1]} or newer, this is {name(cap)}: '
                f'no {reason}')
    return None


def supported(kernel: str, device: torch.device | str | int,
              dtype: torch.dtype | None = None) -> bool:
    """Whether `kernel` can run on `device` with `dtype`.

    False for a non-CUDA device rather than an error: callers route on
    this, and on a CUDA machine a CPU tensor is a normal thing to be
    asked about (it just has no Triton / CuTe path).
    """
    if not isinstance(device, int) and torch.device(device).type != 'cuda':
        return False
    return unsupported_reason(kernel, device, dtype) is None


def require(kernel: str, device: torch.device | str | int,
            dtype: torch.dtype | None = None) -> None:
    """Raises unless `kernel` can run on `device` with `dtype`.

    Raises:
        NotImplementedError: With the architecture, the kernel and the
            missing operation named. Deliberately not a silent fallback
            to another path (AGENTS.md 1.4); the callers that have one
            ask `supported()` instead.
        ValueError: If `device` is not a CUDA device.
        KeyError: If `kernel` has no declared requirement.
    """
    reason = unsupported_reason(kernel, device, dtype)
    if reason is not None:
        raise NotImplementedError(reason)
