"""Compute-capability gating for the kernels, in one place.

Every kernel in this package has a floor on the GPU it can run on, and the
floor is not the same for every kernel or even for every dtype of the same
kernel. Without a single table the floors are implicit in whatever the
kernel happens to do, and an unsupported GPU fails deep inside a JIT
compile ('Unsupported dtype', 'mma16816 requires sm80') with no mention of
which of our kernels asked for what. This module makes the floor a
declaration the entry points check up front, so the error names the
architecture, the kernel and the specific operation that is missing.

The floors come from the hardware, not from measurement here (see
docs/features/arch_support.md for the per-kernel table and for what is
unverified):

  * fp16 tensor cores (HMMA) exist from sm_70 (Volta) on.
  * INT8 tensor cores (IMMA) exist from sm_75 (Turing) on; Volta has
    none, so an INT8 `tl.dot` cannot be lowered there.
  * bf16 exists from sm_80 (Ampere) on -- both the BMMA tensor-core path
    and the `cvt` instructions that convert bf16 to fp32. sm_70 / sm_75
    therefore cannot even *load and widen* a bf16 tensor in a kernel, let
    alone multiply one, which is why a bf16-annotated kernel is not a
    "slow but correct" path below sm_80: it does not lower at all.
  * `cp.async` (and so FlashAttention's pipelined tile loads) is sm_80 on.
  * fp8 arithmetic is sm_89 on; TMA descriptors are sm_90 on.
  * Triton's NVIDIA backend dropped the Volta `mma.m8n8k4` lowering that
    `tl.dot` needs on sm_70; sm_75 keeps the mma16816 path. So sm_70 has
    fp16 tensor cores that Triton can no longer target, and every Triton
    kernel here that calls `tl.dot` has an sm_75 floor rather than an
    sm_70 one.

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
# Triton's NVIDIA backend no longer lowers `tl.dot` for Volta.
TRITON_DOT = (7, 5)

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
        floor: Capability floor that holds for every dtype.
        floor_reason: The operation that sets `floor`, phrased so the
            message says what is missing and not just that it is.
        dtype_floors: Per-dtype floor and reason, on top of `floor`. A
            dtype absent from this mapping is not implemented by the
            kernel at all.
    """

    kernel: str
    floor: tuple[int, int]
    floor_reason: str
    dtype_floors: Mapping[torch.dtype, tuple[tuple[int, int], str]] = (
        dataclasses.field(default_factory=dict))


# bf16 and fp16 in a Triton kernel that multiplies them with tl.dot. The
# fp16 floor is Triton's dot lowering (7, 5), not the hardware's (7, 0).
_TRITON_DOT_DTYPES = {
    torch.bfloat16: (BF16, 'bf16 tl.dot (no BMMA, and no cvt.f32.bf16)'),
    torch.float16: (TRITON_DOT,
                    "fp16 tl.dot (Triton dropped Volta's mma.m8n8k4)"),
    # fp32 keeps working because `tl.dot` takes fp32 operands; it is
    # listed so that routing on `supported()` does not change for a
    # caller that was already passing fp32.
    torch.float32: (TRITON_DOT, 'fp32 tl.dot'),
}
# A Triton kernel that only moves and reduces elements. bf16 is still
# sm_80: the widening to fp32 is a cvt instruction that does not exist
# below it. fp32 and fp16 are plain ALU work on every supported arch.
_TRITON_MOVE_DTYPES = {
    torch.bfloat16: (BF16, 'bf16 load/convert (no cvt.f32.bf16)'),
    torch.float16: (FP16_MMA, 'fp16 load/convert'),
    torch.float32: (FP16_MMA, 'fp32 load/convert'),
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
        floor=TRITON_DOT,
        floor_reason="the kernel's QK^T is a tl.dot",
        dtype_floors=_TRITON_DOT_DTYPES),
    'tile_gather_triton.gather': Requirement(
        kernel='tile_gather_triton.gather',
        floor=FP16_MMA,
        floor_reason='Triton itself (the kernel has no tl.dot)',
        dtype_floors=_TRITON_MOVE_DTYPES),
    'tile_gather_triton.scatter': Requirement(
        kernel='tile_gather_triton.scatter',
        floor=FP16_MMA,
        floor_reason='Triton itself (the kernel has no tl.dot)',
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
        floor_reason=('the probability block is cast to bf16 for its '
                      'tl.dot against the value summaries'),
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
        floor=BF16,
        floor_reason=('the fused kernel quantises the selected blocks to '
                      'INT8 (sm_75 and up) but keeps the pooled summaries '
                      'and the online softmax in bf16 (sm_80 and up)'),
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
