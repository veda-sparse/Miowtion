"""Tests for the capability gate of miowtion.kernels.arch.

No sm_70 / sm_75 hardware exists in this project, so every architecture
here is a monkeypatched `torch.cuda.get_device_capability`. That tests the
dispatch -- which path a capability selects and which error it raises --
and nothing about whether the kernels then run, which only real hardware
can say (docs/features/arch_support.md).
"""

import math

import pytest
import torch

from miowtion.h3 import attention as h3_attention
from miowtion.kernels import arch
from miowtion.kernels import block_heat_triton
from miowtion.kernels import fa4
from miowtion.kernels import reference
from miowtion.kernels import tile_gather_triton
from miowtion.veda import heatmap
from miowtion.veda import tiling

_VOLTA = (7, 0)
_TURING = (7, 5)
_AMPERE = (8, 0)
_ADA = (8, 9)


@pytest.fixture(name='on_arch')
def _on_arch(monkeypatch):
    """Pretends the CUDA device has a given compute capability."""

    def pretend(capability):
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'get_device_capability',
                            lambda device=None: capability)
        return torch.device('cuda')

    return pretend


# The capability matrix: (kernel, capability, dtype) -> can it run. One
# row per claim docs/features/arch_support.md makes.
_MATRIX = [
    # FA4 is sm_80 and up for every dtype: cp.async, not just bf16.
    ('fa4.block_sparse_attention', _VOLTA, torch.float16, False),
    ('fa4.block_sparse_attention', _TURING, torch.float16, False),
    ('fa4.block_sparse_attention', _TURING, torch.bfloat16, False),
    ('fa4.block_sparse_attention', _AMPERE, torch.bfloat16, True),
    ('fa4.dense_attention', _TURING, torch.float16, False),
    ('fa4.dense_attention', _ADA, torch.float16, True),
    # The heat kernel's QK^T is a tl.dot. It compiles from sm_70, but
    # below sm_80 Triton lowers the dot to CUDA-core fma and the kernel
    # loses to the torch reference, so the gate is sm_80 in every dtype
    # (measured: scripts/probe_triton_floor.py).
    ('block_heat_triton.teacher_heat', _VOLTA, torch.float16, False),
    ('block_heat_triton.teacher_heat', _TURING, torch.float16, False),
    ('block_heat_triton.teacher_heat', _TURING, torch.bfloat16, False),
    ('block_heat_triton.teacher_heat', _AMPERE, torch.float16, True),
    ('block_heat_triton.teacher_heat', _AMPERE, torch.bfloat16, True),
    # The tile permutation has no tl.dot, so fp16 runs on Volta; bf16
    # still needs the sm_80 convert instruction.
    ('tile_gather_triton.gather', _VOLTA, torch.float16, True),
    ('tile_gather_triton.gather', _VOLTA, torch.float32, True),
    ('tile_gather_triton.gather', _VOLTA, torch.bfloat16, False),
    ('tile_gather_triton.gather', _TURING, torch.float16, True),
    ('tile_gather_triton.gather', _AMPERE, torch.bfloat16, True),
    ('tile_gather_triton.scatter', _TURING, torch.float16, True),
    ('tile_gather_triton.scatter', _TURING, torch.bfloat16, False),
    # Fused paths that are bf16 by construction.
    ('elementwise_triton', _TURING, torch.bfloat16, False),
    ('elementwise_triton', _AMPERE, torch.bfloat16, True),
    ('compensation_triton.compensate', _TURING, torch.bfloat16, False),
    ('compensation_triton.compensate', _AMPERE, torch.bfloat16, True),
    ('sol_veda.block_sparse_attention', _TURING, torch.bfloat16, False),
    ('sol_veda.block_sparse_attention', _AMPERE, torch.bfloat16, True),
]


@pytest.mark.parametrize('kernel,capability,dtype,ok', _MATRIX)
def test_capability_matrix(on_arch, kernel, capability, dtype, ok):
    device = on_arch(capability)
    assert arch.supported(kernel, device, dtype) is ok
    if ok:
        arch.require(kernel, device, dtype)
    else:
        with pytest.raises(NotImplementedError):
            arch.require(kernel, device, dtype)


@pytest.mark.parametrize('kernel,capability,dtype,ok', _MATRIX)
def test_the_error_names_arch_kernel_and_operation(on_arch, kernel,
                                                   capability, dtype, ok):
    if ok:
        pytest.skip('supported combination')
    device = on_arch(capability)
    reason = arch.unsupported_reason(kernel, device, dtype)
    assert f'sm_{capability[0]}{capability[1]}' in reason
    assert kernel.split('.')[0] in reason
    # The reason has to say what is missing, not only that something is.
    assert any(word in reason for word in
               ('MMA', 'mma', 'fma', 'tl.dot', 'cp.async', 'cvt',
                'conversion', 'SDPA', 'INT8', 'bitwise'))


def test_unsupported_dtype_is_distinguished_from_an_old_arch(on_arch):
    device = on_arch(_ADA)
    reason = arch.unsupported_reason('fa4.block_sparse_attention', device,
                                     torch.float32)
    assert 'does not implement float32' in reason
    assert 'bfloat16' in reason and 'float16' in reason


def test_dtype_is_optional(on_arch):
    device = on_arch(_TURING)
    # Without a dtype only the kernel-wide floor is checked.
    assert arch.unsupported_reason('tile_gather_triton.gather',
                                   device) is None
    assert arch.unsupported_reason('fa4.block_sparse_attention',
                                   device) is not None


def test_a_compile_floor_is_only_set_when_it_is_below_the_gate():
    """`compile_floor` exists to mark a gate that is policy, not a limit.

    Setting it equal to the gate would say "the compiler stops here"
    twice and hide the distinction the field was added for.
    """
    for need in arch.REQUIREMENTS.values():
        if need.compile_floor is None:
            continue
        assert need.compile_floor < need.floor, need.kernel
        # A policy gate has to say why it is above the compile floor.
        assert 'compiles' in need.floor_reason, need.kernel


def test_the_heat_kernel_gate_is_about_tensor_cores_not_compiling():
    # The first version of this table claimed Triton could not compile a
    # dot below sm_75 at all. It can; what it cannot do below sm_80 is
    # reach a tensor core, which is a different (and weaker) statement.
    need = arch.REQUIREMENTS['block_heat_triton.teacher_heat']
    assert need.compile_floor == (7, 0)
    assert need.floor == (8, 0)
    assert 'fma' in need.floor_reason


def test_the_int8_arm_cannot_be_built_below_sm80():
    # Stronger than the bf16 argument: Turing has IMMA, Triton 3.4 still
    # refuses the INT8 dot, so there is nothing to gate on policy here.
    need = arch.REQUIREMENTS['sol_veda.block_sparse_attention']
    assert need.compile_floor is None
    assert 'INT8' in need.floor_reason and 'sm_80' in need.floor_reason


def test_every_requirement_dtype_meets_the_kernel_floor():
    # A dtype floor below the kernel floor would be unreachable and so a
    # lie in the table.
    for need in arch.REQUIREMENTS.values():
        for dtype, (floor, reason) in need.dtype_floors.items():
            assert floor >= need.floor, (need.kernel, dtype)
            assert reason


def test_names_are_printable():
    assert arch.name(_TURING) == 'sm_75 (Turing)'
    assert arch.name(_VOLTA) == 'sm_70 (Volta)'
    assert arch.name((12, 0)).startswith('sm_120')
    # An arch nobody has heard of still prints its number.
    assert arch.name((13, 7)) == 'sm_137'


def test_cpu_is_a_routing_bug_not_a_slow_arch():
    assert arch.supported('tile_gather_triton.gather', 'cpu',
                          torch.float16) is False
    with pytest.raises(ValueError, match='not a CUDA device'):
        arch.capability('cpu')
    with pytest.raises(ValueError, match='not a CUDA device'):
        arch.unsupported_reason('tile_gather_triton.gather', 'cpu')


def test_an_undeclared_kernel_is_an_error():
    with pytest.raises(KeyError, match='no capability requirement'):
        arch.supported('veda.magic', 'cuda')


def test_fa4_reason_names_the_missing_operation(on_arch):
    device = on_arch(_TURING)
    reason = fa4.unavailable_reason(device)
    assert 'sm_75' in reason and 'cp.async' in reason
    assert fa4.available(device) is False


def test_fa4_reason_without_cuda():
    assert fa4.unavailable_reason(torch.device('cpu')) is not None


def test_fa4_reason_on_a_supported_arch_blames_the_package(on_arch):
    device = on_arch(_ADA)
    # No flash_attn on a CPU-only machine, so the arch is not the reason.
    reason = fa4.unavailable_reason(device, torch.bfloat16)
    assert reason is not None and 'sm_89' not in reason


def test_triton_entry_points_refuse_a_cpu_tensor():
    x = torch.zeros(256, 2, 128)
    assert tile_gather_triton.supports(x) is False
    assert block_heat_triton.supports(x) is False


def test_heat_kernel_refuses_mixed_dtypes():
    q = torch.zeros(256, 2, 128, dtype=torch.float16)
    k = torch.zeros(256, 2, 128, dtype=torch.bfloat16)
    lse = torch.zeros(256, 2)
    layout = _layout()
    with pytest.raises(ValueError, match='one tl.dot'):
        block_heat_triton.teacher_heat(q, k, lse, layout,
                                       torch.zeros(1, dtype=torch.long))


def test_sdpa_refuses_bf16_below_sm80(on_arch):
    on_arch(_TURING)
    q = torch.zeros(128, 2, 64, dtype=torch.bfloat16)
    # is_cuda cannot be faked on a CPU tensor, so ask the table the same
    # question the backend asks.
    assert not arch.supported('h3.dense_attention.sdpa', 'cuda', q.dtype)
    assert arch.supported('h3.dense_attention.sdpa', 'cuda', torch.float16)
    # The LSE needs the flash backend, which is sm_80 whatever the dtype.
    assert not arch.supported('h3.dense_attention.sdpa_lse', 'cuda',
                              torch.float16)


def test_math_backend_runs_on_any_arch(on_arch):
    on_arch(_VOLTA)
    q, k, v = (torch.randn(64, 2, 32, dtype=torch.float16) for _ in range(3))
    out, lse = h3_attention.dense_attention(q, k, v, 60, return_lse=True,
                                            backend='math')
    assert out.dtype == torch.float16 and lse.dtype == torch.float32
    assert torch.isfinite(lse[:60]).all()


def _layout(grid=(7, 6, 10), shape='8x4x4', text=130):
    target = tiling.TiledSpan(text, grid, tiling.TileShape.parse(shape))
    used = target.start + target.num_rows
    return tiling.build_tile_layout([target], used, used)


def _fp16_inputs(layout, heads=2, dim=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rows = layout.seq_len
    return [torch.randn(rows, heads, dim, generator=gen).half()
            for _ in range(3)]


def test_fp16_block_sparse_reference_matches_fp32():
    """fp16 is a usable dtype for the reference, not just a compiling one.

    The point of the fp16 path is sm_70 / sm_75, where bf16 does not
    exist. fp16 has the narrower exponent, so what has to hold is that
    the softmax still accumulates in fp32: the reference exponentiates
    fp32 scores, so the only fp16 step is the input rounding.
    """
    layout = _layout()
    q, k, v = _fp16_inputs(layout)
    heads = torch.arange(2)
    qt, kt, vt = (tiling.gather_tiles(t, layout, heads) for t in (q, k, v))
    mask = torch.ones(2, layout.n_tiles, layout.n_tiles, dtype=torch.bool)
    out, lse = reference.block_sparse_attention(
        qt, kt, vt, mask, layout.valid_count, return_lse=True)
    want, want_lse = reference.block_sparse_attention(
        qt.float(), kt.float(), vt.float(), mask, layout.valid_count,
        return_lse=True)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    # fp16 inputs, fp32 math: the gap is the input rounding alone.
    torch.testing.assert_close(out.float(), want, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse, want_lse, rtol=1e-3, atol=1e-3)


def test_fp16_teacher_heat_reference_matches_fp32():
    """The heat reference in fp16, which the kernel's rounding mirrors.

    The kernel rounds each row max to the input dtype because that is
    what the reference's bmm returns; in fp16 that rounding saturates
    only above 65504 pre-scale, which these scores are nowhere near.
    """
    layout = _layout()
    q, k, _ = _fp16_inputs(layout)
    heads = torch.arange(2)
    qt, kt = (tiling.gather_tiles(t, layout, heads) for t in (q, k))
    lse = torch.full((layout.num_slots, 2), math.log(layout.used))
    tiles = torch.arange(layout.n_tiles)
    heat = heatmap.teacher_heat_reference(qt, kt, lse, layout, tiles)
    want = heatmap.teacher_heat_reference(qt.float(), kt.float(), lse,
                                          layout, tiles)
    assert torch.isfinite(heat).all()
    torch.testing.assert_close(heat, want, rtol=5e-2, atol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_rounding_the_row_max_equals_rounding_every_score(dtype):
    """Why the heat kernel may round once per row instead of per score.

    The reference's bmm returns the input dtype, so every score is
    rounded before the max; the kernel keeps fp32 accumulators and
    rounds only the row max. The two agree bitwise because
    round-to-nearest-even is monotonic -- in fp16 as well as in bf16,
    which is what makes following the input dtype legitimate rather
    than an approximation.
    """
    gen = torch.Generator().manual_seed(7)
    scores = torch.randn(64, 128, generator=gen) * 40
    # Ties are where a non-monotonic rounding would show up.
    scores[0, :] = scores[0, 0]
    rounded_first = scores.to(dtype).amax(-1)
    rounded_after = scores.amax(-1).to(dtype)
    assert torch.equal(rounded_first, rounded_after)
