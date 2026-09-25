"""Tests for miowtion.mlx.{block,interop} against the torch reference."""

import dataclasses

import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import block as mlx_block  # noqa: E402
from miowtion.mlx import interop  # noqa: E402

_CONFIG = dataclasses.replace(h3_config.H3Config.tiny(), hidden_size=128,
                              num_heads=4, head_dim=32, ffn_dim=64,
                              rope_freqs_per_axis=4)
_SEQ = 96
_USED = 80


def _torch_block(dtype):
    torch.manual_seed(0)
    block = h3_model.Block(_CONFIG)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if 'norm' in name:
                p.copy_(1 + 0.1 * torch.randn(p.shape))
            else:
                p.normal_(std=0.05)
    block.adaln_proj = None
    return block.to(dtype)


def _inputs(dtype):
    torch.manual_seed(1)
    x = torch.randn(_SEQ, _CONFIG.hidden_size).to(dtype)
    tables = [(0.3 * torch.randn(6, _CONFIG.hidden_size)).to(dtype)
              for _ in range(6)]
    index = torch.randint(0, 6, (_SEQ,))
    positions = torch.rand(_SEQ, 3, dtype=torch.float64) * 30
    cos, sin = h3_model.Rope(_CONFIG).cos_sin(positions)
    return x, tables, index, (cos.to(dtype), sin.to(dtype))


def _reference(dtype, backend='math'):
    block = _torch_block(dtype)
    x, tables, index, rope = _inputs(dtype)
    with torch.no_grad():
        out = block(x, tables, None, index,
                    rope, h3_model.DenseAttention(_USED, backend), 0)
    return block, out


def _mlx_forward(dtype, options=mlx_block.BlockOptions(), weights=None):
    block = _torch_block(dtype)
    x, tables, index, rope = _inputs(dtype)
    if weights is None:
        weights = interop.block_weights_from_torch(block)
    out = mlx_block.block_forward(
        interop.from_torch(x), weights,
        [interop.from_torch(t) for t in tables],
        interop.from_torch(index.to(torch.int32)),
        tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG, options)
    return interop.to_torch(out)


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_interop_roundtrip_is_bitwise():
    for tensor in (torch.randn(7, 5).to(torch.bfloat16), torch.randn(3, 4),
                   torch.randint(0, 9, (11,), dtype=torch.int32)):
        assert torch.equal(interop.to_torch(interop.from_torch(tensor)),
                           tensor)
    with pytest.raises(ValueError):
        interop.from_torch(torch.zeros(2, dtype=torch.int8))


def test_rope_and_swiglu_match_torch_bitwise():
    # Pure elementwise chains: the rounding points are the same, so MLX and
    # torch must agree bit for bit (reductions are compared separately).
    torch.manual_seed(2)
    q = torch.randn(_SEQ, _CONFIG.num_heads, _CONFIG.head_dim).to(
        torch.bfloat16)
    positions = torch.rand(_SEQ, 3, dtype=torch.float64) * 17
    cos, sin = h3_model.Rope(_CONFIG).cos_sin(positions)
    reference = h3_model.apply_rope(q, cos, sin)
    got = mlx_block.apply_rope(interop.from_torch(q), interop.from_torch(cos),
                               interop.from_torch(sin))
    assert torch.equal(interop.to_torch(got), reference)

    gate = (3 * torch.randn(_SEQ, _CONFIG.ffn_dim)).to(torch.bfloat16)
    up = torch.randn(_SEQ, _CONFIG.ffn_dim).to(torch.bfloat16)
    got = mlx_block.swiglu(interop.from_torch(gate), interop.from_torch(up))
    assert torch.equal(interop.to_torch(got),
                       torch.nn.functional.silu(gate) * up)


def test_block_matches_torch_reference():
    _, reference32 = _reference(torch.float32)
    assert _rel_l2(_mlx_forward(torch.float32), reference32) < 1e-5
    _, reference16 = _reference(torch.bfloat16)
    got16 = _mlx_forward(torch.bfloat16)
    # bf16 reductions (RMSNorm statistics, GEMM, attention) accumulate in a
    # different order, so the two bf16 paths are not bitwise equal; both stay
    # within the bf16 rounding error of the fp32 result.
    assert _rel_l2(got16, reference16) < 5e-3
    assert _rel_l2(got16, reference32) <= 1.5 * _rel_l2(reference16,
                                                        reference32)


def test_chunking_does_not_change_the_result():
    whole = _mlx_forward(torch.bfloat16)
    for options in (mlx_block.BlockOptions(head_chunk=1, row_chunk=_SEQ),
                    mlx_block.BlockOptions(head_chunk=2, row_chunk=7,
                                           eval_chunks=True),
                    # Where the graph is evaluated is a memory/latency knob
                    # only; it must not move a single bit of the result.
                    mlx_block.BlockOptions(head_chunk=2, row_chunk=7,
                                           eval_chunks=False)):
        assert _rel_l2(_mlx_forward(torch.bfloat16, options), whole) < 1e-3


def test_quantized_block_error():
    reference = _mlx_forward(torch.bfloat16)
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    errors = {}
    for bits in (8, 4):
        quantized = weights.quantize(bits, group_size=32)
        assert quantized.quantization == (bits, 32)
        errors[bits] = _rel_l2(_mlx_forward(torch.bfloat16, weights=quantized),
                               reference)
        # Dequantized weights must give the same magnitude of error as the
        # quantized matmuls.
        dequantized = _rel_l2(
            _mlx_forward(torch.bfloat16, weights=quantized.dequantize()),
            reference)
        assert dequantized < 2 * errors[bits] + 1e-3
    assert errors[8] < 0.05
    assert errors[8] < errors[4] < 0.5


def test_block_weights_is_strict():
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    tensors = weights.to_tensors()
    assert set(tensors) == {f'{n}.weight' for n in mlx_block.NORM_NAMES
                            + mlx_block.LINEAR_NAMES}
    tensors.pop('mlp.fc1.weight')
    with pytest.raises(KeyError):
        mlx_block.BlockWeights.from_tensors(tensors)


def test_forward_rejects_bad_arguments():
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    x, tables, index, rope = _inputs(torch.bfloat16)
    args = [interop.from_torch(x), weights,
            [interop.from_torch(t) for t in tables],
            interop.from_torch(index.to(torch.int32)),
            tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG]
    for bad in (mlx_block.BlockOptions(head_chunk=3),
                mlx_block.BlockOptions(row_chunk=0)):
        with pytest.raises(ValueError):
            mlx_block.block_forward(*args, bad)
    with pytest.raises(ValueError):
        mlx_block.block_forward(*args[:5], _SEQ + 1, _CONFIG)
    with pytest.raises(ValueError):
        mlx_block.block_forward(args[0], weights, args[2][:5], *args[3:])


def test_padding_rows_do_not_leak():
    # Rows >= used must not influence the output of the real rows.
    out = _mlx_forward(torch.float32)
    torch.manual_seed(1)
    block = _torch_block(torch.float32)
    x, tables, index, rope = _inputs(torch.float32)
    x[_USED:] += 10.0
    weights = interop.block_weights_from_torch(block)
    other = mlx_block.block_forward(
        interop.from_torch(x), weights,
        [interop.from_torch(t) for t in tables],
        interop.from_torch(index.to(torch.int32)),
        tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG)
    assert torch.equal(interop.to_torch(other)[:_USED], out[:_USED])


def _plan(q_block, k_block, budget, seed=0):
    from miowtion.mlx import sparse_attention
    index = sparse_attention.random_index(_USED // q_block, _USED // k_block,
                                          budget, seed=seed)
    return sparse_attention.SparsePlan(index, q_block, k_block)


def test_sparse_plan_with_full_budget_matches_dense():
    # Every key tile selected, in order: the sparse path sees exactly the
    # dense problem, so only the kernel's tiling differs.
    dense = _mlx_forward(torch.float32)
    plan = _plan(16, 16, budget=_USED // 16)
    got = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=plan))
    assert _rel_l2(got, dense) < 1e-5


def test_sparse_plan_restricts_attention_and_keeps_padding_out():
    plan = _plan(20, 16, budget=2)
    sparse = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=plan))
    dense = _mlx_forward(torch.float32)
    # A 40 % budget must actually change the result.
    assert _rel_l2(sparse, dense) > 1e-3
    # Padding rows are outside [0, used), so they still cannot leak in.
    assert sparse.shape == dense.shape


def test_sparse_plan_is_validated():
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    x, tables, index, rope = _inputs(torch.bfloat16)
    args = [interop.from_torch(x), weights,
            [interop.from_torch(t) for t in tables],
            interop.from_torch(index.to(torch.int32)),
            tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG]
    plan = _plan(20, 16, 2)
    bad_plans = [
        dataclasses.replace(plan, q_block=32),  # does not tile `used`
        dataclasses.replace(plan, k_block=32),
        dataclasses.replace(plan, index=plan.index[:-1]),  # wrong tile count
    ]
    for bad in bad_plans:
        with pytest.raises(ValueError):
            mlx_block.block_forward(*args, mlx_block.BlockOptions(sparse=bad))


def _identity_group(heads, plan):
    from miowtion.mlx import sparse_attention
    rows = mx.arange(_USED, dtype=mx.int32)
    return sparse_attention.HeadGroupPlan(heads=heads, gather=rows,
                                          scatter=rows, plan=plan)


def test_layer_plan_splits_the_heads_without_changing_the_result():
    # Two head groups carrying the same selection and the identity
    # permutation must be exactly the single-plan path: the split only
    # decides which heads are gathered together.
    from miowtion.mlx import sparse_attention
    plan = _plan(20, 16, budget=2)
    heads = _CONFIG.num_heads
    layer = sparse_attention.LayerPlan(
        (_identity_group(tuple(range(0, heads, 2)), plan),
         _identity_group(tuple(range(1, heads, 2)), plan)), heads)
    got = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=layer))
    want = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=plan))
    assert torch.equal(got, want)


def test_layer_plan_is_validated():
    from miowtion.mlx import sparse_attention
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    x, tables, index, rope = _inputs(torch.bfloat16)
    args = [interop.from_torch(x), weights,
            [interop.from_torch(t) for t in tables],
            interop.from_torch(index.to(torch.int32)),
            tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG]
    plan = _plan(20, 16, 2)
    heads = _CONFIG.num_heads
    group = _identity_group(tuple(range(heads)), plan)
    bad = [sparse_attention.LayerPlan((_identity_group((0,), plan),), 1),
           sparse_attention.LayerPlan(
               (dataclasses.replace(group, scatter=group.scatter[:-1]),),
               heads)]
    for layer in bad:
        with pytest.raises(ValueError):
            mlx_block.block_forward(*args,
                                    mlx_block.BlockOptions(sparse=layer))


def test_per_head_selection_survives_head_chunking():
    # Veda picks its key tiles per head, and the trunk runs attention
    # head_chunk heads at a time; the chunk's q must meet its own heads'
    # selection, not the whole layer's.
    from miowtion.mlx import sparse_attention
    heads = _CONFIG.num_heads
    index = mx.stack([sparse_attention.random_index(_USED // 20,
                                                    _USED // 16, 2, seed=h)
                      for h in range(heads)])
    plan = sparse_attention.SparsePlan(index, 20, 16)
    want = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=plan))
    # Two heads per call is bitwise the same work, only narrower.
    assert torch.equal(
        _mlx_forward(torch.float32,
                     mlx_block.BlockOptions(sparse=plan, head_chunk=2)),
        want)
    # One head leaves the kernel a batch of one, which reduces in a
    # different order: 1 ulp of fp32, not a different selection.
    alone = _mlx_forward(torch.float32,
                         mlx_block.BlockOptions(sparse=plan, head_chunk=1))
    assert (alone - want).abs().max().item() <= 2.4e-7


def test_a_planner_is_called_per_head_chunk_with_that_chunk_s_heads():
    # A scorer that reads the activations cannot run before the block, so
    # BlockOptions.sparse may be a planner. It must see this chunk's q and
    # k, and its plan (numbered from 0) must run exactly like the static
    # layer plan it is built from.
    from miowtion.mlx import sparse_attention
    heads = _CONFIG.num_heads
    plan = _plan(20, 16, budget=2)
    layer = sparse_attention.LayerPlan(
        (_identity_group(tuple(range(heads)), plan),), heads)
    want = _mlx_forward(torch.float32, mlx_block.BlockOptions(sparse=layer))

    seen = []

    def planner(q, k, head_start):
        seen.append((q.shape, k.shape, head_start))
        chunk = q.shape[1]
        return sparse_attention.LayerPlan(
            (_identity_group(tuple(range(chunk)), plan),), chunk)

    got = _mlx_forward(torch.float32,
                       mlx_block.BlockOptions(sparse=planner, head_chunk=2))
    assert torch.equal(got, want)
    assert [start for _, _, start in seen] == list(range(0, heads, 2))
    assert all(shape == (_SEQ, 2, _CONFIG.head_dim)
               for shapes in seen for shape in shapes[:2])


def test_a_planner_s_plan_is_validated():
    from miowtion.mlx import sparse_attention
    weights = interop.block_weights_from_torch(_torch_block(torch.bfloat16))
    x, tables, index, rope = _inputs(torch.bfloat16)
    args = [interop.from_torch(x), weights,
            [interop.from_torch(t) for t in tables],
            interop.from_torch(index.to(torch.int32)),
            tuple(interop.from_torch(r) for r in rope), _USED, _CONFIG]
    plan = _plan(20, 16, 2)
    # One head too many: the plan does not cover the chunk it was asked for.
    def planner(q, k, head_start):
        return sparse_attention.LayerPlan(
            (_identity_group((0,), plan),), 1)

    with pytest.raises(ValueError):
        mlx_block.block_forward(*args, mlx_block.BlockOptions(
            sparse=planner, head_chunk=2))
