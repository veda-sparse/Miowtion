"""CPU structural and actual tiny-H3 reference-attention distillation tests."""

import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule
from miowtion.train import lora
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_trainer
from miowtion.veda import attention
from miowtion.veda import mask
from miowtion.veda import plan
from miowtion.veda import predictor
from miowtion.veda import tiling


def _tiny_case(checkpointing=True):
    torch.manual_seed(19)
    cfg = h3_config.H3Config.tiny(num_layers=4, num_heads=2)
    model = h3_model.H3DiT(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    model.dense_backend = 'math'
    model.gradient_checkpointing = checkpointing
    geo = geometry.Geometry('16:9', 128, 64, 5, 2, 4, 8, 4)
    packed = layout.pack(torch.ones(4, dtype=torch.long), geo)
    clip = model.clip_inputs(packed, model.refine_text(
        torch.randn(4, cfg.text_dim)), torch.device('cpu'))
    grid_plan = plan.TilePlan.uniform(
        geo, tiling.TileShape(1, 8, 16), cfg.num_layers, cfg.num_heads)
    pred = predictor.TileScorePredictor(cfg.num_layers, cfg.num_heads,
                                         cfg.head_dim)
    clip_tiling = attention.ClipTiling(packed, attention.VedaConfig(
        target_budget=mask.Budget(ratio=0.05)), torch.device('cpu'))
    sparse = attention.SparseStudent(clip_tiling, grid_plan, pred,
                                     allow_reference_kernel=True)
    state = schedule.build_timestep_state(packed, 0.8, 0.5)
    args = (clip, torch.randn(geo.num_video_tokens, cfg.video_patch_dim),
            torch.randn(geo.num_audio_rows, cfg.audio_channels), state)
    return model, pred, sparse, args


def test_real_meta_layers_and_parameter_budget():
    with torch.device('meta'):
        model = h3_model.H3DiT(h3_config.H3Config())
        matched = lora.add_lora(model, 32, 32, layers=[0, 1, 49])
    assert set(matched) == {f'blocks.{i}.attn.{target}'
                            for i in [0, 1, 49]
                            for target in ['qkv_proj', 'out_proj']}
    # Each layer: r*(5376+21504) + r*(7168+5376).
    assert sum(p.numel() for _, p in lora.lora_parameters(model)) == 3784704


def test_real_meta_all_attention_and_mlp_parameter_budget():
    with torch.device('meta'):
        model = h3_model.H3DiT(h3_config.H3Config())
        matched = lora.add_lora(model, 32, 32, targets=lora.TRUNK_TARGETS,
                                layers=list(range(50)))
    assert set(matched) == {f'blocks.{i}.{target}' for i in range(50)
                            for target in lora.TRUNK_TARGETS}
    named = lora.lora_parameters(model)
    assert len(matched) == 200 and len(named) == 400
    # Attention: 39,424*r; SwiGLU fc1/fc2: 53,760*r per block.
    assert sum(p.numel() for _, p in named) == 149094400
    assert all(p.dtype == torch.float32 for _, p in named)
    assert not any(n.startswith('refiner.') for n, _ in named)


@pytest.mark.parametrize('kwargs', [
    {'layers': [50]}, {'layers': [0, 0]}, {'layers': [-1]},
    {'layers': [0], 'targets': ['attn.missing']}, {'rank': 0},
    {'alpha': 0}, {'targets': []},
])
def test_injection_rejects_invalid_before_mutation(kwargs):
    with torch.device('meta'):
        model = h3_model.H3DiT(h3_config.H3Config.tiny())
        with pytest.raises(ValueError):
            lora.add_lora(model, **{'rank': 32, **kwargs})
    assert not lora.lora_parameters(model)


def test_duplicate_injection_rejected():
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].attn = torch.nn.Module()
    model.blocks[0].attn.out_proj = torch.nn.Linear(2, 2)
    lora.add_lora(model, 2, targets=['attn.out_proj'], layers=[0])
    with pytest.raises(ValueError, match='duplicate'):
        lora.add_lora(model, 2, targets=['attn.out_proj'], layers=[0])


@pytest.mark.parametrize('shape', [(17, 8), (2, 9, 8)])
def test_chunked_lora_matches_full_forward_and_gradients(shape):
    torch.manual_seed(101)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].attn = torch.nn.Module()
    module = torch.nn.Linear(8, 13, dtype=torch.float64)
    model.blocks[0].attn.out_proj = module
    lora.add_lora(model, 3, alpha=1.7, layers=[0],
                  targets=['attn.out_proj'])
    module.lora_a.data = module.lora_a.data.double()
    module.lora_b.data = torch.randn(13, 3, dtype=torch.float64) * .1
    source = torch.randn(*shape, dtype=torch.float64)
    upstream = torch.randn(*shape[:-1], 13, dtype=torch.float64)
    results = []
    for chunk_rows in (None, 5):
        module.lora_chunk_rows = chunk_rows
        model.zero_grad(set_to_none=True)
        x = source.clone().requires_grad_()
        output = model.blocks[0].attn.out_proj(x)
        (output * upstream).sum().backward()
        results.append((output.detach(), x.grad,
                        {n: p.grad.clone() for n, p in
                         model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0],
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(results[0][1], results[1][1],
                               rtol=1e-12, atol=1e-12)
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name],
                                   rtol=1e-12, atol=1e-12)
    module.lora_enabled = False
    assert torch.equal(module(source), torch.nn.functional.linear(
        source, module.weight, module.bias))


def test_checkpoint_input_cpu_storage_preserves_outputs_and_gradients():
    model, pred, sparse, args = _tiny_case()
    lora.add_lora(model, 3, layers=[0, 1, 3])
    named = core.freeze_except_lora(model, pred)
    for name, param in named:
        if name.endswith('.lora_b'):
            param.data.normal_(std=.01)
    results = []
    for offload in (False, True):
        model.checkpoint_inputs_to_cpu = offload
        model.zero_grad(set_to_none=True)
        outputs = model(*args, attention_fn=sparse)
        sum(value.square().mean() for value in outputs).backward()
        results.append((tuple(x.detach() for x in outputs),
                        {n: p.grad.clone() for n, p in named}))
    assert all(torch.equal(a, b) for a, b in zip(
        results[0][0], results[1][0]))
    assert all(torch.equal(results[0][1][n], results[1][1][n])
               for n, _ in named)


@pytest.mark.parametrize('group_size', [2, 3, 5])
@pytest.mark.parametrize('full_trunk', [False, True])
def test_nested_checkpoint_groups_preserve_gradients_and_frozen_weights(
        group_size, full_trunk):
    model, pred, sparse, args = _tiny_case()
    layers = list(range(4)) if full_trunk else [0, 1, 3]
    targets = lora.TRUNK_TARGETS if full_trunk else lora.DEFAULT_TARGETS
    lora.add_lora(model, 3, layers=layers, targets=targets)
    named = core.freeze_except_lora(model, pred)
    for name, param in named:
        if name.endswith('.lora_b'):
            param.data.normal_(std=.01)
    frozen = core.frozen_digest(model, pred)
    results = []
    for group in (1, group_size):
        model.checkpoint_group_size = group
        model.zero_grad(set_to_none=True)
        outputs = model(*args, attention_fn=sparse)
        sum(x.square().mean() for x in outputs).backward()
        results.append((tuple(x.detach() for x in outputs),
                        {n: p.grad.clone() for n, p in named}))
        assert core.frozen_digest(model, pred) == frozen
    assert all(torch.equal(a, b) for a, b in zip(
        results[0][0], results[1][0]))
    assert all(torch.equal(results[0][1][n], results[1][1][n])
               for n, _ in named)


@pytest.mark.parametrize('full_trunk', [False, True])
def test_zero_init_freeze_teacher_gradients_and_checkpoint_recompute(
        tmp_path, monkeypatch, full_trunk):
    # Exercise accumulation across several head chunks as on real clips.
    monkeypatch.setattr(attention, '_chunk_heads', lambda _: 1)
    model, pred, sparse, args = _tiny_case()
    baseline = model(*args, attention_fn=sparse)
    layers = list(range(4)) if full_trunk else [0, 1, 3]
    targets = lora.TRUNK_TARGETS if full_trunk else lora.DEFAULT_TARGETS
    lora.add_lora(model, 32, 32, layers=layers, targets=targets)
    named = core.freeze_except_lora(model, pred)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    pred_before = copy.deepcopy(pred.state_dict())
    student_zero = model(*args, attention_fn=sparse)
    assert all(torch.equal(a, b) for a, b in zip(baseline, student_zero))
    with core.frozen_teacher(model):
        target_before = model(*args)
    assert not any(t.requires_grad for t in target_before)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=0.05)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        student = model(*args, attention_fn=sparse)
        loss, video, audio = core.velocity_loss(student, target_before)
        assert torch.isfinite(loss) and video >= 0 and audio >= 0
        loss.backward()
        grads = core.gradient_report(named, require_a=step > 0)
        if step == 0:
            assert all(value == 0 for n, value in grads.items()
                       if n.endswith('.lora_a'))
        # LoRA layer 0 must backpropagate through frozen layer 2.
        assert grads['blocks.0.attn.qkv_proj.lora_b'] > 0
        optimizer.step()
    for name, parameter in model.named_parameters():
        if '.lora_' not in name:
            assert torch.equal(initial[name], parameter), name
            assert parameter.grad is None and not parameter.requires_grad
    assert all(torch.equal(value, pred.state_dict()[name])
               for name, value in pred_before.items())
    assert all(p.grad is None and not p.requires_grad for p in pred.parameters())
    with core.frozen_teacher(model):
        target_after = model(*args)
    assert all(torch.equal(a, b) for a, b in zip(target_before, target_after))
    lora.set_enabled(model, False)
    assert all(torch.equal(a, b) for a, b in zip(
        baseline, model(*args, attention_fn=sparse)))
    lora.set_enabled(model, True)
    # Changing the global hook toggle before recomputation would change
    # this gradient. Compare checkpointed and eager backward numerically.
    optimizer.zero_grad(set_to_none=True)
    core.velocity_loss(model(*args, attention_fn=sparse), target_before)[0].backward()
    checkpoint_grads = {n: p.grad.clone() for n, p in named}
    optimizer.zero_grad(set_to_none=True)
    model.gradient_checkpointing = False
    core.velocity_loss(model(*args, attention_fn=sparse), target_before)[0].backward()
    assert all(torch.equal(checkpoint_grads[n], p.grad) for n, p in named)
    # Independent adapter, with exact output after loading on another model.
    contract = {'rank': 32, 'alpha': 32, 'layers': layers,
                'targets': list(targets)}
    path = str(tmp_path / 'adapter.safetensors')
    core.export_adapter(path, model, contract, 2)
    output = model(*args, attention_fn=sparse)
    lora.reset_parameters(model, 19)
    core.load_sparse_adapter(model, path, contract)
    assert all(torch.equal(a, b) for a, b in zip(
        output, model(*args, attention_fn=sparse)))
    with pytest.raises(ValueError, match='contract'):
        core.load_sparse_adapter(model, path, {**contract, 'rank': 16})
    module = dict(model.named_modules())['blocks.0.attn.qkv_proj']
    module.lora_scale = 0.5
    with pytest.raises(ValueError, match='rank/alpha'):
        core.load_sparse_adapter(model, path, contract)


def test_all_trunk_new_lora_keeps_merged_few_step_teacher():
    model, pred, sparse, args = _tiny_case()
    baseline = copy.deepcopy(model)
    names = ('blocks.0.mlp.fc1', 'blocks.3.attn.out_proj')
    modules = dict(model.named_modules())
    adapter = {name: (torch.randn(2, modules[name].in_features) * .1,
                      torch.randn(modules[name].out_features, 2) * .1)
               for name in names}
    lora.merge_adapter(baseline, adapter)
    # Registration must precede FSDP; values are initialized after Turbo merge.
    lora.add_lora(model, 32, 32, targets=lora.TRUNK_TARGETS,
                  layers=list(range(4)), chunk_rows=7)
    lora.set_enabled(model, False)
    lora.merge_adapter(model, adapter)
    assert all(torch.equal(a, b) for a, b in zip(
        baseline(*args), model(*args)))
    lora.reset_parameters(model, 19)
    lora.set_enabled(model, True)
    named = core.freeze_except_lora(model, pred)
    expected = baseline(*args)
    expected_sparse = baseline(*args, attention_fn=sparse)
    assert all(torch.equal(a, b) for a, b in zip(
        expected_sparse, model(*args, attention_fn=sparse)))
    with torch.no_grad():
        for name, parameter in named:
            if name.endswith('.lora_b'):
                parameter.normal_(std=.01)
    assert any(not torch.equal(a, b) for a, b in zip(
        expected_sparse, model(*args, attention_fn=sparse)))
    with core.frozen_teacher(model):
        actual = model(*args)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if '.lora_' not in n)


def test_teacher_context_restores_after_exception():
    model = torch.nn.Module()
    child = torch.nn.Linear(2, 2)
    model.add_module('child', child)
    child.lora_enabled = True
    with pytest.raises(RuntimeError):
        with core.frozen_teacher(model):
            assert not child.lora_enabled and not torch.is_grad_enabled()
            raise RuntimeError('intentional')
    assert child.lora_enabled and torch.is_grad_enabled()


def test_export_load_in_new_process(tmp_path):
    # New process reconstructs the frozen base independently and loads only A/B.
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].attn = torch.nn.Module()
    model.blocks[0].attn.out_proj = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.blocks[0].attn.out_proj.weight.fill_(0.25)
    lora.add_lora(model, 32, 32, targets=['attn.out_proj'], layers=[0])
    with torch.no_grad():
        model.blocks[0].attn.out_proj.lora_b.fill_(0.1)
    contract = {'rank': 32, 'alpha': 32, 'layers': [0]}
    path = str(tmp_path / 'adapter.safetensors')
    core.export_adapter(path, model, contract, 1)
    expected = model.blocks[0].attn.out_proj(torch.ones(1, 3))
    code = '''
import json, sys, torch
from miowtion.train import lora, sparse_lora
model = torch.nn.Module()
model.blocks = torch.nn.ModuleList([torch.nn.Module()])
model.blocks[0].attn = torch.nn.Module()
model.blocks[0].attn.out_proj = torch.nn.Linear(3, 2, bias=False)
with torch.no_grad():
    model.blocks[0].attn.out_proj.weight.fill_(0.25)
lora.add_lora(model,32,32,targets=['attn.out_proj'],layers=[0])
sparse_lora.load_sparse_adapter(model,sys.argv[1],
    {'rank':32,'alpha':32,'layers':[0]})
torch.save(model.blocks[0].attn.out_proj(torch.ones(1,3)).detach(),sys.argv[2])
'''
    destination = str(tmp_path / 'output.pt')
    subprocess.run([sys.executable, '-c', code, path, destination],
                   check=True, timeout=60, env=os.environ.copy())
    actual = torch.load(destination, weights_only=True)
    assert torch.equal(expected, actual)


def test_velocity_loss_is_target_velocity_and_detaches_teacher():
    video = torch.tensor([1., 3.], requires_grad=True)
    audio = torch.tensor([2.], requires_grad=True)
    target = (torch.zeros(2, requires_grad=True),
              torch.zeros(1, requires_grad=True))
    loss, v, a = core.velocity_loss((video, audio), target, audio_weight=2)
    assert loss == 13 and v == 5 and a == 4
    loss.backward()
    assert all(t.grad is None for t in target)


def test_configs_are_explicit_and_strict():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'configs').glob('sparse_lora*.yaml'))
    assert paths
    assert any(path.name.endswith('_400step.yaml') for path in paths)
    for path in paths:
        cfg = core.SparseLoraConfig.from_yaml(str(path))
        if 'all_attn_mlp' in path.name:
            assert cfg.lora_layers == list(range(50))
            assert cfg.lora_targets == list(lora.TRUNK_TARGETS)
            expected_lr = 1e-3 if 'lr1e-3' in path.stem else 1e-2
            expected_steps = 150 if '150step' in path.stem else 100
            expected_warmup = 20 if 'warmup20' in path.stem else 5
            assert (cfg.keep_ratio, cfg.lr, cfg.steps) == (
                0.1, expected_lr, expected_steps)
            assert cfg.epochs is None and cfg.warmup == expected_warmup
        else:
            assert cfg.lora_layers == [0, 1, 49]
            assert cfg.keep_ratio == 0.05
            assert cfg.lora_targets == list(lora.DEFAULT_TARGETS)
        assert cfg.lora_rank == 32 and cfg.schedule == 'turbo'
        assert cfg.ordered_dq
        expected_sampling = (
            'trajectory' if 'sequential' in path.stem
            or 'all_attn_mlp' in path.stem else 'random')
        assert cfg.timestep_sampling == expected_sampling and cfg.accum == 4
        for change in [{'task': 'ref2va'}, {'num_steps': 50},
                       {'lora_layers': [0, 0]}, {'audio_weight': float('nan')},
                       {'warmup': -1}, {'warmup': 1.5}, {'warmup': True},
                       {'ordered_dq': 'true'},
                       {'lora_targets': ['mlp.missing']},
                       {'steps': None, 'epochs': None},
                       {'steps': 3, 'epochs': 2},
                       {'steps': None, 'epochs': True},
                       {'steps': None, 'epochs': 0},
                       {'timestep_sampling': 'continuous'}]:
            with pytest.raises(ValueError):
                dataclasses.replace(cfg, **change).validate()


@pytest.mark.parametrize('change', [
    {'lora_chunk_rows': 0}, {'lora_chunk_rows': True},
    {'lora_chunk_rows': 2.5}, {'checkpoint_inputs_to_cpu': 'true'},
    {'checkpoint_group_size': 0}, {'checkpoint_group_size': True},
    {'checkpoint_group_size': 1.5},
])
def test_memory_options_reject_invalid_values(change):
    cfg = core.SparseLoraConfig('test', '', '', '', '', '', '', ['16:9@37'])
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, **change).validate()


def test_warmup_is_independent_of_run_extension():
    cfg = core.SparseLoraConfig('', '', '', '', '', '', '', [],
                                lr=1e-4, warmup=5, steps=50)
    rates = [core.learning_rate(cfg, step) for step in range(7)]
    assert rates == pytest.approx([2e-5, 4e-5, 6e-5, 8e-5,
                                  1e-4, 1e-4, 1e-4])
    extended = dataclasses.replace(cfg, steps=400)
    assert rates == [core.learning_rate(extended, step) for step in range(7)]
    assert core.learning_rate(dataclasses.replace(cfg, warmup=0), 0) == cfg.lr


def test_update_report_handles_zero_initialized_b_and_nonfinite_weights():
    a = torch.nn.Parameter(torch.tensor([3., 4.]))
    b = torch.nn.Parameter(torch.zeros(2))
    named = [('lora_a', a), ('lora_b', b)]
    before = {n: p.detach().clone() for n, p in named}
    with torch.no_grad():
        b.copy_(torch.tensor([1., -1.]))
    report = core.update_report(named, before)
    assert report['lora_a']['update_ratio'] == 0
    assert report['lora_b']['update_ratio'] is None
    assert report['lora_b']['update_rms'] == pytest.approx(1.)
    assert torch.equal(before['lora_b'], torch.zeros(2))
    # Strict JSON output: no NaN/Inf to mislead downstream monitoring.
    json.dumps(report, allow_nan=False)
    with torch.no_grad():
        b.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='nonfinite parameter update'):
        core.update_report(named, before)


def test_cpu_cannot_accidentally_start_real_training(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg = core.SparseLoraConfig.from_yaml(str(
        Path(__file__).resolve().parents[2] / 'configs' /
        'sparse_lora_t2va_r32_l0_1_49_smoke.yaml'))
    with pytest.raises(RuntimeError, match='no CUDA'):
        sparse_lora_trainer.SparseLoraTrainer(cfg)
