"""Real CUDA sparse LoRA backward, updates, tail batches and resume."""

import copy
import dataclasses
import json

import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule
from miowtion.kernels import fa4
from miowtion.train import data
from miowtion.train import lora
from miowtion.train import parallel
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_sampling as sampling
from miowtion.train import sparse_lora_trainer as trainer
from miowtion.train import sparse_lora_validation
from miowtion.train import trajectory
from miowtion.veda import attention
from miowtion.veda import plan
from miowtion.veda import predictor
from miowtion.veda import tiling

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not fa4.available(), reason='requires real FA4 block sparsity')]


def _case(tmp_path):
    torch.manual_seed(19)
    cfg = dataclasses.replace(h3_config.H3Config.tiny(
        num_layers=50, num_heads=2), head_dim=128)
    model = h3_model.H3DiT(cfg).cuda()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    model.dense_backend = 'fa4'
    lora.add_lora(model, 32, 32, layers=[0, 1, 49])
    pred = predictor.TileScorePredictor(50, 2, 128).cuda()
    core.freeze_except_lora(model, pred)
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    writer = data.SampleCacheWriter(str(tmp_path / 'cache'))
    writer.add('synthetic_gpu_validation', 't2va', 'validation',
               torch.randn(12, cfg.text_dim), torch.ones(12, dtype=torch.long),
               aspect='16:9', latent_t=7)
    writer.finalize()
    cache = data.SampleCache(str(tmp_path / 'cache'))
    sched = schedule.turbo_schedule(8, schedule.ShiftScales(12., 3.))
    traj = trajectory.Trajectory(model, cache, cache.samples[0], geo, sched,
                                  0, torch.device('cuda'))
    shapes = [tiling.TileShape(1, 8, 16), tiling.TileShape(2, 8, 8)]
    tile_plan = plan.TilePlan(geo.name, geo.video_grid, shapes,
                              [[0, 1] for _ in range(50)])
    table = model.precompute_adaln(traj.inputs().timestep.timesteps)
    return model, pred, traj, tile_plan, table


def test_fifty_layer_sparse_lora_backward_without_update(tmp_path):
    model, pred, traj, tile_plan, table = _case(tmp_path)
    report = sparse_lora_validation.check_state(
        model, pred, traj, tile_plan, table, str(tmp_path / 'validation'),
        ordered_dq=True)
    assert report['status'] == 'passed' and report['optimizer_updates'] == 0
    assert report['adapters_unchanged']
    assert report['all_frozen_values_unchanged']
    assert report['a_gradients_zero_at_init']
    assert set(report['modules']) == {
        f'blocks.{i}.{target}' for i in [0, 1, 49]
        for target in ['attn.qkv_proj', 'attn.out_proj']}
    assert report['student_calls_including_recompute']['sparse'] >= 50
    assert report['student_calls_including_recompute']['dense'] == 0
    assert traj.step == 0


def test_nonzero_initialized_fixture_reaches_a_and_b(tmp_path):
    model, pred, traj, tile_plan, table = _case(tmp_path)
    # A separate synthetic fixture tests A connectivity without running an
    # optimizer step. This is initialization, not a learned adapter.
    with torch.no_grad():
        for name, parameter in lora.lora_parameters(model):
            if name.endswith('.lora_b'):
                parameter.normal_(std=0.01)
    model.gradient_checkpointing = True
    inputs = traj.inputs()
    args = (traj.clip, inputs.video_rows, inputs.audio_rows, inputs.timestep)
    with core.frozen_teacher(model):
        target = model(*args, adaln_table=table)
    sparse = attention.SparseStudent(attention.ClipTiling(
        traj.layout, attention.VedaConfig(ordered_dq=True),
        torch.device('cuda')),
        tile_plan, pred)
    student = model(*args, attention_fn=sparse, adaln_table=table)
    core.velocity_loss(student, target)[0].backward()
    report = core.gradient_report(lora.lora_parameters(model), require_a=True)
    assert all(value > 0 for value in report.values())


def test_updated_fifty_layer_resume_matches_weights_and_gradients(tmp_path):
    """Real optimizer and save/restore, not just a repeated gradient norm."""
    model, pred, traj, tile_plan, table = _case(tmp_path)
    model.gradient_checkpointing = True
    named = lora.lora_parameters(model)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-4,
                                  betas=(.9, .95), weight_decay=0.)
    inputs = traj.inputs()
    args = (traj.clip, inputs.video_rows, inputs.audio_rows, inputs.timestep)
    with core.frozen_teacher(model):
        target = model(*args, adaln_table=table)
    contract = {'ordered_dq': True, 'fixture': 'fifty-layer-gpu'}

    def update():
        optimizer.zero_grad(set_to_none=True)
        sparse = attention.SparseStudent(attention.ClipTiling(
            traj.layout, attention.VedaConfig(ordered_dq=True),
            torch.device('cuda')), tile_plan, pred)
        out = model(*args, attention_fn=sparse, adaln_table=table)
        core.velocity_loss(out, target)[0].backward()
        gradients = {n: p.grad.detach().clone() for n, p in named}
        optimizer.step()
        return gradients

    update()
    update()
    payload = {
        'contract': contract, 'step': 2,
        'weights': {n: p.detach().cpu().clone() for n, p in named},
        'optimizer': {n: {k: v.detach().cpu().clone() for k, v in
                           optimizer.state[p].items()} for n, p in named},
        'param_groups': [{k: copy.deepcopy(v) for k, v in group.items()
                          if k != 'params'} for group in
                         optimizer.param_groups],
    }
    original_gradients = update()
    expected = {n: p.detach().clone() for n, p in named}
    moments = {n: copy.deepcopy(optimizer.state[p]) for n, p in named}
    core.restore_training_payload(named, optimizer, payload, contract)
    replay_gradients = update()
    for n, p in named:
        assert torch.equal(original_gradients[n], replay_gradients[n]), n
        assert torch.equal(expected[n], p), n
        for key, value in moments[n].items():
            assert torch.equal(value.cpu(), optimizer.state[p][key].cpu())


def _tail_trainer(tmp_path, name):
    """Builds the real trainer loop with a synthetic fifty-layer CUDA H3."""
    case_dir = tmp_path / name
    model, pred, traj, tile_plan, _ = _case(case_dir)
    model.gradient_checkpointing = True
    writer = data.SampleCacheWriter(str(case_dir / 'tail_cache'))
    for index in range(5):
        writer.add(str(index), 't2va', 'train',
                   torch.randn(12, model.config.text_dim),
                   torch.ones(12, dtype=torch.long),
                   aspect='16:9', latent_t=7)
    writer.finalize()
    obj = trainer.SparseLoraTrainer.__new__(trainer.SparseLoraTrainer)
    obj.model, obj.predictor = model, pred
    obj.named = lora.lora_parameters(model)
    obj.optimizer = torch.optim.AdamW([p for _, p in obj.named], lr=1e-4)
    obj.config = core.SparseLoraConfig(
        name, '', '', '', '', '', str(case_dir / 'tail_cache'),
        ['16:9@7'], steps=None, epochs=1, accum=4, warmup=5,
        verify_invariants=True, ordered_dq=True, save_every=1)
    obj.env = parallel.DistEnv(0, 1, 0, torch.device('cuda:0'), None)
    obj.cache = data.SampleCache(obj.config.sample_cache)
    obj.samples = sampling.CoverageSampler(obj.cache.samples, 53)
    obj.geometries = data.GeometrySampler(['16:9@7'], 53)
    obj.geometries.geometries = [traj.geometry]
    obj.noise_gen = torch.Generator().manual_seed(59)
    obj.timestep_gen = sampling.timestep_generator(53)
    obj.schedule = traj.schedule

    class LiveTables:
        def get(self, timesteps):
            return model.precompute_adaln(timesteps)

    obj.tables = LiveTables()
    obj.plans = plan.PlanTable([tile_plan])
    obj.run_dir = case_dir / 'run'
    obj.run_dir.mkdir(parents=True)
    obj.step, obj.traj, obj.probe = 0, None, None
    obj.contract = {'fixture': 'fifty-layer-cuda-tail',
                    'sampling': sampling.SAMPLING_VERSION}
    obj.total_steps = sampling.planned_updates(5, 1, 4, None, 1)
    obj.frozen_before = core.frozen_digest(model, pred)
    return obj


def _equal_payload(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal_payload(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _equal_payload(x, y)
    else:
        assert a == b


def test_cuda_partial_accumulation_and_checkpoint_replay(tmp_path):
    primary = _tail_trainer(tmp_path, 'primary')
    primary.train()
    logs = [json.loads(line) for line in
            (primary.run_dir / 'log.jsonl').read_text().splitlines()]
    assert [r['micro_steps_in_update'] for r in logs] == [4, 1]
    assert [r['global_states_per_update'] for r in logs] == [4, 1]
    assert logs[-1]['micro_steps'][0]['gradient_weight'] == 1.
    assert logs[-1]['completed_prompt_visits_global'] == 5
    assert logs[-1]['completed_epochs'] == 1
    assert len({r['sample'] for log in logs for r in log['micro_steps']}) == 5
    replay = _tail_trainer(tmp_path, 'replay')
    replay._restore(str(primary.run_dir / 'step_0000001'))
    replay.train()
    _equal_payload(*(torch.load(obj.run_dir / 'step_0000002/state.pt',
                                map_location='cpu', weights_only=False)
                     for obj in (primary, replay)))


def test_cuda_padding_backward_has_zero_local_gradient(tmp_path):
    obj = _tail_trainer(tmp_path, 'padding')
    obj.samples = sampling.CoverageSampler(obj.cache.samples, 53,
                                           rank=7, world_size=8)
    record = obj._micro_step(loss_scale=2.)
    assert not record['sample_valid'] and record['gradient_weight'] == 0.
    assert record['calls']['sparse'] and not record['calls']['dense']
    assert all(p.grad is not None and bool((p.grad == 0).all())
               for _, p in obj.named)
