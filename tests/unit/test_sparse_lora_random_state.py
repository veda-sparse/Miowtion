"""Actual tiny-H3 random-state distillation, epoch stopping and resume."""

import collections
import json

import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule
from miowtion.train import data
from miowtion.train import lora
from miowtion.train import parallel
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_sampling as sampling
from miowtion.train import sparse_lora_trainer as trainer
from miowtion.veda import attention
from miowtion.veda import plan
from miowtion.veda import predictor
from miowtion.veda import tiling


@pytest.fixture
def build(tmp_path, monkeypatch):
    original_student = attention.SparseStudent

    def cpu_student(*args, **kwargs):
        return original_student(*args, allow_reference_kernel=True, **kwargs)

    monkeypatch.setattr(attention, 'SparseStudent', cpu_student)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *_: 0)
    cfg = h3_config.H3Config.tiny(num_layers=4, num_heads=2)
    geo = geometry.Geometry('16:9', 128, 64, 5, 2, 4, 8, 4)
    torch.manual_seed(29)
    writer = data.SampleCacheWriter(str(tmp_path / 'cache'))
    for index in range(4):
        writer.add(str(index), 't2va', 'train', torch.randn(4, cfg.text_dim),
                   torch.ones(4, dtype=torch.long), aspect='16:9', latent_t=2)
    writer.finalize()

    class LiveTables:
        def __init__(self, model):
            self.model = model

        def get(self, timesteps):
            return self.model.precompute_adaln(timesteps)

    def make(name, accum=1, epochs=None, verify=False):
        obj = trainer.SparseLoraTrainer.__new__(trainer.SparseLoraTrainer)
        torch.manual_seed(41)
        obj.model = h3_model.H3DiT(cfg)
        with torch.no_grad():
            for parameter in obj.model.parameters():
                parameter.normal_(std=0.1)
        obj.model.dense_backend = 'math'
        obj.model.gradient_checkpointing = True
        lora.add_lora(obj.model, 32, 32, layers=[0, 1, 3])
        obj.predictor = predictor.TileScorePredictor(4, 2, 32)
        obj.named = core.freeze_except_lora(obj.model, obj.predictor)
        obj.optimizer = torch.optim.AdamW([p for _, p in obj.named], lr=0.01)
        obj.config = core.SparseLoraConfig(
            name, '', '', '', '', '', str(tmp_path / 'cache'), ['16:9@2'],
            lora_layers=[0, 1, 3], accum=accum, lr=0.01, warmup=5,
            timestep_sampling='random', steps=None if epochs else 2,
            epochs=epochs, verify_invariants=verify, save_every=1)
        obj.env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
        obj.cache = data.SampleCache(str(tmp_path / 'cache'))
        obj.samples = sampling.CoverageSampler(obj.cache.samples, 53)
        obj.geometries = data.GeometrySampler(['16:9@2'], 53)
        obj.geometries.geometries = [geo]
        obj.noise_gen = torch.Generator().manual_seed(59)
        obj.timestep_gen = sampling.timestep_generator(53)
        obj.schedule = schedule.turbo_schedule(
            8, schedule.ShiftScales(12., 3.))
        obj.tables = LiveTables(obj.model)
        obj.plans = plan.PlanTable([plan.TilePlan.uniform(
            geo, tiling.TileShape(1, 8, 16), 4, 2)])
        obj.run_dir = tmp_path / name
        obj.run_dir.mkdir()
        obj.step, obj.traj, obj.probe = 0, None, None
        obj.contract = {'rank': 32, 'layers': [0, 1, 3],
                        'timestep_sampling': 'random', 'epochs': epochs}
        obj.total_steps = sampling.planned_updates(
            obj.samples.batches_per_epoch,
            obj.config.training_states_per_visit, accum,
            obj.config.steps, epochs)
        obj.frozen_before = core.frozen_digest(obj.model, obj.predictor)
        return obj

    return make


@pytest.mark.parametrize('selected', range(8))
def test_selected_state_and_gradients_match_explicit_frozen_rollout(
        build, monkeypatch, selected):
    obj = build('selected')
    # Nonzero adapters expose accidental use of LoRA in teacher preparation.
    with torch.no_grad():
        for name, parameter in obj.named:
            if name.endswith('.lora_b'):
                parameter.normal_(std=0.01)
    sampler_state = obj.samples.state()
    noise_state = obj.noise_gen.get_state()
    obj._new_trajectory()
    with core.frozen_teacher(obj.model):
        for _ in range(selected):
            inputs = obj.traj.inputs()
            table = obj.tables.get(inputs.timestep.timesteps)
            velocity = obj.model(obj.traj.clip, inputs.video_rows,
                                 inputs.audio_rows, inputs.timestep,
                                 adaln_table=table)
            obj.traj.advance(*velocity)
        inputs = obj.traj.inputs()
        table = obj.tables.get(inputs.timestep.timesteps)
        args = (obj.traj.clip, inputs.video_rows, inputs.audio_rows,
                inputs.timestep)
        target = obj.model(*args, adaln_table=table)
    expected_inputs = (inputs.video_rows.clone(), inputs.audio_rows.clone())
    sparse = attention.SparseStudent(obj.clip, obj.plan, obj.predictor)
    loss = core.velocity_loss(obj.model(
        *args, attention_fn=sparse, adaln_table=table), target)[0]
    loss.backward()
    expected_grads = {n: p.grad.clone() for n, p in obj.named}
    obj.optimizer.zero_grad(set_to_none=True)
    obj.samples.load_state(sampler_state)
    obj.noise_gen.set_state(noise_state)
    obj.traj = None
    seen = []

    def check_forward(module, args, kwargs):
        student = kwargs.get('attention_fn') is not None
        seen.append(student)
        assert torch.is_grad_enabled() == student
        assert all(child.lora_enabled == student
                   for child in module.modules()
                   if hasattr(child, 'lora_enabled'))
        if student:
            assert torch.equal(args[1], expected_inputs[0])
            assert torch.equal(args[2], expected_inputs[1])

    handle = obj.model.register_forward_pre_hook(
        check_forward, with_kwargs=True)
    original_randint = torch.randint

    def choose_state(*args, **kwargs):
        if kwargs.get('generator') is obj.timestep_gen:
            return torch.tensor([selected])
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, 'randint', choose_state)
    record = obj._micro_step()
    handle.remove()
    assert record['state'] == selected
    assert record['teacher_preparation_steps'] == selected
    assert record['video_sigma'] == obj.schedule.video[selected]
    assert record['audio_sigma'] == obj.schedule.audio[selected]
    assert record['loss'] == float(loss.detach())
    assert seen.count(False) == selected + 1 and seen.count(True) == 1
    assert obj.traj is None
    assert all(torch.equal(p.grad, expected_grads[n]) for n, p in obj.named)
    assert all(p.grad is None for p in obj.predictor.parameters())


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _equal(x, y)
    else:
        assert a == b


def test_random_prompt_visits_stop_at_two_epochs_and_resume_exactly(build):
    first = build('primary', accum=2, epochs=2, verify=True)
    first.train()
    logs = [json.loads(line) for line in
            (first.run_dir / 'log.jsonl').read_text().splitlines()]
    micro = [r for log in logs for r in log['micro_steps']]
    assert first.step == 4 and len(micro) == 8
    assert collections.Counter(r['sample'] for r in micro) == {
        '0': 2, '1': 2, '2': 2, '3': 2}
    assert len({r['seed'] for r in micro}) == 8
    assert logs[-1]['completed_prompt_visits_global'] == 8
    assert logs[-1]['completed_epochs'] == 2
    assert all(log['frozen_invariants_checked'] for log in logs)
    assert 'completed_trajectories_global' not in logs[-1]
    resumed = build('resumed', accum=2, epochs=2, verify=True)
    resumed._restore(str(first.run_dir / 'step_0000001'))
    assert resumed.traj is None
    resumed.train()
    replay = [json.loads(line) for line in
              (resumed.run_dir / 'log.jsonl').read_text().splitlines()]
    assert [log['micro_steps'] for log in logs[1:]] == [
        log['micro_steps'] for log in replay]
    for step in range(2, 5):
        paths = [obj.run_dir / f'step_{step:07d}/state.pt'
                 for obj in (first, resumed)]
        _equal(*(torch.load(p, weights_only=False) for p in paths))


def test_timestep_rng_shared_across_ranks_and_separate_from_noise():
    generators = [sampling.timestep_generator(53) for _ in range(8)]
    draws = [torch.randint(8, (64,), generator=g) for g in generators]
    assert all(torch.equal(d, draws[0]) for d in draws)
    assert set(draws[0].tolist()) == set(range(8))
    state = generators[0].get_state()
    expected = torch.randint(8, (12,), generator=generators[0])
    torch.randint(1000, (73,), generator=torch.Generator().manual_seed(71))
    restored = sampling.timestep_generator(53)
    restored.set_state(state)
    assert torch.equal(expected, torch.randint(8, (12,), generator=restored))


def test_short_final_update_is_normalized_and_restores_exactly(build):
    first = build('short_primary', accum=3, epochs=1)
    first.train()
    logs = [json.loads(line) for line in
            (first.run_dir / 'log.jsonl').read_text().splitlines()]
    assert first.step == 2
    assert [r['micro_steps_in_update'] for r in logs] == [3, 1]
    assert [r['global_states_per_update'] for r in logs] == [3, 1]
    assert logs[-1]['micro_steps'][0]['gradient_weight'] == 1.
    assert logs[-1]['completed_prompt_visits_global'] == 4
    assert logs[-1]['completed_epochs'] == 1
    replay = build('short_replay', accum=3, epochs=1)
    replay._restore(str(first.run_dir / 'step_0000001'))
    replay.train()
    _equal(*(torch.load(obj.run_dir / 'step_0000002/state.pt',
                        weights_only=False) for obj in (first, replay)))


def test_padding_runs_real_student_backward_with_zero_local_gradient(build):
    obj = build('padding')
    obj.samples = sampling.CoverageSampler(obj.cache.samples, 53, rank=7,
                                           world_size=8)
    record = obj._micro_step(loss_scale=2.)
    assert record['sample_valid'] is False and record['gradient_weight'] == 0.
    assert record['calls']['sparse'] and record['calls']['dense'] == 0
    assert all(p.grad is not None and bool((p.grad == 0).all())
               for _, p in obj.named)
    assert obj.samples.issued_real_global == 4 and obj.traj is None
