"""Strict named-state optimizer restoration; no real GPU claim."""

import copy
import collections
import dataclasses
import json

import pytest
import torch

from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_trainer


def test_resume_can_fork_outputs_but_refuses_rewinding_over_existing_steps(
        tmp_path):
    run = tmp_path / 'original'
    checkpoint = run / 'step_0000001'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'done.json').write_text(json.dumps({'step': 1}))
    (run / 'step_0000003').mkdir()
    with pytest.raises(FileExistsError, match='newer checkpoints'):
        sparse_lora_trainer.validate_run_destination(run, str(checkpoint), 3)
    fork = tmp_path / 'fork'
    sparse_lora_trainer.validate_run_destination(fork, str(checkpoint), 3)
    assert not fork.exists()
    with pytest.raises(ValueError, match='resume step exceeds'):
        sparse_lora_trainer.validate_run_destination(fork, str(checkpoint), 0)
    with pytest.raises(FileNotFoundError, match='completed checkpoint'):
        sparse_lora_trainer.validate_run_destination(fork, str(fork), 3)


def _payload(parameters, optimizer, contract):
    return {'contract': contract, 'step': 1,
            'weights': {n: p.detach().clone() for n, p in parameters},
            'optimizer': {n: copy.deepcopy(optimizer.state[p])
                          for n, p in parameters},
            'param_groups': [{k: v for k, v in group.items() if k != 'params'}
                             for group in optimizer.param_groups]}


def test_optimizer_roundtrip_matches_next_update_bitwise():
    contract = {'task': 't2va', 'rank': 32, 'layers': [0, 1, 49],
                'plan_sha256': 'plan', 'train_sha256': 'train', 'keep_ratio': 0.05}
    a = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer_a = torch.optim.AdamW([a], lr=0.01)
    a.square().sum().backward()
    optimizer_a.step()
    payload = _payload([('lora_a', a)], optimizer_a, contract)
    b = torch.nn.Parameter(torch.ones(4))
    optimizer_b = torch.optim.AdamW([b], lr=0.5)
    assert core.restore_training_payload(
        [('lora_a', b)], optimizer_b, payload, contract) == 1
    assert torch.equal(a, b)
    for parameter, optimizer in [(a, optimizer_a), (b, optimizer_b)]:
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    assert torch.equal(a, b)
    for key, value in optimizer_a.state[a].items():
        assert torch.equal(value, optimizer_b.state[b][key])
    for key in contract:
        changed = {**contract, key: 'incompatible'}
        with pytest.raises(ValueError, match='contract'):
            core.restore_training_payload([('lora_a', b)], optimizer_b,
                                          payload, changed)
    broken = {**payload, 'optimizer': {}}
    with pytest.raises(ValueError, match='complete optimizer'):
        core.restore_training_payload([('lora_a', b)], optimizer_b,
                                      broken, contract)


def test_sampling_transition_only_changes_the_authorized_contract_field():
    old_contract = {'settings': {'timestep_sampling': 'random', 'lr': 1e-4},
                    'assets': {'teacher': 'frozen'}, 'world_size': 4}
    new_contract = copy.deepcopy(old_contract)
    new_contract['settings']['timestep_sampling'] = 'trajectory'
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    payload = _payload([('a', parameter)], optimizer, old_contract)
    payload['rank_states'] = [{'active_trajectory': None,
                               'sampler': {'position': 40},
                               'noise': torch.arange(8)} for _ in range(4)]
    migrated = core.switch_payload_to_trajectory(payload, new_contract)
    assert migrated['contract'] == new_contract
    assert payload['contract'] == old_contract
    assert migrated['rank_states'] is payload['rank_states']
    assert migrated['optimizer'] is payload['optimizer']
    assert migrated['weights'] is payload['weights']
    restored = torch.nn.Parameter(torch.zeros(4))
    restored_optimizer = torch.optim.AdamW([restored], lr=.5)
    assert core.restore_training_payload(
        [('a', restored)], restored_optimizer, migrated, new_contract) == 1
    assert torch.equal(parameter, restored)
    for key, value in optimizer.state[parameter].items():
        assert torch.equal(value, restored_optimizer.state[restored][key])
    for field in ('lr', 'unexpected'):
        bad = copy.deepcopy(new_contract)
        bad['settings'][field] = 'changed'
        with pytest.raises(ValueError, match='contract'):
            core.switch_payload_to_trajectory(payload, bad)
    with pytest.raises(ValueError, match='destination'):
        core.switch_payload_to_trajectory(payload, old_contract)
    bad_payload = copy.deepcopy(payload)
    bad_payload['rank_states'][1]['active_trajectory'] = {'step': 2}
    with pytest.raises(ValueError, match='inactive'):
        core.switch_payload_to_trajectory(bad_payload, new_contract)


def test_real_cpu_trajectory_checkpoint_resumes_mid_rollout(tmp_path,
                                                         monkeypatch):
    """Actual tiny H3, teacher Euler states and reference sparse backward.

    Only the CPU reference kernel is selected explicitly. No CUDA model,
    distributed runtime, or custom kernel is simulated by this test.
    """
    from miowtion.h3 import config as h3_config
    from miowtion.h3 import geometry
    from miowtion.h3 import model as h3_model
    from miowtion.h3 import schedule
    from miowtion.train import adaln
    from miowtion.train import data
    from miowtion.train import lora
    from miowtion.train import parallel
    from miowtion.train import sparse_lora_trainer as trainer
    from miowtion.train import sparse_lora_sampling as sampling
    from miowtion.veda import attention
    from miowtion.veda import plan
    from miowtion.veda import predictor
    from miowtion.veda import tiling

    original = attention.SparseStudent

    def cpu_student(*args, **kwargs):
        return original(*args, allow_reference_kernel=True, **kwargs)

    monkeypatch.setattr(attention, 'SparseStudent', cpu_student)
    cfg = h3_config.H3Config.tiny(num_layers=4, num_heads=2)
    geo = geometry.Geometry('16:9', 128, 64, 5, 2, 4, 8, 4)
    writer = data.SampleCacheWriter(str(tmp_path / 'cache'))
    torch.manual_seed(29)
    for index in range(2):
        writer.add(str(index), 't2va', 'train', torch.randn(4, cfg.text_dim),
                   torch.ones(4, dtype=torch.long), aspect='16:9', latent_t=2)
    writer.finalize()

    class LiveTables:
        def __init__(self, model):
            self.model = model

        def get(self, timesteps):
            return self.model.precompute_adaln(timesteps)

    def build(name):
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
            lora_layers=[0, 1, 3], accum=2, lr=0.01, warmup=5,
            timestep_sampling='trajectory')
        obj.env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
        obj.cache = data.SampleCache(str(tmp_path / 'cache'))
        obj.samples = sampling.CoverageSampler(obj.cache.samples, 53)
        obj.geometries = data.GeometrySampler(['16:9@2'], 53, 'cycle')
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
        obj.contract = {'rank': 32, 'alpha': 32, 'layers': [0, 1, 3]}
        return obj

    def update(obj):
        obj.optimizer.zero_grad(set_to_none=True)
        for group in obj.optimizer.param_groups:
            group['lr'] = core.learning_rate(obj.config, obj.step)
        records = [obj._micro_step() for _ in range(2)]
        core.gradient_report(obj.named, require_a=obj.step > 0)
        before = {n: p.detach().clone() for n, p in obj.named}
        obj.optimizer.step()
        obj.updates = core.update_report(obj.named, before)
        obj.step += 1
        return records

    first = build('first')
    states = []
    for _ in range(3):
        states.extend(r['state'] for r in update(first))
    assert states == list(range(6))
    first._save()
    second = build('second')
    second._restore(str(first.run_dir / 'step_0000003'))
    assert second.step == 3 and second.traj.step == 6
    assert torch.equal(first.traj.video_rows, second.traj.video_rows)
    assert torch.equal(first.traj.audio_rows, second.traj.audio_rows)
    for _ in range(2):
        original_records = update(first)
        restored_records = update(second)
        assert original_records == restored_records
        assert first.updates == second.updates
        assert (first.optimizer.param_groups[0]['lr'] ==
                second.optimizer.param_groups[0]['lr'])
        for (_, a), (_, b) in zip(first.named, second.named):
            assert torch.equal(a, b)
    assert [r['state'] for r in original_records] == [0, 1]
    # A checkpoint at a trajectory boundary has no active state. Replay
    # into a resident model must discard the newer in-memory trajectory.
    for _ in range(3):
        update(first)
    assert first.traj.done and first.step == 8
    first._save()
    expected = update(first)
    expected_weights = {n: p.detach().clone() for n, p in first.named}
    first._restore(str(first.run_dir / 'step_0000008'))
    assert first.traj is None
    assert update(first) == expected
    assert all(torch.equal(p, expected_weights[n]) for n, p in first.named)
    # Exercise the real outer loop and epoch stopping on a CPU tiny H3.
    # Only CUDA accounting is replaced; forward/backward/save are real.
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *_: 0)
    epochs = build('epochs')
    epochs.config = dataclasses.replace(
        epochs.config, steps=None, epochs=2, save_every=1)
    epochs.total_steps = sampling.planned_updates(
        epochs.samples.batches_per_epoch, 8, epochs.config.accum, None, 2)
    epochs.train()
    records = [json.loads(line) for line in
               (epochs.run_dir / 'log.jsonl').read_text().splitlines()]
    assert epochs.step == 16 and epochs.traj.done
    assert records[-1]['completed_epochs'] == 2
    assert records[-1]['completed_trajectories_global'] == 4
    completed = collections.Counter(
        r['sample'] for log in records for r in log['micro_steps']
        if r['state'] == 7)
    assert completed == {'0': 2, '1': 2}
