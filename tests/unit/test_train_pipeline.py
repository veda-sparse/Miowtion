"""CPU integration tests of the training pipeline on a tiny DiT."""

import json
import os

import pytest
import torch
import yaml

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import schedule as h3_schedule
from miowtion.train import adaln
from miowtion.train import checkpoint
from miowtion.train import data
from miowtion.train import trainer as trainer_lib
from miowtion.train import trajectory

from tests.unit import test_h3_model_weights as weights_test

_TINY = h3_config.H3Config.tiny(num_layers=2, num_heads=4)


def _write_release(root, variant='FL2VA'):
    """Tiny checkpoint in the released directory layout."""
    tdir = os.path.join(root, variant, 'transformer')
    os.makedirs(tdir)
    m = weights_test._random_model(_TINY)
    weights_test._write_checkpoint(m, tdir)
    c = _TINY
    with open(os.path.join(tdir, 'config.json'), 'w') as f:
        json.dump({'_class_name': 'MiniMaxH3DiTModel',
                   'hidden_size': c.hidden_size, 'num_layers': c.num_layers,
                   'token_refiner_num_layers': c.num_refiner_layers,
                   'num_attention_heads': c.num_heads,
                   'attention_head_dim': c.head_dim,
                   'ffn_hidden_size': c.ffn_dim, 'text_dim': c.text_dim,
                   'timestep_input_dim': c.freq_dim,
                   'time_embed_hidden_size': c.time_embed_hidden,
                   'time_embed_dim': c.time_embed_dim,
                   'rope_inv_freq_len': c.rope_freqs_per_axis,
                   'latents_dim': c.video_channels,
                   'audio_latents_dim': c.audio_channels,
                   'norm_eps': c.norm_eps, 'qk_norm_eps': c.qk_norm_eps,
                   'final_norm_eps': c.final_norm_eps,
                   'patch_size': [1, 2, 2]}, f)
    with open(os.path.join(root, variant, 'model_index.json'), 'w') as f:
        json.dump({'_minimax_h3': {'sigma_shift_scales': {
            'video': 12.0, 'audio': 3.0}}}, f)
    return m


def _write_samples(directory, n=3):
    writer = data.SampleCacheWriter(directory)
    for i in range(n):
        writer.add(f's{i}', 't2va', 'train',
                   torch.randn(10 + i, _TINY.text_dim),
                   torch.ones(10 + i, dtype=torch.long))
    writer.finalize()


def test_prompt_validation():
    good = ('integrated_multimodal_description: [Shot 1] a cat.\n'
            'overall_soundscape: rain.\nnon_diegetic_music: none.')
    data.validate_prompt(good)
    with pytest.raises(ValueError):
        data.validate_prompt(good.replace('[Shot 1]', ''))
    with pytest.raises(ValueError):
        data.validate_prompt('overall_soundscape: x. ' + good)
    ref = ('subject_definitions: <Subject 1> a man. summary: x. '
           'retention_analysis: y. detailed_description: [Shot 1] z. '
           'overall_soundscape: a. non_diegetic_music: b.')
    data.validate_prompt(ref, 'ref2va')
    with pytest.raises(ValueError):
        data.validate_prompt(ref, 't2va')
    split = data.split_ids([str(i) for i in range(100)], num_test=20)
    assert sum(v == 'test' for v in split.values()) == 20
    assert split == data.split_ids([str(i) for i in range(100)], num_test=20)


def test_sample_cache_roundtrip(tmp_path):
    writer = data.SampleCacheWriter(str(tmp_path))
    hidden = torch.randn(5, 8)
    writer.add('a', 'fl2va', 'train', hidden, torch.tensor([1, 0, 0, 1, 1]),
               keyframes=(0,), aspect='16:9',
               cond_video=torch.randn(4, 96))
    writer.finalize()
    cache = data.SampleCache(str(tmp_path))
    sample = cache.samples[0]
    h, tags = cache.text(sample)
    assert torch.equal(h, hidden.to(torch.bfloat16))
    assert tags.tolist() == [1, 0, 0, 1, 1]
    video, audio = cache.conditions(sample)
    assert video.shape == (4, 96) and audio.shape == (0, 32)
    sampler = data.SampleSampler(cache.samples, seed=0, rank=0)
    assert sampler.next(aspect='16:9').id == 'a'
    with pytest.raises(ValueError):
        sampler.next(aspect='9:16')


def test_adaln_tables_match_live_projection(tmp_path):
    m = _write_release(str(tmp_path))
    tdir = os.path.join(str(tmp_path), 'FL2VA', 'transformer')
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = h3_layout.pack(torch.ones(12, dtype=torch.long), geo)
    sched = h3_schedule.Schedule.build(9, h3_schedule.ShiftScales(12., 3.))
    tables = adaln.AdalnTables()
    tables.build(m, tdir, trajectory.schedule_timestep_sets(
        sched, visual_conditions=False, audio_references=False),
                 torch.device('cpu'))
    for step in range(sched.num_steps):
        state = h3_schedule.build_timestep_state(lay, *sched.timesteps(step))
        live = m.precompute_adaln(state.timesteps)
        cached = tables.get(state.timesteps)
        for a, b in zip(live, cached):
            assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_stage1_trains_and_resumes(tmp_path):
    root = str(tmp_path / 'ckpt')
    _write_release(root)
    cache_dir = str(tmp_path / 'samples')
    _write_samples(cache_dir)
    cfg = {
        'run_name': 'tiny', 'checkpoint_root': root,
        'sample_cache': cache_dir, 'geometries': ['16:9@7'],
        'schedule': 'base', 'num_steps': 4, 'accum': 2, 'steps': 3, 'save_every': 2,
        'warmup': 1, 'keep_ratio': 0.5, 'recall_every': 1,
        'out_dir': str(tmp_path / 'runs'), 'dense_backend': 'math',
        'bootstrap_shape': '1x8x16', 'keep_optimizer': 1,
    }
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(cfg))
    config = trainer_lib.TrainConfig.from_yaml(str(path))
    t = trainer_lib.Trainer(config)
    before = t.predictor.layers[0].proj_q.detach().clone()
    t.train()
    assert not torch.equal(before, t.predictor.layers[0].proj_q)
    log = [json.loads(line) for line in open(
        tmp_path / 'runs' / 'tiny' / 'log.jsonl')]
    steps = [r for r in log if 'step' in r]
    assert [r['step'] for r in steps] == [1, 2, 3]
    assert all(r['kl'] > 0 for r in steps)
    # Training dynamics: per-layer KL, mask quality and its ceiling.
    assert all(len(r['kl_layers']) == 2 for r in steps)
    assert all(0.0 <= r['heat_kept'] <= r['heat_ceiling'] <= 1.0
               for r in steps)
    assert all(r['logit_std'] >= 0 for r in steps)
    # Update diagnostics: cosine needs a previous update.
    assert all(len(r['grad_norm_layers']) == 2 and r['update_ratio'] > 0
               for r in steps)
    assert 'grad_cos' not in steps[0] and -1 <= steps[1]['grad_cos'] <= 1
    latest = checkpoint.latest([str(tmp_path / 'runs' / 'tiny' / 'ckpt')])
    assert latest.endswith('step_0000003')
    # Every checkpoint survives; keep_optimizer 1 dropped the moments of
    # the older one, so only the newest can be resumed from.
    ckpt_dir = tmp_path / 'runs' / 'tiny' / 'ckpt'
    assert sorted(os.listdir(ckpt_dir)) == ['step_0000002', 'step_0000003']
    assert not (ckpt_dir / 'step_0000002' / 'optim.pt').exists()
    assert (ckpt_dir / 'step_0000003' / 'optim.pt').exists()
    assert checkpoint.load(str(ckpt_dir / 'step_0000002'))['optimizer'] == {}
    # A new trainer resumes from the latest complete checkpoint.
    t2 = trainer_lib.Trainer(config)
    assert t2.step == 3
    assert torch.equal(t2.predictor.layers[0].proj_q,
                       t.predictor.layers[0].proj_q)
    assert t2.samples.state() == t.samples.state()


def test_host_optimizer_offload_is_equivalent(tmp_path):
    root = str(tmp_path / 'ckpt')
    _write_release(root)
    cache_dir = str(tmp_path / 'samples')
    _write_samples(cache_dir)
    weights = []
    for offload in (False, True):
        cfg = {
            'run_name': f'offload_{offload}', 'checkpoint_root': root,
            'sample_cache': cache_dir, 'geometries': ['16:9@7'],
            'schedule': 'base', 'num_steps': 4, 'accum': 1, 'steps': 2, 'save_every': 2,
            'warmup': 1, 'keep_ratio': 0.5, 'out_dir': str(tmp_path / 'runs'),
            'dense_backend': 'math', 'bootstrap_shape': '1x8x16',
            'offload_optimizer': offload,
        }
        path = tmp_path / f'cfg_{offload}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        t = trainer_lib.Trainer(trainer_lib.TrainConfig.from_yaml(str(path)))
        t.train()
        weights.append(t.predictor.layers[1].proj_k.detach().clone())
    assert torch.equal(weights[0], weights[1])


def test_turbo_schedule_matches_closed_form():
    from miowtion.h3 import schedule as h3_schedule
    scales = h3_schedule.ShiftScales(12.0, 3.0)
    eight = h3_schedule.turbo_schedule(8, scales)
    assert [round(v, 3) for v in eight.video] == [
        1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.8, 0.632, 0.0]
    assert [round(v, 3) for v in eight.audio] == [
        1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3, 0.0]
    four = h3_schedule.turbo_schedule(4, scales)
    assert four.num_steps == 4 and four.video[2] == pytest.approx(12 / 13)


def _write_adapter(model, path, rank=2, extra=None):
    from safetensors.torch import save_file
    gen = torch.Generator().manual_seed(1)
    tensors = {}
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.startswith(
                 ('blocks.', 'token_refiner.', 'final_layer.adaln'))
             and not n.endswith(('video_out', 'audio_out'))]
    for n in names + list(extra or []):
        weight = dict(model.named_modules())[n].weight if n in dict(
            model.named_modules()) else torch.empty(8, 8)
        tensors[f'{n}.lora_A.weight'] = (torch.randn(
            rank, weight.shape[1], generator=gen) * 0.05).to(torch.bfloat16)
        tensors[f'{n}.lora_B.weight'] = (torch.randn(
            weight.shape[0], rank, generator=gen) * 0.05).to(torch.bfloat16)
    save_file(tensors, path)
    return names


def test_teacher_adapter_tables_equal_full_merge(tmp_path):
    from miowtion.h3 import model as h3_model
    from miowtion.h3 import schedule as h3_schedule
    from miowtion.h3 import weights as h3_weights
    from miowtion.train import lora
    from miowtion.train import parallel
    from miowtion.train import teacher
    root = str(tmp_path / 'ckpt')
    src = _write_release(root)
    adapter_path = str(tmp_path / 'turbo.safetensors')
    _write_adapter(src, adapter_path)
    env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
    t = teacher.build_teacher(root, 'FL2VA', 'turbo', 4, adapter_path, env,
                              visual_conditions=False,
                              audio_references=False)
    t.model.dense_backend = 'math'
    # Reference: keep the AdaLN projections and merge the whole adapter.
    ref = h3_model.H3DiT(_TINY)
    h3_weights.load_dit_weights(ref, os.path.join(root, 'FL2VA',
                                                  'transformer'))
    lora.merge_adapter(ref, lora.load_adapter(adapter_path))
    ref.dense_backend = 'math'
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = h3_layout.pack(torch.ones(12, dtype=torch.long), geo)
    text = torch.randn(12, _TINY.text_dim)
    video = torch.randn(geo.num_video_tokens, 96)
    audio = torch.randn(geo.num_audio_rows, 32)
    for step in range(t.schedule.num_steps):
        state = h3_schedule.build_timestep_state(
            lay, *t.schedule.timesteps(step))
        with torch.no_grad():
            out = t.model(t.model.clip_inputs(lay, t.model.refine_text(text),
                                              torch.device('cpu')),
                          video, audio, state,
                          adaln_table=t.tables.get(state.timesteps))
            expected = ref(ref.clip_inputs(lay, ref.refine_text(text),
                                           torch.device('cpu')),
                           video, audio, state)
        assert torch.equal(out[0], expected[0])
        assert torch.equal(out[1], expected[1])


def test_teacher_adapter_is_strict(tmp_path):
    from miowtion.train import parallel
    from miowtion.train import teacher
    root = str(tmp_path / 'ckpt')
    src = _write_release(root)
    adapter_path = str(tmp_path / 'bad.safetensors')
    _write_adapter(src, adapter_path, extra=['blocks.0.attn.nonexistent'])
    env = parallel.DistEnv(0, 1, 0, torch.device('cpu'), None)
    with pytest.raises(KeyError):
        teacher.build_teacher(root, 'FL2VA', 'turbo', 4, adapter_path, env,
                              visual_conditions=False, audio_references=False)


def test_geometry_cycle_covers_all_and_resumes():
    specs = ['1:1@37', '4:3@72', '16:9@102', '9:16@37']
    sampler = data.GeometrySampler(specs, seed=3, mode='cycle')
    names = [sampler.next().name for _ in range(2 * len(specs))]
    for r in range(2):
        assert sorted(names[r * 4:(r + 1) * 4]) == sorted(
            data.parse_geometry(s).name for s in specs)
    # Resuming from the state continues the same sequence.
    fresh = data.GeometrySampler(specs, seed=3, mode='cycle')
    [fresh.next() for _ in range(5)]
    resumed = data.GeometrySampler(specs, seed=3, mode='cycle')
    resumed.load_state(fresh.state())
    assert [resumed.next().name for _ in range(6)] == [
        fresh.next().name for _ in range(6)]
    with pytest.raises(ValueError):
        data.GeometrySampler(specs, seed=0, mode='round-robin')


def test_update_monitor_and_gradient_stop():
    from miowtion.train import monitor
    from miowtion.veda import predictor as veda_predictor
    pred = veda_predictor.TileScorePredictor(2, 2, 8)
    named = [(f'predictor.{n}', p) for n, p in pred.named_parameters()]
    mon = monitor.UpdateMonitor(named)
    opt = torch.optim.SGD(pred.parameters(), lr=0.1)
    for p in pred.parameters():
        p.grad = torch.ones_like(p)
    first = mon.before_step()
    assert 'grad_cos' not in first and len(first['grad_norm_layers']) == 2
    opt.step()
    ratio = mon.after_step()['update_ratio']
    assert ratio > 0
    for p in pred.parameters():
        p.grad = -torch.ones_like(p)  # opposite direction
    second = mon.before_step()
    assert second['grad_cos'] == pytest.approx(-1.0)
    assert second['grad_cos_layers'] == [-1.0, -1.0]
    # A gradient on a parameter outside the trainable set is rejected.
    trunk = torch.nn.Linear(2, 2)
    model = torch.nn.ModuleDict({'trunk': trunk, 'pred': pred})
    monitor.check_gradient_stop(model, list(model.parameters()))
    trunk.weight.grad = torch.ones_like(trunk.weight)
    with pytest.raises(RuntimeError, match='frozen'):
        monitor.check_gradient_stop(model, list(pred.parameters()))


def _lr_config(**kwargs) -> trainer_lib.TrainConfig:
    base = dict(run_name='lr', checkpoint_root='.', sample_cache='.',
                geometries=['1:1@37'], lr=1e-3, warmup=20, steps=600)
    base.update(kwargs)
    return trainer_lib.TrainConfig(**base)


def test_constant_lr_is_warmup_then_flat():
    config = _lr_config()
    assert trainer_lib.learning_rate(config, 0) == pytest.approx(5e-5)
    assert trainer_lib.learning_rate(config, 19) == pytest.approx(1e-3)
    assert trainer_lib.learning_rate(config, 599) == pytest.approx(1e-3)


def test_cosine_decays_from_the_end_of_the_warmup_to_the_floor():
    config = _lr_config(lr_decay='cosine', lr_min_ratio=0.1)
    # The cosine starts where the warmup ends: no jump at the seam.
    assert trainer_lib.learning_rate(config, 19) == pytest.approx(1e-3)
    # Halfway through the post-warmup span, the cosine is at the midpoint
    # between lr and the floor.
    assert trainer_lib.learning_rate(config, 19 + 290) == pytest.approx(
        0.55e-3, rel=1e-3)
    assert trainer_lib.learning_rate(config, 599) == pytest.approx(1e-4)
    # Monotone after the warmup, and clamped at the floor past the end.
    rates = [trainer_lib.learning_rate(config, s) for s in range(19, 600)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert trainer_lib.learning_rate(config, 900) == pytest.approx(1e-4)
    # The warmup itself is unchanged by the decay.
    plain = _lr_config()
    assert all(trainer_lib.learning_rate(config, s)
               == pytest.approx(trainer_lib.learning_rate(plain, s))
               for s in range(19))


def test_lr_schedule_settings_are_validated():
    with pytest.raises(ValueError, match='lr_decay'):
        _lr_config(lr_decay='linear').validate()
    with pytest.raises(ValueError, match='lr_min_ratio'):
        _lr_config(lr_decay='cosine', lr_min_ratio=1.5).validate()


def test_clipping_is_per_predictor_layer_and_does_not_couple_layers():
    from miowtion.veda import predictor as veda_predictor
    pred = veda_predictor.TileScorePredictor(3, 2, 8)
    named = [(f'predictor.{n}', p) for n, p in pred.named_parameters()]
    groups = trainer_lib._clip_groups(named)
    assert sorted(groups) == ['predictor.layers.0', 'predictor.layers.1',
                              'predictor.layers.2']

    last = 'predictor.layers.2'
    others = [(n, p) for n, p in named if not n.startswith(last)]

    def grads(spike: float) -> list[torch.Tensor]:
        for name, p in named:
            scale = spike if name.startswith(last) else 1.0
            p.grad = torch.full_like(p, 0.3 * scale)
        norms = {name: torch.nn.utils.clip_grad_norm_(ps, 1.0).item()
                 for name, ps in trainer_lib._clip_groups(named).items()}
        assert 'predictor' in trainer_lib.summarize_norms(norms)
        return [p.grad.clone() for _, p in others]

    # The last layer's gradient is what blows up in practice (50-70x the
    # median layer). Clipped per layer, the other layers must not notice.
    calm, spiked = grads(1.0), grads(1000.0)
    assert all(torch.equal(a, b) for a, b in zip(calm, spiked))

    # With one clip over the whole predictor -- what this replaced -- the
    # same spike drags every other layer down with it.
    def global_clip(spike: float) -> list[torch.Tensor]:
        for name, p in named:
            scale = spike if name.startswith(last) else 1.0
            p.grad = torch.full_like(p, 0.3 * scale)
        torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
        return [p.grad.clone() for _, p in others]

    assert not any(torch.equal(a, b)
                   for a, b in zip(global_clip(1.0), global_clip(1000.0)))


def test_summarize_norms_keeps_one_predictor_entry():
    got = trainer_lib.summarize_norms(
        {'predictor.layers.0': 3.0, 'predictor.layers.1': 4.0, 'head': 2.0})
    assert got == {'predictor': pytest.approx(5.0), 'head': 2.0}


def test_synthetic_sample_cache():
    cache = data.SyntheticSampleCache(7, _TINY.text_dim, seed=1)
    sample = cache.samples[0]
    assert sample.task == 't2va' and sample.text_len == 7
    hidden, tags = cache.text(sample)
    assert hidden.shape == (7, _TINY.text_dim)
    assert hidden.dtype == torch.bfloat16
    assert torch.equal(tags, torch.ones(7, dtype=torch.long))
    video, audio = cache.conditions(sample)
    assert video.shape == (0, 96) and audio.shape == (0, 32)
    again = data.SyntheticSampleCache(7, _TINY.text_dim, seed=1)
    assert torch.equal(again.text(sample)[0], hidden)
    with pytest.raises(ValueError):
        data.SyntheticSampleCache(0, _TINY.text_dim)


def test_a_geometry_with_no_sample_is_refused_at_startup():
    """Better than dying whenever `uniform` first draws the orphan."""
    from miowtion.train import trainer as trainer_lib
    samples = [data.Sample(id='a', task='t2va', split='train', text=(0, 4),
                           latent_t=37)]
    checker = trainer_lib.Trainer.__new__(trainer_lib.Trainer)
    checker.samples = data.SampleSampler(samples, seed=0, rank=0)
    checker.geometries = data.GeometrySampler(['16:9@37'], seed=0)
    checker._check_geometry_coverage()          # the corpus covers this one
    checker.geometries = data.GeometrySampler(['16:9@37', '16:9@102'],
                                              seed=0)
    with pytest.raises(ValueError, match='no sample fits'):
        checker._check_geometry_coverage()
