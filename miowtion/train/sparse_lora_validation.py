"""CUDA Sparse LoRA validation with backward and zero optimizer updates.

This module never constructs an optimizer, advances a trajectory, or saves
training checkpoints. Its input can be a tiny synthetic fixture or a real
H3 state, and the report records which model and geometry were exercised.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import time

import torch

from miowtion.h3 import model as h3_model
from miowtion.kernels import fa4
from miowtion.train import lora
from miowtion.train import sparse_lora as core
from miowtion.train import trajectory as trajectory_lib
from miowtion.utils import progress
from miowtion.veda import attention
from miowtion.veda import mask
from miowtion.veda import plan as plan_lib
from miowtion.veda import predictor as predictor_lib


def _errors(actual: tuple, expected: tuple) -> dict:
    result = {}
    for name, a, b in zip(('video', 'audio'), actual, expected):
        delta = a.detach().float() - b.detach().float()
        result[name] = {
            'bitwise_equal': torch.equal(a, b),
            'max_abs': float(delta.abs().max()),
            'mse': float(delta.square().mean()),
        }
    return result


def check_state(model: h3_model.H3DiT,
                predictor: predictor_lib.TileScorePredictor,
                trajectory: trajectory_lib.Trajectory,
                selected_plan: plan_lib.TilePlan,
                adaln_table: object, output_dir: str,
                keep_ratio: float = 0.05,
                hash_frozen: bool = True, ordered_dq: bool = False) -> dict:
    """Checks one CUDA state without changing any parameter values.

    Requires real FA4. Zero-init is compared to the sparse baseline, while
    Dense teacher outputs provide the detached velocity target. At B=0,
    only B is required to have nonzero gradients. This cannot establish
    A's behavior after a real update or the quality of generated videos.
    """
    device = trajectory.video_rows.device
    if device.type != 'cuda' or not fa4.available(device):
        raise RuntimeError('real CUDA FA4 required; no reference fallback')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'running', 'optimizer_created': False,
        'optimizer_updates': 0, 'trajectory_advanced': False,
        'device_name': torch.cuda.get_device_name(device),
        'capability': list(torch.cuda.get_device_capability(device)),
        'model_hidden_size': model.config.hidden_size,
        'model_layers': model.config.num_layers,
        'model_heads': model.config.num_heads,
        'head_dim': model.config.head_dim,
        'geometry': trajectory.geometry.name,
        'canvas': [trajectory.geometry.width, trajectory.geometry.height],
        'frame_count': trajectory.geometry.frame_count,
        'tokens': trajectory.layout.used, 'sample_id': trajectory.sample.id,
        'state_index': trajectory.step, 'keep_ratio': keep_ratio,
        'ordered_dq': ordered_dq,
        'plan_geometry': selected_plan.geometry,
        'plan_sha256': core.canonical_hash(selected_plan.to_json()),
        'stage_seconds': {}, 'stage_peak_memory_bytes': {},
    }

    def record(phase):
        report['phase'] = phase
        report['updated_at'] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        (output / 'progress.json').write_text(json.dumps(report, indent=2))
        progress.log(f'zero-update validation: {phase}')

    def timed(name, function):
        record(name)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        result = function()
        torch.cuda.synchronize(device)
        report['stage_seconds'][name] = time.time() - started
        report['stage_peak_memory_bytes'][name] = (
            torch.cuda.max_memory_allocated(device))
        return result

    try:
        named = core.freeze_except_lora(model, predictor)
        model.gradient_checkpointing = True
        model.zero_grad(set_to_none=True)
        if any(not module.lora_enabled for module in model.modules()
               if hasattr(module, 'lora_enabled')):
            raise ValueError('new LoRA must be enabled for student validation')
        if any(torch.count_nonzero(p) for n, p in named
               if n.endswith('.lora_b')):
            raise ValueError('this check requires zero-initialized B')
        report['modules'] = [n[:-len('.lora_a')] for n, _ in named
                             if n.endswith('.lora_a')]
        report['trainable_parameters'] = sum(p.numel() for _, p in named)
        adapters_before = {n: p.detach().cpu().clone() for n, p in named}
        frozen_before = (timed('hash_frozen_before', lambda: core.frozen_digest(
            model, predictor)) if hash_frozen else None)
        inputs = trajectory.inputs()
        args = (trajectory.clip, inputs.video_rows, inputs.audio_rows,
                inputs.timestep)
        report['timesteps'] = inputs.timestep.timesteps.cpu().tolist()
        tiling = attention.ClipTiling(
            trajectory.layout, attention.VedaConfig(
                ordered_dq=ordered_dq,
                target_budget=mask.Budget(ratio=keep_ratio),
                dense_layers=frozenset()), device)

        def dense_teacher():
            with core.frozen_teacher(model):
                return model(*args, adaln_table=adaln_table)

        target = timed('dense_teacher', dense_teacher)
        if any(t.requires_grad for t in target):
            raise AssertionError('teacher must be detached')

        def sparse_baseline():
            with core.frozen_teacher(model):
                return model(*args, attention_fn=attention.SparseStudent(
                    tiling, selected_plan, predictor), adaln_table=adaln_table)

        baseline = timed('sparse_baseline', sparse_baseline)

        def zero_init_student():
            with torch.no_grad():
                return model(*args, attention_fn=attention.SparseStudent(
                    tiling, selected_plan, predictor), adaln_table=adaln_table)

        zero = timed('zero_init_student', zero_init_student)
        report['zero_init_errors'] = _errors(zero, baseline)
        if not all(v['bitwise_equal'] for v in
                   report['zero_init_errors'].values()):
            raise AssertionError('zero-init student changed sparse baseline')
        del zero

        student_attention = attention.SparseStudent(
            tiling, selected_plan, predictor)
        student = timed('student_with_grad', lambda: model(
            *args, attention_fn=student_attention, adaln_table=adaln_table))
        report['grad_forward_errors'] = _errors(student, baseline)
        loss, video, audio = core.velocity_loss(student, target)
        if not torch.isfinite(loss):
            raise AssertionError('nonfinite velocity loss')
        report['loss'] = float(loss.detach())
        report['video_velocity_mse'] = float(video.detach())
        report['audio_velocity_mse'] = float(audio.detach())
        timed('backward', loss.backward)
        report['gradient_norms'] = core.gradient_report(named)
        report['a_gradients_zero_at_init'] = all(
            value == 0 for name, value in report['gradient_norms'].items()
            if name.endswith('.lora_a'))
        report['student_calls_including_recompute'] = student_attention.calls
        if (student_attention.calls['dense'] or
                not student_attention.calls['sparse']):
            raise AssertionError('student did not stay fully sparse')
        if any(p.grad is not None for n, p in model.named_parameters()
               if '.lora_' not in n):
            raise AssertionError('backbone or output head received gradients')
        if any(p.grad is not None for p in predictor.parameters()):
            raise AssertionError('predictor received gradients')
        report['adapters_unchanged'] = all(torch.equal(
            adapters_before[n], p.detach().cpu()) for n, p in named)
        if not report['adapters_unchanged']:
            raise AssertionError('adapter values changed without an update')
        after = timed('dense_teacher_after_backward', dense_teacher)
        report['teacher_stability_errors'] = _errors(after, target)
        if not all(v['bitwise_equal'] for v in
                   report['teacher_stability_errors'].values()):
            raise AssertionError('fixed-state teacher output drifted')
        if hash_frozen:
            frozen_after = timed('hash_frozen_after', lambda:
                                 core.frozen_digest(model, predictor))
            report['frozen_before_sha256'] = frozen_before
            report['frozen_after_sha256'] = frozen_after
            unchanged = frozen_before == frozen_after
            report['all_frozen_values_unchanged'] = unchanged
            if frozen_before != frozen_after:
                raise AssertionError('frozen model/predictor values changed')
        report['status'] = 'passed'
        record('complete')
        return report
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        record('failed')
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2))


def check_native_state(config: core.SparseLoraConfig, sample_cache: str,
                       sample_id: str, geometry: str, output_dir: str,
                       seed: int = 0) -> dict:
    """Loads real backbone/Turbo/bundle for one zero-update diagnostic state.

    Cache provenance is reported as a diagnostic input, not silently
    adopted as new training data. No optimizer or Trainer is constructed.
    """
    from miowtion.train import data
    from miowtion.train import parallel
    from miowtion.train import teacher
    from miowtion.train import sparse_lora_trainer
    from miowtion.veda import bundle

    config.validate()
    if not torch.cuda.is_available() or not fa4.available(torch.device('cuda')):
        raise RuntimeError('real CUDA sparse backend required')
    env = parallel.init_distributed()
    if env.world_size != 1:
        raise ValueError('this zero-update diagnostic currently uses one GPU')
    resolved = data.parse_geometry(geometry)
    cache = data.SampleCache(sample_cache)
    sample = next(s for s in cache.samples if s.id == sample_id)
    if (sample.task != 't2va' or sample.keyframes or sample.references
            or sample.latent_t not in (None, resolved.latent_t)
            or sample.aspect not in (None, resolved.aspect)):
        raise ValueError('diagnostic sample/geometry is not compatible T2VA')
    assets = sparse_lora_trainer.asset_contract(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'mode': 'native_zero_update_validation', 'optimizer_updates': 0,
        'sample_cache_index_sha256': core.sha256_file(
            Path(sample_cache, 'index.json')),
        'sample_id': sample_id, 'historical_cache_split': sample.split,
        'original_prompt_available': False,
        'used_for_new_training': False, 'seed': seed, 'assets': assets,
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    def inject(model):
        lora.add_lora(model, config.lora_rank, config.lora_alpha,
                      config.lora_targets, layers=config.lora_layers)

    progress.log('native validation: load backbone/Turbo; no optimizer')
    built = teacher.build_teacher(
        config.checkpoint_root, 'FL2VA', 'turbo', 8, config.teacher_adapter,
        env, visual_conditions=False, audio_references=False,
        offload_blocks=0, mlp_chunk_rows=config.mlp_chunk_rows,
        before_shard=inject)
    model = built.model
    model.dense_backend = config.dense_backend
    lora.reset_parameters(model, config.seed)
    published = bundle.load(config.predictor_bundle, env.device)
    selected = published.plans.select(resolved)
    if (selected.geometry != resolved.name
            or selected.grid != resolved.video_grid):
        raise ValueError('native check requires an exact published plan')
    core.freeze_except_lora(model, published.predictor)
    traj = trajectory_lib.Trajectory(
        model, cache, sample, resolved, built.schedule, seed, env.device)
    table = built.tables.get(traj.inputs().timestep.timesteps)
    return check_state(model, published.predictor, traj, selected, table,
                       output_dir, config.keep_ratio,
                       ordered_dq=config.ordered_dq)
