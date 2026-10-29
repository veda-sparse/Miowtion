"""T2VA velocity distillation with frozen published Veda and Turbo.

This is separate from Stage 2: no KL backward, predictor updates, head
snapshots, or Veda2 substitution. CUDA entry refuses reference attention.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.h3 import schedule as h3_schedule
from miowtion.train import checkpoint
from miowtion.train import data
from miowtion.train import lora
from miowtion.train import parallel
from miowtion.train import sparse_lora as core
from miowtion.train import sparse_lora_data as split_data
from miowtion.train import sparse_lora_sampling as sampling
from miowtion.train import teacher
from miowtion.train import trajectory
from miowtion.utils import progress
from miowtion.veda import attention
from miowtion.veda import bundle
from miowtion.veda import mask
from miowtion.veda import plan


def asset_contract(config: core.SparseLoraConfig,
                   hash_backbone: bool = False) -> dict:
    """Fingerprints weights and actual plans; full shard hashes at training.

    CPU preparation checks shard existence/size and headers, but labels
    that weaker backbone identity explicitly. Training hashes all shards.
    """
    root = Path(config.checkpoint_root, config.variant)
    transformer = root / 'transformer'
    index = transformer / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map']
    files = sorted(set(weight_map.values()))
    shards = []
    for filename in files:
        path = transformer / filename
        record = {'file': filename, 'bytes': path.stat().st_size}
        if hash_backbone:
            progress.log(f'hash frozen backbone shard {filename}')
            record['sha256'] = core.sha256_file(path)
        shards.append(record)
    metadata = bundle.read_metadata(config.predictor_bundle)
    table = plan.PlanTable([plan.TilePlan.from_json(p)
                           for p in json.loads(metadata['plans']).values()])
    geometries = data.GeometrySampler(config.geometries, config.seed)
    cfg = h3_config.H3Config.from_pretrained(str(transformer))
    declared = tuple(int(metadata[k]) for k in
                     ('num_layers', 'num_heads', 'head_dim'))
    if declared != (cfg.num_layers, cfg.num_heads, cfg.head_dim):
        raise ValueError('published predictor/backbone architecture mismatch')
    selected = {}
    for geometry in geometries.geometries:
        chosen = table.select(geometry)
        if (chosen.geometry != geometry.name
                or chosen.grid != geometry.video_grid):
            raise ValueError('experiment requires exact published plans')
        if len(chosen.head_shape) != cfg.num_layers or any(
                len(row) != cfg.num_heads for row in chosen.head_shape):
            raise ValueError('tile plan layer/head count mismatch')
        selected[geometry.name] = core.canonical_hash(chosen.to_json())
    scales = h3_schedule.ShiftScales.from_checkpoint(str(root))
    sched = teacher.make_schedule(config.schedule, config.num_steps, scales)
    return {
        'transformer_config_sha256': core.sha256_file(
            transformer / 'config.json'),
        'transformer_index_sha256': core.sha256_file(index),
        'backbone_identity': ('full-sha256' if hash_backbone
                              else 'index-and-sizes'),
        'backbone_shards': shards,
        'turbo_sha256': core.sha256_file(config.teacher_adapter),
        'predictor_sha256': core.sha256_file(config.predictor_bundle),
        'predictor_storage': metadata.get('dtype', 'float32'),
        'predictor_source_weights': metadata.get('source_weights'),
        'bundle_training_keep_ratio': float(metadata['keep_ratio']),
        'selected_plans_sha256': selected,
        'video_sigmas': list(sched.video), 'audio_sigmas': list(sched.audio),
    }


def build_contract(config: core.SparseLoraConfig, assets: dict,
                    modules: list[str]) -> dict:
    """Immutable mechanism and data identity used by exports and resume."""
    settings = dataclasses.asdict(config)
    # Disabled options preserve the identity of earlier unoptimized runs.
    if config.lora_chunk_rows is None:
        settings.pop('lora_chunk_rows')
    if not config.checkpoint_inputs_to_cpu:
        settings.pop('checkpoint_inputs_to_cpu')
    if config.checkpoint_group_size == 1:
        settings.pop('checkpoint_group_size')
    if config.duration_balance_window is None:
        settings.pop('duration_balance_window')
        settings.pop('duration_balance_prefix')
    for name in ('run_name', 'out_dir', 'steps', 'save_every',
                 'checkpoint_root',
                 'teacher_adapter', 'predictor_bundle', 'sample_cache',
                 'train_manifest', 'validation_manifest'):
        settings.pop(name)
    return {
        'format': core.FORMAT, 'settings': settings,
        'sampling': (sampling.BALANCED_SAMPLING_VERSION
                     if config.duration_balance_window is not None
                     else sampling.SAMPLING_VERSION),
        'modules': modules, 'assets': assets,
        'train_manifest_sha256': core.sha256_file(config.train_manifest),
        'validation_manifest_sha256': core.sha256_file(
            config.validation_manifest),
        'teacher': 'dense-without-new-lora', 'student_dense_layers': [],
        'train_predictor': False, 'train_output_heads': False,
        'train_backbone': False, 'train_turbo': False,
        'qkv_layout': 'miowtion-grouped-q-k-v',
    }


def check_manifests(config: core.SparseLoraConfig) -> tuple[list, list]:
    """Validates independent content, explicit splits and geometry coverage."""
    train = split_data.read_manifest(config.train_manifest)
    validation = split_data.read_manifest(config.validation_manifest)
    split_data.assert_disjoint(train, validation)
    if any(r['split'] != 'train' for r in train) or any(
            r['split'] != 'validation' for r in validation):
        raise ValueError('manifest split labels are inconsistent')
    configured = {g.name for g in data.GeometrySampler(
        config.geometries, config.seed).geometries}
    requested = {f'{r["aspect"].replace(":", "x")}_t{r["latent_t"]}'
                 for r in train}
    if requested != configured:
        raise ValueError('configured geometries must match training manifest')
    return train, validation


def validate_cache(config: core.SparseLoraConfig, cache: data.SampleCache,
                    train: list[dict], validation: list[dict]) -> list:
    """Links exact prompts and assignments to the encoded states used."""
    encoding_path = Path(config.sample_cache, 'sparse_lora_encoding.json')
    encoding = json.loads(encoding_path.read_text())
    if (encoding['variant'] != 'FL2VA' or encoding['text_layers'] != 50
            or encoding['chat_template'] or encoding['special_tokens']):
        raise ValueError('incompatible text encoder contract')
    samples = {s.id: s for s in cache.samples}
    if len(samples) != len(cache.samples):
        raise ValueError('duplicate ids in sample cache')
    for record in train + validation:
        sample = samples[record['id']]
        if (encoding['prompts'].get(record['id']) !=
                core.canonical_hash(record['prompt'])):
            raise ValueError(f'encoded prompt mismatch: {record["id"]}')
        if ((sample.task, sample.split, sample.aspect, sample.latent_t) !=
                ('t2va', record['split'], record['aspect'], record['latent_t'])
                or sample.keyframes or sample.references):
            raise ValueError(f'encoded assignment mismatch: {record["id"]}')
    return [samples[r['id']] for r in train]


def validate_run_destination(run_dir: Path, resume: str | None,
                             total_steps: int) -> None:
    """Refuses overwrites before expensive loading or optimizer updates."""
    if resume is None:
        if run_dir.exists():
            raise FileExistsError('unique run_name required; resume explicitly')
        return
    marker = Path(resume, 'done.json')
    if not marker.is_file():
        raise FileNotFoundError('resume requires completed checkpoint')
    resume_step = json.loads(marker.read_text())['step']
    if type(resume_step) is not int or not 0 <= resume_step <= total_steps:
        raise ValueError('resume step exceeds requested workload or is invalid')
    if any(int(p.name[5:]) > resume_step
           for p in run_dir.glob('step_[0-9]*') if p.name[5:].isdigit()):
        raise FileExistsError('newer checkpoints already exist; '
                              'resume in a new run directory')


def cpu_preflight(config: core.SparseLoraConfig,
                   load_predictor: bool = True) -> dict:
    """Checks real metadata and 50-layer meta structure without DiT loading."""
    config.validate()
    progress.log('CPU preflight: validate content splits')
    train, validation = check_manifests(config)
    samples = [data.Sample(r['id'], 't2va', 'train', (0, 0),
                           aspect=r['aspect'], latent_t=r['latent_t'])
               for r in train]
    coverage = sampling.CoverageSampler(
        samples, config.seed,
        duration_balance_window=config.duration_balance_window,
        duration_balance_prefix=config.duration_balance_prefix)
    progress.log('CPU preflight: fingerprint Turbo/bundle and resolve plans')
    assets = asset_contract(config)
    transformer = Path(config.checkpoint_root, 'FL2VA', 'transformer')
    cfg = h3_config.H3Config.from_pretrained(str(transformer))
    progress.log('CPU preflight: inject adapters into the real meta structure')
    with torch.device('meta'):
        model = h3_model.H3DiT(cfg)
        modules = lora.add_lora(
            model, config.lora_rank, config.lora_alpha,
            config.lora_targets, layers=config.lora_layers,
            chunk_rows=config.lora_chunk_rows)
    model.requires_grad_(False)
    named = lora.lora_parameters(model)
    report = {
        'cpu_preflight': 'passed', 'gpu_backward': 'not_run',
        'module_names': modules,
        'trainable_parameters': sum(p.numel() for _, p in named),
        'parameter_shapes': {n: list(p.shape) for n, p in named},
        'train_inputs': len(train), 'validation_inputs': len(validation),
        'sampling': (sampling.BALANCED_SAMPLING_VERSION
                     if config.duration_balance_window is not None
                     else sampling.SAMPLING_VERSION),
        'single_gpu_planned_updates': sampling.planned_updates(
            coverage.batches_per_epoch, config.training_states_per_visit,
            config.accum,
            config.steps, config.epochs),
        'single_gpu_batches_per_epoch': coverage.batches_per_epoch,
        'assets': assets, 'sample_cache_ready': False,
        'frozen': ['backbone', 'merged Turbo', 'predictor/projector',
                   'video/audio output heads', 'text encoder', 'VAEs'],
    }
    if load_predictor:
        progress.log('CPU preflight: stream/validate all predictor tensors')
        report['published_predictor_validation'] = bundle.validate_contents(
            config.predictor_bundle)
        report['full_resident_predictor_load'] = 'not_required_by_cpu_preflight'
    if Path(config.sample_cache, 'index.json').is_file():
        cache = data.SampleCache(config.sample_cache)
        validate_cache(config, cache, train, validation)
        report['sample_cache_ready'] = True
    return report


class SparseLoraTrainer:
    """Shared-model teacher-forced trajectories; only new LoRA is optimized."""

    def __init__(self, config: core.SparseLoraConfig,
                 resume: str | None = None):
        from miowtion.kernels import fa4
        config.validate()
        # Never load tens of GB onto a CPU-only machine accidentally.
        if not torch.cuda.is_available():
            raise RuntimeError('no CUDA resource; use --cpu-preflight')
        self.env = parallel.init_distributed()
        if (config.ordered_dq and torch.cuda.get_device_capability(
                self.env.device)[0] not in fa4.PATCHED_MAJOR_ARCHS):
            raise ValueError('ordered dQ verified path requires SM8x/SM120')
        if not fa4.available(self.env.device):
            raise RuntimeError('real FA4 sparse backward required; no fallback')
        if config.offload_blocks and not self.env.distributed:
            raise ValueError('single-GPU inference BlockStreamer is not '
                             'backward-safe; use FSDP offload or no offload')
        self.config = config
        self.run_dir = Path(config.out_dir, config.run_name)
        train, validation = check_manifests(config)
        self.cache = data.SampleCache(config.sample_cache)
        train_samples = validate_cache(config, self.cache, train, validation)
        # Validate the complete schedule before hashing/loading frozen weights.
        self.samples = sampling.CoverageSampler(
            train_samples, config.seed, self.env.rank, self.env.world_size,
            duration_balance_window=config.duration_balance_window,
            duration_balance_prefix=config.duration_balance_prefix)
        self.total_steps = sampling.planned_updates(
            self.samples.batches_per_epoch, config.training_states_per_visit,
            config.accum, config.steps, config.epochs)
        validate_run_destination(self.run_dir, resume, self.total_steps)
        assets = (asset_contract(config, hash_backbone=True)
                  if self.env.is_main else None)
        assets = parallel.broadcast_object(assets, self.env)
        self.modules = []

        def inject(model):
            self.modules.extend(lora.add_lora(
                model, config.lora_rank, config.lora_alpha,
                config.lora_targets, layers=config.lora_layers,
                chunk_rows=config.lora_chunk_rows))
            # FSDP needs registered parameters before sharding; keep the
            # new branch inactive until Turbo is merged and values reset.
            lora.set_enabled(model, False)

        built = teacher.build_teacher(
            config.checkpoint_root, 'FL2VA', 'turbo', 8,
            config.teacher_adapter, self.env, visual_conditions=False,
            audio_references=False, offload_blocks=config.offload_blocks,
            prefetch=config.prefetch, mlp_chunk_rows=config.mlp_chunk_rows,
            before_shard=inject)
        self.model = built.model
        self.model.dense_backend = config.dense_backend
        self.model.gradient_checkpointing = True
        self.model.checkpoint_inputs_to_cpu = config.checkpoint_inputs_to_cpu
        self.model.checkpoint_group_size = config.checkpoint_group_size
        self.tables, self.schedule = built.tables, built.schedule
        lora.reset_parameters(self.model, config.seed)
        lora.set_enabled(self.model, True)
        published = bundle.load(config.predictor_bundle, self.env.device)
        self.predictor, self.plans = published.predictor, published.plans
        self.named = core.freeze_except_lora(self.model, self.predictor)
        self.contract = build_contract(config, assets, self.modules)
        self.contract['encoded_text_sha256'] = core.sha256_file(
            Path(config.sample_cache, 'text.safetensors'))
        self.contract['cache_index_sha256'] = core.sha256_file(
            Path(config.sample_cache, 'index.json'))
        self.contract['world_size'] = self.env.world_size
        self.optimizer = torch.optim.AdamW(
            [p for _, p in self.named], lr=config.lr, betas=tuple(config.betas),
            weight_decay=config.weight_decay)
        self.geometries = data.GeometrySampler(config.geometries, config.seed,
                                               mode='cycle')
        self.noise_gen = torch.Generator().manual_seed(
            config.seed * 977 + self.env.rank)
        self.timestep_gen = sampling.timestep_generator(config.seed)
        self.step = 0
        self.traj = None
        self.probe = None
        if resume:
            self._restore(resume)
        if self.env.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / 'contract.json').write_text(
                json.dumps(self.contract, indent=2))
            (self.run_dir / 'config.json').write_text(
                json.dumps(dataclasses.asdict(config), indent=2))
        self._log({'event': 'initialized', 'modules': self.modules,
                   'world_size': self.env.world_size,
                   'nominal_global_slots_per_update':
                       config.accum * self.env.world_size,
                   'warmup_updates': config.warmup,
                   'planned_updates': self.total_steps,
                   'requested_epochs': config.epochs,
                   'timestep_sampling': config.timestep_sampling,
                   'training_states_per_prompt_visit':
                       config.training_states_per_visit,
                   'trainable_parameters': sum(
                       p.numel() for _, p in self.named),
                   'contract_sha256': core.canonical_hash(self.contract)})
        self.frozen_before = None
        if config.verify_invariants:
            progress.log('smoke: digest all frozen weights and buffers')
            self.frozen_before = core.frozen_digest(self.model, self.predictor)

    def _log(self, record: dict) -> None:
        if self.env.is_main:
            with (self.run_dir / 'log.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        progress.log(json.dumps(record))

    def _new_trajectory(self, state: dict | None = None) -> None:
        if state is None:
            sample = self.samples.next()
            self.sample_valid = self.samples.last_valid
            geometry = next(g for g in self.geometries.geometries
                            if (g.aspect, g.latent_t) ==
                            (sample.aspect, sample.latent_t))
            seed = int(torch.randint(2**31 - 1, (1,), generator=self.noise_gen))
        else:
            self.sample_valid = state['sample_valid']
            if self.sample_valid != self.samples.last_valid:
                raise ValueError('active trajectory validity/cursor mismatch')
            geometry = next(g for g in self.geometries.geometries
                            if g.name == state['geometry'])
            sample = next(s for s in self.samples.samples
                          if s.id == state['sample'])
            seed = state['seed']
        self.traj_seed = seed
        self.traj = trajectory.Trajectory(
            self.model, self.cache, sample, geometry, self.schedule, seed,
            self.env.device)
        if state is not None:
            self.traj.step = state['step']
            self.traj.video_rows.copy_(state['video'].to(self.env.device))
            self.traj.audio_rows.copy_(state['audio'].to(self.env.device))
        self.clip = attention.ClipTiling(
            self.traj.layout, attention.VedaConfig(
                target_budget=mask.Budget(ratio=self.config.keep_ratio),
                ordered_dq=self.config.ordered_dq,
                dense_layers=frozenset()), self.env.device)
        self.plan = self.plans.select(geometry)
        progress.log(f'trajectory {sample.id}, seed={seed}, '
                     f'{geometry.name}, starting state={self.traj.step}')

    def _roll_teacher_to(self, selected: int) -> None:
        """Builds the selected noisy state with only frozen dense forwards."""
        if not 0 <= selected < self.schedule.num_steps:
            raise ValueError('selected state is outside the Turbo schedule')
        with core.frozen_teacher(self.model):
            while self.traj.step < selected:
                inputs = self.traj.inputs()
                table = self.tables.get(inputs.timestep.timesteps)
                velocity = self.model(
                    self.traj.clip, inputs.video_rows, inputs.audio_rows,
                    inputs.timestep, adaln_table=table)
                self.traj.advance(*velocity)
                progress.log(f'teacher preparation '
                             f'{self.traj.step}/{selected}')

    def _micro_step(self, loss_scale: float | None = None) -> dict:
        random_state = self.config.timestep_sampling == 'random'
        if random_state:
            self._new_trajectory()
            selected = int(torch.randint(
                self.schedule.num_steps, (1,), generator=self.timestep_gen))
            progress.log(f'sample Turbo state {selected}/'
                         f'{self.schedule.num_steps - 1}')
            self._roll_teacher_to(selected)
        elif self.traj is None or self.traj.done:
            self._new_trajectory()
        inputs = self.traj.inputs()
        table = self.tables.get(inputs.timestep.timesteps)
        args = (self.traj.clip, inputs.video_rows, inputs.audio_rows,
                inputs.timestep)
        with core.frozen_teacher(self.model):
            target = self.model(*args, adaln_table=table)
        if self.config.verify_invariants and self.probe is None:
            probe_args = (args[0], args[1].detach().clone(),
                          args[2].detach().clone(), args[3])
            self.probe = (probe_args, table, tuple(t.clone() for t in target))
        student_attention = attention.SparseStudent(
            self.clip, self.plan, self.predictor)
        student = self.model(*args, attention_fn=student_attention,
                             adaln_table=table)
        if self.config.verify_invariants and self.step == 0:
            with core.frozen_teacher(self.model):
                baseline_attention = attention.SparseStudent(
                    self.clip, self.plan, self.predictor)
                baseline = self.model(*args, attention_fn=baseline_attention,
                                      adaln_table=table)
            if not all(torch.equal(s, b) for s, b in zip(student, baseline)):
                raise RuntimeError('zero-init differs from sparse baseline')
        loss, video, audio = core.velocity_loss(
            student, target, self.config.audio_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite velocity loss')
        # Keep LoRA enabled until checkpoint recomputation finishes here.
        scale = 1 / self.config.accum if loss_scale is None else loss_scale
        scale *= int(self.sample_valid)
        (loss * scale).backward()
        record = {
            'sample': self.traj.sample.id, 'seed': self.traj_seed,
            'sample_valid': self.sample_valid, 'gradient_weight': scale,
            'geometry': self.traj.geometry.name,
            'timestep_sampling': self.config.timestep_sampling,
            'teacher_preparation_steps': self.traj.step if random_state else 0,
            'state': self.traj.step, 'video_sigma': self.schedule.video[
                self.traj.step],
            'audio_sigma': self.schedule.audio[self.traj.step],
            'video_mse': float(video.detach()),
            'audio_mse': float(audio.detach()),
            'loss': float(loss.detach()), 'calls': student_attention.calls,
        }
        calls = student_attention.calls
        if calls['dense'] or not calls['sparse']:
            raise RuntimeError('student unexpectedly used dense attention')
        if random_state:
            # Each selected state completes one prompt visit. No unused
            # tail rollout, no retained graph, no continuing it next micro.
            self.traj = None
        else:
            self.traj.advance(*target)
        return record

    def train(self) -> None:
        """Runs updates, saving raw adapter and restart state together."""
        config = self.config
        while self.step < self.total_steps:
            torch.cuda.synchronize(self.env.device)
            torch.cuda.reset_peak_memory_stats(self.env.device)
            started = time.perf_counter()
            lr = core.learning_rate(config, self.step)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            self.optimizer.zero_grad(set_to_none=True)
            records = []
            active = (self.schedule.num_steps - self.traj.step
                      if self.traj is not None and not self.traj.done else 0)
            batch = sampling.update_plan(
                self.samples, config.training_states_per_visit,
                config.accum, self.step, config.epochs, active)
            for micro in range(batch.micro_steps):
                records.append(self._micro_step(batch.loss_scale))
                progress.log(f'update {self.step + 1}/{self.total_steps}, '
                             f'micro {micro + 1}/{batch.micro_steps}, '
                             f'loss={records[-1]["loss"]:.6g}')
            gradients = core.gradient_report(self.named,
                                               require_a=self.step > 0)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.named], config.grad_clip,
                error_if_nonfinite=True)
            before = {n: p.detach().clone() for n, p in self.named}
            self.optimizer.step()
            updates = core.update_report(self.named, before)
            del before
            self.step += 1
            torch.cuda.synchronize(self.env.device)
            update_seconds = time.perf_counter() - started
            update_peak = torch.cuda.max_memory_allocated(self.env.device)
            if config.verify_invariants:
                args, table, before = self.probe
                with core.frozen_teacher(self.model):
                    after = self.model(*args, adaln_table=table)
                if not all(torch.equal(a, b) for a, b in zip(before, after)):
                    raise RuntimeError('dense teacher drifted after update')
                if self.step in (1, self.total_steps):
                    after_digest = core.frozen_digest(
                        self.model, self.predictor)
                    if after_digest != self.frozen_before:
                        raise RuntimeError('frozen weights or buffers changed')
            record = {'event': 'update', 'step': self.step, 'lr': lr,
                      'seconds': update_seconds, 'micro_steps': records,
                      'global_states_per_update':
                          batch.real_states,
                      'global_slots_per_update': batch.slots,
                      'global_padding_states_per_update':
                          batch.slots - batch.real_states,
                      'micro_steps_in_update': batch.micro_steps,
                      'local_trajectories_in_update': len({
                          (r['sample'], r['seed']) for r in records
                          if r['sample_valid']}),
                      'parameter_updates': updates,
                      'gradient_norm': float(norm.full_tensor() if hasattr(
                          norm, 'full_tensor') else norm),
                      'gradient_norm_by_parameter': gradients,
                      'max_memory_allocated_bytes':
                          update_peak,
                      'frozen_invariants_checked': config.verify_invariants}
            completed = self.samples.issued_real_global
            if self.traj is not None and not self.traj.done:
                completed -= self.samples.last_batch_size
            record['completed_prompt_visits_global'] = completed
            if config.timestep_sampling == 'trajectory':
                record['completed_trajectories_global'] = completed
            record['completed_epochs'] = (
                completed // len(self.samples.samples))
            self._log(record)
            if (self.step % config.save_every == 0
                    or self.step == self.total_steps):
                self._save()

    def _rank_state(self) -> dict:
        active = None
        if self.traj is not None and not self.traj.done:
            active = {'sample': self.traj.sample.id, 'seed': self.traj_seed,
                      'sample_valid': self.sample_valid,
                      'geometry': self.traj.geometry.name,
                      'step': self.traj.step,
                      'video': self.traj.video_rows.cpu(),
                      'audio': self.traj.audio_rows.cpu()}
        return {'sampler': self.samples.state(),
                'noise': self.noise_gen.get_state(),
                'timestep_rng': self.timestep_gen.get_state(),
                'active_trajectory': active,
                'cpu_rng': torch.get_rng_state(),
                'cuda_rng': (torch.cuda.get_rng_state(self.env.device)
                             if self.env.device.type == 'cuda' else None)}

    def _save(self) -> None:
        states = [self._rank_state()]
        if self.env.distributed:
            states = [None] * self.env.world_size
            dist.all_gather_object(states, self._rank_state())
        weights = {n: checkpoint._full(p) for n, p in self.named}
        optimizer = {n: {k: checkpoint._full(v) for k, v in
                          self.optimizer.state[p].items()}
                     for n, p in self.named}
        directory = self.run_dir / f'step_{self.step:07d}'
        if self.env.is_main:
            directory.mkdir(exist_ok=False)
        if self.env.distributed:
            dist.barrier()
        core.export_adapter(str(directory / 'sparse_lora.safetensors'),
                            self.model, self.contract, self.step)
        if self.env.is_main:
            payload = {'step': self.step, 'contract': self.contract,
                       'weights': weights, 'optimizer': optimizer,
                       'rank_states': states,
                       'param_groups': [{k: v for k, v in group.items()
                                         if k != 'params'} for group in
                                        self.optimizer.param_groups]}
            path = directory / 'state.pt'
            torch.save(payload, str(path) + '.tmp')
            Path(str(path) + '.tmp').replace(path)
            marker = {'step': self.step, 'state_sha256': core.sha256_file(path),
                      'adapter_sha256': core.sha256_file(
                          directory / 'sparse_lora.safetensors')}
            (directory / 'done.json').write_text(json.dumps(marker))
        if self.env.distributed:
            dist.barrier()

    def _restore(self, directory: str) -> None:
        marker = json.loads(Path(directory, 'done.json').read_text())
        path = Path(directory, 'state.pt')
        if core.sha256_file(path) != marker['state_sha256']:
            raise ValueError('resume checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.step = core.restore_training_payload(
            self.named, self.optimizer, payload, self.contract)
        state = payload['rank_states'][self.env.rank]
        self.samples.load_state(state['sampler'])
        self.noise_gen.set_state(state['noise'])
        self.timestep_gen.set_state(state['timestep_rng'])
        # Also support replay into an already-resident debug model.
        # A boundary checkpoint must clear any newer in-memory rollout.
        self.traj = None
        if state['active_trajectory'] is not None:
            if self.config.timestep_sampling == 'random':
                raise ValueError('random-state resume cannot retain a rollout')
            self._new_trajectory(state['active_trajectory'])
        torch.set_rng_state(state['cpu_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], self.env.device)
