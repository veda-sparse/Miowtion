"""Veda predictor training (stage 1) and optional LoRA recovery (stage 2).

Stage 1 (predictor only): the trunk is frozen and runs under no_grad; each
micro-step is one dense teacher forward whose attention layers also train
the predictor against the teacher heat (TeacherCollector). No sparse kernel
is involved.

Stage 2 (optional): predictor + trunk LoRA (qkv/out) + output heads. Per
micro-step a frozen-teacher dense forward (LoRA off, output heads restored
from the snapshot taken at start) collects the KL and the target velocity;
then the sparse student (LoRA on, gradient checkpointing, FA4 block-sparse
kernel required) is trained with MSE(v_s, v_t) + MSE(a_s, a_t). The student
never collects KL: under checkpoint recomputation every append would run
twice and hang on a freed graph. Stage 2 must start from a converged stage-1
predictor (its EMA weights).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from miowtion.kernels import fa4
from miowtion.train import checkpoint
from miowtion.train import data
from miowtion.train import lora
from miowtion.train import optim
from miowtion.train import parallel
from miowtion.train import teacher
from miowtion.train import trajectory as traj_lib
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

MAX_EXTRA_PARAMS = 1_000_000_000


@dataclasses.dataclass
class TrainConfig:
    """Run configuration (YAML keys have the same names)."""

    run_name: str
    checkpoint_root: str
    sample_cache: str
    geometries: list[str]
    stage: int = 1
    variant: str = 'FL2VA'
    tasks: list[str] = dataclasses.field(default_factory=lambda: ['t2va'])
    plan_dir: str | None = None
    bootstrap_shape: str = '4x8x4'
    schedule: str = 'base'
    num_steps: int = 49
    teacher_adapter: str | None = None
    keep_ratio: float = 0.1
    ref_keep_ratio: float | None = None
    tile_conditions: bool = False
    teacher_q_tiles: float = 1.0
    recall_every: int = 4
    accum: int = 4
    steps: int = 400
    lr: float = 1e-4
    warmup: int = 25
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema: float = 0.995
    seed: int = 0
    out_dir: str = 'runs'
    persistent_dir: str | None = None
    save_every: int = 50
    init_from: str | None = None
    init_drop_prefixes: list[str] = dataclasses.field(default_factory=list)
    lora_rank: int = 64
    kl_weight: float = 1.0
    mlp_chunk_rows: int | None = None
    offload_blocks: int = 0
    prefetch: int = 1
    seq_bucket: int | None = None
    dense_backend: str = 'auto'
    offload_optimizer: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f'unknown config keys {sorted(unknown)}')
        if 'betas' in raw:
            raw['betas'] = tuple(raw['betas'])
        return cls(**raw)

    def validate(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError(f'stage must be 1 or 2, got {self.stage}')
        for task in self.tasks:
            if data.VARIANT_OF_TASK[task] != self.variant:
                raise ValueError(f'task {task} needs the '
                                 f'{data.VARIANT_OF_TASK[task]} checkpoint')
        if self.stage == 2 and not self.init_from:
            raise ValueError('stage 2 must start from a stage-1 checkpoint')


class Trainer:
    """Owns the model, predictor, optimizer and the trajectory stream."""

    def __init__(self, config: TrainConfig):
        config.validate()
        self.config = config
        self.env = parallel.init_distributed()
        self.device = self.env.device
        self.run_dir = os.path.join(config.out_dir, config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        progress.log(f'run {config.run_name}: stage {config.stage}, variant '
                     f'{config.variant}, tasks {config.tasks}, geometries '
                     f'{config.geometries}, {config.steps} updates x '
                     f'{config.accum} micro-steps, output {self.run_dir}')
        teacher_ = teacher.build_teacher(
            config.checkpoint_root, config.variant, config.schedule,
            config.num_steps, config.teacher_adapter, self.env,
            visual_conditions=config.variant == 'Ref2VA'
            or 'fl2va' in config.tasks,
            audio_references=config.variant == 'Ref2VA',
            offload_blocks=config.offload_blocks, prefetch=config.prefetch,
            mlp_chunk_rows=config.mlp_chunk_rows,
            before_shard=(lambda m: lora.add_lora(m, config.lora_rank))
            if config.stage == 2 else None)
        self.model = teacher_.model
        self.model.dense_backend = config.dense_backend
        if config.stage == 2:
            lora.reset_parameters(self.model, seed=config.seed)
        self.schedule = teacher_.schedule
        self.tables = teacher_.tables

        cfg = self.model.config
        # The predictor is replicated: every rank must start from the same
        # weights, so its initialization is seeded identically everywhere.
        torch.manual_seed(config.seed)
        self.predictor = veda_predictor.TileScorePredictor(
            cfg.num_layers, cfg.num_heads, cfg.head_dim).to(self.device)
        self.plans = (veda_plan.PlanTable.load_dir(config.plan_dir)
                      if config.plan_dir else None)
        self.veda_config = veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=config.keep_ratio),
            ref_budget=(veda_mask.Budget(ratio=config.ref_keep_ratio)
                        if config.ref_keep_ratio else None),
            tile_conditions=config.tile_conditions,
            teacher_q_tiles=config.teacher_q_tiles,
            recall_every=config.recall_every)

        self.trainable = [(f'predictor.{n}', p)
                          for n, p in self.predictor.named_parameters()]
        if config.stage == 2:
            self.trainable += lora.lora_parameters(self.model)
            self.trainable += [
                (n, p) for n, p in self.model.named_parameters()
                if n.startswith(('final_layer.video_out.',
                                 'final_layer.audio_out.'))]
            for _, p in self.trainable:
                p.requires_grad_(True)
            self.head_snapshot = {
                n: p.detach().clone() for n, p in self.model.named_parameters()
                if n.startswith(('final_layer.video_out.',
                                 'final_layer.audio_out.'))}
            self.model.gradient_checkpointing = True
            if not fa4.available(self.device):
                raise RuntimeError('stage 2 needs the FA4 block-sparse kernel '
                                   '(SM90/SM100); refusing to run the '
                                   'reference kernel')
        self._check_param_budget()
        # The optimizer (and EMA, checkpoints) work on `opt_params`: the
        # trainable parameters themselves, or their pinned host masters.
        self.masters = (optim.HostMasters(self.trainable)
                        if config.offload_optimizer else None)
        self.opt_params = (self.masters.named() if self.masters
                           else self.trainable)
        self.clip_groups = _param_groups(self.trainable)
        self.optimizer = torch.optim.AdamW(
            [{'params': ps, 'name': g}
             for g, ps in _param_groups(self.opt_params).items()],
            lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay)
        self.ema = checkpoint.Ema(self.opt_params, config.ema)
        self.ckpt = checkpoint.CheckpointManager(
            os.path.join(self.run_dir, 'ckpt'), config.persistent_dir,
            self.env)

        self.geometries = data.GeometrySampler(config.geometries, config.seed)
        self.cache = data.SampleCache(config.sample_cache)
        self.samples = data.SampleSampler(
            self.cache.select('train', config.tasks), config.seed,
            self.env.rank)
        self.noise_gen = torch.Generator().manual_seed(
            config.seed * 977 + self.env.rank)
        self.step = 0
        self._restore()
        self.trajectory = None

    # --- setup -----------------------------------------------------------

    def _check_param_budget(self) -> None:
        # numel() of a sharded DTensor is its global size.
        extra = sum(p.numel() for _, p in self.trainable)
        if extra > MAX_EXTRA_PARAMS:
            raise ValueError(f'{extra} trainable parameters exceed the 1B '
                             'budget')
        self._log({'event': 'trainable_params', 'count': extra})

    def _restore(self) -> None:
        config = self.config
        ckpt_dirs = [os.path.join(self.run_dir, 'ckpt'), config.persistent_dir]
        latest = checkpoint.latest(ckpt_dirs)
        if latest is not None:
            payload = checkpoint.load(latest)
            checkpoint.resume(self.opt_params, self.ema, self.optimizer,
                              payload)
            self.step = payload['step']
            self.geometries.load_state(payload['shared_generator'])
            rank_states = payload['rank_states']
            state = rank_states[self.env.rank % len(rank_states)]
            self.samples.load_state(state['sampler'])
            # Generators of ranks beyond the old world start fresh.
            if self.env.rank < len(rank_states):
                self.noise_gen.set_state(state['noise'])
            self._log({'event': 'resumed', 'from': latest})
        elif config.init_from:
            payload = checkpoint.load(config.init_from)
            predictor_only = [(n, p) for n, p in self.opt_params
                              if n.startswith('predictor.')]
            checkpoint.init_weights(predictor_only, payload, use_ema=True,
                                    drop_prefixes=config.init_drop_prefixes)
            self.ema = checkpoint.Ema(self.opt_params, config.ema)
            self._log({'event': 'initialized', 'from': config.init_from})
        if self.masters is not None:
            self.masters.push_params()

    # --- trajectory stream -------------------------------------------------

    def _next_trajectory(self):
        geometry = self.geometries.next()
        sample = self.samples.next(aspect=geometry.aspect,
                                   latent_t=geometry.latent_t)
        progress.log(f'new trajectory: sample {sample.id} ({sample.task}), '
                     f'geometry {geometry.name}, {self.schedule.num_steps} '
                     'steps')
        seed = int(torch.randint(2**31 - 1, (1,), generator=self.noise_gen))
        self.trajectory = traj_lib.Trajectory(
            self.model, self.cache, sample, geometry, self.schedule, seed,
            self.device, self.config.seq_bucket)
        if self.plans is not None:
            self.plan = self.plans.select(geometry)
        else:
            cfg = self.model.config
            self.plan = veda_plan.PlanTable([veda_plan.TilePlan.uniform(
                geometry, tiling.TileShape.parse(self.config.bootstrap_shape),
                cfg.num_layers, cfg.num_heads)]).select(geometry)
        self.clip_tiling = veda_attention.ClipTiling(
            self.trajectory.layout, self.veda_config, self.device)

    # --- one update ------------------------------------------------------

    def _micro_step(self, stats: dict) -> None:
        if self.trajectory is None or self.trajectory.done:
            self._next_trajectory()
        traj = self.trajectory
        self._micro_start = time.time()
        inputs = traj.inputs()
        table = self.tables.get(inputs.timestep.timesteps)
        num_layers = self.model.config.num_layers
        kl_scale = 1.0 / (num_layers * self.config.accum)
        if self.config.stage == 2:
            kl_scale *= self.config.kl_weight
        collector = veda_attention.TeacherCollector(
            self.clip_tiling, self.plan, self.predictor, self.noise_gen,
            grad_scale=kl_scale, dense_backend=self.config.dense_backend)
        teacher_ctx = (_FrozenTeacher(self) if self.config.stage == 2
                       else contextlib.nullcontext())
        with teacher_ctx, torch.no_grad():
            video_t, audio_t = self.model(traj.clip, inputs.video_rows,
                                          inputs.audio_rows, inputs.timestep,
                                          collector, table)
        stats['kl'].append(sum(collector.stats.kl) / num_layers)
        stats['recall'] += collector.stats.recall
        progress.log(f'  update {self.step + 1}/{self.config.steps} micro '
                     f'{len(stats["kl"])}/{self.config.accum}: traj step '
                     f'{traj.step + 1}/{self.schedule.num_steps}, '
                     f'{traj.layout.used} tokens, kl {stats["kl"][-1]:.4f}, '
                     f'{time.time() - self._micro_start:.1f}s')
        if self.config.stage == 2:
            student = veda_attention.SparseStudent(
                self.clip_tiling, self.plan, self.predictor)
            video_s, audio_s = self.model(traj.clip, inputs.video_rows,
                                          inputs.audio_rows, inputs.timestep,
                                          student, table)
            loss = (F.mse_loss(video_s, video_t) + F.mse_loss(audio_s, audio_t))
            (loss / self.config.accum).backward()
            stats['mse'].append(loss.item())
        traj.advance(video_t, audio_t)

    def train(self) -> None:
        config = self.config
        self._updates = progress.Progress('updates', config.steps - self.step)
        while self.step < config.steps:
            start = time.time()
            stats = {'kl': [], 'recall': [], 'mse': []}
            for _ in range(config.accum):
                self._micro_step(stats)
            replicated = [p for n, p in self.trainable
                          if not n.startswith('blocks.')]
            parallel.all_reduce_gradients(replicated, self.env)
            norms = {name: torch.nn.utils.clip_grad_norm_(
                params, config.grad_clip).item()
                     for name, params in self.clip_groups.items()}
            for group in self.optimizer.param_groups:
                group['lr'] = config.lr * min(1.0, (self.step + 1)
                                              / max(1, config.warmup))
            if self.masters is not None:
                self.masters.pull_grads()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=self.masters is None)
            for _, p in self.trainable:
                p.grad = None
            if self.masters is not None:
                self.masters.push_params()
            self.ema.update(self.opt_params)
            self.step += 1
            self._log_step(stats, norms, time.time() - start)
            self._updates.update(f'kl {self._last_record["kl"]:.4f} recall '
                                 f'{self._last_record["recall"]:.3f}')
            if self.step % config.save_every == 0 or self.step == config.steps:
                self._save()
        self.ckpt.wait()

    # --- bookkeeping -------------------------------------------------------

    def _save(self) -> None:
        progress.log(f'saving checkpoint at update {self.step}')
        rank_state = {'sampler': self.samples.state(),
                      'noise': self.noise_gen.get_state()}
        self.ckpt.save(self.step, self.opt_params, self.ema, self.optimizer,
                       rank_state, self.geometries.state(),
                       dataclasses.asdict(self.config))

    def _reduce_mean(self, values: list[float]) -> float:
        finite = [v for v in values if not math.isnan(v)]
        t = torch.tensor([sum(finite), len(finite)], dtype=torch.float64,
                         device=self.device)
        if self.env.distributed:
            dist.all_reduce(t)
        return (t[0] / t[1]).item() if t[1] > 0 else float('nan')

    def _log_step(self, stats: dict, norms: dict, seconds: float) -> None:
        record = {'step': self.step,
                  'kl': self._reduce_mean(stats['kl']),
                  'recall': self._reduce_mean(stats['recall']),
                  'lr': self.optimizer.param_groups[0]['lr'],
                  'grad_norm': norms, 'seconds': round(seconds, 2),
                  'geometry': self.trajectory.geometry.name,
                  'max_mem_gib': round(torch.cuda.max_memory_allocated()
                                       / 2**30, 2)
                  if torch.cuda.is_available() else 0.0}
        if stats['mse']:
            record['mse'] = self._reduce_mean(stats['mse'])
        self._last_record = record
        self._log(record)

    def _log(self, record: dict) -> None:
        if not self.env.is_main:
            return
        line = json.dumps(record)
        print(line, flush=True)
        with open(os.path.join(self.run_dir, 'log.jsonl'), 'a') as f:
            f.write(line + '\n')


def _param_groups(named) -> dict[str, list]:
    """Parameter groups clipped separately: predictor, lora, head."""
    groups = {}
    for name, p in named:
        key = ('predictor' if name.startswith('predictor.') else
               'lora' if '.lora_' in name else 'head')
        groups.setdefault(key, []).append(p)
    return groups


class _FrozenTeacher:
    """LoRA off and output heads swapped to their start-of-run snapshot."""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.saved = {}

    @torch.no_grad()
    def __enter__(self):
        params = dict(self.trainer.model.named_parameters())
        for name, snap in self.trainer.head_snapshot.items():
            self.saved[name] = params[name].detach().clone()
            params[name].copy_(snap)
        lora.set_enabled(self.trainer.model, False)
        return self

    @torch.no_grad()
    def __exit__(self, *args):
        params = dict(self.trainer.model.named_parameters())
        for name, value in self.saved.items():
            params[name].copy_(value)
        lora.set_enabled(self.trainer.model, True)
        return False
