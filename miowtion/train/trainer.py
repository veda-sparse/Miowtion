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

import collections.abc
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
from miowtion.train import muon
from miowtion.train import monitor as monitor_lib
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
    # None only with random_weights_seed (random prompt rows instead).
    sample_cache: str | None
    geometries: list[str]
    stage: int = 1
    variant: str = 'FL2VA'
    tasks: list[str] = dataclasses.field(default_factory=lambda: ['t2va'])
    plan_dir: str | None = None
    bootstrap_shape: str = '4x8x4'
    schedule: str = 'base'
    num_steps: int = 49
    teacher_adapter: str | None = None
    geometry_sampling: str = 'uniform'  # or 'cycle', see GeometrySampler
    # Denoising steps per trajectory before the next one starts; None rolls
    # the whole schedule. Only for smoke tests that must visit many
    # geometries quickly (every trajectory then stays near the noise end).
    trajectory_steps: int | None = None
    keep_ratio: float = 0.1
    ref_keep_ratio: float | None = None
    # Absolute per-query tile budgets take precedence over the ratio fields.
    keep_tiles: int | None = None
    ref_keep_tiles: int | None = None
    tile_conditions: bool = False
    teacher_q_tiles: float = 1.0
    recall_every: int = 1  # mask diagnostics on every n-th layer
    # VedaConfig.collect_bytes in MiB: the head-chunk bound of the teacher
    # heat. The default fits a 24 GB card; raise it where memory allows.
    veda_collect_mib: int = veda_attention.DEFAULT_COLLECT_BYTES // 2**20
    # Veda2's two score terms (docs/features/veda2.md). They start at the
    # closed-form coefficients of the block log-mass expansion, which the
    # plain predictor structurally cannot express, and train from there.
    # second_order_rank 0 and count_term false is Veda1 exactly.
    second_order_rank: int = 0
    count_term: bool = False
    # Moments from 'ablate_sol.py --second-moments', required when
    # second_order_rank is below head_dim: only full rank has a closed-form
    # warm start. None at full rank uses that closed form.
    second_order_moments: str | None = None
    # Hold the second-cumulant head at its warm start and train only the
    # base projections. The head is a bilinear in so_q and so_k, so its
    # contribution is quadratic in the parameters and a step size tuned
    # for the base projections runs it away: measured, logit spread grew
    # 6x in 48 updates. Freezing it isolates the question that has not
    # been answered yet -- whether training the base against the right
    # target and features moves the predictor towards the oracle.
    freeze_second_order: bool = False
    # Hold the per-head log B gain at its closed-form 1.0. Veda2 added it
    # and it is the predictor's only non-matrix parameter, so `optimizer:
    # muon` cannot train it: Muon orthogonalizes matrices and refuses 1-D
    # tensors. Freezing it is cheap rather than a compromise -- 56 scalars
    # a layer against 2 * 56 * 384 * 128 matrix entries, so about 1e-5 of
    # the capacity -- and its initialisation is already the exact
    # coefficient. Its one job is to shrink towards 0 on geometries whose
    # tile row counts barely vary, and every geometry in these runs has
    # real spread (16:9@37: std(log B) 0.664). Never set implicitly: a
    # config that wants Muon has to say this out loud.
    freeze_count_gain: bool = False
    # Stop a run that is not improving instead of paying for the rest.
    # On by default, and deliberately so: a Veda2 run degraded for 59
    # updates before anyone looked, which was an hour of GPU spent on a
    # result that was visible by update 20. Set abort_patience to 0 to
    # disable, which is a choice a config now has to make out loud.
    # See train.monitor.EarlyAbort: the comparison is between trailing
    # means, because the watched metric is not comparable across the
    # cycled geometries, so single updates swing more than progress does.
    # Patience 20 with window 8 tolerates a long plateau; it is there to
    # catch a decline, not to prune a slow run.
    abort_window: int = 8
    abort_patience: int = 20
    abort_metric: str = 'kept_over_ceiling'
    # Falls below the starting trailing mean by this much and the run
    # stops, whatever its age. This is the criterion that catches what
    # actually went wrong here, and unlike patience it is safe on a long
    # run. Give at least one of the two.
    abort_max_drop: float = 0.05
    # 'max' distils against the block's peak probability (Veda1), 'sum'
    # against its attention mass. Mass is what determines the output
    # error, and it is also what the predictor's own initialization
    # estimates, so the two Veda2 changes belong together.
    heat_reduce: str = 'max'
    # 'forward' is KL(teacher || student), mass-covering, which is what
    # Veda1 trained with; 'reverse' is KL(student || teacher),
    # mode-seeking, which is what a top-k read-out actually wants. See
    # heatmap.seer_kl for the measured failure of the forward direction.
    kl_direction: str = 'forward'
    # Weight of the expected-retained-mass loss derived from reading
    # attention as one-sided entropic optimal transport (see
    # heatmap.transport_loss). It is the only objective here that is
    # invariant to the per-row affine maps a top-k read-out is invariant
    # to; the other three each lost to a gauge direction instead.
    transport_weight: float = 0.0
    # Weight of the oracle top-k BCE added to the seer KL (0 = KL only).
    # The KL fits the whole teacher distribution; only the top-k ordering
    # reaches the kernel. See heatmap.oracle_bce().
    topk_weight: float = 0.0
    accum: int = 4
    steps: int = 400
    lr: float = 1e-4
    warmup: int = 25
    lr_decay: str = 'none'  # or 'cosine', see learning_rate()
    lr_min_ratio: float = 0.1  # floor of the cosine, as a fraction of lr
    # 'adamw' or 'muon'. Muon orthogonalizes each head's update matrix, so
    # the step size is set by lr * muon_rms rather than by the gradient's
    # magnitude (miowtion.train.muon). The predictor is a stack of per-head
    # matrices and nothing else, which is what Muon is for; stage 2's heads
    # and LoRA stay on AdamW either way.
    optimizer: str = 'adamw'
    muon_momentum: float = 0.95
    muon_rms: float = 0.2      # RMS every orthogonalized update is scaled to
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema: float = 0.995
    seed: int = 0
    out_dir: str = 'runs'
    persistent_dir: str | None = None
    save_every: int = 50
    keep_optimizer: int = 2  # newest checkpoints keeping Adam moments
    init_from: str | None = None
    init_drop_prefixes: list[str] = dataclasses.field(default_factory=list)
    lora_rank: int = 64
    # Weight of the seer KL. It used to apply in stage 2 only, but the KL
    # and the thing the kernel reads can pull apart in stage 1 too: with a
    # mass target, 25 updates took the KL from 134 to 48 while recall went
    # 0.714 -> 0.654, the KL buying its reduction by flattening towards
    # the teacher's bulk. Being able to turn it down against topk_weight
    # is how that gets balanced. 1.0 is the historical behaviour.
    kl_weight: float = 1.0
    mlp_chunk_rows: int | None = None
    offload_blocks: int = 0
    prefetch: int = 1
    seq_bucket: int | None = None
    dense_backend: str = 'auto'
    offload_optimizer: bool = False
    monitor_updates: bool = True
    # Benchmark-only: a random teacher (miowtion.h3.synthetic) and random
    # prompt rows of random_text_len, so a card's training speed can be
    # measured without the weights. The predictor learns nothing useful.
    random_weights_seed: int | None = None
    random_text_len: int = 589

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
        if self.transport_weight < 0:
            raise ValueError('transport_weight must be >= 0: '
                             f'{self.transport_weight}')
        if self.transport_weight and self.heat_reduce != 'sum':
            raise ValueError("transport_weight needs heat_reduce 'sum': "
                             'the loss is about transported mass, and the '
                             'block maximum is not a mass')
        if self.kl_direction not in ('forward', 'reverse'):
            raise ValueError("kl_direction must be 'forward' or 'reverse': "
                             f'{self.kl_direction!r}')
        if self.heat_reduce not in ('max', 'sum'):
            raise ValueError("heat_reduce must be 'max' or 'sum': "
                             f'{self.heat_reduce!r}')
        if self.second_order_rank < 0:
            raise ValueError('second_order_rank must be >= 0: '
                             f'{self.second_order_rank}')
        if self.stage not in (1, 2):
            raise ValueError(f'stage must be 1 or 2, got {self.stage}')
        if self.optimizer not in ('adamw', 'muon'):
            raise ValueError("optimizer must be 'adamw' or 'muon', got "
                             f'{self.optimizer!r}')
        if (self.sample_cache is None) != (self.random_weights_seed
                                           is not None):
            raise ValueError('sample_cache must be null exactly when '
                             'random_weights_seed is set')
        for task in self.tasks:
            if data.VARIANT_OF_TASK[task] != self.variant:
                raise ValueError(f'task {task} needs the '
                                 f'{data.VARIANT_OF_TASK[task]} checkpoint')
        if self.stage == 2 and not self.init_from:
            raise ValueError('stage 2 must start from a stage-1 checkpoint')
        if self.geometry_sampling not in data.GEOMETRY_SAMPLING_MODES:
            raise ValueError(f'geometry_sampling must be one of '
                             f'{data.GEOMETRY_SAMPLING_MODES}')
        if self.trajectory_steps is not None and not (
                1 <= self.trajectory_steps <= self.num_steps):
            raise ValueError(f'trajectory_steps must be in [1, '
                             f'{self.num_steps}], got {self.trajectory_steps}')
        if self.lr_decay not in LR_DECAYS:
            raise ValueError(f'lr_decay must be one of {sorted(LR_DECAYS)}, '
                             f'got {self.lr_decay!r}')
        if not 0.0 <= self.lr_min_ratio <= 1.0:
            raise ValueError(f'lr_min_ratio must be in [0, 1], got '
                             f'{self.lr_min_ratio}')
        for name in ('keep_tiles', 'ref_keep_tiles'):
            count = getattr(self, name)
            if count is not None and (type(count) is not int or count <= 0):
                raise ValueError(f'{name} must be a positive integer')

    def budgets(self) -> tuple[veda_mask.Budget, veda_mask.Budget | None]:
        """Resolve target/reference budgets for this training run."""
        target = (veda_mask.Budget(tiles=self.keep_tiles)
                  if self.keep_tiles is not None else
                  veda_mask.Budget(ratio=self.keep_ratio))
        reference = (veda_mask.Budget(tiles=self.ref_keep_tiles)
                     if self.ref_keep_tiles is not None else
                     veda_mask.Budget(ratio=self.ref_keep_ratio)
                     if self.ref_keep_ratio is not None else None)
        return target, reference


LR_DECAYS = ('none', 'cosine')


def learning_rate(config: TrainConfig, step: int) -> float:
    """Learning rate for a 0-based update index.

    Linear warmup over `warmup` updates, then either a constant rate or a
    cosine falling to `lr_min_ratio * lr` at `steps`. The two are multiplied
    rather than chained, so the cosine starts at the end of the warmup and
    the curve has no jump there.

    Args:
        config: Run configuration.
        step: Update index, 0-based; may exceed `config.steps` (the cosine
            is clamped at its floor).

    Returns:
        The rate to write into every parameter group.
    """
    lr = config.lr * min(1.0, (step + 1) / max(1, config.warmup))
    if config.lr_decay == 'none':
        return lr
    done = (step + 1 - config.warmup) / max(1, config.steps - config.warmup)
    done = min(1.0, max(0.0, done))
    cosine = 0.5 * (1.0 + math.cos(math.pi * done))
    return lr * (config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine)


def _require_matrix_params(
        named: collections.abc.Sequence[
            tuple[str, torch.nn.Parameter]]) -> None:
    """Rejects 1-D trainables before Muon gets a chance to.

    Group names stopped being enough when Veda2 added `count_gain`, which
    is [num_heads] and sits inside the 'predictor' group: the group-level
    check passes, and Muon's own shape check then fires several minutes
    in, after the 33B of weights have loaded, saying only
    `got shape (56,)`. This says which parameter and what to do instead.

    Args:
        named: The trainable (name, parameter) pairs the optimizer will own.

    Raises:
        ValueError: If any trainable parameter has fewer than two dims.
    """
    flat = [(n, p) for n, p in named if p.requires_grad and p.ndim < 2]
    if not flat:
        return
    name, param = flat[0]
    raise ValueError(
        f'optimizer muon cannot train {len(flat)} non-matrix parameters, '
        f'e.g. {name} with shape {tuple(param.shape)}: Muon orthogonalizes '
        'matrices. Set freeze_count_gain: true to hold them at their '
        'initialisation, or use optimizer: adamw.')


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
            if config.stage == 2 else None,
            random_weights_seed=config.random_weights_seed)
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
            cfg.num_layers, cfg.num_heads, cfg.head_dim,
            second_order_rank=config.second_order_rank,
            count_term=config.count_term).to(self.device)
        if config.second_order_rank:
            self._warm_start_second_order(config, cfg)
        self.plans = (veda_plan.PlanTable.load_dir(config.plan_dir)
                      if config.plan_dir else None)
        target_budget, ref_budget = config.budgets()
        self.veda_config = veda_attention.VedaConfig(
            target_budget=target_budget,
            ref_budget=ref_budget,
            heat_reduce=config.heat_reduce,
            kl_weight=config.kl_weight,
            transport_weight=config.transport_weight,
            kl_direction=config.kl_direction,
            tile_conditions=config.tile_conditions,
            teacher_q_tiles=config.teacher_q_tiles,
            recall_every=config.recall_every,
            collect_bytes=config.veda_collect_mib * 2**20)

        self.trainable = [(f'predictor.{n}', p)
                          for n, p in self.predictor.named_parameters()]
        if config.freeze_second_order:
            held = [n for n, p in self.predictor.named_parameters()
                    if n.endswith(('so_q', 'so_k'))]
            for name, param in self.predictor.named_parameters():
                if name.endswith(('so_q', 'so_k')):
                    param.requires_grad_(False)
            self.trainable = [(n, p) for n, p in self.trainable
                              if not n.endswith(('so_q', 'so_k'))]
            progress.log(f'second-order head frozen: {len(held)} tensors '
                         'held at the warm start')
        if config.freeze_count_gain:
            held = [n for n, p in self.predictor.named_parameters()
                    if n.endswith('count_gain')]
            for name, param in self.predictor.named_parameters():
                if name.endswith('count_gain'):
                    param.requires_grad_(False)
            self.trainable = [(n, p) for n, p in self.trainable
                              if not n.endswith('count_gain')]
            progress.log(f'count gain frozen: {len(held)} tensors held at '
                         'the closed-form 1.0')
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
        self.clip_groups = _clip_groups(self.trainable)
        groups = _param_groups(self.opt_params)
        if config.optimizer == 'muon':
            # Every stage-1 parameter is a [heads, 3D, D] stack of per-head
            # matrices, which is exactly what Muon orthogonalizes. Anything
            # else a stage adds (LoRA, the output heads) keeps AdamW: they
            # are not per-head matrices and Muon refuses 1-D parameters.
            non_matrix = {g: ps for g, ps in groups.items() if g != 'predictor'}
            if non_matrix:
                raise ValueError(
                    f'optimizer muon needs a predictor-only run; this one '
                    f'also trains {sorted(non_matrix)}')
            _require_matrix_params(self.opt_params)
            self.optimizer = muon.Muon(
                groups['predictor'], lr=config.lr,
                momentum=config.muon_momentum, rms_target=config.muon_rms,
                weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                [{'params': ps, 'name': g} for g, ps in groups.items()],
                lr=config.lr, betas=config.betas,
                weight_decay=config.weight_decay)
        self.ema = checkpoint.Ema(self.opt_params, config.ema)
        self.ckpt = checkpoint.CheckpointManager(
            os.path.join(self.run_dir, 'ckpt'), config.persistent_dir,
            self.env, config.keep_optimizer)

        self.geometries = data.GeometrySampler(
            config.geometries, config.seed, config.geometry_sampling)
        if config.random_weights_seed is None:
            self.cache = data.SampleCache(config.sample_cache)
        else:
            self.cache = data.SyntheticSampleCache(
                config.random_text_len, cfg.text_dim,
                config.random_weights_seed)
        self.samples = data.SampleSampler(
            self.cache.select('train', config.tasks), config.seed,
            self.env.rank)
        self._check_geometry_coverage()
        self.noise_gen = torch.Generator().manual_seed(
            config.seed * 977 + self.env.rank)
        self.step = 0
        self._restore()
        self.start_step = self.step
        # Diagnostics are computed where the optimizer steps (host masters
        # when offloaded), so they cost no device memory.
        self.monitor = (monitor_lib.UpdateMonitor(self.opt_params)
                        if config.monitor_updates else None)
        self.trajectory = None

    # --- setup -----------------------------------------------------------

    def _warm_start_second_order(self, config: 'TrainConfig', cfg) -> None:
        """Puts the second-cumulant head on its closed-form coefficients.

        Starting there rather than at noise is the same idea the plain
        predictor already uses: its N(0, 1e-4) projections make an
        untrained score equal mean-pooled QK, the zero-order term of the
        block log-mass. This extends that to the next term, so training
        starts from the best estimate instead of discovering it.

        Raises:
            ValueError: If a rank below head_dim comes without moments.
        """
        if config.second_order_rank == cfg.head_dim and not (
                config.second_order_moments):
            self.predictor.init_exact_second_order_()
            progress.log('second-order head warm-started at the exact '
                         'diagonal cumulant (full rank)')
            return
        if not config.second_order_moments:
            raise ValueError(
                f'second_order_rank {config.second_order_rank} is below '
                f'head_dim {cfg.head_dim}, which has no closed-form warm '
                'start; pass second_order_moments from '
                "'ablate_sol.py --second-moments'")
        moments = torch.load(config.second_order_moments, map_location='cpu')
        self.predictor.init_low_rank_second_order_(moments['c_u'],
                                                   moments['c_v'])
        progress.log(f'second-order head warm-started from '
                     f'{config.second_order_moments} '
                     f"(geometry {moments.get('geometry')})")


    def _check_geometry_coverage(self) -> None:
        """Refuses a geometry no sample can serve, before the first step.

        A prompt is written for a duration, so a sample only fits a
        geometry whose latent_t it declares (data.SampleSampler). Without
        this the run dies whenever the sampler first happens to draw the
        unusable geometry -- with `uniform` sampling that is a random
        number of updates in, after the teacher has been loaded and the
        tables built.

        Raises:
            ValueError: Naming every geometry with no compatible sample.
        """
        samples = self.samples.samples
        orphans = []
        for geometry in self.geometries.geometries:
            if not any(
                    (s.aspect in (None, geometry.aspect))
                    and (s.latent_t in (None, geometry.latent_t))
                    for s in samples):
                orphans.append(geometry.name)
        if orphans:
            have = sorted({s.latent_t for s in samples if s.latent_t})
            raise ValueError(
                f'no sample fits {orphans}: the cache holds prompts written '
                f'for latent_t {have}. Drop those geometries or encode '
                'prompts for them.')

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
            # Parameters this stage added are absent from a published
            # checkpoint by construction, and the warm start above already
            # put them on their closed-form values. Everything else must
            # still be present.
            new_suffixes = []
            if config.second_order_rank:
                new_suffixes += ['so_q', 'so_k']
            if config.count_term:
                new_suffixes.append('count_gain')
            checkpoint.init_weights(predictor_only, payload, use_ema=True,
                                    drop_prefixes=config.init_drop_prefixes,
                                    new_suffixes=new_suffixes)
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
        limit = self.config.trajectory_steps
        if (self.trajectory is None or self.trajectory.done
                or (limit is not None and self.trajectory.step >= limit)):
            self._next_trajectory()
        traj = self.trajectory
        self._micro_start = time.time()
        inputs = traj.inputs()
        table = self.tables.get(inputs.timestep.timesteps)
        num_layers = self.model.config.num_layers
        # Pure normalization. The KL's own weight lives in VedaConfig, so
        # that kl_weight 0 means 'BCE only' rather than 'no loss at all'.
        kl_scale = 1.0 / (num_layers * self.config.accum)
        collector = veda_attention.TeacherCollector(
            self.clip_tiling, self.plan, self.predictor, self.noise_gen,
            grad_scale=kl_scale, dense_backend=self.config.dense_backend,
            topk_weight=self.config.topk_weight)
        teacher_ctx = (_FrozenTeacher(self) if self.config.stage == 2
                       else contextlib.nullcontext())
        with teacher_ctx, torch.no_grad():
            video_t, audio_t = self.model(traj.clip, inputs.video_rows,
                                          inputs.audio_rows, inputs.timestep,
                                          collector, table)
        resolved = collector.stats.resolve()  # one device transfer
        stats['kl'].append(sum(resolved['kl']) / num_layers)
        stats['kl_layers'].append(resolved['kl'])
        for name in ('topk_bce', 'transport', 'logit_std', 'recall',
                     'heat_kept', 'heat_ceiling'):
            # A zero-weighted loss records nothing, so the key is absent.
            stats[name] += resolved.get(name, [])
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
        abort = (monitor_lib.EarlyAbort(config.abort_window,
                                        config.abort_patience,
                                        config.abort_metric,
                                        config.abort_max_drop)
                 if (config.abort_patience or config.abort_max_drop)
                 else None)
        while self.step < config.steps:
            start = time.time()
            stats = {name: [] for name in
                     ('kl', 'kl_layers', 'topk_bce', 'transport',
                      'logit_std', 'recall', 'heat_kept',
                      'heat_ceiling', 'mse')}
            for _ in range(config.accum):
                self._micro_step(stats)
            replicated = [p for n, p in self.trainable
                          if not n.startswith('blocks.')]
            parallel.all_reduce_gradients(replicated, self.env)
            if self.step == self.start_step:
                monitor_lib.check_gradient_stop(
                    self.model, [p for _, p in self.trainable])
            norms = summarize_norms(
                {name: torch.nn.utils.clip_grad_norm_(
                    params, config.grad_clip).item()
                 for name, params in self.clip_groups.items()})
            lr = learning_rate(config, self.step)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            if self.masters is not None:
                self.masters.pull_grads()
            diagnostics = (self.monitor.before_step()
                           if self.monitor is not None else {})
            self.optimizer.step()
            if self.monitor is not None:
                diagnostics.update(self.monitor.after_step())
            self.optimizer.zero_grad(set_to_none=self.masters is None)
            for _, p in self.trainable:
                p.grad = None
            if self.masters is not None:
                self.masters.push_params()
            self.ema.update(self.opt_params)
            self.step += 1
            self._log_step(stats, norms, time.time() - start, diagnostics)
            self._updates.update(self._progress_line(self._last_record))
            if abort is not None:
                reason = abort.update(self._last_record)
                if reason:
                    self._log({'event': 'aborted', 'step': self.step,
                               'reason': reason})
                    progress.log(f'aborting at update {self.step}: {reason}')
                    self._save()
                    return
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

    def _log_step(self, stats: dict, norms: dict, seconds: float,
                  diagnostics: dict) -> None:
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
        if isinstance(self.optimizer, muon.Muon):
            # Measured, not derived: the scale formula assumes a full-rank
            # update, and a rank-deficient gradient lands below the target
            # (miowtion.train.muon). This is the number lr is calibrated
            # against, so it has to come from the step that just ran.
            record['update_rms'] = round(self.optimizer.last_update_rms(), 5)
            record['update_align'] = round(self.optimizer.last_alignment(), 4)
        for name in ('topk_bce', 'transport', 'heat_kept', 'heat_ceiling',
                     'logit_std'):
            if stats[name]:
                record[name] = round(self._reduce_mean(stats[name]), 5)
        # The geometry-normalized one. heat_kept alone is not comparable
        # across geometries, because the ceiling itself moves with the
        # tile count, and the geometries are cycled. This is the number to
        # watch: the KL can fall while it stays flat or drops, which is
        # the loss paying itself down by flattening.
        if record.get('heat_ceiling'):
            record['kept_over_ceiling'] = round(
                record['heat_kept'] / record['heat_ceiling'], 5)
        # Per-layer KL is rank 0's own micro-steps: it says where in depth
        # the predictor is behind, and averaging it across ranks would only
        # hide that they saw different geometries.
        if stats['kl_layers']:
            columns = list(zip(*stats['kl_layers']))
            record['kl_layers'] = [float(f'{sum(c) / len(c):.4g}')
                                   for c in columns]
        record.update(diagnostics)
        self._last_record = record
        self._log(record)

    def _progress_line(self, record: dict) -> str:
        """The short status: the loss, and the thing the kernel reads."""
        parts = [f"kl {record.get('kl', float('nan')):.4f}"]
        for key, fmt in (('transport', '.5f'), ('recall', '.3f'),
                         ('kept_over_ceiling', '.4f')):
            if key in record:
                parts.append(f'{key} {record[key]:{fmt}}')
        return '  '.join(parts)

    def _log(self, record: dict) -> None:
        if not self.env.is_main:
            return
        line = json.dumps(record)
        print(line, flush=True)
        with open(os.path.join(self.run_dir, 'log.jsonl'), 'a') as f:
            f.write(line + '\n')


def _param_groups(named) -> dict[str, list]:
    """Optimizer parameter groups: predictor, lora, head."""
    groups = {}
    for name, p in named:
        key = ('predictor' if name.startswith('predictor.') else
               'lora' if '.lora_' in name else 'head')
        groups.setdefault(key, []).append(p)
    return groups


def _clip_groups(named) -> dict[str, list]:
    """Parameter groups clipped independently, one per predictor layer.

    The layer predictors are independent models: separate parameters,
    one KL term each, no gradient crossing between them. A single clip
    over all of them is therefore a coupling, and an asymmetric one --
    the last layer's gradient runs 50-70x the median layer's and carries
    ~98% of the total norm, so a shared clip would be triggered by that
    one layer and would shrink the step of the other 49 with it. lora
    and head stay whole: those are one model.
    """
    groups = {}
    for name, p in named:
        key = (name.rsplit('.', 1)[0] if name.startswith('predictor.') else
               'lora' if '.lora_' in name else 'head')
        groups.setdefault(key, []).append(p)
    return groups


def summarize_norms(norms: dict[str, float]) -> dict[str, float]:
    """Collapses the per-layer predictor norms back into one number.

    Clipping is per layer, but the log keeps a single `predictor` entry
    (the norm the whole predictor would report, i.e. the root sum of
    squares of the groups). Per-layer detail is already logged as
    `grad_norm_layers`.
    """
    out, layers = {}, []
    for name, value in norms.items():
        if name.startswith('predictor.'):
            layers.append(value)
        else:
            out[name] = value
    if layers:
        out['predictor'] = math.sqrt(sum(v * v for v in layers))
    return out


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
