"""Frozen-teacher Sparse LoRA contracts, loss, and independent artifacts.

Adapter parameters are registered before FSDP sharding, then initialized
after the existing Turbo is merged into base weights. Disabling these hooks
therefore removes only the new Sparse LoRA.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributed import tensor as dtensor

from miowtion.train import checkpoint
from miowtion.train import lora

FORMAT = 'miowtion-sparse-lora-v1'


@dataclasses.dataclass
class SparseLoraConfig:
    """Explicit T2VA-only recovery configuration; no Stage-1 training."""

    run_name: str
    checkpoint_root: str
    teacher_adapter: str
    predictor_bundle: str
    train_manifest: str
    validation_manifest: str
    sample_cache: str
    geometries: list[str]
    out_dir: str = 'runs'
    task: str = 't2va'
    variant: str = 'FL2VA'
    schedule: str = 'turbo'
    num_steps: int = 8
    timestep_sampling: str = 'random'
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_layers: list[int] = dataclasses.field(
        default_factory=lambda: [0, 1, 49])
    lora_targets: list[str] = dataclasses.field(
        default_factory=lambda: list(lora.DEFAULT_TARGETS))
    keep_ratio: float = 0.05
    audio_weight: float = 1.0
    steps: int | None = 50
    epochs: int | None = None
    accum: int = 4
    lr: float = 1e-4
    warmup: int = 0
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 20261015
    save_every: int = 10
    dense_backend: str = 'auto'
    mlp_chunk_rows: int | None = 8192
    offload_blocks: int = 0
    prefetch: int = 1
    verify_invariants: bool = False
    ordered_dq: bool = False
    lora_chunk_rows: int | None = None
    checkpoint_inputs_to_cpu: bool = False
    checkpoint_group_size: int = 1
    duration_balance_window: int | None = None
    duration_balance_prefix: int = 0

    @property
    def training_states_per_visit(self) -> int:
        """Number of student backwards for one prompt visit in an epoch."""
        return 1 if self.timestep_sampling == 'random' else self.num_steps

    @classmethod
    def from_yaml(cls, path: str) -> SparseLoraConfig:
        """Loads a strict YAML schema, rejecting accidental Stage-2 flags."""
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError('configuration must be a mapping')
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f'unknown Sparse LoRA keys: {sorted(unknown)}')
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """Checks the supported mechanism, without assuming GPU readiness."""
        if (self.task, self.variant, self.schedule, self.num_steps) != (
                't2va', 'FL2VA', 'turbo', 8):
            raise ValueError('this experiment requires T2VA/FL2VA/Turbo8')
        if self.timestep_sampling not in ('random', 'trajectory'):
            raise ValueError('timestep_sampling must be random or trajectory')
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise ValueError('run_name must be a single directory name')
        if (type(self.lora_rank) is not int or self.lora_rank <= 0
                or not math.isfinite(self.lora_alpha)
                or self.lora_alpha <= 0):
            raise ValueError('rank and alpha must be positive')
        if (not self.lora_layers
                or len(set(self.lora_layers)) != len(self.lora_layers)
                or any(type(i) is not int or i < 0 for i in self.lora_layers)):
            raise ValueError('LoRA layers must be unique nonnegative ints')
        if (not self.lora_targets or len(set(self.lora_targets)) !=
                len(self.lora_targets) or not set(self.lora_targets) <=
                set(lora.TRUNK_TARGETS)):
            raise ValueError('only trunk attention/MLP projections supported')
        if not 0 < self.keep_ratio < 1:
            raise ValueError('keep_ratio must be strictly between 0 and 1')
        if (self.steps is None) == (self.epochs is None):
            raise ValueError('set exactly one of steps or epochs')
        for name in ('steps', 'epochs'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be a positive integer or null')
        for name in ('accum', 'save_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError('warmup must be a nonnegative integer')
        for name in ('lr', 'grad_clip', 'audio_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError('betas must contain two values in [0, 1)')
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError('weight_decay must be finite and nonnegative')
        if not self.geometries or len(set(self.geometries)) != len(
                self.geometries):
            raise ValueError('geometries must be nonempty and unique')
        if self.dense_backend not in ('auto', 'fa4', 'sdpa', 'math'):
            raise ValueError('unknown dense backend')
        if self.offload_blocks < 0 or self.prefetch < 1:
            raise ValueError('invalid offload settings')
        if self.mlp_chunk_rows is not None and self.mlp_chunk_rows < 1:
            raise ValueError('mlp_chunk_rows must be positive or null')
        if type(self.ordered_dq) is not bool:
            raise ValueError('ordered_dq must be Boolean')
        if (self.lora_chunk_rows is not None
                and (type(self.lora_chunk_rows) is not int
                     or self.lora_chunk_rows < 1)):
            raise ValueError('lora_chunk_rows must be positive or null')
        if type(self.checkpoint_inputs_to_cpu) is not bool:
            raise ValueError('checkpoint_inputs_to_cpu must be Boolean')
        if (type(self.checkpoint_group_size) is not int
                or self.checkpoint_group_size < 1):
            raise ValueError('checkpoint_group_size must be a positive integer')
        if self.duration_balance_window is None:
            if self.duration_balance_prefix != 0:
                raise ValueError('duration prefix requires a balance window')
        elif (type(self.duration_balance_window) is not int
              or self.duration_balance_window < 1
              or type(self.duration_balance_prefix) is not int
              or not 0 <= self.duration_balance_prefix <=
              self.duration_balance_window):
            raise ValueError('invalid duration balance window/prefix')


def learning_rate(config: SparseLoraConfig, step: int) -> float:
    """Linear warmup by absolute zero-based update; then constant rate.

    Independent of the requested stopping step, so extending a resumed
    run preserves the schedule. Zero disables warmup for short smoke runs.
    """
    return config.lr * min(1.0, (step + 1) / max(1, config.warmup))


@torch.no_grad()
def update_report(named: list[tuple[str, nn.Parameter]],
                  before: dict[str, torch.Tensor]) -> dict:
    """Measures actual A/B changes, including the zero-initialized B case.

    Snapshots contain only new adapters, never frozen backbone weights.
    DTensor reductions measure global norms on every rank. A relative
    update is undefined for zero weights and is reported as null; update
    RMS remains meaningful. These are parameter, not function, changes.
    """
    report = {}
    for name, parameter in named:
        original = before[name]
        values = torch.stack((original.float().norm(),
                              parameter.float().norm(),
                              (parameter - original).float().norm()))
        if isinstance(values, dtensor.DTensor):
            values = values.full_tensor()
        prior, current, change = values.cpu().tolist()
        if not all(math.isfinite(v) for v in (prior, current, change)):
            raise RuntimeError(f'nonfinite parameter update: {name}')
        report[name] = {
            'before_norm': prior, 'after_norm': current,
            'update_rms': change / math.sqrt(parameter.numel()),
            'update_ratio': change / prior if prior else None,
        }
    return report


def sha256_file(path: str | Path) -> str:
    """Hashes files incrementally, including large source weight shards."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    """Stable identity for JSON contracts independent of dict insertion."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_except_lora(model: nn.Module, predictor: nn.Module
                       ) -> list[tuple[str, nn.Parameter]]:
    """Freezes backbone, merged Turbo, heads, and predictor explicitly."""
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()
    trainable = lora.lora_parameters(model)
    if not trainable:
        raise ValueError('no new Sparse LoRA parameters installed')
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    return trainable


@contextlib.contextmanager
def frozen_teacher(model: nn.Module) -> Iterator[None]:
    """Disable only new hooks for a detached dense teacher evaluation.

    Exit before building the student graph; keep student hooks enabled
    through backward so non-reentrant checkpoint recomputation sees them.
    """
    modules = [m for m in model.modules() if hasattr(m, 'lora_enabled')]
    enabled = [m.lora_enabled for m in modules]
    try:
        lora.set_enabled(model, False)
        with torch.no_grad():
            yield
    finally:
        for module, state in zip(modules, enabled):
            module.lora_enabled = state


def velocity_loss(student: tuple[torch.Tensor, torch.Tensor],
                  teacher: tuple[torch.Tensor, torch.Tensor],
                  audio_weight: float = 1.0
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE of target-only fp32 velocities, separately normalized by modality.

    H3DiT.forward already selects target video/audio positions, excluding
    text, padding and conditions. Do not reinterpret these as epsilon.
    """
    if any(s.shape != t.shape or s.numel() == 0
           for s, t in zip(student, teacher)):
        raise ValueError('nonempty matching target video/audio shapes required')
    video = F.mse_loss(student[0].float(), teacher[0].detach().float())
    audio = F.mse_loss(student[1].float(), teacher[1].detach().float())
    return video + audio_weight * audio, video, audio


def gradient_report(named: list[tuple[str, nn.Parameter]],
                    require_a: bool = False) -> dict:
    """Requires finite connected gradients and a nonzero B in every target.

    A may be zero at B=0 initialization. Later updates must reach A too.
    DTensor statistics use global norms (collectives on every rank).
    """
    report = {}
    for name, parameter in named:
        if parameter.grad is None:
            raise RuntimeError(f'missing gradient: {name}')
        grad = parameter.grad
        norm = grad.float().norm()
        norm = norm.full_tensor() if isinstance(norm, dtensor.DTensor) else norm
        report[name] = float(norm)
        if not math.isfinite(report[name]):
            raise RuntimeError(f'nonfinite gradient: {name}')
        if report[name] == 0 and (name.endswith('.lora_b') or require_a):
            raise RuntimeError(f'zero gradient: {name}')
    return report


def assert_contract(actual: dict, expected: dict) -> None:
    """Refuses changed architecture, weights, data or training semantics."""
    if actual != expected:
        keys = sorted(k for k in actual.keys() | expected.keys()
                      if actual.get(k) != expected.get(k))
        raise ValueError(f'incompatible Sparse LoRA contract: {keys}')


def restore_training_payload(named: list[tuple[str, nn.Parameter]],
                             optimizer: torch.optim.Optimizer,
                             payload: dict, contract: dict) -> int:
    """Validates named weights/moments before restoring any state."""
    assert_contract(payload['contract'], contract)
    parameters = dict(named)
    if set(parameters) != set(payload['weights']) or any(
            p.shape != payload['weights'][n].shape for n, p in named):
        raise ValueError('resume parameter names/shapes mismatch')
    if set(payload['optimizer']) != set(parameters):
        raise ValueError('resume requires complete optimizer state')
    if len(payload['param_groups']) != len(optimizer.param_groups):
        raise ValueError('resume optimizer group mismatch')
    for name, parameter in named:
        moments = payload['optimizer'][name]
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= moments.keys():
            raise ValueError(f'incomplete AdamW state: {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            if moments[key].shape != parameter.shape:
                raise ValueError(f'optimizer shape mismatch: {name}/{key}')
    checkpoint.resume(named, checkpoint.Ema([], 0.0), optimizer,
                      {**payload, 'ema': {}})
    return payload['step']


def switch_payload_to_trajectory(payload: dict, contract: dict) -> dict:
    """Explicitly migrates a saved random-state run to sequential states.

    Only timestep_sampling may change. LoRA, Adam moments, step, data
    cursor and all RNG states are retained. This is an authorized change
    of experiment, not an equivalent replay or a relaxed ordinary resume.
    A random-state boundary must not contain an active partial rollout.
    """
    old = payload['contract']
    expected_old = {**contract, 'settings': {
        **contract['settings'], 'timestep_sampling': 'random'}}
    if contract['settings'].get('timestep_sampling') != 'trajectory':
        raise ValueError('transition destination must use trajectory sampling')
    assert_contract(old, expected_old)
    if any(state.get('active_trajectory') is not None
           for state in payload['rank_states']):
        raise ValueError('transition requires an inactive random rollout')
    return {**payload, 'contract': contract}


def frozen_digest(model: nn.Module, predictor: nn.Module) -> str:
    """Collective full-content digest for a costly real-model smoke check.

    The whole frozen model is streamed one tensor at a time to CPU, never
    cloned as a second complete model. Includes buffers and output heads.
    """
    digest = hashlib.sha256()
    for prefix, module in [('model', model), ('predictor', predictor)]:
        for name, value in list(module.named_parameters()) + list(
                module.named_buffers()):
            if '.lora_' in name:
                continue
            value = checkpoint._full(value).contiguous()
            identity = f'{prefix}.{name}:{value.shape}:{value.dtype}'
            digest.update(identity.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def export_adapter(path: str, model: nn.Module, contract: dict,
                   step: int) -> None:
    """Collective independent raw LoRA export; rank zero writes the file.

    A/B names match load_adapter; QKV B rows use Miowtion's grouped Q/K/V
    order, NOT the release's interleaved rows or Comfy ConvRot weights.
    Scale is alpha/rank from metadata, never guessed by a generic merger.
    """
    tensors = {}
    for name, parameter in lora.lora_parameters(model):
        destination = name.replace('.lora_a', '.lora_A.weight').replace(
            '.lora_b', '.lora_B.weight')
        tensors[destination] = checkpoint._full(parameter).contiguous()
    if (torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0):
        return
    metadata = {'format': FORMAT, 'contract': json.dumps(contract),
                'step': str(step), 'weights': 'raw',
                'qkv_layout': 'miowtion-grouped-q-k-v'}
    save_file(tensors, path + '.tmp', metadata=metadata)
    Path(path + '.tmp').replace(path)


@torch.no_grad()
def load_sparse_adapter(model: nn.Module, path: str, contract: dict) -> None:
    """Loads into matching, already-injected hooks; no base-weight merge."""
    with safe_open(path, framework='pt', device='cpu') as stream:
        metadata = stream.metadata()
    if (metadata.get('format') != FORMAT or metadata.get('qkv_layout') !=
            'miowtion-grouped-q-k-v'):
        raise ValueError('unknown Sparse LoRA format/QKV layout')
    assert_contract(json.loads(metadata['contract']), contract)
    adapter = lora.load_adapter(path)
    destinations = {n[:-len('.lora_a')] for n, _ in
                    lora.lora_parameters(model) if n.endswith('.lora_a')}
    if set(adapter) != destinations:
        raise ValueError('Sparse LoRA module set mismatch')
    modules = dict(model.named_modules())
    settings = contract.get('settings', contract)
    rank = settings.get('lora_rank', settings.get('rank'))
    alpha = settings.get('lora_alpha', settings.get('alpha'))
    if rank is None or alpha is None:
        raise ValueError('adapter contract must specify rank and alpha')
    for name, (a, b) in adapter.items():
        if (a.shape != modules[name].lora_a.shape or
                b.shape != modules[name].lora_b.shape):
            raise ValueError(f'Sparse LoRA shape mismatch: {name}')
        if (a.shape[0] != rank or modules[name].lora_scale != alpha / rank):
            raise ValueError(f'Sparse LoRA rank/alpha mismatch: {name}')
    for name, (a, b) in adapter.items():
        checkpoint._assign(modules[name].lora_a, a)
        checkpoint._assign(modules[name].lora_b, b)
