"""Training checkpoints: EMA, atomic writes, resume vs. init.

Contents: trainable weights and their EMA (full tensors), optimizer moments
keyed by parameter name (never by flat index), every rank's sampler position
and noise generator, the rank-shared generator (hash-checked across ranks
before saving), step and config.

Write protocol: local temp file -> atomic rename -> completion marker with
the file size -> background copy to persistent storage. Files without a
marker are never loaded. The process must wait for the last copy before
exiting, otherwise the final checkpoint dies with it.

Two mutually exclusive load modes:
  * resume: continue the same run, restore everything;
  * init:   start a new stage from weights only; every stored tensor must
    land on a trainable parameter unless its prefix is explicitly dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import tensor as dtensor
from torch.distributed.tensor import _utils as dtensor_utils

from miowtion.train import parallel

_FILE = 'state.pt'
_OPTIM = 'optim.pt'
_MARKER = 'done.json'


class Ema:
    """Exponential moving average of trainable parameters (fp32)."""

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]],
                 decay: float):
        self.decay = decay
        self.shadow = {name: p.detach().float().clone()
                       for name, p in named_params}

    @torch.no_grad()
    def update(self, named_params: Iterable[tuple[str, nn.Parameter]]) -> None:
        for name, p in named_params:
            self.shadow[name].lerp_(p.detach().float(), 1.0 - self.decay)


def _full(t):
    """Full tensor on CPU; collective for sharded DTensors."""
    if isinstance(t, dtensor.DTensor):
        t = t.full_tensor()
    return t.detach().cpu() if torch.is_tensor(t) else t


@torch.no_grad()
def _assign(dst: torch.Tensor, full: torch.Tensor) -> None:
    """Copies a full tensor into a plain or dim-0-sharded destination."""
    if isinstance(dst, dtensor.DTensor):
        shape, offset = dtensor_utils.compute_local_shape_and_global_offset(
            dst.shape, dst.device_mesh, dst.placements)
        full = full[offset[0]:offset[0] + shape[0]]
        dst = dst.to_local()
    dst.copy_(full.to(dst.device, dst.dtype))


def _state_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _gather(obj, env: parallel.DistEnv) -> list:
    if not env.distributed:
        return [obj]
    out = [None] * env.world_size
    dist.all_gather_object(out, obj)
    return out


class CheckpointManager:
    """Writes checkpoints locally and mirrors them to persistent storage."""

    def __init__(self, local_dir: str, persistent_dir: str | None,
                 env: parallel.DistEnv, keep_optimizer: int = 2):
        """Initializes the manager.

        Args:
            local_dir: Directory of step_* checkpoints.
            persistent_dir: Optional mirror (never pruned).
            env: Distributed environment.
            keep_optimizer: How many of the newest checkpoints keep their
                Adam moments. No checkpoint is ever deleted: weights and
                EMA are the training history and must survive the whole
                run. The moments are half of a checkpoint's bytes and are
                only read to resume, which only ever happens from the
                newest one, so older ones drop them.
        """
        if keep_optimizer < 1:
            raise ValueError(f'keep_optimizer must be >= 1, got '
                             f'{keep_optimizer}')
        self.local_dir = local_dir
        self.persistent_dir = persistent_dir
        self.env = env
        self.keep_optimizer = keep_optimizer
        self._copy_thread: threading.Thread | None = None

    def save(self, step: int, named_params: Sequence[tuple[str, nn.Parameter]],
             ema: Ema, optimizer: torch.optim.Optimizer,
             rank_state: dict, shared_generator: torch.Tensor,
             config: dict) -> None:
        """Collective: every rank must call it."""
        rank_states = _gather(rank_state, self.env)
        hashes = _gather(_state_hash(shared_generator), self.env)
        if len(set(hashes)) != 1:
            raise RuntimeError('rank-shared generator diverged across ranks: '
                               f'{hashes}')
        # Gathering sharded tensors is collective: all ranks, before the
        # rank-0 early exit.
        names = {id(p): name for name, p in named_params}
        weights = {n: _full(p).float() for n, p in named_params}
        ema_state = {n: _full(t) for n, t in ema.shadow.items()}
        opt_state = {names[id(p)]: {k: _full(v)
                                    for k, v in optimizer.state[p].items()}
                     for group in optimizer.param_groups
                     for p in group['params'] if p in optimizer.state}
        if not self.env.is_main:
            return
        payload = {
            'step': step,
            'weights': weights,
            'ema': ema_state,
            'param_groups': [{k: v for k, v in g.items() if k != 'params'}
                             for g in optimizer.param_groups],
            'rank_states': rank_states,
            'shared_generator': shared_generator,
            'config': config,
        }
        directory = os.path.join(self.local_dir, f'step_{step:07d}')
        os.makedirs(directory, exist_ok=True)
        # The moments go in their own file so that pruning them later does
        # not have to rewrite (and briefly double) the weights.
        _write(os.path.join(directory, _OPTIM), opt_state)
        path = _write(os.path.join(directory, _FILE), payload)
        with open(os.path.join(directory, _MARKER), 'w') as f:
            json.dump({'step': step, 'bytes': os.path.getsize(path)}, f)
        self._start_copy(directory)
        self._prune()

    def _prune(self) -> None:
        """Drops the Adam moments of all but the newest keep_optimizer.

        Checkpoint directories themselves are never removed.
        """
        for root in (self.local_dir, self.persistent_dir):
            if not root or not os.path.isdir(root):
                continue
            complete = sorted(
                d for d in os.listdir(root) if d.startswith('step_')
                and os.path.exists(os.path.join(root, d, _MARKER)))
            for name in complete[:-self.keep_optimizer]:
                moments = os.path.join(root, name, _OPTIM)
                if os.path.exists(moments):
                    os.remove(moments)

    def _start_copy(self, directory: str) -> None:
        if not self.persistent_dir:
            return
        self.wait()
        target = os.path.join(self.persistent_dir, os.path.basename(directory))

        def copy():
            tmp = target + '.partial'
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(directory, tmp)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)

        self._copy_thread = threading.Thread(target=copy, daemon=False)
        self._copy_thread.start()

    def wait(self) -> None:
        """Blocks until the background copy finished (call before exit)."""
        if self._copy_thread is not None:
            self._copy_thread.join()
            self._copy_thread = None


def latest(directories: Sequence[str]) -> str | None:
    """Newest complete checkpoint directory, preferring local copies."""
    found = {}
    for root in directories:
        if not root or not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            marker = os.path.join(root, name, _MARKER)
            path = os.path.join(root, name, _FILE)
            if not (name.startswith('step_') and os.path.exists(marker)):
                continue
            with open(marker) as f:
                meta = json.load(f)
            if os.path.getsize(path) != meta['bytes']:
                continue
            found.setdefault(meta['step'], os.path.join(root, name))
    return found[max(found)] if found else None


def _write(path: str, payload) -> str:
    """Atomic torch.save; returns the path."""
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    return path


def load(directory: str) -> dict:
    """Payload of a checkpoint; 'optimizer' is empty once pruned."""
    if not os.path.exists(os.path.join(directory, _MARKER)):
        raise FileNotFoundError(f'{directory} has no completion marker')
    payload = torch.load(os.path.join(directory, _FILE), map_location='cpu',
                         weights_only=False)
    moments = os.path.join(directory, _OPTIM)
    payload['optimizer'] = (
        torch.load(moments, map_location='cpu', weights_only=False)
        if os.path.exists(moments) else {})
    return payload


@torch.no_grad()
def init_weights(named_params: Sequence[tuple[str, nn.Parameter]],
                 payload: dict, use_ema: bool = True,
                 drop_prefixes: Sequence[str] = ()) -> None:
    """New-stage init: weights only, strict.

    Raises:
        KeyError: If a stored tensor has no destination (and is not
            dropped), or if a destination is missing from the file.
    """
    source = payload['ema'] if use_ema else payload['weights']
    source = {k: v for k, v in source.items()
              if not k.startswith(tuple(drop_prefixes))}
    params = dict(named_params)
    unused = sorted(set(source) - set(params))
    if unused:
        raise KeyError(f'checkpoint tensors without a destination: '
                       f'{unused[:5]} ({len(unused)})')
    missing = sorted(set(params) - set(source))
    if missing:
        raise KeyError(f'parameters missing from checkpoint: {missing[:5]}')
    for name, value in source.items():
        _assign(params[name], value)


@torch.no_grad()
def resume(named_params: Sequence[tuple[str, nn.Parameter]], ema: Ema,
           optimizer: torch.optim.Optimizer, payload: dict) -> None:
    """Restores weights, EMA and optimizer moments by parameter name."""
    params = dict(named_params)
    for name, value in payload['weights'].items():
        _assign(params[name], value)
    for name, value in payload['ema'].items():
        _assign(ema.shadow[name], value)
    for group, saved in zip(optimizer.param_groups, payload['param_groups']):
        group.update(saved)
    for name, state in payload['optimizer'].items():
        p = params[name]
        restored = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.dim() > 0:
                slot = torch.zeros_like(p)  # same placement as the param
                _assign(slot, value)
                value = slot
            restored[key] = value
        optimizer.state[p] = restored


def rank_state_for(payload: dict, rank: int) -> dict:
    """World-size changes: new rank r inherits old rank r % old_world."""
    states = payload['rank_states']
    return states[rank % len(states)]
