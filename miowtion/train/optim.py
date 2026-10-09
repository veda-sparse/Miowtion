"""Host-offloaded optimizer state for replicated trainable parameters.

On small-memory GPUs the predictor's AdamW moments, fp32 gradients and EMA
(about 5.5 GB for 275M parameters) do not fit next to the frozen trunk.
Their cost on the host is negligible next to a training update (a CPU AdamW
step over 275M parameters takes well under a second versus tens of seconds
per update), so the master weights, moments and EMA live in pinned host
memory and only the parameters used by the forward stay on the device.

Update: clip gradients on the device -> copy them to the host masters ->
optimizer step on the host -> copy masters back to the device parameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributed import tensor as dtensor


def _pinned(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory() if torch.cuda.is_available() else t


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class HostMasters:
    """Pinned host copies of device parameters, used by the optimizer."""

    def __init__(self, named_params: Sequence[tuple[str, nn.Parameter]]):
        for name, p in named_params:
            if isinstance(p, dtensor.DTensor):
                raise ValueError(f'{name}: sharded parameters cannot use '
                                 'host optimizer offload')
        self.device = dict(named_params)
        self.host = {name: nn.Parameter(_pinned(p.detach().float().cpu()))
                     for name, p in named_params}

    def named(self) -> list[tuple[str, nn.Parameter]]:
        return list(self.host.items())

    @torch.no_grad()
    def pull_grads(self) -> None:
        """Device gradients -> host masters (missing gradients are zero)."""
        for name, p in self.device.items():
            host = self.host[name]
            if host.grad is None:
                host.grad = _pinned(torch.zeros_like(host))
            if p.grad is None:
                host.grad.zero_()
            else:
                host.grad.copy_(p.grad, non_blocking=True)
        _synchronize()

    @torch.no_grad()
    def push_params(self) -> None:
        """Host masters -> device parameters."""
        for name, p in self.device.items():
            p.copy_(self.host[name], non_blocking=True)
        _synchronize()


class _StateView:
    """Routes `optimizer.state[p]` to whichever optimizer owns `p`.

    The checkpoint reads `optimizer.state[p]` when saving and assigns to
    it when resuming, so a plain ChainMap is not enough: an assignment
    has to land in the optimizer that will actually step that parameter.
    """

    def __init__(self, parts: Sequence[torch.optim.Optimizer]):
        self._parts = list(parts)

    def _owner(self, param) -> torch.optim.Optimizer:
        for part in self._parts:
            for group in part.param_groups:
                if any(p is param for p in group['params']):
                    return part
        raise KeyError('parameter belongs to no sub-optimizer')

    def __getitem__(self, param):
        return self._owner(param).state[param]

    def __setitem__(self, param, value) -> None:
        self._owner(param).state[param] = value

    def __contains__(self, param) -> bool:
        return any(param in part.state for part in self._parts)


class Hybrid:
    """One optimizer interface over several, split by parameter shape.

    Muon orthogonalizes matrices, so it cannot own a 1-D parameter, and
    Veda2 gave the predictor exactly one: `count_gain`, a per-head gain
    on log B_j. Freezing it works but gives up a parameter that has a
    real job (shrinking towards 0 on geometries whose tile row counts
    barely vary). Routing it to AdamW instead costs nothing and keeps it
    trainable.

    Each sub-optimizer's groups carry an `lr_scale`, because a learning
    rate does not mean the same thing on both sides: Muon's step has RMS
    `lr * rms_target` whatever the gradient is, while AdamW's is about
    `lr` whatever the weight is. The trainer's schedule sets one `lr` and
    every group scales it, so the ratio between the two is fixed by the
    configuration rather than drifting with the warmup.
    """

    def __init__(self, parts: Sequence[torch.optim.Optimizer]):
        if not parts:
            raise ValueError('Hybrid needs at least one optimizer')
        self.parts = list(parts)
        self.state = _StateView(self.parts)

    @property
    def param_groups(self) -> list[dict]:
        """The sub-optimizers' own group dicts, not copies.

        Mutating `group['lr']` therefore reaches the optimizer that uses
        it, which is how the trainer's schedule works.
        """
        return [g for part in self.parts for g in part.param_groups]

    def step(self) -> None:
        for part in self.parts:
            part.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for part in self.parts:
            part.zero_grad(set_to_none=set_to_none)

    def last_update_rms(self) -> float:
        """The Muon side's measured update RMS, or 0 if there is none."""
        for part in self.parts:
            if hasattr(part, 'last_update_rms'):
                return part.last_update_rms()
        return 0.0

    def last_alignment(self) -> float:
        for part in self.parts:
            if hasattr(part, 'last_alignment'):
                return part.last_alignment()
        return 0.0
