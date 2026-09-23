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
