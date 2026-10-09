"""Update diagnostics of the predictor: gradient direction and step size.

Logged every update (on the tensors the optimizer steps, i.e. the host
masters when the optimizer is offloaded, so no device memory is used):
  * per-layer gradient norm of the predictor (proj_q and proj_k together);
  * cosine between this update's gradient and the previous update's, per
    layer and overall: how consistent the gradient direction is. Values near
    0 mean the batch gradient is dominated by noise;
  * ||delta w|| / ||w|| of the optimizer step (relative update size).
"""

from __future__ import annotations

import collections
import re
from collections.abc import Sequence

import torch
from torch import nn

_LAYER = re.compile(r'predictor\.layers\.(\d+)\.')


class UpdateMonitor:
    """Gradient direction / update size of the predictor parameters."""

    def __init__(self, named_params: Sequence[tuple[str, nn.Parameter]]):
        layers = collections.defaultdict(list)
        for name, param in named_params:
            match = _LAYER.match(name)
            if match:
                layers[int(match.group(1))].append(param)
        if not layers:
            raise ValueError('no predictor parameters to monitor')
        self.layers = [layers[i] for i in sorted(layers)]
        self._prev_grads: list[list[torch.Tensor]] | None = None
        self._before: list[list[torch.Tensor]] | None = None

    @torch.no_grad()
    def before_step(self) -> dict:
        """Gradient statistics; call after reduction, before the step."""
        grads = [[(p.grad if p.grad is not None
                   else torch.zeros_like(p)).detach().float().clone()
                  for p in params] for params in self.layers]
        norms, dots, prev_norms = [], [], []
        for i, layer in enumerate(grads):
            norms.append(sum(g.square().sum() for g in layer).sqrt().item())
            if self._prev_grads is not None:
                dots.append(sum((g * h).sum() for g, h in
                                zip(layer, self._prev_grads[i])).item())
                prev_norms.append(sum(h.square().sum() for h in
                                      self._prev_grads[i]).sqrt().item())
        record = {'grad_norm_layers': [float(f'{n:.4g}') for n in norms]}
        if dots:
            cos = [d / max(a * b, 1e-30)
                   for d, a, b in zip(dots, norms, prev_norms)]
            total = sum(dots) / max(
                (sum(n * n for n in norms) ** 0.5)
                * (sum(n * n for n in prev_norms) ** 0.5), 1e-30)
            record['grad_cos'] = round(total, 4)
            record['grad_cos_layers'] = [round(c, 3) for c in cos]
        self._prev_grads = grads
        self._before = [[p.detach().float().clone() for p in params]
                        for params in self.layers]
        return record

    @torch.no_grad()
    def after_step(self) -> dict:
        """Relative size of the optimizer step (call right after it)."""
        delta = weight = 0.0
        for params, before in zip(self.layers, self._before):
            for p, b in zip(params, before):
                delta += (p.detach().float() - b).square().sum().item()
                weight += b.square().sum().item()
        self._before = None
        return {'update_ratio': (delta / max(weight, 1e-30)) ** 0.5}


def check_gradient_stop(model: nn.Module,
                        trainable: Sequence[nn.Parameter]) -> None:
    """Raises if a parameter outside `trainable` received a gradient.

    Stage 1 must only train the predictor: the trunk is frozen and the
    predictor reads detached activations.
    """
    allowed = {id(p) for p in trainable}
    leaked = [name for name, p in model.named_parameters()
              if id(p) not in allowed and p.grad is not None]
    if leaked:
        raise RuntimeError(f'gradient reached frozen parameters: '
                           f'{leaked[:5]} ({len(leaked)})')


class EarlyAbort:
    """Stops a run that is not improving, instead of paying for the rest.

    A Veda2 stage-1 run degraded monotonically for 59 updates before
    anyone looked: recall 0.73 -> 0.64, the watched share 0.919 -> 0.882,
    and the loss itself rising. An hour of GPU went into a result that was
    visible by update 20.

    The comparison is between trailing means, not single updates, because
    the metric is not comparable across the cycled geometries: each one has
    its own ceiling, so consecutive updates differ by more than any
    plausible improvement. A window of twice the geometry count sees each
    geometry about twice.

    Attributes:
        window: Updates per trailing mean.
        patience: Updates without a new best trailing mean before aborting.
        metric: Record field to watch; larger is better.
    """

    def __init__(self, window: int = 8, patience: int = 10,
                 metric: str = 'kept_over_ceiling'):
        if window < 1:
            raise ValueError(f'window must be >= 1: {window}')
        if patience < 1:
            raise ValueError(f'patience must be >= 1: {patience}')
        self.window = window
        self.patience = patience
        self.metric = metric
        self._history: list[float] = []
        self._best: float | None = None
        self._best_at = 0

    def update(self, record: dict) -> str | None:
        """Feeds one update's record.

        Args:
            record: The per-update log record.

        Returns:
            A reason to stop, or None to carry on. The reason is phrased
            for a human reading the log, with the numbers in it.
        """
        value = record.get(self.metric)
        if value is None:
            return None
        self._history.append(float(value))
        if len(self._history) < self.window:
            return None
        trailing = sum(self._history[-self.window:]) / self.window
        step = len(self._history)
        if self._best is None or trailing > self._best:
            self._best, self._best_at = trailing, step
            return None
        if step - self._best_at >= self.patience:
            return (f'{self.metric} has not improved for '
                    f'{step - self._best_at} updates: trailing mean over '
                    f'{self.window} is {trailing:.4f} against a best of '
                    f'{self._best:.4f} at update {self._best_at}')
        return None
