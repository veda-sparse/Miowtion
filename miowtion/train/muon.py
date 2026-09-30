"""Muon for the tile-score predictor: per-head orthogonalized momentum.

Muon replaces Adam's per-element second moment with a spectral one: the
momentum buffer is orthogonalized (all singular values driven to 1) by a
Newton-Schulz iteration, so every direction of the update carries the same
step size. It is a matrix optimizer, and the predictor is a stack of
matrices -- `proj_q` / `proj_k` of shape [num_heads, 3 * head_dim,
head_dim] -- so the fit is unusually direct: no embeddings, no biases, no
norms, nothing that has to be left to AdamW.

**The heads are orthogonalized one by one.** A parameter here is not one
matrix but `num_heads` independent ones stacked on dim 0: head h scores
only its own attention head, and `LayerPredictor.embed` indexes them with
`proj.index_select(0, heads)`. Flattening the stack into a single matrix
would mix heads that never interact, so the Newton-Schulz iteration runs
batched over the leading dimensions instead (torch.matmul broadcasts, so
this costs nothing extra).

Update size is set explicitly rather than inherited. After
orthogonalization an [m, n] update has Frobenius norm sqrt(min(m, n)) and
therefore RMS 1 / sqrt(max(m, n)) -- a number that depends on the shape,
which is not what a learning rate should have to absorb. Scaling by
`rms_target * sqrt(max(m, n))` makes the RMS of every update exactly
`rms_target` whatever the matrix looks like, so `lr` means the same thing
across layers and across models (the convention Moonlight uses to put Muon
and AdamW on one axis). `Muon.last_update_rms()` reports the measured
value so the calibration can be checked rather than assumed.

The RMS target is exact only for a full-rank update. Orthogonalizing a
rank-deficient gradient produces a rank-deficient update whose Frobenius
norm is sqrt(rank), not sqrt(min(m, n)), so its RMS falls short of the
target by sqrt(rank / min(m, n)) -- a constant gradient, rank 1, comes out
around 0.03 instead of 0.2. That is Muon working as intended (it cannot
manufacture directions the gradient does not contain), which is why the
measured RMS is logged every update rather than trusted from the formula.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

# Quintic Newton-Schulz coefficients. They do not converge to the exact
# polar factor: the iteration is tuned to push every singular value into
# roughly [0.7, 1.3] in five steps, which is what the update direction
# needs, and chasing exactness would cost iterations for no benefit
# (Jordan et al., the Muon reference implementation).
_NS_COEFFS = (3.4445, -4.7750, 2.0315)
# Eight, not the reference implementation's five. Measured on a [384, 128]
# matrix with condition number 1000 (what a from-scratch predictor's early
# gradients can look like): five steps leave the smallest singular value at
# 0.12, eight reach 0.68, and twelve change nothing -- the iteration has hit
# its fixed point. Each step is two matmuls per head and the whole thing is
# milliseconds against a 4-minute update, so there is nothing to save here.
_NS_STEPS = 8
# Guards the normalization of a zero (or denormal) gradient.
_EPS = 1e-7


@torch.no_grad()
def orthogonalize(matrices: torch.Tensor, steps: int = _NS_STEPS
                  ) -> torch.Tensor:
    """Newton-Schulz orthogonalization, batched over leading dimensions.

    Args:
        matrices: [..., m, n] float tensor; every [m, n] slice is treated as
            an independent matrix.
        steps: Iterations of the quintic map.

    Returns:
        [..., m, n] with the singular values of each slice driven to ~1.

    Raises:
        ValueError: If `matrices` has fewer than two dimensions.
    """
    if matrices.ndim < 2:
        raise ValueError(f'need at least 2 dims, got {matrices.shape}')
    a, b, c = _NS_COEFFS
    x = matrices.to(torch.float32)
    # The iteration is written for wide matrices; a tall one is transposed
    # and transposed back, which is equivalent because (X^T)^+ = (X^+)^T.
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    # Spectral norm <= Frobenius norm, so this makes the iteration's
    # contraction region safe without computing the norm it really wants.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + _EPS)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if transposed else x).to(matrices.dtype)


def update_scale(shape: torch.Size, rms_target: float) -> float:
    """Factor making an orthogonalized [..., m, n] update have RMS
    `rms_target`.

    An orthogonal [m, n] matrix has Frobenius norm sqrt(min(m, n)) spread
    over m * n entries, i.e. RMS 1 / sqrt(max(m, n)).
    """
    return rms_target * math.sqrt(max(shape[-2], shape[-1]))


class Muon(torch.optim.Optimizer):
    """Muon over per-head matrix parameters.

    Args:
        params: Parameters, each [..., m, n] with at least 2 dims.
        lr: Learning rate; the step is `lr * rms_target` in RMS.
        momentum: Heavy-ball coefficient of the buffer.
        nesterov: Look ahead one momentum step before orthogonalizing.
        rms_target: RMS every orthogonalized update is scaled to.
        weight_decay: Decoupled, applied as `p *= 1 - lr * weight_decay`.
        ns_steps: Newton-Schulz iterations.

    Raises:
        ValueError: On a non-positive lr or rms_target, or a parameter with
            fewer than two dimensions (which has no singular values to
            equalize and belongs in AdamW).
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 5e-3,
                 momentum: float = 0.95, nesterov: bool = True,
                 rms_target: float = 0.2, weight_decay: float = 0.0,
                 ns_steps: int = _NS_STEPS):
        if lr <= 0:
            raise ValueError(f'lr must be positive, got {lr}')
        if rms_target <= 0:
            raise ValueError(f'rms_target must be positive, got {rms_target}')
        super().__init__(list(params), {
            'lr': lr, 'momentum': momentum, 'nesterov': nesterov,
            'rms_target': rms_target, 'weight_decay': weight_decay,
            'ns_steps': ns_steps})
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim < 2:
                    raise ValueError(
                        f'Muon needs matrix parameters, got shape '
                        f'{tuple(p.shape)}; 1-D parameters (biases, norms) '
                        'belong in AdamW')
        self._last_rms: dict[int, float] = {}
        self._last_alignment: dict[int, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        """One optimizer step.

        Returns:
            The closure's value, or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._last_rms.clear()
        self._last_alignment.clear()
        for group in self.param_groups:
            lr = group['lr']
            scale_rms = group['rms_target']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(p.grad)
                direction = (p.grad.add(buf, alpha=group['momentum'])
                             if group['nesterov'] else buf)
                update = orthogonalize(direction, group['ns_steps'])
                update = update.mul_(update_scale(p.shape, scale_rms))
                self._last_rms[id(p)] = float(
                    update.pow(2).mean().sqrt().item())
                # How much of the step serves the batch that just arrived.
                # The buffer averages ~1/(1-momentum) updates, and gradients
                # from different request geometries are close to orthogonal
                # (measured), so a fixed-RMS update gets divided among every
                # direction the buffer still holds. This says whether that
                # is costing anything: near 1 the step follows the current
                # gradient, near 0 it is spending itself on stale ones.
                self._last_alignment[id(p)] = float(torch.nn.functional
                    .cosine_similarity(update.flatten(), p.grad.flatten(),
                                       dim=0).item())
                if group['weight_decay']:
                    p.mul_(1 - lr * group['weight_decay'])
                p.add_(update, alpha=-lr)
        return loss

    def last_alignment(self) -> float:
        """Mean cosine between the last step and the gradient it came from.

        Not a health metric on its own: momentum is meant to carry
        information the current gradient lacks. It is here because Muon
        re-normalizes the buffer, so the cost of mixing orthogonal
        directions is not visible in the loss -- a step spread over several
        geometries advances each of them by a fraction of its RMS budget.
        Returns 0.0 before the first step.
        """
        if not self._last_alignment:
            return 0.0
        return sum(self._last_alignment.values()) / len(self._last_alignment)

    def last_update_rms(self) -> float:
        """RMS of the scaled updates of the last step, averaged over params.

        Reported so the calibration can be verified instead of assumed: it
        should equal `rms_target` to within the Newton-Schulz tolerance.
        Returns 0.0 before the first step.
        """
        if not self._last_rms:
            return 0.0
        return sum(self._last_rms.values()) / len(self._last_rms)
