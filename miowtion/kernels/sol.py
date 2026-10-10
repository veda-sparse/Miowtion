"""Sol-Attn as a fourth attention mode, for quality comparison only.

Sol-Attn (arXiv 2607.24027, NVIDIA) routes key blocks by an on-the-fly
threshold inside a single online-softmax pass and reuses the
below-threshold score columns to approximate the contribution of the
blocks it skipped. It is therefore a different design from Veda on two
axes at once: the budget is dynamic rather than fixed, and dropped blocks
are compensated rather than discarded. Both axes were measured offline
here and neither paid (see docs/features/sol_ablation.md); this module
exists so that the comparison can also be made on actual video.

Upstream is used as a dependency, not copied: the package lives at
`techniques/sparse_backends` of NVlabs/Sana, pinned in pyproject.toml and
registered in docs/dependencies.md. Its CuTe kernels cover SM89, SM90,
SM100 and SM120; on SM120 (our box) the CuTe path compiles and is what
these numbers use, which is worth stating because the package also has a
Triton reference path that is correct but not representative.

Two properties of the upstream API constrain how it can be used:

* The kernels are **forward-only**, so this is an inference mode and
  `return_lse` is unsupported -- the teacher heat maps cannot be built
  through it.
* Head dimension is pinned to 128 and the layout is [B, T, H, 128]. H3's
  head dimension is 128, so the only adaptation needed is the batch axis.

Sparsity is set by `tau`, a threshold coefficient, and *not* by a budget:
larger tau routes fewer blocks exactly. A density target therefore has to
be reached by calibrating tau, and the achieved density is a property of
the clip rather than of the configuration. That is the whole point of the
method and also why it cannot be put at exactly the same budget as a
fixed top-k router; comparisons must report the density Sol actually
used.
"""

from __future__ import annotations

import os

import torch

_UNAVAILABLE: str | None = None
try:  # pragma: no cover - import guard, exercised by availability tests
    from sol_attn import sol_attn as _sol_attn
    from sol_attn import interface as _sol_interface
except Exception as error:  # noqa: BLE001 - any import failure disables it
    _sol_attn = None
    _sol_interface = None
    _UNAVAILABLE = str(error)

# Upstream routes at 64-token key-block granularity (sol_attn.interface).
BLOCK_SIZE = 64

# Filled by attention() when MIOWTION_SOL_DENSITY is set, so a comparison
# run can state the sparsity it reached instead of the one it asked for.
DENSITY_LOG: list[float] = []

# tau -> the densities it would have reached on every call so far, filled
# when MIOWTION_SOL_DENSITY is a comma-separated list of candidates.
DENSITY_CURVE: dict[float, list[float]] = {}


def available(device: torch.device | str | None = None) -> bool:
    """Whether Sol-Attn can run for tensors on `device`.

    Args:
        device: Target device; CPU is always False because the package is
            CUDA-only.

    Returns:
        True when the package imported and the device is CUDA.
    """
    if _sol_attn is None:
        return False
    if device is None:
        return torch.cuda.is_available()
    return torch.device(device).type == 'cuda'


def unavailable_reason() -> str | None:
    """The import error that disabled Sol-Attn, or None if it is usable."""
    return _UNAVAILABLE


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              used: int, tau: float = 1.0,
              thresh_type: str = 'exact') -> torch.Tensor:
    """Sol-Attn over rows [0, used) of a packed sequence.

    Args:
        q: [S, H, 128] bf16, post QK-norm and RoPE.
        k: [S, H, 128] bf16.
        v: [S, H, 128] bf16.
        used: Number of real rows; the rest are padding and stay zero.
        tau: Upstream threshold coefficient. Larger routes fewer blocks
            exactly, i.e. more sparsity. It is not a budget.
        thresh_type: Upstream routing threshold; 'exact' uses the full
            covariance.

    Returns:
        out: [S, H, 128] bf16, zero on padding rows.

    Raises:
        RuntimeError: If the package is not importable or the device is
            not CUDA.
        ValueError: If the head dimension is not 128, which upstream
            requires, or tau is not positive.
    """
    if not available(q.device):
        raise RuntimeError(
            'Sol-Attn is unavailable: '
            f'{_UNAVAILABLE or "tensors are not on CUDA"}')
    if q.shape[-1] != 128:
        raise ValueError('Sol-Attn pins the head dimension to 128, got '
                         f'{q.shape[-1]}')
    if tau <= 0:
        raise ValueError(f'tau must be > 0, got {tau}')
    # Upstream defaults to 'diag' and the ComfyUI node takes that
    # default, while this wrapper defaults to 'exact'. The two are
    # different operating points -- the published tau-to-density mapping
    # (1.0 keeps about 16% of key blocks) does not match what we measure
    # under 'exact' (2.6%) -- so the choice has to be switchable without
    # editing a config, to settle which mapping belongs to which.
    thresh_type = os.environ.get('MIOWTION_SOL_THRESH', thresh_type)
    if thresh_type not in ('diag', 'exact'):
        raise ValueError("thresh_type must be 'diag' or 'exact', got "
                         f'{thresh_type!r}')
    # A density target has to be verified, not assumed: tau is a
    # threshold, so the sparsity it reaches is a property of the
    # activations. Setting MIOWTION_SOL_DENSITY makes every call record
    # the routed fraction so a run reports the sparsity it actually used.
    probe = os.environ.get('MIOWTION_SOL_DENSITY')
    if probe:
        # A bare value records the density of the tau in use; a
        # comma-separated list records the whole curve in one pass, which
        # matters because a probe run otherwise pays the adapter merge
        # and the AdaLN tables once per candidate.
        taus = ([tau] if probe in ('1', 'true', 'yes')
                else [float(x) for x in probe.split(',')])
        for candidate in taus:
            DENSITY_CURVE.setdefault(candidate, []).append(
                density(q, k, v, used, candidate, thresh_type))
        DENSITY_LOG.append(DENSITY_CURVE[taus[0]][-1])
    out = torch.zeros_like(q)
    # Upstream wants [B, T, H, D]; a packed sequence is one batch element.
    rows = _sol_attn(q[:used][None].contiguous(),
                     k[:used][None].contiguous(),
                     v[:used][None].contiguous(),
                     tau=tau, thresh_type=thresh_type)
    out[:used] = rows[0].to(out.dtype)
    return out


def density(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, used: int,
            tau: float, thresh_type: str = 'diag') -> float:
    """Fraction of key blocks Sol-Attn routes exactly, at this tau.

    Needed because tau is a threshold and not a budget: to compare
    against a fixed-budget router at the same sparsity, tau has to be
    calibrated until this matches, and the value it reaches has to be
    reported rather than assumed.

    This reproduces upstream's own decision rather than approximating
    it. Upstream computes a per-(query block, head) threshold
    `mean + tau * std` over the block summaries and routes a key block
    exactly when the mean proxy score over the query block exceeds it
    (`triton_ref/fwd.py`: `exact = (sum(scores, 0) / q_len >
    route_threshold)`); the block summaries and the threshold both come
    from upstream's `prepare`, so only the comparison is reimplemented
    here.

    Args:
        q: [S, H, 128] bf16.
        k: [S, H, 128] bf16.
        v: [S, H, 128] bf16.
        used: Number of real rows.
        tau: Threshold coefficient to evaluate.
        thresh_type: 'diag' (upstream default) or 'exact'.

    Returns:
        The routed fraction over all (query block, key block, head)
        triples of the real rows.

    Raises:
        RuntimeError: If the package is not importable.
    """
    if _sol_attn is None:
        raise RuntimeError(f'Sol-Attn is unavailable: {_UNAVAILABLE}')
    from sol_attn.triton_ref import preprocess as _pre

    qb, kb, vb = (x[:used][None].contiguous() for x in (q, k, v))
    scale = float(q.shape[-1]) ** -0.5
    kc, _, threshold = _pre.prepare(qb, kb, vb, tau=tau, scale=scale,
                                    thresh_type=thresh_type)
    # kc: [B, n_blocks, H, D] block summaries; threshold: [B, n_q, H].
    blocks = kc.shape[1]
    groups = threshold.shape[1]
    rows = qb.shape[1]
    # Mean query of each threshold group, which is what the mean proxy
    # score over a query block reduces to (the score is bilinear).
    pad = (-rows) % groups if groups else 0
    per = (rows + pad) // groups
    qg = torch.zeros(1, groups * per, qb.shape[2], qb.shape[3],
                     dtype=torch.float32, device=qb.device)
    qg[:, :rows] = qb.float()
    qg = qg.view(1, groups, per, qb.shape[2], qb.shape[3]).mean(2)
    # [B, groups, H, blocks] proxy scores, then compare to the threshold.
    proxy = torch.einsum('bghd,bkhd->bghk', qg, kc.float()) * scale
    routed = proxy > threshold.float().unsqueeze(-1)
    return float(routed.float().mean())
