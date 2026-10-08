"""Offline ablation of block-selection targets and per-row key budgets.

Two hypotheses are checked on a frozen backbone, from exported q / k / v
only: no training and no kernel work.

H1 (selection target): picking blocks by their contribution to the output
error beats picking them by attention quality (what Veda's scorer is
distilled against today).
H2 (budget allocation): a per-row variable budget beats a fixed top-k.

Both reduce to one exact identity. For query token `u` in query tile `i`,
with the kept set `S_i`, the dropped set `U_i` and `c` in {0, 1} selecting
whether Sol's zero-order compensation is applied to the dropped blocks:

    O_u^(M,c) - O_u = -(Z_u / D_u) * sum_{j in U_i} xi_uj,
    xi_uj = (dN_uj - dE_uj * O_u) / Z_u,

where dN_uj = N_uj - c * Nhat_uj and dE_uj = E_uj - c * Ehat_uj are the
residuals of the block's unnormalized weighted V and mass. Direct discard
is the c = 0 special case, so both settings share one code path.

Projected through this head's slice of the output projection, the triangle
inequality gives a bound that is linear in the selection, so the ideal
block importance is

    Omega_ij^(c) = sum_{u in I_i} || xi_uj^(c) W_o^h ||_2.

Per-row top-k over Omega minimizes that bound at a fixed k; a single
global threshold on Omega minimizes it at a fixed total budget. Those two
are the oracle row of the 2x2 in H2.

Implementation notes that matter:

- Everything is normalized by the dense log-sum-exp, so the tables hold
  a_uj = E_uj / Z_u and n_uj = N_uj / Z_u. Then sum_j a_uj == 1 and
  sum_j n_uj == O_u, which keeps fp32 well scaled and makes "keep
  everything" exactly zero error.
- || x W_o^h || is never formed in the model's hidden space. With
  G = W_o^h W_o^h^T = S^T S (S being the head's column slice of
  out_proj.weight) and G = L L^T, we have || x W_o^h || = || x L ||, so the
  whole ablation stays in head_dim. `whitening` builds L once per head.
- Omega needs n_uj with its head_dim axis intact, i.e. [U, N, D] per query
  tile, which costs a factor N more than attention itself. Query tiles are
  therefore sampled (`rows`), exactly as the tile-plan search does.
- Two passes over the full attention of each (layer, head, step): pass one
  builds the [R, N] score tables, pass two scores every mask. All masks
  reuse the same {a, n, ahat, nhat}, so no mask recomputes attention.
- Global (text / audio-conditioning) key tiles are always kept and are
  counted against the budget. Global *query* rows attend densely in Veda,
  so they have no selection to ablate and are not sampled here.

References:
    Sol-Attn, arXiv 2607.24027 (training-free, mu + beta * sigma threshold
    plus zero-order compensation).
    Veda, arXiv 2605.30325 (distilled scorer, fixed top-k, direct discard).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Sequence

import torch

from miowtion.h3 import attention as h3_attention
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import tiling

TILE = tiling.TILE_SIZE

# Pass-two keeps [U, N, D] fp32 intermediates per query-tile chunk; pass one
# keeps two of them. 1 GiB bounds the chunk so that long clips (103k tokens,
# 806 tiles) do not add gigabytes on top of the teacher's activations.
_CHUNK_BYTES = 1 << 30

# Jitter added to the diagonal of G before the Cholesky, relative to its
# mean diagonal. S^T S is PSD but can be numerically indefinite when
# hidden_size >> head_dim and the slice is near rank-deficient.
_CHOLESKY_JITTER = 1e-6


def whitening(out_proj_weight: torch.Tensor, head: int,
              head_dim: int) -> torch.Tensor:
    """Lower-triangular L with L L^T = W_o^h W_o^h^T, for || x W_o^h ||.

    Args:
        out_proj_weight: [hidden_size, num_heads * head_dim] of the layer's
            output projection (any float dtype).
        head: Head index.
        head_dim: Head dimension.

    Returns:
        [head_dim, head_dim] fp32 L, so that `x @ L` has the same row norms
        as `x @ W_o^h`.

    Raises:
        ValueError: If the slice is out of range.
    """
    cols = out_proj_weight.shape[1]
    start = head * head_dim
    if start + head_dim > cols:
        raise ValueError(f'head {head} x {head_dim} exceeds {cols} columns')
    s = out_proj_weight[:, start:start + head_dim].float()
    gram = s.transpose(0, 1) @ s
    jitter = _CHOLESKY_JITTER * gram.diagonal().mean().clamp(min=1e-30)
    eye = torch.eye(head_dim, dtype=gram.dtype, device=gram.device)
    try:
        return torch.linalg.cholesky(gram + jitter * eye)
    except RuntimeError:
        # Rank-deficient slice: an eigenvalue-clamped square root preserves
        # row norms just as well and never fails.
        evals, evecs = torch.linalg.eigh(gram)
        return evecs * evals.clamp(min=0.0).sqrt()[None, :]


@dataclasses.dataclass
class HeadTables:
    """Block score tables of one (layer, head, step) over sampled rows.

    All tables are fp32 [R, N] over the sampled query tiles `rows` and all
    N = n_video_tiles + n_global_tiles key tiles, except as noted.

    Attributes:
        rows: [R] int64 sampled video query tile ids.
        attn_mass: A_ij, the block's attention mass summed over the tile.
        max_prob: Mx_ij, the largest probability in the block (Veda's
            current distillation target).
        omega0: Omega_ij^(0), ideal importance under direct discard.
        omega1: Omega_ij^(1), ideal importance under Sol compensation.
        proxy: s_tilde_ij, the mean-pooled proxy score Sol thresholds.
        mu: [R] row mean of `proxy` over selectable video tiles.
        sigma: [R] row standard deviation of `proxy`, same support.
        valid_rows: [R] int64 real query tokens in each sampled tile.
        kv_ok: [N] bool, key tile holds at least one real row.
        n_video_tiles: Key tiles in the video quadrant; tiles at or after
            this index are global and always kept.
    """

    rows: torch.Tensor
    attn_mass: torch.Tensor
    max_prob: torch.Tensor
    omega0: torch.Tensor
    omega1: torch.Tensor
    proxy: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    valid_rows: torch.Tensor
    kv_ok: torch.Tensor
    n_video_tiles: int

    @property
    def n_tiles(self) -> int:
        return self.attn_mass.shape[1]

    def score(self, name: str) -> torch.Tensor:
        """The [R, N] table named by one of the H1 oracle scores."""
        table = {'A': self.attn_mass, 'Mx': self.max_prob,
                 'Omega0': self.omega0, 'Omega1': self.omega1,
                 'proxy': self.proxy}.get(name)
        if table is None:
            raise ValueError(f'unknown score {name!r}')
        return table


def _chunk_rows(n_tiles: int, head_dim: int) -> int:
    """Query tiles per chunk under `_CHUNK_BYTES` of [U, N, D] fp32."""
    per_row = 2 * n_tiles * head_dim * 4
    return max(1, _CHUNK_BYTES // (per_row * TILE))


def _tile_order(x: torch.Tensor, layout: tiling.TileLayout,
                head: int) -> torch.Tensor:
    """[N * TILE, D] fp32 of one head in tile order, padding zeroed."""
    out = x.index_select(0, layout.gather_index)
    out = (out[:, head] if out.dim() == 3 else out).float()
    if layout.pad_slots.numel():
        out = out.index_fill(0, layout.pad_slots, 0.0)
    return out


def _slots_of(rows: torch.Tensor) -> torch.Tensor:
    """[R * TILE] slot ids of the query tiles `rows`, in tile order."""
    return (rows[:, None] * TILE
            + torch.arange(TILE, device=rows.device)[None]).reshape(-1)


def _probabilities(q_chunk: torch.Tensor, k_t: torch.Tensor,
                   lse_chunk: torch.Tensor, scale: float,
                   slot_valid: torch.Tensor,
                   row_valid: torch.Tensor) -> torch.Tensor:
    """[U, N * TILE] fp32 exp(s - lse); invalid rows and keys are zero."""
    s = (q_chunk @ k_t.transpose(0, 1)) * scale
    p = torch.exp(s - lse_chunk[:, None])
    p = p * slot_valid[None, :]
    return p * row_valid[:, None]


@torch.no_grad()
def head_tables(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                lse: torch.Tensor, layout: tiling.TileLayout,
                rows: torch.Tensor, head: int,
                whiten: torch.Tensor) -> HeadTables:
    """Pass one: the [R, N] block score tables of one head.

    Args:
        q: [S, H, D] packed queries (any float dtype).
        k: [S, H, D] packed keys.
        v: [S, H, D] packed values.
        lse: [S, H] fp32 dense log-sum-exp of the packed rows.
        layout: Tile permutation.
        rows: [R] int64 video query tile ids to sample.
        head: Head index into q / k / v / lse.
        whiten: [D, D] fp32 L from `whitening`.

    Returns:
        The score tables.

    Raises:
        ValueError: If `rows` is not inside the video quadrant.
    """
    if rows.numel() and int(rows.max()) >= layout.n_video_tiles:
        raise ValueError('rows must be video query tiles; global query rows '
                         'attend densely and have no selection to ablate')
    device = q.device
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    n_tiles = layout.n_tiles
    q_t = _tile_order(q, layout, head)
    k_t = _tile_order(k, layout, head)
    v_t = _tile_order(v, layout, head)
    lse_t = _tile_order(lse[:, head, None], layout, 0).reshape(-1)
    slot_valid = layout.slot_valid.to(torch.float32)
    counts = layout.valid_count.to(torch.float32).clamp(min=1.0)  # [N]

    # Zero-order block summaries: mean key and summed value per key tile.
    k_mean = (k_t.view(n_tiles, TILE, head_dim).sum(1)
              / counts[:, None])                                   # [N, D]
    v_sum = v_t.view(n_tiles, TILE, head_dim).sum(1)               # [N, D]
    v_white = (v_t @ whiten).view(n_tiles, TILE, head_dim)
    v_sum_white = v_sum @ whiten                                   # [N, D]
    q_mean = (q_t.view(n_tiles, TILE, head_dim).sum(1)
              / counts[:, None])[:layout.n_video_tiles]

    slots = _slots_of(rows)
    row_valid_all = layout.slot_valid.index_select(0, slots).to(torch.float32)
    chunk = _chunk_rows(n_tiles, head_dim) * TILE
    parts: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, slots.numel(), chunk):
        sl = slots[start:start + chunk]
        q_c = q_t.index_select(0, sl)
        lse_c = lse_t.index_select(0, sl)
        rv = row_valid_all[start:start + chunk]
        p = _probabilities(q_c, k_t, lse_c, scale, slot_valid, rv)
        pb = p.view(-1, n_tiles, TILE)
        a = pb.sum(-1)                                             # [U, N]
        mx = pb.amax(-1)                                           # [U, N]
        n_white = torch.einsum('unb,nbd->und', pb, v_white)        # [U, N, D]
        o_white = n_white.sum(1)                                   # [U, D]
        del p, pb
        # xi^(0) = n_uj - a_uj * O_u, already whitened.
        xi = n_white - a[:, :, None] * o_white[:, None, :]
        omega0 = xi.norm(dim=-1)                                   # [U, N]
        # Sol's zero-order estimate of the dropped blocks.
        s_hat = (q_c @ k_mean.transpose(0, 1)) * scale
        p_hat = torch.exp(s_hat - lse_c[:, None]) * rv[:, None]    # [U, N]
        a_hat = p_hat * counts[None, :]
        n_hat_white = p_hat[:, :, None] * v_sum_white[None, :, :]
        xi -= n_hat_white - (a_hat[:, :, None]
                             * o_white[:, None, :])
        omega1 = xi.norm(dim=-1)
        del xi, n_white, n_hat_white
        parts.append((a, mx, omega0, omega1))

    def rows_of(index: int) -> torch.Tensor:
        """[R, N]: per-tile sum (or max) of a [U, N] quantity."""
        stacked = torch.cat([p[index] for p in parts], 0)
        grouped = stacked.view(rows.numel(), TILE, n_tiles)
        return grouped.amax(1) if index == 1 else grouped.sum(1)

    attn_mass, max_prob, omega0, omega1 = (rows_of(i) for i in range(4))
    proxy = torch.full((rows.numel(), n_tiles), float('nan'), device=device)
    proxy[:, :layout.n_video_tiles] = (
        q_mean.index_select(0, rows) @ k_mean[:layout.n_video_tiles]
        .transpose(0, 1)) * scale
    support = layout.kv_ok[:layout.n_video_tiles]
    sel = proxy[:, :layout.n_video_tiles][:, support]
    mu = sel.mean(-1)
    sigma = sel.std(-1, unbiased=False)
    return HeadTables(
        rows=rows, attn_mass=attn_mass, max_prob=max_prob, omega0=omega0,
        omega1=omega1, proxy=proxy, mu=mu, sigma=sigma,
        valid_rows=layout.valid_count.index_select(0, rows).long(),
        kv_ok=layout.kv_ok.clone(), n_video_tiles=layout.n_video_tiles)


@torch.no_grad()
def relative_errors(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    lse: torch.Tensor, layout: tiling.TileLayout,
                    rows: torch.Tensor, head: int, whiten: torch.Tensor,
                    masks: dict[str, torch.Tensor],
                    compensate: Sequence[int] = (0, 1)
                    ) -> dict[tuple[str, int], float]:
    """Pass two: epsilon(M, c) of every mask, from one shared attention scan.

    Args:
        q: [S, H, D] packed queries.
        k: [S, H, D] packed keys.
        v: [S, H, D] packed values.
        lse: [S, H] fp32 dense log-sum-exp.
        layout: Tile permutation.
        rows: [R] int64 sampled video query tile ids.
        head: Head index.
        whiten: [D, D] fp32 L from `whitening`.
        masks: name -> [R, N] bool keep mask over all key tiles.
        compensate: Which c values to evaluate.

    Returns:
        (mask name, c) -> relative Frobenius error in W_o space.

    Raises:
        ValueError: If a mask has the wrong shape.
    """
    device = q.device
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    n_tiles = layout.n_tiles
    for name, m in masks.items():
        if tuple(m.shape) != (rows.numel(), n_tiles):
            raise ValueError(f'mask {name!r} has shape {tuple(m.shape)}, '
                             f'expected {(rows.numel(), n_tiles)}')
    q_t = _tile_order(q, layout, head)
    k_t = _tile_order(k, layout, head)
    v_t = _tile_order(v, layout, head)
    lse_t = _tile_order(lse[:, head, None], layout, 0).reshape(-1)
    slot_valid = layout.slot_valid.to(torch.float32)
    counts = layout.valid_count.to(torch.float32).clamp(min=1.0)
    k_mean = (k_t.view(n_tiles, TILE, head_dim).sum(1) / counts[:, None])
    v_white = v_t @ whiten
    v_sum_white = v_white.view(n_tiles, TILE, head_dim).sum(1)

    slots = _slots_of(rows)
    row_valid_all = layout.slot_valid.index_select(0, slots).to(torch.float32)
    keys = [(name, c) for name in masks for c in compensate]
    err = {key: torch.zeros((), dtype=torch.float64, device=device)
           for key in keys}
    norm = torch.zeros((), dtype=torch.float64, device=device)
    tiny = torch.finfo(torch.float32).tiny
    chunk = _chunk_rows(n_tiles, head_dim) * TILE
    for start in range(0, slots.numel(), chunk):
        sl = slots[start:start + chunk]
        q_c = q_t.index_select(0, sl)
        lse_c = lse_t.index_select(0, sl)
        rv = row_valid_all[start:start + chunk]
        # Query tile of each row in this chunk, to index the [R, N] masks.
        owner = torch.div(torch.arange(start, start + sl.numel(),
                                       device=device), TILE,
                          rounding_mode='floor')
        p = _probabilities(q_c, k_t, lse_c, scale, slot_valid, rv)
        pb = p.view(-1, n_tiles, TILE)
        a = pb.sum(-1)
        o_white = p @ v_white                                      # [U, D]
        s_hat = (q_c @ k_mean.transpose(0, 1)) * scale
        p_hat = torch.exp(s_hat - lse_c[:, None]) * rv[:, None]
        a_hat = p_hat * counts[None, :]
        norm += (o_white.square().sum(-1) * rv).double().sum()
        for name, mask in masks.items():
            keep = mask.index_select(0, owner)                     # [U, N]
            keep_f = keep.to(torch.float32)
            num = (p * keep_f.repeat_interleave(TILE, dim=1)) @ v_white
            den = (a * keep_f).sum(-1)
            drop = 1.0 - keep_f
            num_hat = (p_hat * drop) @ v_sum_white
            den_hat = (a_hat * drop).sum(-1)
            for c in compensate:
                o_sparse = ((num + c * num_hat)
                            / (den + c * den_hat).clamp(min=tiny)[:, None])
                diff = (o_sparse - o_white).square().sum(-1) * rv
                err[(name, c)] += diff.double().sum()
        del p, pb
    denom = norm.clamp(min=tiny).sqrt()
    return {key: float((val.sqrt() / denom).item())
            for key, val in err.items()}


# --- Budget allocation rules (section 3 of the validation plan) ----------
#
# Every rule returns a [R, N] bool keep mask over all key tiles, with the
# global tiles forced on. Rules are compared at the same *average* density
# inside one (layer, head, step), so that only the allocation across rows
# differs, never the total cost.


def video_budget(density: float, n_tiles: int, n_global: int) -> float:
    """Video key tiles per row at an overall keep `density`.

    Global key tiles are always kept and count against the budget, so the
    video quadrant gets whatever is left.

    Raises:
        ValueError: If the global tiles alone already exceed the density.
    """
    budget = density * n_tiles - n_global
    if budget < 1.0:
        raise ValueError(f'density {density} leaves {budget:.2f} video tiles '
                         f'per row after {n_global} global tiles')
    return budget


def _row_ranks(statistic: torch.Tensor,
               support: torch.Tensor) -> torch.Tensor:
    """[R, M] int64 rank of each entry within its row, best = 0.

    Entries outside `support` are ranked after every supported entry, so a
    rank-based budget never spends on them.
    """
    keyed = statistic.masked_fill(~support[None, :], float('-inf'))
    order = keyed.argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order,
                   torch.arange(order.shape[-1], device=order.device)
                   .expand_as(order))
    return ranks


def _with_globals(video_keep: torch.Tensor, kv_ok: torch.Tensor,
                  n_video: int) -> torch.Tensor:
    """[R, N] mask: the video decision, plus every real global tile."""
    rows = video_keep.shape[0]
    full = torch.zeros((rows, kv_ok.numel()), dtype=torch.bool,
                       device=video_keep.device)
    full[:, :n_video] = video_keep
    full[:, n_video:] = kv_ok[None, n_video:]
    return full


def fixed_topk_mask(scores: torch.Tensor, budget: float,
                    kv_ok: torch.Tensor, n_video: int,
                    rows: torch.Tensor) -> torch.Tensor:
    """R1 / oracle top-k: the same budget for every row.

    A fractional budget is spread over rows with Veda's Bresenham pattern,
    so the average density is exactly `budget / n_tiles` and the pattern
    follows the tile id, as in training.

    Args:
        scores: [R, N] any score monotone in importance.
        budget: Video key tiles per row (may be fractional).
        kv_ok: [N] bool, key tile has a real row.
        n_video: Key tiles in the video quadrant.
        rows: [R] int64 query tile ids (for the Bresenham phase).

    Returns:
        [R, N] bool keep mask.
    """
    support = kv_ok[:n_video]
    ranks = _row_ranks(scores[:, :n_video], support)
    k_lo, _, frac = veda_mask.split_budget(budget, int(support.sum()))
    extra = veda_mask.bresenham_extra(n_video, frac, scores.device)
    allowed = k_lo + extra.index_select(0, rows).long()
    keep = (ranks < allowed[:, None]) & support[None, :]
    return _with_globals(keep, kv_ok, n_video)


def threshold_mask(statistic: torch.Tensor, kv_ok: torch.Tensor,
                   n_video: int, threshold: float,
                   k_min: int | None = None,
                   k_max: int | None = None) -> torch.Tensor:
    """One global threshold on `statistic`, optionally clamped per row.

    This is the shape of every variable-budget rule: R2 thresholds the row
    z-score, R3 thresholds `proxy - mu - alpha * sigma^2 / 2`, and the
    oracle variable budget thresholds Omega itself.

    Args:
        statistic: [R, N] the quantity compared against `threshold`; only
            the first `n_video` columns are read.
        kv_ok: [N] bool.
        n_video: Key tiles in the video quadrant.
        threshold: Keep where `statistic > threshold`.
        k_min: Top up rows that fall below this many video tiles.
        k_max: Truncate rows above this many video tiles.

    Returns:
        [R, N] bool keep mask.
    """
    support = kv_ok[:n_video]
    stat = statistic[:, :n_video]
    keep = (stat > threshold) & support[None, :]
    if k_min is None and k_max is None:
        return _with_globals(keep, kv_ok, n_video)
    ranks = _row_ranks(stat, support)
    if k_min is not None:
        keep |= (ranks < k_min) & support[None, :]
    if k_max is not None:
        keep &= ranks < k_max
    return _with_globals(keep, kv_ok, n_video)


def calibrate_threshold(statistic: torch.Tensor, kv_ok: torch.Tensor,
                        n_video: int, budget: float) -> float:
    """Threshold whose average kept count per row is `budget`.

    The pooled (1 - rho) quantile of the supported entries, taken as the
    midpoint between the k-th and (k+1)-th largest so that a strict `>`
    keeps exactly k of them when there are no ties.

    Raises:
        ValueError: If the support is empty.
    """
    support = kv_ok[:n_video]
    values = statistic[:, :n_video][:, support].reshape(-1)
    total = values.numel()
    if total == 0:
        raise ValueError('empty video support')
    count = int(round(budget * statistic.shape[0]))
    count = min(max(count, 1), total)
    top = values.topk(min(count + 1, total)).values
    if top.numel() > count:
        return float(0.5 * (top[count - 1] + top[count]))
    return float(top[-1]) - 1.0  # Keep everything.


def zscore_statistic(tables: HeadTables) -> torch.Tensor:
    """[R, N] row z-score of the proxy, Sol's thresholded quantity."""
    sigma = tables.sigma.clamp(min=torch.finfo(torch.float32).tiny)
    return (tables.proxy - tables.mu[:, None]) / sigma[:, None]


def sigma_adaptive_statistic(tables: HeadTables,
                             alpha: float) -> torch.Tensor:
    """[R, N] `proxy - mu - alpha * sigma^2 / 2`, R3's thresholded quantity.

    Dropping the log-sum-exp of a Gaussian row leaves `mu + sigma^2 / 2` as
    the natural offset, which makes the optimal beta scale with sigma:
    sharp rows keep fewer blocks, flat rows hand more to the compensation.
    `alpha` is swept because block-mean scores have a smaller variance than
    token-level ones, so the coefficient need not be 1.
    """
    return (tables.proxy - tables.mu[:, None]
            - alpha * tables.sigma[:, None].square() / 2.0)


def allocation_masks(tables: HeadTables, density: float,
                     alphas: Sequence[float] = (0.5, 1.0, 2.0),
                     k_min: int | None = None, k_max: int | None = None,
                     oracle: str = 'Omega0') -> dict[str, torch.Tensor]:
    """The section 3.2 rules plus the 2x2 oracle row, all at `density`.

    Args:
        tables: Score tables of one (layer, head, step).
        density: Overall keep fraction of all key tiles per row.
        alphas: R3's sigma coefficients to sweep.
        k_min: Lower clamp on R3's per-row count.
        k_max: Upper clamp on R3's per-row count.
        oracle: Which Omega table forms the oracle row of the 2x2.

    Returns:
        Rule name -> [R, N] bool keep mask.
    """
    n_video, kv_ok = tables.n_video_tiles, tables.kv_ok
    budget = video_budget(density, tables.n_tiles, tables.n_tiles - n_video)
    omega = tables.score(oracle)
    masks = {
        'R1_topk': fixed_topk_mask(tables.proxy, budget, kv_ok, n_video,
                                   tables.rows),
        'oracle_topk': fixed_topk_mask(omega, budget, kv_ok, n_video,
                                       tables.rows),
    }
    z = zscore_statistic(tables)
    masks['R2_zscore'] = threshold_mask(
        z, kv_ok, n_video, calibrate_threshold(z, kv_ok, n_video, budget))
    for alpha in alphas:
        stat = sigma_adaptive_statistic(tables, alpha)
        thr = calibrate_threshold(stat, kv_ok, n_video, budget)
        masks[f'R3_sigma_a{alpha:g}'] = threshold_mask(
            stat, kv_ok, n_video, thr, k_min=k_min, k_max=k_max)
    masks['oracle_threshold'] = threshold_mask(
        omega, kv_ok, n_video,
        calibrate_threshold(omega, kv_ok, n_video, budget))
    # R2': one beta for the whole model, not calibrated per head.
    beta = _normal_quantile(1.0 - density)
    masks['R2g_zscore_global'] = threshold_mask(z, kv_ok, n_video, beta)
    return masks


def selection_masks(tables: HeadTables, density: float) -> dict[
        str, torch.Tensor]:
    """The H1 masks: per-row top-k under each of the four oracle scores."""
    n_video, kv_ok = tables.n_video_tiles, tables.kv_ok
    budget = video_budget(density, tables.n_tiles, tables.n_tiles - n_video)
    return {name: fixed_topk_mask(tables.score(name), budget, kv_ok,
                                  n_video, tables.rows)
            for name in ('A', 'Mx', 'Omega0', 'Omega1')}


def _normal_quantile(p: float) -> float:
    """Phi^{-1}(p) via the error function."""
    if not 0.0 < p < 1.0:
        raise ValueError(f'quantile out of range: {p}')
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item()


# --- Mechanism diagnostics (section 4) -----------------------------------


def kept_per_row(mask: torch.Tensor, n_video: int) -> torch.Tensor:
    """[R] int64 video key tiles kept by each row."""
    return mask[:, :n_video].sum(-1)


def load_imbalance(mask: torch.Tensor, n_video: int) -> float:
    """max / mean of the per-row kept count: the kernel's slowest block."""
    kept = kept_per_row(mask, n_video).float()
    return float(kept.max() / kept.mean().clamp(min=1e-30))


def row_moments(tables: HeadTables) -> dict[str, torch.Tensor]:
    """D1: per-row skewness and excess kurtosis of the proxy z-score."""
    support = tables.kv_ok[:tables.n_video_tiles]
    z = zscore_statistic(tables)[:, :tables.n_video_tiles][:, support]
    return {'skew': z.pow(3).mean(-1),
            'excess_kurtosis': z.pow(4).mean(-1) - 3.0}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman correlation of two 1-D tensors (ties broken by order)."""
    if a.numel() < 2:
        return float('nan')
    ranks = []
    for x in (a, b):
        order = x.argsort(stable=True)
        r = torch.empty_like(order, dtype=torch.float32)
        r.scatter_(0, order, torch.arange(x.numel(), dtype=torch.float32,
                                          device=x.device))
        ranks.append(r - r.mean())
    num = (ranks[0] * ranks[1]).sum()
    den = ranks[0].norm() * ranks[1].norm()
    return float(num / den.clamp(min=1e-30))


def proxy_omega_spearman(tables: HeadTables, zones: Sequence[
        tuple[float, float]] = ((-1.0, 1.0), (1.0, float('inf'))),
        oracle: str = 'Omega0') -> dict[str, float]:
    """D2: Spearman of the proxy against Omega inside z-score zones.

    The claim under test: in the bulk the block-mean proxy's ranking is
    close to noise, while in the right tail it is informative. If so, a
    fixed top-k wastes budget filling k from the bulk.
    """
    support = tables.kv_ok[:tables.n_video_tiles]
    z = zscore_statistic(tables)[:, :tables.n_video_tiles]
    proxy = tables.proxy[:, :tables.n_video_tiles]
    omega = tables.score(oracle)[:, :tables.n_video_tiles]
    out = {}
    for lo, hi in zones:
        sel = (z > lo) & (z <= hi) & support[None, :]
        out[f'z({lo:g},{hi:g}]'] = _spearman(proxy[sel], omega[sel])
    return out


def dropped_mass(tables: HeadTables, mask: torch.Tensor) -> torch.Tensor:
    """D5: [R] attention mass the mask discards, per query tile."""
    return (tables.attn_mass * (~mask).to(tables.attn_mass.dtype)).sum(-1)


# --- Driver: one AttentionFn that ablates while the teacher rolls dense ---


@dataclasses.dataclass(frozen=True)
class Record:
    """Everything measured for one (clip, step, layer, head, density).

    Attributes:
        clip: Sample id.
        step: Denoising step.
        layer: Layer index.
        head: Head index.
        density: Overall keep fraction the rules were calibrated to.
        shape: Tile shape of this head, as a string.
        errors: 'mask|c' -> relative error in W_o space.
        kept_mean: 'mask' -> mean video key tiles kept per row.
        kept_max_over_mean: 'mask' -> load imbalance.
        spearman: z-score zone -> Spearman of proxy against Omega0.
        skew: Mean per-row skewness of the proxy z-score.
        excess_kurtosis: Mean per-row excess kurtosis.
        dropped_mass_r1: Mean attention mass R1 discards.
    """

    clip: str
    step: int
    layer: int
    head: int
    density: float
    shape: str
    errors: dict[str, float]
    kept_mean: dict[str, float]
    kept_max_over_mean: dict[str, float]
    spearman: dict[str, float]
    skew: float
    excess_kurtosis: float
    dropped_mass_r1: float

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AblationConfig:
    """Knobs of the offline ablation.

    Attributes:
        densities: Overall keep fractions to sweep.
        alphas: R3's sigma coefficients.
        query_tiles: Video query tiles sampled per (layer, head).
        k_min: Lower clamp on R3's per-row count; None disables it.
        k_max: Upper clamp on R3's per-row count; None disables it.
        oracle: Which Omega table forms the oracle row of the 2x2.
        heads: Heads to measure; None means every head.
    """

    densities: Sequence[float] = (0.05, 0.1, 0.2)
    alphas: Sequence[float] = (0.5, 1.0, 2.0)
    query_tiles: int = 8
    k_min: int | None = None
    k_max: int | None = None
    oracle: str = 'Omega0'
    heads: Sequence[int] | None = None


def head_report(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                lse: torch.Tensor, layout: tiling.TileLayout,
                rows: torch.Tensor, head: int, whiten: torch.Tensor,
                config: AblationConfig) -> list[dict]:
    """Both hypotheses for one head, one entry per density.

    Args:
        q: [S, H, D] packed queries.
        k: [S, H, D] packed keys.
        v: [S, H, D] packed values.
        lse: [S, H] fp32 dense log-sum-exp.
        layout: Tile permutation of this head.
        rows: [R] int64 sampled video query tiles.
        head: Head index.
        whiten: [D, D] fp32 L from `whitening`.
        config: Ablation knobs.

    Returns:
        One dict per density, with the fields `Record` expects.
    """
    tables = head_tables(q, k, v, lse, layout, rows, head, whiten)
    moments = row_moments(tables)
    spearman = proxy_omega_spearman(tables, oracle=config.oracle)
    out = []
    for density in config.densities:
        masks = selection_masks(tables, density)
        masks.update(allocation_masks(tables, density, config.alphas,
                                      config.k_min, config.k_max,
                                      config.oracle))
        errors = relative_errors(q, k, v, lse, layout, rows, head, whiten,
                                 masks)
        out.append({
            'density': float(density),
            'errors': {f'{name}|{c}': value
                       for (name, c), value in errors.items()},
            'kept_mean': {
                name: float(kept_per_row(m, tables.n_video_tiles)
                            .float().mean())
                for name, m in masks.items()},
            'kept_max_over_mean': {
                name: load_imbalance(m, tables.n_video_tiles)
                for name, m in masks.items()},
            'spearman': spearman,
            'skew': float(moments['skew'].mean()),
            'excess_kurtosis': float(moments['excess_kurtosis'].mean()),
            # Per-token mean, so tiles with fewer real rows do not skew it.
            'dropped_mass_r1': float(
                (dropped_mass(tables, masks['R1_topk'])
                 / tables.valid_rows.clamp(min=1)).mean()),
        })
    return out


def _dense_weight(weight: torch.Tensor) -> torch.Tensor:
    """The out_proj weight, refusing a sharded parameter.

    Raises:
        ValueError: If the weight is a DTensor. Gathering it inside the
            attention callback would be a collective, and ranks of this
            ablation process different clips, so they would not meet.
    """
    if type(weight).__name__ == 'DTensor':
        raise ValueError('run the ablation in a single process: the output '
                         'projection is sharded, and gathering it inside '
                         'the attention callback would deadlock')
    return weight


class AblationScorer:
    """AttentionFn that returns dense attention and ablates each head.

    Only the layers of the steps being measured should use this function;
    the trajectory itself must stay on the dense teacher, or the ablation
    would grade a sparse model with itself.
    """

    def __init__(self, layout, plan, clip, config: AblationConfig,
                 out_proj: 'callable', clip_id: str, step: int, seed: int,
                 dense_backend: str = 'auto'):
        """
        Args:
            layout: Packed layout of the clip.
            plan: `veda.plan.TilePlan` giving each head's tile shape.
            clip: `veda.attention.ClipTiling` of this clip.
            config: Ablation knobs.
            out_proj: layer index -> [hidden, num_heads * head_dim] weight.
            clip_id: Sample id, recorded with every row.
            step: Denoising step, recorded with every row.
            seed: Query-tile sampling seed.
            dense_backend: Backend of the dense attention.
        """
        self.layout = layout
        self.plan = plan
        self.clip = clip
        self.config = config
        self.out_proj = out_proj
        self.clip_id = clip_id
        self.step = step
        self.seed = seed
        self.dense_backend = dense_backend
        self.records: list[Record] = []
        self.num_layers = plan.num_layers
        self._progress = None

    def _rows(self, tile_layout: tiling.TileLayout,
              layer_index: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.seed * 1000 + layer_index)
        count = min(self.config.query_tiles, tile_layout.n_video_tiles)
        rows = torch.randperm(tile_layout.n_video_tiles, generator=gen)
        return rows[:count].sort().values.to(self.clip.device)

    @torch.no_grad()
    def __call__(self, q, k, v, layer_index):
        if self._progress is None:
            self._progress = progress.Progress(
                f'  step {self.step}: ablating {q.shape[1]} heads x '
                f'{len(self.config.densities)} densities over layers',
                total=self.num_layers, every=1)
        out, lse = h3_attention.dense_attention(
            q, k, v, self.layout.used, return_lse=True,
            backend=self.dense_backend)
        weight = _dense_weight(self.out_proj(layer_index))
        head_dim = q.shape[-1]
        wanted = (set(self.config.heads) if self.config.heads is not None
                  else None)
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            rows = self._rows(tile_layout, layer_index)
            for head in group.heads.tolist():
                if wanted is not None and head not in wanted:
                    continue
                whiten = whitening(weight, head, head_dim)
                for entry in head_report(q, k, v, lse.float(), tile_layout,
                                         rows, head, whiten, self.config):
                    self.records.append(Record(
                        clip=self.clip_id, step=self.step,
                        layer=layer_index, head=head,
                        shape=str(group.shape), **entry))
        self._progress.update(f'layer {layer_index}: '
                              f'{len(self.records)} rows')
        return out


# --- Aggregation and the two gates (section 6) ---------------------------


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize(records: Sequence[Record], eta: float = 0.15,
              g2_margin: float = 0.05) -> list[dict]:
    """Per-density aggregate of the ablation, with both gate verdicts.

    G1 (selection target): the fraction of (layer, head, step) where
    `eps(Omega0) <= (1 - eta) * eps(Mx)` at c = 0. Below a majority, the
    selection target is not the bottleneck and swapping Veda's
    distillation target is not worth training.
    G2 (budget allocation): the relative gap between the oracle's fixed
    top-k and the oracle's global threshold. Under `g2_margin` a variable
    budget has little to offer even with a perfect score, so R2 and R3 need
    no further comparison.

    Args:
        records: Rows from the ablation run.
        eta: Required relative improvement for G1.
        g2_margin: Gap under which G2 fails.

    Returns:
        One dict per density, sorted by density.
    """
    by_density: dict[float, list[Record]] = {}
    for record in records:
        by_density.setdefault(record.density, []).append(record)
    out = []
    for density in sorted(by_density):
        rows = by_density[density]
        names = sorted({n for r in rows for n in r.errors})
        medians = {n: _median([r.errors[n] for r in rows if n in r.errors])
                   for n in names}
        g1_hits = [r for r in rows
                   if 'Omega0|0' in r.errors and 'Mx|0' in r.errors
                   and r.errors['Omega0|0'] <= (1 - eta) * r.errors['Mx|0']]
        gaps = [(r.errors['oracle_topk|0'] - r.errors['oracle_threshold|0'])
                / max(r.errors['oracle_topk|0'], 1e-30)
                for r in rows if 'oracle_topk|0' in r.errors
                and 'oracle_threshold|0' in r.errors]
        g2_gap = _median(gaps) if gaps else float('nan')
        out.append({
            'density': density,
            'heads': len(rows),
            'median_error': medians,
            'g1_pass_fraction': len(g1_hits) / max(1, len(rows)),
            'g1_pass': len(g1_hits) > len(rows) / 2,
            'g2_oracle_gap': g2_gap,
            'g2_pass': bool(g2_gap > g2_margin),
            'median_spearman': {
                zone: _median([r.spearman[zone] for r in rows
                               if zone in r.spearman
                               and not math.isnan(r.spearman[zone])])
                for zone in sorted({z for r in rows for z in r.spearman})},
            'median_skew': _median([r.skew for r in rows]),
            'median_excess_kurtosis': _median(
                [r.excess_kurtosis for r in rows]),
            'median_load_imbalance': {
                n: _median([r.kept_max_over_mean[n] for r in rows
                            if n in r.kept_max_over_mean])
                for n in sorted({n for r in rows
                                 for n in r.kept_max_over_mean})},
        })
    return out


def save_records(path: str, records: Sequence[Record], meta: dict) -> None:
    """Atomically writes the rows and their summary next to the metadata."""
    payload = {'meta': meta,
               'records': [r.to_json() for r in records],
               'summary': summarize(records)}
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=1)
    os.replace(tmp, path)


def load_records(path: str) -> tuple[list[Record], dict]:
    """Reads back what `save_records` wrote."""
    with open(path) as handle:
        payload = json.load(handle)
    return [Record(**row) for row in payload['records']], payload['meta']


# --- Run driver (single process; the ablation is cheap) ------------------


@dataclasses.dataclass
class RunConfig:
    """One ablation run (YAML keys have the same names).

    Attributes:
        run_name: Output directory name under out_dir.
        checkpoint_root: Checkpoint root containing FL2VA/ and Ref2VA/.
        sample_cache: Encoded prompts to measure (test split excluded).
        geometries: Geometry specs measured one after another with the
            model loaded once, e.g. ['16:9@37'].
        variant: 'FL2VA' or 'Ref2VA'.
        tasks: Cached sample tasks to draw from.
        num_clips: Prompts per geometry.
        steps: Denoising steps to measure (early, two middle, late).
        schedule / num_steps: 'base' + 49 or 'turbo' + 4 / 8.
        teacher_adapter: Few-step LoRA merged into the teacher (turbo).
        plan: Tile plan json; None uses `tile_shape` for every head.
        tile_shape: Shape used when `plan` is None.
        densities / alphas / query_tiles / k_min / k_max / oracle / heads:
            See `AblationConfig`.
        eta: G1's required relative improvement.
        g2_margin: G2's minimum oracle gap.
        seed: Query-tile sampling seed.
        out_dir: Root of run directories.
        mlp_chunk_rows / offload_blocks / prefetch: Memory knobs.
        dense_backend: Backend of the dense attention.
    """

    run_name: str
    checkpoint_root: str
    sample_cache: str
    geometries: list[str]
    variant: str = 'FL2VA'
    tasks: list[str] = dataclasses.field(default_factory=lambda: ['t2va'])
    num_clips: int = 4
    steps: list[int] = dataclasses.field(
        default_factory=lambda: [0, 12, 25, 40])
    schedule: str = 'base'
    num_steps: int = 49
    teacher_adapter: str | None = None
    plan: str | None = None
    tile_shape: str = '4x4x8'
    densities: list[float] = dataclasses.field(
        default_factory=lambda: [0.05, 0.1, 0.2])
    alphas: list[float] = dataclasses.field(
        default_factory=lambda: [0.5, 1.0, 2.0])
    query_tiles: int = 8
    k_min: int | None = None
    k_max: int | None = None
    oracle: str = 'Omega0'
    heads: list[int] | None = None
    eta: float = 0.15
    g2_margin: float = 0.05
    seed: int = 0
    out_dir: str = 'runs'
    mlp_chunk_rows: int | None = None
    offload_blocks: int = 0
    prefetch: int = 1
    dense_backend: str = 'auto'

    def ablation(self) -> AblationConfig:
        return AblationConfig(
            densities=tuple(self.densities), alphas=tuple(self.alphas),
            query_tiles=self.query_tiles, k_min=self.k_min,
            k_max=self.k_max, oracle=self.oracle,
            heads=tuple(self.heads) if self.heads is not None else None)


def ablate_clip(model, cache, sample, geometry, schedule, tables,
                config: RunConfig, plan, device) -> list[Record]:
    """Rolls one clip with the dense teacher, ablating the chosen steps."""
    from miowtion.h3 import model as h3_model  # pylint: disable=import-outside-toplevel
    from miowtion.train import trajectory  # pylint: disable=import-outside-toplevel

    traj = trajectory.Trajectory(model, cache, sample, geometry, schedule,
                                 config.seed, device)
    veda_config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=max(config.densities)))
    clip_tiling = veda_attention.ClipTiling(traj.layout, veda_config, device)
    ablation = config.ablation()
    records: list[Record] = []
    last = max(config.steps)
    progress.log(f'clip {sample.id}: {traj.layout.used} tokens, rolling '
                 f'steps 0..{last} (ablating {list(config.steps)})')
    steps = progress.Progress(f'clip {sample.id}: denoise steps', last + 1)
    while not traj.done and traj.step <= last:
        inputs = traj.inputs()
        if traj.step in config.steps:
            attention_fn = AblationScorer(
                traj.layout, plan, clip_tiling, ablation,
                out_proj=lambda i: model.blocks[i].attn.out_proj.weight,
                clip_id=sample.id, step=traj.step,
                seed=config.seed + traj.step,
                dense_backend=config.dense_backend)
        else:
            attention_fn = h3_model.DenseAttention(traj.layout.used,
                                                   config.dense_backend)
        with torch.no_grad():
            video_v, audio_v = model(traj.clip, inputs.video_rows,
                                     inputs.audio_rows, inputs.timestep,
                                     attention_fn,
                                     tables.get(inputs.timestep.timesteps))
        if isinstance(attention_fn, AblationScorer):
            records += attention_fn.records
        steps.update(f'step {traj.step} '
                     f'{"ablated" if traj.step in config.steps else "dense"}')
        traj.advance(video_v, audio_v)
    return records


def run_ablation(config: RunConfig) -> None:
    """Measures both hypotheses over the configured geometries and clips.

    Raises:
        ValueError: If a geometry has fewer cached samples than num_clips,
            or if the run is launched with more than one rank.
    """
    from miowtion.train import data  # pylint: disable=import-outside-toplevel
    from miowtion.train import parallel  # pylint: disable=import-outside-toplevel
    from miowtion.train import teacher  # pylint: disable=import-outside-toplevel

    env = parallel.init_distributed()
    if env.world_size != 1:
        raise ValueError('the ablation reads unsharded output projections; '
                         f'launch it with one rank, not {env.world_size}')
    teacher_ = teacher.build_teacher(
        config.checkpoint_root, config.variant, config.schedule,
        config.num_steps, config.teacher_adapter, env,
        visual_conditions=config.variant == 'Ref2VA'
        or 'fl2va' in config.tasks,
        audio_references=config.variant == 'Ref2VA',
        offload_blocks=config.offload_blocks, prefetch=config.prefetch,
        mlp_chunk_rows=config.mlp_chunk_rows)
    model, schedule, tables = (teacher_.model, teacher_.schedule,
                               teacher_.tables)
    model.dense_backend = config.dense_backend
    cache = data.SampleCache(config.sample_cache)
    out_dir = os.path.join(config.out_dir, config.run_name)
    os.makedirs(out_dir, exist_ok=True)
    progress.log(f'Sol-Attn ablation over {config.geometries}, densities '
                 f'{config.densities}, steps {config.steps}, '
                 f'{config.num_clips} clips, output {out_dir}')
    for spec in config.geometries:
        geometry = data.parse_geometry(spec)
        samples = [s for s in cache.select('train', config.tasks)
                   if s.aspect in (None, geometry.aspect)
                   and s.latent_t in (None, geometry.latent_t)]
        samples = samples[:config.num_clips]
        if len(samples) < config.num_clips:
            raise ValueError(f'only {len(samples)} samples for '
                             f'{geometry.name}')
        if config.plan:
            plan = veda_plan.TilePlan.load(config.plan)
        else:
            plan = veda_plan.TilePlan.uniform(
                geometry, tiling.TileShape.parse(config.tile_shape),
                model.config.num_layers, model.config.num_heads)
        path = os.path.join(out_dir, f'{geometry.name}.json')
        if os.path.exists(path):
            progress.log(f'{geometry.name}: already measured, skipped')
            continue
        records: list[Record] = []
        clips = progress.Progress(f'{geometry.name}: clips', len(samples))
        for sample in samples:
            records += ablate_clip(model, cache, sample, geometry, schedule,
                                   tables, config, plan, env.device)
            clips.update(f'{sample.id}: {len(records)} rows so far')
        save_records(path, records, {
            'geometry': geometry.name,
            'grid': list(geometry.video_grid),
            'densities': list(config.densities),
            'alphas': list(config.alphas),
            'steps': list(config.steps),
            'query_tiles': config.query_tiles,
            'oracle': config.oracle,
            'schedule': config.schedule,
            'num_steps': config.num_steps,
            'teacher_adapter': config.teacher_adapter,
            'plan': config.plan or f'uniform {config.tile_shape}',
            'variant': config.variant,
            'clips': [s.id for s in samples],
        })
        progress.log(f'saved {path} ({len(records)} rows)')
        for row in summarize(records, config.eta, config.g2_margin):
            progress.log(
                f"  density {row['density']:.2f}: G1 "
                f"{'pass' if row['g1_pass'] else 'FAIL'} "
                f"({row['g1_pass_fraction']:.0%} of heads), G2 "
                f"{'pass' if row['g2_pass'] else 'FAIL'} "
                f"(oracle gap {row['g2_oracle_gap']:+.1%})")
