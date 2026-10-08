"""Offline ablation of block-selection targets and per-row key budgets.

Two hypotheses are checked on a frozen backbone, from exported q / k / v
only: no training and no kernel work.

H1 (selection target): picking blocks by their contribution to the output
error beats picking them by attention quality (what Veda's scorer is
distilled against today).
H2 (budget allocation): a per-row variable budget beats a fixed top-k.

Both reduce to one exact identity. For query token `u` in query tile `i`,
with the kept set `S_i`, the dropped set `U_i` and `c` in {0, 1} selecting
whether Sol's zero-order compensation is applied to the dropped blocks
(c = 2 is the equal-cost upper bound on it, see `relative_errors`):

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


# The log mass of a block is a cumulant expansion of the pooled score:
#   log E_uj = log B_j + s_mean + 1/2 var + ...,
# with s_mean = q_u . Kbar_j / sqrt(d) and var = q_u^T Cov_j q_u / d.
# Veda's predictor, before training, computes exactly `s_mean` and nothing
# else, so the whole second-order term is signal it cannot express. These
# variants measure how much of it matters.


@torch.no_grad()
def _proxy_variants(q_t: torch.Tensor, k_t: torch.Tensor,
                    q_mean: torch.Tensor, k_mean: torch.Tensor,
                    counts: torch.Tensor, layout: tiling.TileLayout,
                    rows: torch.Tensor, scale: float,
                    head_dim: int) -> dict[str, torch.Tensor]:
    """[R, N] proxy score variants; global columns stay NaN.

    Args:
        q_t: [N * TILE, D] fp32 tile-ordered queries, padding zeroed.
        k_t: [N * TILE, D] fp32 tile-ordered keys.
        q_mean: [n_video, D] per-tile mean query.
        k_mean: [N, D] per-tile mean key.
        counts: [N] fp32 real rows per tile, clamped to 1.
        layout: Tile layout.
        rows: [R] int64 sampled video query tiles.
        scale: 1 / sqrt(head_dim).
        head_dim: D.

    Returns:
        'proxy', 'proxy_logb', 'proxy_diag', 'proxy_full'.
    """
    n_tiles = layout.n_tiles
    n_video = layout.n_video_tiles
    device = q_t.device

    def blank() -> torch.Tensor:
        return torch.full((rows.numel(), n_tiles), float('nan'),
                          device=device)

    qm = q_mean.index_select(0, rows)                             # [R, D]
    km = k_mean[:n_video]                                         # [V, D]
    zeroth = (qm @ km.transpose(0, 1)) * scale
    log_b = torch.log(counts[:n_video])[None, :]

    # Second moments. The query side needs E[q q^T] over the tile, the key
    # side the central covariance of the block.
    qt = q_t.view(n_tiles, TILE, head_dim)
    kt = k_t.view(n_tiles, TILE, head_dim)
    q_sq = (qt.square().sum(1) / counts[:, None]).index_select(0, rows)
    k_sq = kt.square().sum(1) / counts[:, None]
    k_var = (k_sq - k_mean.square())[:n_video]                    # [V, D]
    diag_term = 0.5 * scale * scale * (q_sq @ k_var.transpose(0, 1))

    # Full term: <E[q q^T], Cov_j> is a rank D^2 bilinear, so it is one
    # matmul over flattened outer products.
    q_outer = torch.einsum('nbi,nbj->nij', qt, qt) / counts[:, None, None]
    q_outer = q_outer.index_select(0, rows).reshape(rows.numel(), -1)
    k_outer = torch.einsum('nbi,nbj->nij', kt, kt) / counts[:, None, None]
    k_outer = k_outer - k_mean[:, :, None] * k_mean[:, None, :]
    k_outer = k_outer[:n_video].reshape(n_video, -1)
    full_term = 0.5 * scale * scale * (q_outer @ k_outer.transpose(0, 1))

    out = {}
    for name, video in (('proxy', zeroth),
                        ('proxy_logb', zeroth + log_b),
                        ('proxy_diag', zeroth + log_b + diag_term),
                        ('proxy_full', zeroth + log_b + full_term)):
        table = blank()
        table[:, :n_video] = video
        out[name] = table
    return out


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
        proxy: s_tilde_ij, the mean-pooled proxy score Sol thresholds,
            which is also what Veda's predictor computes before training.
        proxy_logb: `proxy` plus log of the block's real row count, the
            zero-order estimate of the block's log mass.
        proxy_diag: `proxy_logb` plus the diagonal second-order cumulant
            term. One extra pooled feature per side, so the predictor's
            n^2 bilinear stays at rank head_dim.
        proxy_full: `proxy_logb` plus the full second-order term. Needs a
            rank head_dim^2 bilinear, far too expensive to serve; it is
            the ceiling of any second-order correction.
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
    proxy_logb: torch.Tensor
    proxy_diag: torch.Tensor
    proxy_full: torch.Tensor
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
                 'proxy': self.proxy, 'proxy_logb': self.proxy_logb,
                 'proxy_diag': self.proxy_diag,
                 'proxy_full': self.proxy_full}.get(name)
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
    variants = _proxy_variants(q_t, k_t, q_mean, k_mean, counts, layout,
                               rows, scale, head_dim)
    proxy = variants['proxy']
    support = layout.kv_ok[:layout.n_video_tiles]
    sel = proxy[:, :layout.n_video_tiles][:, support]
    mu = sel.mean(-1)
    sigma = sel.std(-1, unbiased=False)
    return HeadTables(
        rows=rows, attn_mass=attn_mass, max_prob=max_prob, omega0=omega0,
        omega1=omega1, proxy=proxy,
        proxy_logb=variants['proxy_logb'],
        proxy_diag=variants['proxy_diag'],
        proxy_full=variants['proxy_full'], mu=mu, sigma=sigma,
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

    Sol's compensation factors into a guessed mass times the block's mean
    value, `Nhat_uj = exp(q_u . Kbar_j / sqrt d) * Vhat_j`. So c = 2
    replaces only the guess by the true mass `E_uj`, keeping the block-mean
    direction. It costs exactly what c = 1 costs (one scalar per dropped
    block) and is therefore the equal-cost ceiling on *any* predictor of
    the correction weight, trained or not. Its denominator is exact, so
    all of its remaining error is direction inside the dropped blocks,
    which a per-block scalar cannot reach.

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
        compensate: Which c values to evaluate: 0 discards, 1 is Sol's
            zero-order compensation, 2 is its equal-cost ceiling.

    Returns:
        (mask name, c) -> relative Frobenius error in W_o space.

    Raises:
        ValueError: If a mask has the wrong shape, or c is not 0, 1 or 2.
    """
    if any(c not in (0, 1, 2) for c in compensate):
        raise ValueError(f'compensate must be 0, 1 or 2: {compensate}')
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
    # c = 2 weights the same direction by the true mass, so it needs the
    # block mean rather than the block sum.
    v_mean_white = v_sum_white / counts[:, None]

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
            # c = 1: guessed mass. c = 2: true mass, same direction.
            extra = {
                1: ((p_hat * drop) @ v_sum_white, (a_hat * drop).sum(-1)),
                2: ((a * drop) @ v_mean_white, (a * drop).sum(-1)),
            }
            for c in compensate:
                if c == 0:
                    num_c, den_c = num, den
                else:
                    num_x, den_x = extra[c]
                    num_c, den_c = num + num_x, den + den_x
                o_sparse = num_c / den_c.clamp(min=tiny)[:, None]
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


def scorer_masks(tables: HeadTables, density: float) -> dict[
        str, torch.Tensor]:
    """Per-row top-k under each servable score, at one density.

    `proxy` is what Veda's predictor computes before training, so these
    bracket what a better *feature set* could buy without changing the
    kernel or the budget rule. `proxy_diag` costs one more pooled feature
    per side; `proxy_full` is not servable and marks the ceiling of any
    second-order correction.
    """
    n_video, kv_ok = tables.n_video_tiles, tables.kv_ok
    budget = video_budget(density, tables.n_tiles, tables.n_tiles - n_video)
    return {f'{name}_topk': fixed_topk_mask(tables.score(name), budget,
                                            kv_ok, n_video, tables.rows)
            for name in ('proxy', 'proxy_logb', 'proxy_diag', 'proxy_full')}


def recall_against(mask: torch.Tensor, reference: torch.Tensor,
                   n_video: int) -> float:
    """Fraction of the reference's kept video tiles that `mask` also keeps.

    The forced diagonal is not excluded here, unlike the training-time
    recall: these masks have no forced diagonal, so there is nothing free
    to remove.
    """
    hit = (mask[:, :n_video] & reference[:, :n_video]).sum().double()
    total = reference[:, :n_video].sum().double().clamp(min=1)
    return float(hit / total)


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
        recall_vs_omega: 'mask' -> overlap with the ideal-importance set.
        recall_vs_mass: 'mask' -> overlap with the block-mass set.
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
    # Added after the first runs; defaulted so their files still load.
    recall_vs_omega: dict[str, float] = dataclasses.field(
        default_factory=dict)
    recall_vs_mass: dict[str, float] = dataclasses.field(
        default_factory=dict)

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
        budget_rules: Include H2's variable-budget rules. Turn them off
            once that question is settled: they are most of the pass-two
            cost and none of the remaining signal.
    """

    densities: Sequence[float] = (0.05, 0.1, 0.2)
    alphas: Sequence[float] = (0.5, 1.0, 2.0)
    query_tiles: int = 8
    k_min: int | None = None
    k_max: int | None = None
    oracle: str = 'Omega0'
    heads: Sequence[int] | None = None
    budget_rules: bool = True


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
        if config.budget_rules:
            masks.update(allocation_masks(tables, density, config.alphas,
                                          config.k_min, config.k_max,
                                          config.oracle))
        masks.update(scorer_masks(tables, density))
        errors = relative_errors(q, k, v, lse, layout, rows, head, whiten,
                                 masks, compensate=(0, 1, 2))
        n_video = tables.n_video_tiles
        out.append({
            'density': float(density),
            'recall_vs_omega': {
                name: recall_against(m, masks[config.oracle], n_video)
                for name, m in masks.items()},
            'recall_vs_mass': {
                name: recall_against(m, masks['A'], n_video)
                for name, m in masks.items()},
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
            # proxy_topk is R1_topk by another name and is always present.
            'dropped_mass_r1': float(
                (dropped_mass(tables, masks['proxy_topk'])
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


# --- Predictor initialization probe --------------------------------------
#
# The ablation above scores block-score *tables*. This probe scores actual
# predictor modules, in the metrics the stage-1 training loop already
# watches (recall, heat_kept), so a claim about the predictor's starting
# point can be checked without training anything.
#
# At initialization every projection is N(0, 1e-4), so the parameters are
# effectively layer-independent and one LayerPredictor stands in for all
# of them. That is why this costs one teacher rollout rather than a
# 275M-parameter model per variant.


def init_variants(head_dim: int,
                  device: torch.device | str = 'cpu') -> dict[str, 'object']:
    """The predictor initializations Veda2 compares, one layer each.

    Args:
        head_dim: Head dimension of the model under probe.
        device: Where to put them; must match the activations, since the
            per-head index_select happens on the device.

    Returns:
        'veda1' is today's predictor, whose untrained score is mean-pooled
        QK, the zero-order term of log block mass. 'veda2' adds the
        low-rank second-order head at full rank with the exact warm start,
        so its untrained score is the complete diagonal estimate.
    """
    from miowtion.veda import predictor as veda_predictor  # pylint: disable=import-outside-toplevel
    out = {}
    for name, rank in (('veda1', 0), ('veda2', head_dim)):
        torch.manual_seed(0)
        pred = veda_predictor.TileScorePredictor(1, _PROBE_HEADS, head_dim,
                                                 second_order_rank=rank)
        if rank:
            pred.init_exact_second_order_()
        out[name] = pred.to(device)
    return out


# The probe's stand-in predictor needs at least as many heads as the model.
_PROBE_HEADS = 128


class PredictorInitProbe:
    """AttentionFn scoring predictor initializations against the teacher.

    Only the layers being probed should use this function; the trajectory
    itself stays on the dense teacher.
    """

    def __init__(self, layout, plan, clip, variants: dict,
                 query_tiles: int, seed: int, step: int,
                 targets: Sequence[str] = ('max', 'sum'),
                 dense_backend: str = 'auto'):
        """
        Args:
            layout: Packed layout of the clip.
            plan: `veda.plan.TilePlan` giving each head's tile shape.
            clip: `veda.attention.ClipTiling` of this clip.
            variants: From `init_variants`.
            query_tiles: Video query tiles supervised per layer.
            seed: Query-tile sampling seed.
            step: Denoising step, recorded with every row.
            targets: Teacher heat reductions to score against.
            dense_backend: Backend of the dense attention.
        """
        self.layout = layout
        self.plan = plan
        self.clip = clip
        self.variants = variants
        self.query_tiles = query_tiles
        self.seed = seed
        self.step = step
        self.targets = tuple(targets)
        self.dense_backend = dense_backend
        self.rows: list[dict] = []

    @torch.no_grad()
    def __call__(self, q, k, v, layer_index):
        from miowtion.veda import heatmap as veda_heatmap  # pylint: disable=import-outside-toplevel
        out, lse = h3_attention.dense_attention(
            q, k, v, self.layout.used, return_lse=True,
            backend=self.dense_backend)
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            blocks = self.clip.blocks(tile_layout)
            gen = torch.Generator().manual_seed(self.seed * 1000
                                                + layer_index)
            count = min(self.query_tiles, tile_layout.n_video_tiles)
            rows = torch.randperm(tile_layout.n_video_tiles,
                                  generator=gen)[:count]
            rows = rows.sort().values.to(q.device)
            q_t, k_t = (tiling.gather_tiles(t, tile_layout, group.heads)
                        for t in (q, k))
            lse_t = lse.index_select(0, tile_layout.gather_index)
            lse_t = lse_t.index_select(1, group.heads).contiguous()
            if tile_layout.pad_slots.numel():
                lse_t.index_fill_(0, tile_layout.pad_slots, 0.0)
            logits = {}
            for name, pred in self.variants.items():
                # A one-layer stand-in answers for every layer; a trained
                # bundle has all of them. Same for the head axis: the
                # stand-in is built wide so any real head indexes into it.
                layer = min(layer_index, len(pred.layers) - 1)
                heads = group.heads.clamp(
                    max=pred.layers[layer].proj_q.shape[0] - 1)
                full = pred.scores(layer, q_t, k_t, tile_layout, heads)
                logits[name] = full.index_select(1, rows).contiguous()
            # Spread of the scores over the video quadrant. Terms added
            # to a *trained* base have to compete with it, so comparing
            # this across variants is how a scale mismatch shows up.
            spread = {}
            for name, score in logits.items():
                video = score[:, :, :tile_layout.n_video_tiles]
                spread[name] = float(video.float().std())
            for target in self.targets:
                heat = veda_heatmap.teacher_heat(q_t, k_t, lse_t,
                                                 tile_layout, rows,
                                                 reduce=target)
                for name, score in logits.items():
                    stats = veda_heatmap.mask_diagnostics(
                        score, heat, tile_layout, blocks, rows)
                    self.rows.append({
                        'step': self.step, 'layer': layer_index,
                        'variant': name, 'target': target,
                        'logit_std': spread[name],
                        **{key: float(value) for key, value
                           in stats.items()}})
        return out


def init_probe_by_clip(rows: Sequence[dict],
                       target: str = 'sum') -> list[dict]:
    """Per-clip hardness and per-variant kept share, hardest first.

    `heat_ceiling` is the share of block heat the *oracle* keeps at this
    budget, so a low value means the teacher's attention is not
    concentrated enough for the budget and no predictor can do well. That
    is the measurable definition of a hard clip.
    """
    clips: dict[str, list[dict]] = {}
    for row in rows:
        if row.get('target') == target and 'clip' in row:
            clips.setdefault(row['clip'], []).append(row)
    out = []
    for clip, group in clips.items():
        kept = {}
        for row in group:
            kept.setdefault(row['variant'], []).append(row['heat_kept'])
        out.append({
            'clip': clip,
            'heat_ceiling': _median([r['heat_ceiling'] for r in group]),
            'heat_kept': {name: _median(values)
                          for name, values in sorted(kept.items())},
        })
    out.sort(key=lambda row: row['heat_ceiling'])
    return out


def summarize_init_probe(rows: Sequence[dict]) -> list[dict]:
    """Median recall / heat_kept per (teacher target, variant).

    `heat_ceiling` is the oracle's own share at this budget, so
    `heat_kept / heat_ceiling` is the part a predictor could still win.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        buckets.setdefault((row['target'], row['variant']), []).append(row)
    out = []
    for (target, variant), group in sorted(buckets.items()):
        def med(field: str) -> float:
            values = [g[field] for g in group
                      if not math.isnan(g.get(field, float('nan')))]
            return _median(values) if values else float('nan')
        kept, ceiling = med('heat_kept'), med('heat_ceiling')
        out.append({
            'target': target, 'variant': variant, 'layers': len(group),
            'recall': med('recall'), 'heat_kept': kept,
            'heat_ceiling': ceiling, 'logit_std': med('logit_std'),
            'kept_over_ceiling': kept / ceiling if ceiling else float('nan'),
        })
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


def scorer_report(records: Sequence[Record], density: float,
                  oracle: str = 'Omega0',
                  baseline: str = 'proxy_topk',
                  compensate: int = 0) -> list[dict]:
    """How far each servable score closes the gap to the oracle.

    The plan's own metric for a scorer: the fraction of the distance
    between the proxy and the oracle that it recovers,

        (eps_baseline - eps_variant) / (eps_baseline - eps_oracle).

    1.0 means it scores as well as knowing the answer; 0.0 means it adds
    nothing over the untrained pooled score.

    Args:
        records: Rows from an ablation run.
        density: Which density to report.
        oracle: Mask name of the oracle, the far end of the gap.
        baseline: Mask name of the near end.
        compensate: Which c value to read.

    Returns:
        One dict per mask, sorted by median error.

    Raises:
        ValueError: If no record has that density, or the ends are absent.
    """
    rows = [r for r in records if r.density == density]
    if not rows:
        raise ValueError(f'no record at density {density}')
    names = sorted({n.rsplit('|', 1)[0] for r in rows for n in r.errors})
    key = lambda n: f'{n}|{compensate}'  # noqa: E731
    for end in (oracle, baseline):
        if key(end) not in rows[0].errors:
            raise ValueError(f'{end!r} is missing at c={compensate}')
    med = {n: _median([r.errors[key(n)] for r in rows if key(n) in r.errors])
           for n in names}
    span = med[baseline] - med[oracle]
    out = []
    for name in names:
        recalls = [r.recall_vs_omega.get(name) for r in rows
                   if name in r.recall_vs_omega]
        mass = [r.recall_vs_mass.get(name) for r in rows
                if name in r.recall_vs_mass]
        out.append({
            'mask': name,
            'median_error': med[name],
            'gap_closed': ((med[baseline] - med[name]) / span
                           if abs(span) > 1e-30 else float('nan')),
            'recall_vs_omega': _median(recalls) if recalls else float('nan'),
            'recall_vs_mass': _median(mass) if mass else float('nan'),
        })
    out.sort(key=lambda row: row['median_error'])
    return out


# --- Static per-head budget allocation -----------------------------------
#
# H2 asked whether a row inside a head should get its own budget, and the
# answer was no: the proxy cannot rank well enough to place the extra
# blocks, and the load imbalance costs more than the error it saves.
#
# Lifting the same question to heads changes both objections. A per-head
# budget is a static number calibrated offline, exactly like the tile plan
# already is, so every row of a head keeps a uniform budget: no imbalance,
# no runtime decision, no kernel change. And the allocator is not guessing
# from a proxy, it is reading measured error curves.


def error_curve(records: Sequence[Record], mask: str = 'proxy_topk',
                compensate: int = 0) -> dict[tuple[int, int],
                                             list[tuple[float, float]]]:
    """(layer, head) -> sorted [(density, median error)] over clips/steps.

    Args:
        records: Rows from an ablation run over several densities.
        mask: Which mask's error to read.
        compensate: Which c value to read.

    Returns:
        One curve per head, with the median taken over clips and steps.

    Raises:
        ValueError: If no record carries that mask and c.
    """
    key = f'{mask}|{compensate}'
    buckets: dict[tuple[int, int, float], list[float]] = {}
    for record in records:
        if key not in record.errors:
            continue
        slot = (record.layer, record.head, record.density)
        buckets.setdefault(slot, []).append(record.errors[key])
    if not buckets:
        raise ValueError(f'no record carries {key!r}')
    curves: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (layer, head, density), values in buckets.items():
        curves.setdefault((layer, head), []).append(
            (density, _median(values)))
    for curve in curves.values():
        curve.sort()
    return curves


def _interpolate_log_log(curve: Sequence[tuple[float, float]],
                         density: float) -> float:
    """Median error at `density`, linear in log density and log error.

    Relative error against a fixed budget behaves like a power law in the
    keep ratio over this range, so interpolating the logs is both smoother
    and monotone, which the allocator below relies on.
    """
    xs = [math.log(d) for d, _ in curve]
    ys = [math.log(max(e, 1e-12)) for _, e in curve]
    x = math.log(density)
    if x <= xs[0]:
        lo, hi = 0, 1
    elif x >= xs[-1]:
        lo, hi = len(xs) - 2, len(xs) - 1
    else:
        hi = next(i for i in range(1, len(xs)) if xs[i] >= x)
        lo = hi - 1
    span = xs[hi] - xs[lo]
    t = 0.0 if span == 0 else (x - xs[lo]) / span
    return math.exp(ys[lo] + t * (ys[hi] - ys[lo]))


def allocate_budget(curves: dict[tuple[int, int],
                                 list[tuple[float, float]]],
                    mean_density: float, grid: Sequence[float] | None = None,
                    tolerance: float = 1e-4) -> dict[tuple[int, int], float]:
    """Per-head densities minimizing total error at a fixed mean density.

    Lagrangian on a discrete grid: for a price `lam` on density every head
    independently picks the grid point minimizing `error + lam * density`,
    and `lam` is bisected until the mean lands on `mean_density`. With a
    convex curve this is the exact optimum of the grid-restricted problem,
    and the usual marginal-value condition holds: at the solution every
    head's last block buys the same error reduction.

    Args:
        curves: From `error_curve`.
        mean_density: The budget to spend, averaged over heads.
        grid: Candidate densities; defaults to 48 log-spaced points inside
            the measured range.
        tolerance: Stop bisecting once the mean is this close below the
            target.

    Returns:
        (layer, head) -> density. The mean never exceeds `mean_density`:
        a budget that overspends is not a budget, and on a discrete grid
        the exact target is usually not attainable. With thousands of
        heads the shortfall is far below one grid step.

    Raises:
        ValueError: If there are no curves, or the target is outside grid.
    """
    if not curves:
        raise ValueError('no error curves to allocate over')
    if grid is None:
        lo = min(d for c in curves.values() for d, _ in c)
        hi = max(d for c in curves.values() for d, _ in c)
        steps = 48
        grid = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo))
                         * i / (steps - 1.0)) for i in range(steps)]
        # Pin the ends: exp(log(x)) drifts, and a target equal to the
        # measured minimum would then look out of range.
        grid[0], grid[-1] = lo, hi
    grid = sorted(grid)
    if not grid[0] <= mean_density <= grid[-1]:
        raise ValueError(f'mean density {mean_density} outside the grid '
                         f'[{grid[0]:.4f}, {grid[-1]:.4f}]')
    table = {head: [_interpolate_log_log(curve, d) for d in grid]
             for head, curve in curves.items()}
    # Summing n copies of an exact grid value drifts above it, so the
    # feasibility test needs a relative slack rather than a bare <=.
    ceiling = mean_density * (1.0 + 1e-9) + 1e-12

    def pick(lam: float) -> dict[tuple[int, int], float]:
        out = {}
        for head, errs in table.items():
            best = min(range(len(grid)), key=lambda i: errs[i] + lam * grid[i])
            out[head] = grid[best]
        return out

    # A larger price buys less density, so the mean is monotone in lam.
    low, high = 0.0, 1.0
    best = None
    while True:
        chosen = pick(high)
        if sum(chosen.values()) / len(table) <= ceiling:
            best = chosen
            break
        high *= 4.0
        if high > 1e12:
            raise ValueError('no price drives the mean density down; the '
                             'curves are probably not decreasing')
    for _ in range(200):
        mid = 0.5 * (low + high)
        chosen = pick(mid)
        achieved = sum(chosen.values()) / len(chosen)
        if achieved <= ceiling:
            best = chosen
            high = mid
            if mean_density - achieved <= tolerance:
                break
        else:
            low = mid
    return best


def allocation_report(curves: dict[tuple[int, int],
                                   list[tuple[float, float]]],
                      mean_density: float) -> dict:
    """What a static per-head budget buys over a uniform one.

    Returns:
        Total error under the uniform and the allocated budgets, the
        relative saving, and the spread of the chosen densities.
    """
    uniform = sum(_interpolate_log_log(c, mean_density)
                  for c in curves.values())
    chosen = allocate_budget(curves, mean_density)
    allocated = sum(_interpolate_log_log(curves[h], d)
                    for h, d in chosen.items())
    densities = sorted(chosen.values())
    n = len(densities)
    return {
        'mean_density': mean_density,
        'heads': n,
        'uniform_total_error': uniform,
        'allocated_total_error': allocated,
        'relative_saving': (uniform - allocated) / max(uniform, 1e-30),
        'achieved_mean_density': sum(densities) / n,
        'density_p10': densities[int(0.10 * (n - 1))],
        'density_median': densities[n // 2],
        'density_p90': densities[int(0.90 * (n - 1))],
        'density_min': densities[0],
        'density_max': densities[-1],
    }


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
    """Reads back what `save_records` wrote.

    Unknown keys are dropped and missing optional ones defaulted, so a
    file written by an older or newer schema still loads.
    """
    with open(path) as handle:
        payload = json.load(handle)
    known = {f.name for f in dataclasses.fields(Record)}
    records = [Record(**{k: v for k, v in row.items() if k in known})
               for row in payload['records']]
    return records, payload['meta']


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
        predictors: Bundles to add to the init probe, each written
            `name=path`. The first one's plans are used for the geometry,
            so the probe sees the tile shapes they were trained against.
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
    predictors: list[str] = dataclasses.field(default_factory=list)
    densities: list[float] = dataclasses.field(
        default_factory=lambda: [0.05, 0.1, 0.2])
    alphas: list[float] = dataclasses.field(
        default_factory=lambda: [0.5, 1.0, 2.0])
    query_tiles: int = 8
    k_min: int | None = None
    k_max: int | None = None
    oracle: str = 'Omega0'
    heads: list[int] | None = None
    budget_rules: bool = True
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
            heads=tuple(self.heads) if self.heads is not None else None,
            budget_rules=self.budget_rules)


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

def run_init_probe(config: RunConfig) -> None:
    """Scores predictor initializations on one clip per geometry.

    Raises:
        ValueError: If launched with more than one rank, or a geometry has
            no cached sample.
    """
    from miowtion.h3 import model as h3_model  # pylint: disable=import-outside-toplevel
    from miowtion.train import data  # pylint: disable=import-outside-toplevel
    from miowtion.train import parallel  # pylint: disable=import-outside-toplevel
    from miowtion.train import teacher  # pylint: disable=import-outside-toplevel
    from miowtion.train import trajectory  # pylint: disable=import-outside-toplevel

    env = parallel.init_distributed()
    if env.world_size != 1:
        raise ValueError('run the init probe with one rank, not '
                         f'{env.world_size}')
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
    variants = init_variants(model.config.head_dim, env.device)
    loaded = []
    for spec in config.predictors:
        from miowtion.veda import bundle as veda_bundle  # pylint: disable=import-outside-toplevel
        if '=' not in spec:
            raise ValueError(f'predictors entries are name=path: {spec!r}')
        name, path = spec.split('=', 1)
        bundle = veda_bundle.load(path, env.device)
        variants[name] = bundle.predictor
        loaded.append(bundle)
        progress.log(
            f'loaded {name} from {path} (keep ratio {bundle.keep_ratio}, '
            f'second_order_rank {bundle.predictor.second_order_rank}, '
            f'count_term {bundle.predictor.count_term})')
    trained = loaded[0] if loaded else None
    for spec in config.geometries:
        geometry = data.parse_geometry(spec)
        samples = [s for s in cache.select('train', config.tasks)
                   if s.aspect in (None, geometry.aspect)
                   and s.latent_t in (None, geometry.latent_t)]
        if not samples:
            raise ValueError(f'no cached sample for {geometry.name}')
        samples = samples[:config.num_clips]
        if trained is not None:
            # The trained predictor was distilled against these shapes;
            # scoring it on any others would grade the wrong thing.
            plan = trained.plans.select(geometry)
        else:
            plan = veda_plan.TilePlan.uniform(
                geometry, tiling.TileShape.parse(config.tile_shape),
                model.config.num_layers, model.config.num_heads)
        veda_config = veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=config.densities[0]))
        progress.log(f'init probe {geometry.name}: {len(samples)} clips, '
                     f'budget ratio {config.densities[0]}, steps '
                     f'{list(config.steps)}')
        rows: list[dict] = []
        clips = progress.Progress(f'{geometry.name}: clips', len(samples))
        for sample in samples:
            traj = trajectory.Trajectory(model, cache, sample, geometry,
                                         schedule, config.seed, env.device)
            clip_tiling = veda_attention.ClipTiling(
                traj.layout, veda_config, env.device)
            last = max(config.steps)
            while not traj.done and traj.step <= last:
                inputs = traj.inputs()
                if traj.step in config.steps:
                    attention_fn = PredictorInitProbe(
                        traj.layout, plan, clip_tiling, variants,
                        config.query_tiles, config.seed, traj.step,
                        dense_backend=config.dense_backend)
                else:
                    attention_fn = h3_model.DenseAttention(
                        traj.layout.used, config.dense_backend)
                with torch.no_grad():
                    video_v, audio_v = model(
                        traj.clip, inputs.video_rows, inputs.audio_rows,
                        inputs.timestep, attention_fn,
                        tables.get(inputs.timestep.timesteps))
                if isinstance(attention_fn, PredictorInitProbe):
                    rows += [{**row, 'clip': sample.id}
                             for row in attention_fn.rows]
                traj.advance(video_v, audio_v)
            clips.update(f'{sample.id}: {len(rows)} rows')
        path = os.path.join(out_dir, f'{geometry.name}_init.json')
        summary = summarize_init_probe(rows)
        tmp = path + '.tmp'
        with open(tmp, 'w') as handle:
            json.dump({'meta': {'geometry': geometry.name,
                                'clips': [s.id for s in samples],
                                'budget_ratio': config.densities[0],
                                'steps': list(config.steps),
                                'query_tiles': config.query_tiles,
                                'tile_shape': config.tile_shape,
                                'predictors': list(config.predictors)},
                       'rows': rows, 'summary': summary,
                       'by_clip': init_probe_by_clip(rows)}, handle,
                      indent=1)
        os.replace(tmp, path)
        progress.log(f'saved {path} ({len(rows)} rows)')
        for row in summary:
            progress.log(
                f"  target {row['target']:4s} {row['variant']:14s}: "
                f"recall {row['recall']:.4f}  heat_kept {row['heat_kept']:.4f}"
                f"  kept/ceiling {row['kept_over_ceiling']:.4f}  "
                f"logit_std {row['logit_std']:.4f}")

# --- Second-moment calibration for the low-rank head --------------------
#
# `predictor.init_low_rank_second_order_` needs E[sq_q^T sq_q] and
# E[var_k^T var_k] per (layer, head) to know which directions of the
# diagonal cumulant term are worth spending rank on. Both are cheap: one
# teacher rollout, and D^2 per tile.


class SecondMomentProbe:
    """AttentionFn accumulating the pooled second moments of one clip.

    Only the video quadrant is weighted, because that is the only part of
    the score `select_video_blocks` reads.
    """

    def __init__(self, layout, plan, clip, num_heads: int, head_dim: int,
                 dense_backend: str = 'auto'):
        """
        Args:
            layout: Packed layout of the clip.
            plan: `veda.plan.TilePlan` giving each head's tile shape.
            clip: `veda.attention.ClipTiling` of this clip.
            num_heads: Heads per layer.
            head_dim: Head dimension.
            dense_backend: Backend of the dense attention.
        """
        self.layout = layout
        self.plan = plan
        self.clip = clip
        self.dense_backend = dense_backend
        shape = (plan.num_layers, num_heads, head_dim, head_dim)
        self.c_u = torch.zeros(shape, dtype=torch.float64)
        self.c_v = torch.zeros(shape, dtype=torch.float64)
        self.tiles = torch.zeros(plan.num_layers, num_heads,
                                 dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, q, k, v, layer_index):
        from miowtion.veda import predictor as veda_predictor  # pylint: disable=import-outside-toplevel
        out, _ = h3_attention.dense_attention(
            q, k, v, self.layout.used, return_lse=True,
            backend=self.dense_backend)
        for group in self.plan.head_groups(layer_index, self.clip.device):
            tile_layout = self.clip.get(group.shape)
            n_video = tile_layout.n_video_tiles
            q_t, k_t = (tiling.gather_tiles(t, tile_layout, group.heads)
                        for t in (q, k))
            sq_q = veda_predictor.pool_tiles(
                q_t, tile_layout, veda_predictor.SECOND_RAW)[:, :n_video]
            var_k = veda_predictor.pool_tiles(
                k_t, tile_layout,
                veda_predictor.SECOND_CENTRAL)[:, :n_video]
            heads = group.heads.cpu()
            self.c_u[layer_index].index_add_(
                0, heads, torch.einsum('hnd,hne->hde', sq_q, sq_q)
                .double().cpu())
            self.c_v[layer_index].index_add_(
                0, heads, torch.einsum('hnd,hne->hde', var_k, var_k)
                .double().cpu())
            self.tiles[layer_index].index_add_(
                0, heads, torch.full((heads.numel(),), float(n_video),
                                     dtype=torch.float64))
        return out

    def moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean second moments as fp32 [L, H, D, D], ready for the init."""
        counts = self.tiles.clamp(min=1.0)[:, :, None, None]
        return ((self.c_u / counts).float(), (self.c_v / counts).float())


def run_second_moments(config: RunConfig) -> None:
    """Collects and saves the calibration moments for one geometry.

    Raises:
        ValueError: If launched with more than one rank, or a geometry has
            no cached sample.
    """
    from miowtion.h3 import model as h3_model  # pylint: disable=import-outside-toplevel
    from miowtion.train import data  # pylint: disable=import-outside-toplevel
    from miowtion.train import parallel  # pylint: disable=import-outside-toplevel
    from miowtion.train import teacher  # pylint: disable=import-outside-toplevel
    from miowtion.train import trajectory  # pylint: disable=import-outside-toplevel

    env = parallel.init_distributed()
    if env.world_size != 1:
        raise ValueError('run the calibration with one rank, not '
                         f'{env.world_size}')
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
    plans = None
    if config.predictors:
        from miowtion.veda import bundle as veda_bundle  # pylint: disable=import-outside-toplevel
        path = config.predictors[0].split('=', 1)[-1]
        plans = veda_bundle.load(path, 'cpu').plans
    for spec in config.geometries:
        geometry = data.parse_geometry(spec)
        samples = [s for s in cache.select('train', config.tasks)
                   if s.aspect in (None, geometry.aspect)
                   and s.latent_t in (None, geometry.latent_t)]
        if not samples:
            raise ValueError(f'no cached sample for {geometry.name}')
        samples = samples[:config.num_clips]
        plan = (plans.select(geometry) if plans is not None
                else veda_plan.TilePlan.uniform(
                    geometry, tiling.TileShape.parse(config.tile_shape),
                    model.config.num_layers, model.config.num_heads))
        veda_config = veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=config.densities[0]))
        probe = None
        progress.log(f'second-moment calibration {geometry.name}: '
                     f'{len(samples)} clips, steps {list(config.steps)}')
        clips = progress.Progress(f'{geometry.name}: clips', len(samples))
        for sample in samples:
            traj = trajectory.Trajectory(model, cache, sample, geometry,
                                         schedule, config.seed, env.device)
            clip_tiling = veda_attention.ClipTiling(traj.layout,
                                                    veda_config, env.device)
            if probe is None:
                probe = SecondMomentProbe(
                    traj.layout, plan, clip_tiling,
                    model.config.num_heads, model.config.head_dim,
                    config.dense_backend)
            else:
                probe.layout, probe.clip = traj.layout, clip_tiling
            last = max(config.steps)
            while not traj.done and traj.step <= last:
                inputs = traj.inputs()
                if traj.step in config.steps:
                    attention_fn = probe
                else:
                    attention_fn = h3_model.DenseAttention(
                        traj.layout.used, config.dense_backend)
                with torch.no_grad():
                    video_v, audio_v = model(
                        traj.clip, inputs.video_rows, inputs.audio_rows,
                        inputs.timestep, attention_fn,
                        tables.get(inputs.timestep.timesteps))
                traj.advance(video_v, audio_v)
            clips.update(sample.id)
        c_u, c_v = probe.moments()
        path = os.path.join(out_dir, f'{geometry.name}_moments.pt')
        torch.save({'c_u': c_u, 'c_v': c_v,
                    'geometry': geometry.name,
                    'clips': [s.id for s in samples],
                    'steps': list(config.steps)}, path)
        progress.log(f'saved {path} ({tuple(c_u.shape)})')
