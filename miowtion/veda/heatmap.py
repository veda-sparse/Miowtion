"""Teacher block heat maps, the seer KL loss and mask diagnostics.

Teacher heat of (query tile i, key tile j) is, with `reduce='max'`, the
maximum true attention probability inside the block:
    heat[i, j] = max_{r in i, c in j} exp(q_r . k_c / sqrt(D) - lse_r),
and with `reduce='sum'` the block's total attention mass:
    heat[i, j] = sum_{r in i, c in j} exp(q_r . k_c / sqrt(D) - lse_r).

Which one to distil against is a measured question, not a free choice; see
docs/features/veda2.md. Mass is also the quantity the predictor's own
initialization already estimates, since mean-pooled QK is the zero-order
term of log mass, while max has no such correspondence.
The per-row LSE comes for free from the dense teacher's flash attention; it
is exact, and because every downstream consumer (row normalization, top-k)
is invariant to a per-row constant, even an approximate LSE would only cost
a per-row scale.

Heat is O(S^2 D) regardless of sparsity and dominates the training step, so
only a sampled subset of query tiles is supervised (the KL is a mean over
query tiles, so the subsample is unbiased).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from miowtion.veda import mask as veda_mask
from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE


@torch.no_grad()
def teacher_heat_reference(q: torch.Tensor, k: torch.Tensor,
                           lse: torch.Tensor, layout: tiling.TileLayout,
                           q_tiles: torch.Tensor, q_chunk: int = 2048,
                           k_chunk: int = 8192,
                           reduce: str = 'max') -> torch.Tensor:
    """Chunked torch implementation of the teacher heat (reference).

    Args:
        q: [N, H', D] bf16 tile-ordered queries.
        k: [N, H', D] bf16 tile-ordered keys.
        lse: [N, H'] fp32 tile-ordered teacher LSE (0 on padding slots).
        layout: Tile layout.
        q_tiles: [R] int64 query tiles to supervise.
        q_chunk: Query rows per chunk (multiple of 128).
        k_chunk: Key rows per chunk (multiple of 128).
        reduce: 'max' for the block's peak probability, 'sum' for its mass.

    Returns:
        [H', R, n_tiles] fp32 heat, 0 on padding query rows / empty keys.

    Raises:
        ValueError: On an unknown reduction.
    """
    if reduce not in ('max', 'sum'):
        raise ValueError(f"reduce must be 'max' or 'sum': {reduce!r}")
    heads, head_dim = q.shape[1], q.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    valid = layout.slot_valid.bool()
    rows = (q_tiles[:, None] * _TILE + torch.arange(
        _TILE, device=q.device)[None]).view(-1)
    q_sel = q.index_select(0, rows).transpose(0, 1)  # [H', R*128, D]
    lse_sel = lse.index_select(0, rows).transpose(0, 1)  # [H', R*128]
    row_valid = valid.index_select(0, rows)
    k_t = k.transpose(0, 1)  # [H', N, D]
    n_tiles = layout.n_tiles
    heat = q.new_zeros((heads, q_tiles.numel(), n_tiles), dtype=torch.float32)
    for q0 in range(0, rows.numel(), q_chunk):
        q1 = min(q0 + q_chunk, rows.numel())
        parts = []
        for k0 in range(0, layout.num_slots, k_chunk):
            k1 = min(k0 + k_chunk, layout.num_slots)
            s = torch.bmm(q_sel[:, q0:q1], k_t[:, k0:k1].transpose(1, 2))
            s = s.masked_fill(~valid[None, None, k0:k1], float('-inf'))
            blocks = s.view(heads, q1 - q0, -1, _TILE)
            if reduce == 'max':
                # Scale after the max: a positive scale commutes with max.
                parts.append(blocks.amax(-1))
            else:
                p = torch.exp(blocks.float() * scale
                              - lse_sel[:, q0:q1, None, None])
                parts.append(p.sum(-1))
        agg = torch.cat(parts, -1)                        # [H', rq, n]
        if reduce == 'max':
            heat_rows = torch.exp(agg.float() * scale
                                  - lse_sel[:, q0:q1, None])
        else:
            heat_rows = agg
        heat_rows = heat_rows.masked_fill(~row_valid[None, q0:q1, None], 0.0)
        per_tile = heat_rows.view(heads, -1, _TILE, n_tiles)
        heat[:, q0 // _TILE:q1 // _TILE] = (per_tile.amax(2)
                                            if reduce == 'max'
                                            else per_tile.sum(2))
    return heat


def teacher_heat(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor,
                 layout: tiling.TileLayout, q_tiles: torch.Tensor,
                 reduce: str = 'max') -> torch.Tensor:
    """Teacher heat; the fused Triton kernel on CUDA, else the reference."""
    from miowtion.kernels import block_heat_triton  # pylint: disable=import-outside-toplevel
    if block_heat_triton.supports(q):
        return block_heat_triton.teacher_heat(q, k, lse, layout, q_tiles,
                                              reduce)
    return teacher_heat_reference(q, k, lse, layout, q_tiles, reduce=reduce)


# Floor on the teacher's probability in the reverse direction. log t is
# -inf on a block the teacher gives no mass, which would make any student
# mass there an infinite penalty; this caps that at ~14 per unit of mass,
# which is still strongly zero-forcing but finite.
_REVERSE_FLOOR = 1e-6

# Stand-in for -inf when sorting standardised scores: -inf would make the
# masked columns' gradient NaN if they were ever read, and the scores are
# standardised, so nothing real comes anywhere near this.
_LARGE = 1e4


def seer_kl(logits: torch.Tensor, heat: torch.Tensor,
            layout: tiling.TileLayout,
            direction: str = 'forward') -> torch.Tensor:
    """Seer KL over key tiles, averaged over rows then heads.

    Which direction matters, because the two have opposite failure modes
    and only one of them matches a top-k read-out.

    'forward' is KL(teacher || student): mass-covering. Every block the
    teacher touches has to get student probability, so the cheapest way to
    reduce it is to spread out. Measured in stage 1 against a mass target,
    25 updates took it from 134 to 48 while recall fell 0.714 -> 0.654;
    the loss was being paid down by flattening, not by ordering. It also
    explains the absolute scale: the penalty blows up wherever the teacher
    has a little mass and the student has almost none.

    'reverse' is KL(student || teacher): mode-seeking. The student is only
    punished for mass where the teacher has none, so it concentrates on
    the teacher's dominant blocks, which is what the kernel then reads out
    as a top-k.

    Args:
        logits: [H', R, n_tiles] fp32 student block logits (with grad).
        heat: [H', R, n_tiles] fp32 teacher heat.
        layout: Tile layout (empty key tiles are excluded).
        direction: 'forward' or 'reverse'.

    Returns:
        Scalar loss.

    Raises:
        ValueError: On an unknown direction.
    """
    if direction not in ('forward', 'reverse'):
        raise ValueError(f"direction must be 'forward' or 'reverse': "
                         f'{direction!r}')
    col_ok = layout.kv_ok[None, None, :]
    logp = F.log_softmax(logits.masked_fill(~col_ok, float('-inf')), dim=-1)
    target = heat.masked_fill(~col_ok, 0.0).clamp(min=0.0)
    total = target.sum(-1, keepdim=True)
    row_ok = total[..., 0] > 0
    target = target / total.clamp(min=torch.finfo(torch.float32).tiny)
    if direction == 'forward':
        positive = (target > 0) & col_ok
        terms = torch.where(
            positive,
            target * (torch.log(torch.where(positive, target, 1.0))
                      - torch.where(positive, logp, 0.0)), 0.0)
    else:
        # Weighted by the student's own mass, so the sum is over columns
        # it actually uses, and the teacher's zeros are floored.
        log_target = torch.log(target.clamp(min=_REVERSE_FLOOR))
        terms = torch.where(col_ok, logp.exp() * (logp - log_target), 0.0)
    per_row = terms.sum(-1)  # [H', R]
    per_head = (per_row * row_ok).sum(-1) / row_ok.sum(-1).clamp(min=1)
    return per_head.mean()


# Width of the soft top-k boundary, in units of the row-standardised
# score. The score is standardised, so this is comparable across heads and
# geometries. 0.15 makes the band about 0.035 * n_tiles wide at the 10%
# quantile of a standard normal (pdf 0.175 at z = 1.28), i.e. roughly a
# dozen blocks out of 360 -- wide enough for a gradient, narrow enough
# that it is a gradient about the decision.
_TRANSPORT_TAU = 0.15


def transport_loss(logits: torch.Tensor, heat: torch.Tensor,
                   layout: tiling.TileLayout,
                   blocks: list[veda_mask.ColumnBlock],
                   q_tiles: torch.Tensor,
                   temperature: float = _TRANSPORT_TAU) -> torch.Tensor:
    """Negative retained transport mass, relative to the attainable mass.

    Read attention as one-sided entropic optimal transport (arXiv
    2508.08369): per query row, the forward pass solves

        max_p <p, s> + H(p)   over the simplex,   giving p = softmax(s),

    so p_uv is the mass transported from query u to key v. Block sparsity
    is a *support constraint* on that plan, and the constrained optimum is

        F(S) = log sum_{v in S} exp(s_uv) = log E_u(S),

    the log-partition function over the kept keys. So

        argmax_{|S| = k} F(S)  =  top-k by block attention mass,

    exactly, with no approximation, and the value left on the table is
    -log(1 - r_u) in the dropped mass fraction r_u. That is why mass is
    the right distillation target and why `heat_kept` is the right metric:
    they are the EOT objective itself, not proxies for it.

    This loss is that objective, relaxed only in the selection. A hard
    top-k is not differentiable, so the k-hot indicator is replaced by a
    sigmoid about the row's own k-th largest score,

        z_u = (s_u - mean_u) / std_u,    theta_u = k_u-th largest z_u,
        pi_u = sigmoid((z_u - theta_u) / tau),
        L_u  = - <pi_u, a_u> / <oracle_u, a_u>,

    where a_u is the teacher's block mass and `oracle_u` is the hard top-k
    by a_u at the same budget. Three properties follow, and all three are
    what the four earlier objectives lacked:

    * As tau -> 0 this is *exactly* `-heat_kept / heat_ceiling`, the metric
      the run is judged on. The loss and the metric are one object, so they
      cannot move in opposite directions -- which is what happened to the
      forward KL (KL 134 -> 48 while recall 0.714 -> 0.654) and to the
      centred BCE (loss up on three of four geometries).
    * It is exactly invariant to a per-row positive affine map of the
      scores, which is the full gauge a top-k read-out is blind to.
      Standardising removes the offset and the scale; theta is a function
      of z, so the sigmoid's argument is invariant too. The gradient in
      both gauge directions is therefore zero, not merely small.
    * Its gradient is concentrated at the rank-k boundary, because that is
      where sigmoid' peaks. A softmax policy over the whole row was the
      first thing tried here and is the wrong relaxation: measured on
      standardised scores at n = 360, it spreads over 220 blocks and puts
      only 39% of its weight inside the budget, so most of the gradient
      asks about ranks the kernel never reads.

    The forced diagonal is excluded, from the policy and from the ceiling
    both. It is a kernel rule rather than a predictor decision, so it is
    free to get right and would only dilute the loss -- the same reason
    `oracle_bce` and `mask_diagnostics` drop it.

    Args:
        logits: [H', R, n_tiles] fp32 student block logits (with grad).
        heat: [H', R, n_tiles] fp32 teacher heat; must be the *mass*
            reduction, since the derivation is about transported mass.
        layout: Tile layout (empty key tiles are excluded).
        blocks: Column blocks from column_blocks(), which carry the budget.
        q_tiles: [R] int64 video query tile ids.
        temperature: Width of the soft boundary in standardised score
            units. Must be > 0.

    Returns:
        Scalar loss, averaged over rows then heads. -1 is the oracle
        selection and 0 is a selection holding no mass at all. The policy
        is rescaled to spend exactly the per-row budget, so a flat
        predictor cannot score below -1 by keeping everything a little;
        finite temperature still leaves an O(tau) slack, so values a shade
        below -1 are possible and are not a bug.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f'temperature must be > 0: {temperature}')
    n_video = layout.n_video_tiles
    with torch.no_grad():
        oracle = veda_mask.select_video_blocks(heat, layout, blocks, q_tiles)
        chosen = torch.zeros(*oracle.index.shape[:2], n_video,
                             dtype=torch.bool, device=logits.device)
        chosen.scatter_(2, oracle.index, oracle.keep)
        diag = F.one_hot(q_tiles, n_video).bool()[None]
        valid = layout.kv_ok[None, None, :n_video] & ~diag
        chosen = chosen & valid
        mass = heat[:, :, :n_video].float().masked_fill(~valid, 0.0)
        mass = mass.clamp(min=0.0)
        # The attainable mass at this budget, which is what makes the loss
        # read as a fraction and makes -1 mean 'the oracle's own choice'.
        ceiling = (mass * chosen).sum(-1, keepdim=True)
        budget = chosen.sum(-1)  # [H', R], varies per row by layout rule
        row_ok = (ceiling[..., 0] > 0) & (budget > 0)

    scores = logits[:, :, :n_video].float()
    # Row-standardise over the valid columns only. This is the gauge
    # projection: its backward removes the constant and the radial
    # component of the gradient, so no update can be spent on them.
    counted = valid.to(scores.dtype)
    rows = counted.sum(-1, keepdim=True).clamp(min=1.0)
    centre = (scores * counted).sum(-1, keepdim=True) / rows
    centred = (scores - centre) * counted
    var = (centred.square() * counted).sum(-1, keepdim=True) / rows
    z = centred / var.clamp(min=1e-12).sqrt()

    # The row's own decision boundary, detached: it is where the top-k
    # cut falls, not a quantity to optimise. Placed *midway* between the
    # k-th and (k+1)-th largest score, not on the k-th: sitting on it
    # would give the marginal column sigmoid(0) = 0.5 and break the
    # tau -> 0 limit, which is the whole point of the loss. The budget
    # varies per row, so this is a sort and a gather rather than a topk,
    # and invalid columns are pushed below every real one.
    with torch.no_grad():
        ranked, _ = torch.sort(z.masked_fill(~valid, -_LARGE), dim=-1,
                               descending=True)
        last = ranked.shape[-1] - 1
        kth = (budget - 1).clamp(min=0, max=last).unsqueeze(-1)
        theta = 0.5 * (ranked.gather(-1, kth)
                       + ranked.gather(-1, (kth + 1).clamp(max=last)))

    policy = torch.sigmoid((z - theta) / temperature) * counted
    # Rescale so the policy spends exactly the budget. Without this a flat
    # predictor wins: z = 0 everywhere puts sigmoid at 0.5 on *every*
    # column, so it 'keeps' half the row and scores below -1, i.e. better
    # than the oracle. That is the forward-KL failure mode -- a loss
    # reducible by flattening -- reintroduced through the relaxation.
    # At tau -> 0 the policy is already exactly k-hot, so this is a no-op
    # in the limit and the loss still equals the metric.
    policy = policy * (budget.unsqueeze(-1).to(policy.dtype)
                       / policy.sum(-1, keepdim=True).clamp(min=1e-6))
    retained = (policy * mass).sum(-1) / ceiling[..., 0].clamp(
        min=torch.finfo(torch.float32).tiny)
    per_head = -(retained * row_ok).sum(-1) / row_ok.sum(-1).clamp(min=1)
    return per_head.mean()


def oracle_bce(logits: torch.Tensor, heat: torch.Tensor,
               layout: tiling.TileLayout,
               blocks: list[veda_mask.ColumnBlock],
               q_tiles: torch.Tensor) -> torch.Tensor:
    """Balanced BCE of the student logits against the oracle's top-k set.

    The seer KL fits the teacher's whole distribution, but the only thing
    the kernel ever reads out of the predictor is which blocks land in the
    top-k. The two objectives came apart in practice: over 600 updates the
    KL fell 30% while the kept heat moved 3% and the logit spread shrank
    monotonically (3.73 -> 2.51), i.e. the KL was being minimized by
    flattening towards the bulk instead of sharpening the ordering. This
    term states the objective directly -- is this block in the oracle's set
    or not -- and is meant to be added to the KL, not to replace it.

    Positives and negatives are averaged separately and then halved: the
    oracle keeps about `keep_ratio` of the columns, so an unbalanced mean
    would be ~90% negatives and the all-negative predictor would already
    look good.

    The forced diagonal is excluded. It is a kernel rule, not a predictor
    decision, so it is free to get right and would only dilute the loss --
    the same reason mask_diagnostics() removes it from recall.

    Args:
        logits: [H', R, n_tiles] fp32 student block logits (with grad).
        heat: [H', R, n_tiles] fp32 teacher heat.
        layout: Tile layout.
        blocks: Column blocks from column_blocks().
        q_tiles: [R] int64 video query tile ids.

    Returns:
        Scalar loss; 0 when no column is both valid and off-diagonal.
    """
    n_video = layout.n_video_tiles
    with torch.no_grad():
        oracle = veda_mask.select_video_blocks(heat, layout, blocks, q_tiles)
        target = torch.zeros(*oracle.index.shape[:2], n_video,
                             dtype=torch.bool, device=logits.device)
        target.scatter_(2, oracle.index, oracle.keep)
        diag = F.one_hot(q_tiles, n_video).bool()[None]
        valid = layout.kv_ok[None, None, :n_video] & ~diag
        positive = target & valid
        negative = ~target & valid
    scores = logits[:, :, :n_video].float()
    # Centre each row before the sigmoid. The kernel reads a within-row
    # ordering, so the loss has to be invariant to a per-row offset for the
    # same reason the top-k is. Without this the BCE reads absolute logit
    # magnitude, and the second-order score term is a sum of non-negative
    # products, i.e. a large positive offset: measured, it put the loss at
    # 537 with a gradient of +-1 per element, so grad_norm hit 7522 against
    # a clip of 1.0 and every update was scaled to nothing.
    counted = valid.to(scores.dtype)
    rows = counted.sum(-1, keepdim=True).clamp(min=1.0)
    centre = (scores * counted).sum(-1, keepdim=True) / rows
    scores = scores - centre
    terms = F.binary_cross_entropy_with_logits(
        scores, target.to(scores.dtype), reduction='none')
    n_pos = positive.sum().clamp(min=1)
    n_neg = negative.sum().clamp(min=1)
    loss = 0.5 * ((terms * positive).sum() / n_pos
                  + (terms * negative).sum() / n_neg)
    # No Python branch on the counts: that would synchronize.
    return torch.where(valid.any(), loss, torch.zeros_like(loss))


@torch.no_grad()
def mask_diagnostics(logits: torch.Tensor, heat: torch.Tensor,
                     layout: tiling.TileLayout,
                     blocks: list[veda_mask.ColumnBlock],
                     q_tiles: torch.Tensor) -> dict[str, torch.Tensor]:
    """Predictor vs. oracle on the video quadrant, as device scalars.

    Every value stays on the device: the caller resolves a whole micro
    step's diagnostics with one transfer, because a `.item()` here would
    synchronize once per layer and head group.

    Returns:
        recall: overlap of the predicted and the oracle top-k set. Both
            sets have the same size per row, so recall equals precision.
            The forced diagonal is not a predictor decision and is removed
            from both; NaN when the budget holds nothing else.
        heat_kept: share of a row's total video block heat that the
            predicted tiles carry, averaged over rows and heads. The
            diagonal counts, because the kernel does compute it. This is
            what recall is a proxy for: recall counts blocks, this weighs
            them by how much attention they actually hold.
        heat_ceiling: the same for the oracle's own selection, i.e. the
            most any predictor could keep at this budget. A low ceiling
            means the teacher itself is not concentrated enough for the
            budget, which no amount of training can fix.
    """
    n_video = layout.n_video_tiles
    pred = veda_mask.select_video_blocks(logits, layout, blocks, q_tiles)
    oracle = veda_mask.select_video_blocks(heat, layout, blocks, q_tiles)

    def dense(sel: veda_mask.Selection) -> torch.Tensor:
        out = torch.zeros(*sel.index.shape[:2], n_video, dtype=torch.bool,
                          device=sel.index.device)
        return out.scatter_(2, sel.index, sel.keep)

    pred_set, oracle_set = dense(pred), dense(oracle)
    video_heat = heat[:, :, :n_video].float().clamp(min=0.0)
    total = video_heat.sum(-1)
    row_ok = total > 0
    tiny = torch.finfo(torch.float32).tiny

    def share(selected: torch.Tensor) -> torch.Tensor:
        row = (video_heat * selected).sum(-1) / total.clamp(min=tiny)
        return (row * row_ok).sum() / row_ok.sum().clamp(min=1)

    diag = F.one_hot(q_tiles, n_video).bool()[None]
    pred_off, oracle_off = pred_set & ~diag, oracle_set & ~diag
    found = (pred_off & oracle_off).sum()
    possible = oracle_off.sum()
    # No Python branch on `possible`: that would synchronize.
    recall = torch.where(possible > 0, found / possible.clamp(min=1),
                         torch.nan)
    return {'recall': recall, 'heat_kept': share(pred_set),
            'heat_ceiling': share(oracle_set)}
