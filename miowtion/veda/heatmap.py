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


def seer_kl(logits: torch.Tensor, heat: torch.Tensor,
            layout: tiling.TileLayout) -> torch.Tensor:
    """KL(teacher || student) over key tiles, averaged over rows then heads.

    Args:
        logits: [H', R, n_tiles] fp32 student block logits (with grad).
        heat: [H', R, n_tiles] fp32 teacher heat.
        layout: Tile layout (empty key tiles are excluded).

    Returns:
        Scalar loss.
    """
    col_ok = layout.kv_ok[None, None, :]
    logp = F.log_softmax(logits.masked_fill(~col_ok, float('-inf')), dim=-1)
    target = heat.masked_fill(~col_ok, 0.0).clamp(min=0.0)
    total = target.sum(-1, keepdim=True)
    row_ok = total[..., 0] > 0
    target = target / total.clamp(min=torch.finfo(torch.float32).tiny)
    positive = (target > 0) & col_ok
    terms = torch.where(positive,
                        target * (torch.log(torch.where(positive, target, 1.0))
                                  - torch.where(positive, logp, 0.0)), 0.0)
    per_row = terms.sum(-1)  # [H', R]
    per_head = (per_row * row_ok).sum(-1) / row_ok.sum(-1).clamp(min=1)
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
