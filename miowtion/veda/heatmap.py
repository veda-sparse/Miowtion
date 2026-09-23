"""Teacher block heat maps, the seer KL loss and mask recall.

Teacher heat of (query tile i, key tile j) is the maximum true attention
probability inside the block:
    heat[i, j] = max_{r in i, c in j} exp(q_r . k_c / sqrt(D) - lse_r).
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
                           k_chunk: int = 8192) -> torch.Tensor:
    """Chunked torch implementation of the teacher heat (reference).

    Args:
        q: [N, H', D] bf16 tile-ordered queries.
        k: [N, H', D] bf16 tile-ordered keys.
        lse: [N, H'] fp32 tile-ordered teacher LSE (0 on padding slots).
        layout: Tile layout.
        q_tiles: [R] int64 query tiles to supervise.
        q_chunk: Query rows per chunk (multiple of 128).
        k_chunk: Key rows per chunk (multiple of 128).

    Returns:
        [H', R, n_tiles] fp32 heat, 0 on padding query rows / empty keys.
    """
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
        row_max = []
        for k0 in range(0, layout.num_slots, k_chunk):
            k1 = min(k0 + k_chunk, layout.num_slots)
            s = torch.bmm(q_sel[:, q0:q1], k_t[:, k0:k1].transpose(1, 2))
            s = s.masked_fill(~valid[None, None, k0:k1], float('-inf'))
            # Scale after the max: a positive scale commutes with max.
            row_max.append(s.view(heads, q1 - q0, -1, _TILE).amax(-1))
        tile_max = torch.cat(row_max, -1).float() * scale  # [H', rq, n]
        heat_rows = torch.exp(tile_max - lse_sel[:, q0:q1, None])
        heat_rows = heat_rows.masked_fill(~row_valid[None, q0:q1, None], 0.0)
        heat[:, q0 // _TILE:q1 // _TILE] = heat_rows.view(
            heads, -1, _TILE, n_tiles).amax(2)
    return heat


def teacher_heat(q: torch.Tensor, k: torch.Tensor, lse: torch.Tensor,
                 layout: tiling.TileLayout,
                 q_tiles: torch.Tensor) -> torch.Tensor:
    """Teacher heat; the fused Triton kernel on CUDA, else the reference."""
    if q.is_cuda:
        from miowtion.kernels import block_heat_triton  # pylint: disable=import-outside-toplevel
        if block_heat_triton.available():
            return block_heat_triton.teacher_heat(q, k, lse, layout, q_tiles)
    return teacher_heat_reference(q, k, lse, layout, q_tiles)


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


@torch.no_grad()
def mask_recall(logits: torch.Tensor, heat: torch.Tensor,
                layout: tiling.TileLayout,
                blocks: list[veda_mask.ColumnBlock],
                q_tiles: torch.Tensor) -> torch.Tensor:
    """Overlap of predicted and oracle top-k sets, diagonal excluded.

    Both sets have the same size per row, so recall equals precision. Only
    the video -> video quadrant counts; the forced diagonal is not a
    predictor decision and is removed from both sets. Returns NaN when the
    budget holds nothing but the diagonal.
    """
    n_video = layout.n_video_tiles
    pred = veda_mask.select_video_blocks(logits, layout, blocks, q_tiles)
    oracle = veda_mask.select_video_blocks(heat, layout, blocks, q_tiles)

    def dense(sel: veda_mask.Selection) -> torch.Tensor:
        out = torch.zeros(*sel.index.shape[:2], n_video, dtype=torch.bool,
                          device=sel.index.device)
        return out.scatter_(2, sel.index, sel.keep)

    diag = F.one_hot(q_tiles, n_video).bool()[None]
    pred_set, oracle_set = dense(pred) & ~diag, dense(oracle) & ~diag
    total = oracle_set.sum()
    if total == 0:  # the budget holds only the diagonal
        return torch.tensor(float('nan'))
    return (pred_set & oracle_set).sum() / total
