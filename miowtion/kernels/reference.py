"""Reference block-sparse attention (CPU / unit tests).

Semantics every kernel must match:
  * inputs are tile-ordered [N, H', D]; key tile j contributes only its first
    valid_count[j] rows (the padding suffix never enters the softmax);
  * a query row attends to the valid rows of the key tiles kept for its
    query tile; rows with no key at all output 0;
  * outputs of padding query rows are unspecified (they are scattered to a
    discarded row).
"""

from __future__ import annotations

import math

import torch

from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           block_mask: torch.Tensor,
                           valid_count: torch.Tensor) -> torch.Tensor:
    """Masked softmax attention expanded to token granularity, fp32 math.

    Args:
        q: [N, H', D] tile-ordered queries.
        k: [N, H', D] tile-ordered keys.
        v: [N, H', D] tile-ordered values.
        block_mask: [H', n_tiles, n_tiles] bool.
        valid_count: [n_tiles] int valid prefix length of each key tile.

    Returns:
        [N, H', D] in q.dtype.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    key_valid = (torch.arange(_TILE, device=q.device)[None, :]
                 < valid_count[:, None]).view(-1)  # [N]
    token_mask = block_mask.repeat_interleave(_TILE, 1).repeat_interleave(
        _TILE, 2) & key_valid[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    scores = scores.masked_fill(~token_mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.einsum('hqk,khd->qhd', probs, v.float())
    return out.to(q.dtype)
