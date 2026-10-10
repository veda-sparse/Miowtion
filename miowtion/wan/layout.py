"""Packed layout for Wan2.1, so Veda's tiling can run on it.

Veda never touches the audio or reference quadrants: across
`veda/tiling.py`, `veda/mask.py`, `veda/attention.py` and
`veda/search.py` it reads twelve layout fields and all of them are
geometry-agnostic -- `n_video_tiles`, `gather_index`, `used`,
`pad_slots`, `valid_count`, `kv_ok`, `seq_len`, `n_tiles`,
`n_ref_tiles`, `spans`, `slot_valid`, `scatter_index` -- and
`tiling.build_tile_layout` derives every one of them from the spans
alone. So a second model needs a `PackedLayout` whose spans are right,
not a reimplementation of the interface, which is what
docs/dependencies.md used to claim.

Wan2.1-T2V-1.3B is text-to-video: one target span, no keyframes, no
reference visuals, no audio. Its DiT patches the latent by [1, 2, 2]
(transformer/config.json), so the token grid is
(latent_t, latent_h // 2, latent_w // 2); the VAE is 8x spatial and 4x
temporal with the first frame kept, so latent_t = (frames - 1) // 4 + 1.
"""

from __future__ import annotations

import torch

from miowtion.h3 import layout as h3_layout

# Wan's DiT patch size, from transformer/config.json. Only the spatial
# halving matters for the token grid; the temporal factor is 1.
PATCH = (1, 2, 2)
# AutoencoderKLWan: 8x spatial, 4x temporal with the first frame kept.
VAE_SPATIAL = 8
VAE_TEMPORAL = 4


def token_grid(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """(T, H, W) token grid of a Wan clip.

    Args:
        width: Pixel width, a multiple of 16 (8 for the VAE, 2 for the patch).
        height: Pixel height, same constraint.
        frames: Frame count; Wan keeps the first frame and groups the rest
            in fours, so this should be 4k + 1.

    Returns:
        The DiT token grid.

    Raises:
        ValueError: If the sizes do not divide, which would silently drop
            a row or a column of tokens.
    """
    step = VAE_SPATIAL * PATCH[1]
    if width % step or height % step:
        raise ValueError(
            f'{width}x{height} must divide {step} (VAE {VAE_SPATIAL} x '
            f'patch {PATCH[1]}); a remainder would drop tokens silently')
    if frames % VAE_TEMPORAL != 1:
        raise ValueError(
            f'{frames} frames is not 4k + 1; Wan keeps the first frame and '
            'groups the rest in fours')
    return ((frames - 1) // VAE_TEMPORAL + 1,
            height // VAE_SPATIAL // PATCH[1],
            width // VAE_SPATIAL // PATCH[2])


def packed_layout(width: int, height: int, frames: int,
                  text_len: int = 512) -> h3_layout.PackedLayout:
    """A `PackedLayout` for one Wan clip: text, then one video span.

    The audio and reference fields exist because the type is shared with
    H3; they are empty here and Veda reads none of them.

    Args:
        width: Pixel width.
        height: Pixel height.
        frames: Frame count.
        text_len: Prompt tokens before the video rows.

    Returns:
        A layout whose single span is the target.
    """
    grid = token_grid(width, height, frames)
    video_rows = grid[0] * grid[1] * grid[2]
    used = text_len + video_rows
    span = h3_layout.VideoSpan(start=text_len, grid=grid, role='target')
    img_pos = torch.arange(text_len, used)
    empty = torch.zeros(0, dtype=torch.long)
    return h3_layout.PackedLayout(
        seq_len=used,
        used=used,
        text_len=text_len,
        img_pos=img_pos,
        audio_pos=empty,
        update_mask=torch.ones(video_rows, dtype=torch.bool),
        audio_update_mask=torch.zeros(0, dtype=torch.bool),
        position_ids=torch.arange(used),
        token_tags=torch.zeros(used, dtype=torch.long),
        cu_seqlens=torch.tensor([0, used, used], dtype=torch.int32),
        spans=(span,),
        visual_cond_shapes=(),
        audio_cond_lengths=(),
    )
