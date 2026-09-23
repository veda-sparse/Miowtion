"""Latent patchify helpers, initial noise and condition noise augmentation.

All random draws happen on CPU in fp32 so a seed reproduces the same noise on
any device.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import schedule as h3_schedule

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32


def patchify(latent: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] latent -> [B*T*(H/2)*(W/2), C*4] rows (1x2x2 patch).

    Rows are t-major, then h, then w; each row is ordered (c, ph, pw).
    """
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, 1, h // 2, 2, w // 2, 2)
    x = torch.einsum('nctrhpwq->nthwcrpq', x)
    return x.reshape(b * t * (h // 2) * (w // 2), c * 4).contiguous()


def unpatchify(rows: torch.Tensor, latent_t: int, latent_h: int,
               latent_w: int) -> torch.Tensor:
    """Inverse of patchify for one sample: rows -> [1, C, T, H, W]."""
    c = rows.shape[-1] // 4
    x = rows.reshape(1, latent_t, latent_h // 2, latent_w // 2, c, 1, 2, 2)
    x = torch.einsum('nthwcrpq->nctrhpwq', x)
    return x.reshape(1, c, latent_t, latent_h, latent_w).contiguous()


def initial_noise(geometry: h3_geometry.Geometry,
                  seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """sigma = 1 state of the target rows.

    Video noise is drawn on the raw [1, 24, T, H, W] latent and then
    patchified (not drawn directly as rows); audio re-seeds a fresh
    generator with the same seed.

    Returns:
        video_rows: [N_video, 96] fp32.
        audio_rows: [2 * audio_t, 32] fp32, channel-major.
    """
    gen = torch.Generator().manual_seed(int(seed))
    video = torch.randn(1, VIDEO_CHANNELS, geometry.latent_t,
                        geometry.latent_h, geometry.latent_w, generator=gen,
                        dtype=torch.float32)
    gen = torch.Generator().manual_seed(int(seed))
    audio = torch.randn(geometry.num_audio_rows, AUDIO_CHANNELS,
                        generator=gen, dtype=torch.float32)
    return patchify(video), audio


def augment_visual_conditions(clean_rows: torch.Tensor,
                              shapes: Sequence[tuple[int, int, int]],
                              target_latent_t: int, seed: int) -> torch.Tensor:
    """Noise-augments visual condition rows (keyframes, reference visuals).

    Each condition draws from a fresh generator seeded with `seed`, on a
    [1, 24, target_latent_t + len(shapes), H, W] latent, keeps the first
    latent_t frames and mixes 0.999 * clean + 0.001 * noise.

    Args:
        clean_rows: [N, 96] clean patchified condition latents, in the
            layout's img_pos order.
        shapes: Latent (T, H, W) of each condition.
        target_latent_t: Latent frames of the target video.
        seed: Request seed.

    Returns:
        [N, 96] fp32 augmented rows.
    """
    ratio = torch.tensor(h3_schedule.VISUAL_COND_TIMESTEP,
                         dtype=torch.float32)
    out = []
    offset = 0
    for latent_t, latent_h, latent_w in shapes:
        gen = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(1, VIDEO_CHANNELS, target_latent_t + len(shapes),
                            latent_h, latent_w, generator=gen,
                            dtype=torch.float32)[:, :, :latent_t]
        noise_rows = patchify(noise)
        clean = clean_rows[offset:offset + noise_rows.shape[0]].float()
        if clean.shape != noise_rows.shape:
            raise ValueError(f'condition rows {tuple(clean.shape)} do not '
                             f'match shape {(latent_t, latent_h, latent_w)}')
        out.append(ratio * clean + (1.0 - ratio) * noise_rows)
        offset += noise_rows.shape[0]
    if offset != clean_rows.shape[0]:
        raise ValueError(f'{clean_rows.shape[0]} condition rows, shapes '
                         f'cover {offset}')
    return torch.cat(out) if out else clean_rows.float()
