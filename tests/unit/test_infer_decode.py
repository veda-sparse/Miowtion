"""Tests for miowtion.infer.decode (latent layout and helpers)."""

import numpy as np
import pytest
import torch

from miowtion.h3 import geometry
from miowtion.h3 import noise
from miowtion.infer import decode


def test_video_latent_inverts_encode_normalization():
    geo = geometry.geometry_from_latent_t('16:9', 37)
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(1, 24, geo.latent_t, geo.latent_h, geo.latent_w,
                    generator=gen)
    mean, std = torch.randn(24, generator=gen), torch.rand(24) + 0.5
    # As in encode.ConditionEncoder: normalize, then patchify.
    rows = noise.patchify((z - mean.view(1, -1, 1, 1, 1))
                          / std.view(1, -1, 1, 1, 1))
    torch.testing.assert_close(decode.video_latent(rows, geo, mean, std), z,
                               rtol=1e-6, atol=1e-6)
    ident = decode.video_latent(noise.patchify(z), geo, torch.zeros(24),
                                torch.ones(24))
    assert torch.equal(ident, z)


def test_audio_latent_is_channel_major():
    audio_t, channels = 7, 32
    z = torch.randn(2, channels, audio_t)
    # Row c * audio_t + t holds channel c at frame t.
    rows = torch.cat([z[0].t(), z[1].t()])
    out = decode.audio_latent(rows, torch.zeros(channels),
                              torch.ones(channels))
    assert torch.equal(out, z)


def test_psnr_per_frame():
    a = np.zeros((2, 4, 4, 3), np.uint8)
    b = a.copy()
    b[1] = 255
    psnr = decode.psnr_per_frame(a, b)
    assert psnr[0] > 90 and abs(psnr[1]) < 1e-6


def test_title_and_side_by_side_layout():
    pytest.importorskip('PIL')  # the `encode` extra
    frames = np.full((3, 64, 96, 3), 7, np.uint8)
    titled = decode.add_title(frames, 'Dense')
    bar = titled.shape[1] - 64
    assert bar % 2 == 0 and bar >= 32
    assert np.array_equal(titled[:, bar:], frames)  # content untouched
    assert titled[:, :bar].max() == 255  # white text drawn on the bar
    both = decode.side_by_side([titled, titled])
    assert both.shape == (3, titled.shape[1], 2 * 96 + 8, 3)
    assert np.array_equal(both[:, :, :96], titled)
    assert np.array_equal(both[:, :, -96:], titled)
