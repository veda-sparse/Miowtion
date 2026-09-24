"""Tests for miowtion.infer.decode (latent layout and helpers)."""

import math
import shutil
import subprocess

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


@pytest.mark.skipif(shutil.which('ffmpeg') is None
                    or shutil.which('ffprobe') is None, reason='needs ffmpeg')
def test_mp4_keeps_the_full_audio(tmp_path):
    fps, seconds, rate = 24, 2.0, 32000
    # Full-size frames: the truncation shows once video outpaces audio.
    frames = np.random.default_rng(0).integers(
        0, 256, (int(fps * seconds), 768, 1344, 3), dtype=np.uint8)
    wave = torch.sin(torch.linspace(0, 2000, int(rate * seconds)))
    path = str(tmp_path / 'a.mp4')
    decode.write_mp4(path, frames, [(torch.stack([wave, wave]) * 0.5, 'x'),
                                    (torch.stack([wave, wave]) * 0.2, 'y')],
                     rate, fps)
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,'
         'duration', '-of', 'csv=p=0', path], capture_output=True,
        text=True, check=True).stdout.split()
    durations = {}
    for line in out:
        kind, value = line.split(',')
        durations.setdefault(kind, []).append(float(value))
    assert abs(durations['video'][0] - seconds) < 0.05
    assert len(durations['audio']) == 2
    assert all(abs(d - seconds) < 0.1 for d in durations['audio'])


def test_assign_jobs_balances_longest_first():
    from miowtion.infer import pipeline
    # One device takes everything, in longest-first order.
    assert pipeline.assign_jobs([1.0, 5.0, 3.0], 1) == [[1, 2, 0]]
    # Two devices: 5 | 3 + 1.
    assert pipeline.assign_jobs([1.0, 5.0, 3.0], 2) == [[1], [2, 0]]
    # Ties go to the lowest index, so the split is deterministic.
    assert pipeline.assign_jobs([2.0, 2.0, 2.0, 2.0], 2) == [[0, 2], [1, 3]]
    assert pipeline.assign_jobs([], 3) == [[], [], []]
    with pytest.raises(ValueError):
        pipeline.assign_jobs([1.0], 0)


def test_geometry_cost_grows_with_tokens_squared():
    from miowtion.infer import pipeline
    short = geometry.geometry_from_latent_t('16:9', 37)
    long = geometry.geometry_from_latent_t('16:9', 102)
    square = geometry.geometry_from_latent_t('1:1', 37)
    assert pipeline.geometry_cost(long) > pipeline.geometry_cost(short)
    assert pipeline.geometry_cost(short) > pipeline.geometry_cost(square)
    ratio = pipeline.geometry_cost(long) / pipeline.geometry_cost(short)
    tokens = [math.prod(g.video_grid) for g in (long, short)]
    assert ratio == pytest.approx((tokens[0] / tokens[1]) ** 2)


def test_video_frame_chunking_is_exact(monkeypatch):
    """The de-normalization is elementwise, so chunking must not change it."""
    class _Processor:
        @staticmethod
        def revert_tensor(x):
            return (x * 0.5 + 0.25).clamp(0, 1)

    class _Vae:
        processor = _Processor()

        @staticmethod
        def parameters():
            yield torch.zeros(1)

        @staticmethod
        def decode_base(z, frame_num):
            del frame_num
            gen = torch.Generator().manual_seed(z.shape[2])
            return torch.randn(1, 3, 70, 8, 12, generator=gen)

    decoder = decode.Decoder.__new__(decode.Decoder)
    decoder.device = torch.device('cpu')
    decoder.video_vae = _Vae()
    decoder.video_mean, decoder.video_std = torch.zeros(24), torch.ones(24)
    geo = geometry.geometry_from_latent_t('1:1', 37)
    rows = torch.randn(geo.latent_t * (geo.latent_h // 2)
                       * (geo.latent_w // 2), 96)
    monkeypatch.setattr(decode, '_REVERT_CHUNK_FRAMES', 1000)
    whole = decoder.video(rows, geo)
    monkeypatch.setattr(decode, '_REVERT_CHUNK_FRAMES', 7)
    assert np.array_equal(decoder.video(rows, geo), whole)
    assert whole.shape == (70, 8, 12, 3)
