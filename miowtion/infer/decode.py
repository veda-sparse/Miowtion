"""Latents -> pixels / waveform with the release's VAEs, and mp4 muxing.

Decoding inverts miowtion.train.encode: rows are unpatchified and
de-normalized with the VAE's latents_mean / latents_std before the release's
own decoders run. The audio VAE is mono; the two stereo channels are decoded
as a batch of two (rows are channel-major).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import wave
from collections.abc import Sequence

import numpy as np
import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import noise
from miowtion.train import encode

# Title bar height as a fraction of the frame height, and the font (looked up
# by name in the system font directories; Pillow's default font otherwise).
_TITLE_BAR_FRACTION = 0.08
_TITLE_FONT = 'DejaVuSans-Bold.ttf'
# Separator between side-by-side videos, in pixels (even, for yuv420p).
_SEPARATOR = 8
# The Turbo LoRA's reference generator scales the waveform down only when
# its std exceeds 1/5, to avoid clipping on loud outputs.
_LOUDNESS_STD_FACTOR = 5.0


def _latent_stats(component_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(os.path.join(component_dir, 'config.json')) as f:
        stats = json.load(f)
    return (torch.tensor(stats['latents_mean'], dtype=torch.float32),
            torch.tensor(stats['latents_std'], dtype=torch.float32))


def video_latent(rows: torch.Tensor, geometry: h3_geometry.Geometry,
                 mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N_video, 96] normalized rows -> [1, 24, T, H, W] VAE latent."""
    z = noise.unpatchify(rows.float().cpu(), geometry.latent_t,
                         geometry.latent_h, geometry.latent_w)
    return z * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)


def audio_latent(rows: torch.Tensor, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    """[2 * audio_t, C] channel-major rows -> [2, C, audio_t] latent.

    Row c * audio_t + t holds stereo channel c at latent frame t; the mono
    audio VAE decodes the two channels as a batch.
    """
    z = rows.float().cpu().view(2, -1, mean.numel()).transpose(1, 2)
    return z * std.view(1, -1, 1) + mean.view(1, -1, 1)


class Decoder:
    """Video and audio VAE decoders of one release variant."""

    def __init__(self, variant_dir: str, device: torch.device):
        self.device = device
        video_pkg = encode.import_release_package(
            variant_dir, 'video_vae.minimax_h3_video_vae')
        audio_pkg = encode.import_release_package(
            variant_dir, 'audio_vae.minimax_h3_audio_vae')
        video_dir = os.path.join(variant_dir, 'video_vae')
        audio_dir = os.path.join(variant_dir, 'audio_vae')
        self.video_vae = video_pkg.MiniMaxH3VideoVAE.from_pretrained(
            video_dir).to(device).eval()
        self.audio_vae = audio_pkg.MiniMaxH3AudioVAE.from_pretrained(
            audio_dir).to(device).eval()
        self.video_mean, self.video_std = _latent_stats(video_dir)
        self.audio_mean, self.audio_std = _latent_stats(audio_dir)
        with open(os.path.join(audio_dir, 'config.json')) as f:
            self.sample_rate = int(json.load(f)['sample_rate'])

    @torch.no_grad()
    def video(self, rows: torch.Tensor,
              geometry: h3_geometry.Geometry) -> np.ndarray:
        """[N_video, 96] rows -> [frames, H, W, 3] uint8."""
        z = video_latent(rows, geometry, self.video_mean, self.video_std)
        dtype = next(self.video_vae.parameters()).dtype
        recon = self.video_vae.decode_base(z.to(self.device, dtype),
                                           frame_num=geometry.frame_count)
        pixels = self.video_vae.processor.revert_tensor(recon.float())
        frames = pixels[0].permute(1, 2, 3, 0).mul(255).round()
        return frames.to(torch.uint8).cpu().numpy()

    @torch.no_grad()
    def audio(self, rows: torch.Tensor) -> torch.Tensor:
        """[2 * audio_t, 32] channel-major rows -> [2, samples] fp32."""
        z = audio_latent(rows, self.audio_mean, self.audio_std)
        dtype = next(self.audio_vae.parameters()).dtype
        wav = self.audio_vae.decode(z.to(self.device, dtype))[:, 0].float()
        std = wav.std() * _LOUDNESS_STD_FACTOR
        if std > 1.0:
            wav = wav / std
        return wav.clamp(-1.0, 1.0).cpu()


def _write_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    pcm = (waveform.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(pcm.shape[0])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.t().contiguous().numpy().tobytes())


def write_mp4(path: str, frames: np.ndarray,
              audio_tracks: Sequence[tuple[torch.Tensor, str]],
              sample_rate: int, fps: int = h3_geometry.FPS) -> None:
    """Encodes [F, H, W, 3] uint8 frames with ffmpeg (H.264 + AAC).

    Args:
        path: Output .mp4.
        frames: [F, H, W, 3] uint8, H and W even.
        audio_tracks: (waveform [2, samples], title) per audio stream, in
            order; the first is the default track.
        sample_rate: Audio sample rate.
        fps: Video frame rate.
    """
    _, height, width, _ = frames.shape
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-']
        maps = ['-map', '0:v']
        for i, (waveform, title) in enumerate(audio_tracks):
            wav_path = os.path.join(tmp, f'audio{i}.wav')
            _write_wav(wav_path, waveform, sample_rate)
            cmd += ['-i', wav_path]
            maps += ['-map', f'{i + 1}:a', f'-metadata:s:a:{i}',
                     f'title={title}']
        # No -frames:v: it ends the whole output as soon as the video frames
        # are written, cutting the audio short (the pipe ends at EOF anyway).
        cmd += maps + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16']
        if audio_tracks:
            cmd += ['-c:a', 'aac', '-b:a', '192k']
        cmd.append(path)
        subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                       check=True)


def add_title(frames: np.ndarray, title: str) -> np.ndarray:
    """Prepends a black bar with the centered white title to every frame."""
    from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel
    num_frames, height, width, _ = frames.shape
    bar_height = 2 * max(16, round(height * _TITLE_BAR_FRACTION / 2))
    size = round(bar_height * 0.6)
    try:
        font = ImageFont.truetype(_TITLE_FONT, size)
    except OSError:
        font = ImageFont.load_default(size=size)
    bar = Image.new('RGB', (width, bar_height), 'black')
    draw = ImageDraw.Draw(bar)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    draw.text(((width - (right - left)) / 2 - left,
               (bar_height - (bottom - top)) / 2 - top), title,
              fill='white', font=font)
    out = np.empty((num_frames, bar_height + height, width, 3), np.uint8)
    out[:, :bar_height] = np.asarray(bar)
    out[:, bar_height:] = frames
    return out


def side_by_side(videos: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenates equally sized [F, H, W, 3] videos left to right."""
    num_frames, height = videos[0].shape[:2]
    separator = np.zeros((num_frames, height, _SEPARATOR, 3), np.uint8)
    parts = []
    for i, video in enumerate(videos):
        if video.shape[:2] != (num_frames, height):
            raise ValueError('side-by-side videos differ in frames/height')
        parts += ([separator] if i else []) + [video]
    return np.concatenate(parts, axis=2)


def psnr_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PSNR (dB) of every frame pair of two [F, H, W, 3] uint8 videos."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = (diff ** 2).reshape(diff.shape[0], -1).mean(1)
    return 10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))
