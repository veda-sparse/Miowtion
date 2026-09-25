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
# Frames whose de-normalization runs at once. The step is elementwise per
# frame, so chunking is exact; unchunked, the fp32 copy of a 14.4 s 16:9
# clip (345 frames of 1344x768) is 4 GB and exhausts a 24 GB GPU.
_REVERT_CHUNK_FRAMES = 32
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

    def __init__(self, variant_dir: str, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        """Loads both VAEs on `device`.

        Args:
            variant_dir: Release variant directory (FL2VA / Ref2VA).
            device: Where the decoders run.
            dtype: Video VAE weight dtype. The checkpoints are fp32, but
                the video
                decoder is a 2.4B-parameter ViT3D: fp32 leaves the tensor
                cores idle and SDPA falls back off its flash kernel, so
                decoding a 14.4 s 16:9 clip takes 165 s and 16.1 GiB
                against 50.9 s and 9.3 GiB in bf16, at 51.3 dB PSNR.
        """
        self.device = device
        video_pkg = encode.import_release_package(
            variant_dir, 'video_vae.minimax_h3_video_vae')
        audio_pkg = encode.import_release_package(
            variant_dir, 'audio_vae.minimax_h3_audio_vae')
        video_dir = os.path.join(variant_dir, 'video_vae')
        audio_dir = os.path.join(variant_dir, 'audio_vae')
        self.video_vae = video_pkg.MiniMaxH3VideoVAE.from_pretrained(
            video_dir).to(device=device, dtype=dtype).eval()
        # The audio VAE stays fp32: it is small enough that its dtype does
        # not show up in the decode time, so there is nothing to trade.
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
        out = np.empty((recon.shape[2], recon.shape[3], recon.shape[4], 3),
                       np.uint8)
        for start in range(0, recon.shape[2], _REVERT_CHUNK_FRAMES):
            rows_ = slice(start, start + _REVERT_CHUNK_FRAMES)
            pixels = self.video_vae.processor.revert_tensor(
                recon[:, :, rows_].float())
            frames = pixels[0].permute(1, 2, 3, 0).mul(255).round()
            out[rows_] = frames.to(torch.uint8).cpu().numpy()
        return out

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


class DiffusersDecoder:
    """The same two VAEs, from the diffusers-port release layout.

    The original release ships the VAEs as python packages inside the
    checkpoint (what `Decoder` imports) and needs diffusers 0.32.2 to run
    them; the re-published port ships plain diffusers components
    (`vae/`, `audio_vae/`) whose classes only exist in diffusers >= 0.36.
    The two pins cannot live in one environment, so this class imports
    diffusers lazily and nothing else in the repo depends on it.

    The rows, the normalization and the returned types are identical to
    `Decoder`, so callers (and the mp4 writer) do not care which one they
    got.
    """

    def __init__(self, variant_dir: str, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        """Loads both VAEs on `device`.

        Args:
            variant_dir: Directory holding `vae/` and `audio_vae/`.
            device: Where the decoders run (`mps` on Apple silicon).
            dtype: Video VAE weight dtype; see Decoder for why bf16.

        Raises:
            ImportError: When the installed diffusers has no MiniMax-H3
                autoencoder (it landed in 0.36).
        """
        import diffusers  # pylint: disable=import-outside-toplevel

        missing = [n for n in ('AutoencoderKLMiniMaxH3',
                               'AutoencoderKLMiniMaxH3Audio')
                   if not hasattr(diffusers, n)]
        if missing:
            raise ImportError(
                f'diffusers {diffusers.__version__} has no {missing[0]}; '
                'the diffusers-port VAEs need diffusers >= 0.36')
        self.device = device
        video_dir = os.path.join(variant_dir, 'vae')
        audio_dir = os.path.join(variant_dir, 'audio_vae')
        self.video_vae = diffusers.AutoencoderKLMiniMaxH3.from_pretrained(
            video_dir, torch_dtype=dtype).to(device).eval()
        # Small enough that its dtype does not show up in the decode time.
        self.audio_vae = (
            diffusers.AutoencoderKLMiniMaxH3Audio.from_pretrained(
                audio_dir, torch_dtype=torch.float32).to(device).eval())
        self.video_mean, self.video_std = _latent_stats(video_dir)
        self.audio_mean, self.audio_std = _latent_stats(audio_dir)
        with open(os.path.join(audio_dir, 'config.json')) as f:
            self.sample_rate = int(json.load(f)['sampling_rate'])

    @torch.no_grad()
    def video(self, rows: torch.Tensor,
              geometry: h3_geometry.Geometry) -> np.ndarray:
        """[N_video, 96] rows -> [frames, H, W, 3] uint8."""
        z = video_latent(rows, geometry, self.video_mean, self.video_std)
        dtype = next(self.video_vae.parameters()).dtype
        recon = self.video_vae.decode(z.to(self.device, dtype)).sample
        # The decoder emits 4 * (T - 1) + 1 frames; a geometry whose frame
        # count is not that (aligned durations) keeps its leading frames.
        recon = recon[:, :, :geometry.frame_count]
        out = np.empty((recon.shape[2], recon.shape[3], recon.shape[4], 3),
                       np.uint8)
        for start in range(0, recon.shape[2], _REVERT_CHUNK_FRAMES):
            rows_ = slice(start, start + _REVERT_CHUNK_FRAMES)
            # diffusers returns [-1, 1]; the release package's
            # revert_tensor does the same affine map.
            pixels = recon[0, :, rows_].float().permute(1, 2, 3, 0)
            frames = pixels.add(1.0).mul(127.5).round().clamp(0.0, 255.0)
            out[rows_] = frames.to(torch.uint8).cpu().numpy()
        return out

    @torch.no_grad()
    def audio(self, rows: torch.Tensor) -> torch.Tensor:
        """[2 * audio_t, 32] channel-major rows -> [2, samples] fp32."""
        z = audio_latent(rows, self.audio_mean, self.audio_std)
        dtype = next(self.audio_vae.parameters()).dtype
        wav = self.audio_vae.decode(z.to(self.device, dtype)).sample
        wav = wav[:, 0].float()
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
