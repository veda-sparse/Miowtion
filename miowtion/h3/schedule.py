"""Flow-matching sampling schedule and per-row timestep assignment.

Conventions:
  * sigma = 1 is pure noise, sigma = 0 is clean.
  * The DiT receives t = 1 - sigma.
  * The model predicts v = x0 - noise; an Euler step is
    x <- x + (sigma_cur - sigma_next) * v, computed in fp32.
  * Video and audio have separate shift factors, read from the checkpoint's
    model_index.json (`_minimax_h3.sigma_shift_scales`).
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import layout as h3_layout

# Visual conditions (keyframes, reference images/videos) are mixed as
# 0.999 * clean + 0.001 * noise and fed with timestep max(t_video, 0.999);
# audio references stay clean with timestep 1.0.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


@dataclasses.dataclass(frozen=True)
class ShiftScales:
    video: float
    audio: float

    @classmethod
    def from_checkpoint(cls, variant_dir: str) -> ShiftScales:
        """Reads `<checkpoint>/<FL2VA|Ref2VA>/model_index.json`."""
        with open(os.path.join(variant_dir, 'model_index.json')) as f:
            scales = json.load(f)['_minimax_h3']['sigma_shift_scales']
        return cls(video=float(scales['video']), audio=float(scales['audio']))


def shift_sigmas(num_points: int, shift: float) -> list[float]:
    """Shifted sigma grid from 1 to 0.

    Args:
        num_points: Points of the base grid linspace(1, 0, num_points) in
            fp32. 50 points give the 49-step base schedule; 9 points give
            the 8-step few-step schedule.
        shift: Shift factor s in sigma' = s * b / (1 + (s - 1) * b).

    Returns:
        Sigmas ending in 0.0 (appended if the shifted grid does not).
    """
    if num_points < 2 or shift <= 0:
        raise ValueError(f'bad schedule num_points={num_points} shift={shift}')
    base = torch.linspace(1.0, 0.0, num_points, dtype=torch.float32)
    shifted = shift * base / (1 + (shift - 1) * base)
    shifted = torch.unique_consecutive(shifted)
    if shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.zeros(1)])
    return [float(v) for v in shifted.tolist()]


@dataclasses.dataclass(frozen=True)
class Schedule:
    """Paired video/audio sigma schedules of one trajectory."""

    video: tuple[float, ...]
    audio: tuple[float, ...]

    @classmethod
    def build(cls, num_points: int, scales: ShiftScales) -> Schedule:
        video = shift_sigmas(num_points, scales.video)
        audio = shift_sigmas(num_points, scales.audio)
        if len(video) != len(audio):
            raise ValueError('video/audio schedules differ in length')
        return cls(tuple(video), tuple(audio))

    @property
    def num_steps(self) -> int:
        return len(self.video) - 1

    def timesteps(self, step: int) -> tuple[float, float]:
        """(t_video, t_audio) fed to the DiT at `step`."""
        return 1.0 - self.video[step], 1.0 - self.audio[step]


def shift(u: float, scale: float) -> float:
    return scale * u / (1.0 + (scale - 1.0) * u)


def unshift(sigma: float, scale: float) -> float:
    return sigma / (scale + sigma * (1.0 - scale))


def turbo_schedule(num_steps: int, scales: ShiftScales) -> 'Schedule':
    """Few-step (Turbo LoRA) grid, closed form in float64.

    Video: sigma_i = shift(1 - i/n, s_video). Audio: the same base point
    re-shifted with s_audio, i.e. time_shift(sigma_video, s_video ->
    s_audio). Each stream then takes Euler steps on its own clock, as the
    Turbo LoRA's sampler does.
    """
    if num_steps < 1:
        raise ValueError(f'num_steps must be >= 1, got {num_steps}')
    video = [shift(1.0 - i / num_steps, scales.video)
             for i in range(num_steps + 1)]
    audio = [shift(unshift(s, scales.video), scales.audio) for s in video]
    audio[-1] = 0.0
    return Schedule(tuple(video), tuple(audio))


def euler_step_(x: torch.Tensor, velocity: torch.Tensor, sigma_cur: float,
                sigma_next: float) -> None:
    """In-place Euler step x += (sigma_cur - sigma_next) * v in fp32."""
    if x.dtype != torch.float32:
        raise ValueError(f'state must be fp32, got {x.dtype}')
    x.add_(velocity.float(), alpha=float(sigma_cur) - float(sigma_next))


@dataclasses.dataclass(frozen=True)
class TimestepState:
    """Per-forward timestep inputs of the DiT.

    Attributes:
        timesteps: [M] fp32 distinct timesteps, sorted ascending. M varies
            (e.g. at the first step t_video == t_audio == 0 collapse into one
            slot), so it must never be assumed to be 2.
        slot: [seq_len] int64 index into `timesteps` for every packed row.
        adaln_index: [seq_len] int64 AdaLN table row,
            clamp(tag, 0) + MODALITY_NUM * slot.
    """

    timesteps: torch.Tensor
    slot: torch.Tensor
    adaln_index: torch.Tensor

    def to(self, device: torch.device) -> TimestepState:
        return TimestepState(self.timesteps.to(device), self.slot.to(device),
                             self.adaln_index.to(device))


def build_timestep_state(layout: h3_layout.PackedLayout, t_video: float,
                         t_audio: float) -> TimestepState:
    """Assigns a timestep to every packed row.

    Text, padding and target video rows use t_video; target audio rows use
    t_audio; visual conditions use max(t_video, 0.999); audio conditions use
    max(t_audio, 1.0).
    """
    per_row = torch.full((layout.seq_len,), float(t_video),
                         dtype=torch.float32)
    per_row[layout.target_audio_pos] = float(t_audio)
    per_row[layout.cond_img_pos] = max(float(t_video), VISUAL_COND_TIMESTEP)
    per_row[layout.cond_audio_pos] = max(float(t_audio), AUDIO_COND_TIMESTEP)
    timesteps, slot = torch.unique(per_row, sorted=True, return_inverse=True)
    tags = layout.token_tags.clamp(min=0)
    return TimestepState(timesteps, slot,
                         tags + h3_config.MODALITY_NUM * slot)
