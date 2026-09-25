"""The denoise loop on MLX: schedule, Euler steps, one clip end to end.

The loop itself is tiny; what matters is where the weights come from. Every
step reads the whole trunk again (33B parameters do not stay resident), so
`generate` asks for a fresh block source per step and never holds more than
the prefetcher's slots. The AdaLN tables of every step are built once
beforehand (model.precompute_adaln), so the 26 GB of AdaLN projections are
read once per clip rather than once per step.

The trajectory state (the video and audio rows being denoised) is kept in
MLX and advanced with the same fp32 Euler step as h3.schedule.euler_step_;
the noise that starts it comes from torch, since only torch's generator
reproduces a seed bit for bit.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import schedule as h3_schedule
from miowtion.mlx import block as mlx_block
from miowtion.mlx import dit
from miowtion.mlx import model as mlx_model
from miowtion.utils import progress

_FP32 = mx.float32


def euler_step(x: mx.array, velocity: mx.array, sigma_cur: float,
               sigma_next: float) -> mx.array:
    """x + (sigma_cur - sigma_next) * v in fp32.

    Not bitwise equal to h3.schedule.euler_step_: torch's `add_(v, alpha)`
    emits a fused multiply-add (one rounding), MLX rounds the product and
    the sum separately, so results differ by at most one ulp (measured
    2.4e-07 on values of order 1). MLX has no fma op and no fp64 on the
    GPU, so the only way to reproduce torch here would be to route the
    step through a GEMM, which is not worth pinning a trajectory to the
    accumulation order of a matmul. One ulp of fp32 is far below the bf16
    trunk's own spread.

    Args:
        x: [N, C] fp32 state.
        velocity: [N, C] fp32 velocity.
        sigma_cur: Sigma of this step.
        sigma_next: Sigma of the next step.

    Returns:
        [N, C] fp32.

    Raises:
        ValueError: When either input is not fp32.
    """
    if x.dtype != _FP32 or velocity.dtype != _FP32:
        raise ValueError(f'state and velocity must be fp32, got {x.dtype} '
                         f'and {velocity.dtype}')
    delta = np.float32(float(sigma_cur) - float(sigma_next))
    return x + mx.array(delta) * velocity


def timestep_inputs(layout: h3_layout.PackedLayout, t_video: float,
                    t_audio: float) -> mlx_model.TimestepInputs:
    """The packed timestep assignment of one step, as MLX arrays.

    The assignment itself (which row gets which timestep) is
    h3.schedule.build_timestep_state: it is pure indexing on the host and
    identical for both backends.
    """
    state = h3_schedule.build_timestep_state(layout, t_video, t_audio)
    return mlx_model.TimestepInputs(
        timesteps=state.timesteps.numpy(),
        slot=mx.array(state.slot.numpy().astype(np.int32)),
        adaln_index=mx.array(state.adaln_index.numpy().astype(np.int32)))


def schedule_timestep_sets(layout: h3_layout.PackedLayout,
                           schedule: h3_schedule.Schedule
                           ) -> list[np.ndarray]:
    """The distinct timestep set of every step of the schedule."""
    return [timestep_inputs(layout, *schedule.timesteps(step)).timesteps
            for step in range(schedule.num_steps)]


class Trajectory:
    """Rolls one clip from sigma = 1 to the end of the schedule.

    Attributes:
        layout: The packed layout of the clip.
        schedule: The sigma schedule being rolled out.
        step: Index of the next step.
    """

    def __init__(self, layout: h3_layout.PackedLayout,
                 schedule: h3_schedule.Schedule, video: mx.array,
                 audio: mx.array, num_cond_video: int = 0,
                 num_cond_audio: int = 0):
        """Takes the initial rows (conditions first, then the noise).

        Args:
            layout: Packed layout of the clip.
            schedule: Sigma schedule.
            video: [Nv, video_patch_dim] fp32 rows in img_pos order.
            audio: [Na, audio_channels] fp32 rows in audio_pos order.
            num_cond_video: Leading video rows that are conditions and are
                never advanced.
            num_cond_audio: Leading audio rows that are references.

        Raises:
            ValueError: On non-fp32 rows.
        """
        if video.dtype != _FP32 or audio.dtype != _FP32:
            raise ValueError('rows must be fp32')
        self.layout = layout
        self.schedule = schedule
        self.video_rows = video
        self.audio_rows = audio
        self.num_cond_video = num_cond_video
        self.num_cond_audio = num_cond_audio
        self.step = 0

    @property
    def done(self) -> bool:
        return self.step >= self.schedule.num_steps

    def timestep(self) -> mlx_model.TimestepInputs:
        return timestep_inputs(self.layout,
                               *self.schedule.timesteps(self.step))

    def advance(self, video_v: mx.array, audio_v: mx.array) -> None:
        """Euler step of the target rows with the predicted velocity."""
        for rows, cond, v, sigmas in (
                ('video_rows', self.num_cond_video, video_v,
                 self.schedule.video),
                ('audio_rows', self.num_cond_audio, audio_v,
                 self.schedule.audio)):
            state = getattr(self, rows)
            stepped = euler_step(state[cond:], v, sigmas[self.step],
                                 sigmas[self.step + 1])
            setattr(self, rows, mx.concatenate([state[:cond], stepped])
                    if cond else stepped)
        mx.eval(self.video_rows, self.audio_rows)
        self.step += 1


@dataclasses.dataclass(frozen=True)
class Generated:
    """One generated clip.

    Attributes:
        video: [Nt, video_patch_dim] fp32 denoised target video rows.
        audio: [Nta, audio_channels] fp32 denoised target audio rows.
        step_seconds: Wall seconds of every denoise step.
    """

    video: mx.array
    audio: mx.array
    step_seconds: tuple[float, ...]

    @property
    def seconds(self) -> float:
        return sum(self.step_seconds)


def generate(weights: dit.NonTrunkWeights,
             blocks: Callable[[], mlx_model.BlockSource],
             clip: mlx_model.ClipInputs, traj: Trajectory,
             tables: mlx_model.AdalnTables, config: h3_config.H3Config,
             options: mlx_block.BlockOptions = mlx_block.BlockOptions(),
             ) -> Generated:
    """Rolls `traj` to the end of its schedule.

    Args:
        weights: The resident non-trunk weights.
        blocks: Called once per step; returns a source that yields blocks
            0 .. num_layers - 1 (a slab BlockPrefetcher). It is called per
            step because a prefetcher is consumed by one pass over the
            trunk.
        clip: Trajectory-static inputs (model.clip_inputs).
        traj: The trajectory to roll (mutated).
        tables: AdaLN tables covering every step's timestep set.
        config: Architecture.
        options: Block chunking and the optional Veda plan.

    Returns:
        The denoised target rows and the per-step wall times.

    Raises:
        ValueError: When the trajectory is already finished.
    """
    if traj.done:
        raise ValueError('trajectory is already at the end of its schedule')
    steps = progress.Progress('denoise', traj.schedule.num_steps, every=1)
    step_seconds = []
    while not traj.done:
        start = time.time()
        timestep = traj.timestep()
        video_v, audio_v = mlx_model.velocity(
            weights, blocks(), clip, tables.get(timestep.timesteps),
            timestep, traj.video_rows, traj.audio_rows, config, options)
        traj.advance(video_v, audio_v)
        step_seconds.append(time.time() - start)
        steps.update(f'step {traj.step - 1}: {step_seconds[-1]:.1f} s')
    return Generated(traj.video_rows[traj.num_cond_video:],
                     traj.audio_rows[traj.num_cond_audio:],
                     tuple(step_seconds))


def steps_summary(step_seconds: Sequence[float]) -> str:
    """One line: total, mean and the slowest step."""
    if not step_seconds:
        return 'no steps'
    return (f'{sum(step_seconds):.1f} s total, '
            f'{np.mean(step_seconds):.1f} s/step, '
            f'max {max(step_seconds):.1f} s')
