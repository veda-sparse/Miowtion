"""The sampler is the training loop: one trajectory from pure noise.

Every state along a teacher-driven Euler trajectory is one training
micro-step. The trajectory always advances with the teacher's velocity, so
the predictor fits where the original model attends, never a sparse
student's drift.

A few-step teacher must be rolled out on its own grid (e.g. 8 steps: sigma
>= 0.632 only); rolling it on the 50-step grid would query it at noise
levels it was never trained on.
"""

from __future__ import annotations

import dataclasses

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import noise
from miowtion.h3 import schedule as h3_schedule
from miowtion.train import data


@dataclasses.dataclass
class StepInputs:
    step: int
    timestep: h3_schedule.TimestepState
    video_rows: torch.Tensor  # [Nv, 96] fp32, conditions + target
    audio_rows: torch.Tensor  # [Na, 32] fp32, conditions + target


class Trajectory:
    """Rolls one clip from sigma = 1 to the end of the schedule."""

    def __init__(self, model: h3_model.H3DiT, cache: data.SampleCache,
                 sample: data.Sample, geometry: h3_geometry.Geometry,
                 schedule: h3_schedule.Schedule, seed: int,
                 device: torch.device, seq_len: int | None = None):
        """Builds the layout, refined text and initial noise.

        Args:
            model: The teacher DiT.
            cache: Sample cache.
            sample: Encoded presentation.
            geometry: Target geometry (shared by all ranks for this step).
            schedule: Sigma schedule to roll out on.
            seed: Noise seed (per rank and trajectory).
            device: Model device.
            seq_len: Optional fixed bucket length.
        """
        hidden, tags = cache.text(sample)
        self.sample = sample
        self.geometry = geometry
        self.schedule = schedule
        self.layout = h3_layout.pack(tags, geometry, sample.keyframes,
                                     sample.references, seq_len)
        self.clip = model.clip_inputs(
            self.layout, model.refine_text(hidden.to(device)), device)
        video, audio = noise.initial_noise(geometry, seed)
        cond_video, cond_audio = cache.conditions(sample)
        expected = int((~self.layout.update_mask).sum())
        if cond_video.shape[0] != expected:
            raise ValueError(f'{sample.id}: {cond_video.shape[0]} condition '
                             f'rows, layout needs {expected}')
        if cond_video.shape[0]:
            cond_video = noise.augment_visual_conditions(
                cond_video, self.layout.visual_cond_shapes,
                geometry.latent_t, seed)
        # Audio references stay clean (their timestep is pinned to 1.0).
        self.video_rows = torch.cat([cond_video.float(), video]).to(device)
        self.audio_rows = torch.cat([cond_audio.float(), audio]).to(device)
        self.n_cond_video = cond_video.shape[0]
        self.n_cond_audio = cond_audio.shape[0]
        self.step = 0

    @property
    def done(self) -> bool:
        return self.step >= self.schedule.num_steps

    def inputs(self) -> StepInputs:
        t_video, t_audio = self.schedule.timesteps(self.step)
        state = h3_schedule.build_timestep_state(self.layout, t_video,
                                                 t_audio)
        return StepInputs(self.step, state.to(self.video_rows.device),
                          self.video_rows, self.audio_rows)

    def timestep_sets(self) -> list[torch.Tensor]:
        """Distinct timestep sets of every step (for AdaLN tables)."""
        return [h3_schedule.build_timestep_state(
            self.layout, *self.schedule.timesteps(s)).timesteps
                for s in range(self.schedule.num_steps)]

    @torch.no_grad()
    def advance(self, video_v: torch.Tensor, audio_v: torch.Tensor) -> None:
        """Euler step of the target rows with the teacher velocity."""
        cur, nxt = (self.schedule.video[self.step],
                    self.schedule.video[self.step + 1])
        h3_schedule.euler_step_(self.video_rows[self.n_cond_video:], video_v,
                                cur, nxt)
        cur, nxt = (self.schedule.audio[self.step],
                    self.schedule.audio[self.step + 1])
        h3_schedule.euler_step_(self.audio_rows[self.n_cond_audio:], audio_v,
                                cur, nxt)
        self.step += 1


def schedule_timestep_sets(schedule: h3_schedule.Schedule,
                           visual_conditions: bool,
                           audio_references: bool) -> list[torch.Tensor]:
    """Every distinct timestep set a run can need, without building clips.

    A step's set depends only on which row kinds exist: target video
    (t_video), target audio (t_audio), visual conditions (floor 0.999) and
    audio references (floor 1.0). All enabled combinations are returned,
    since t2va and fl2va samples may be mixed in one run.
    """
    sets = {}
    for step in range(schedule.num_steps):
        t_video, t_audio = schedule.timesteps(step)
        for visual in {False, visual_conditions}:
            for audio in {False, audio_references}:
                values = [t_video, t_audio]
                if visual:
                    values.append(max(t_video,
                                      h3_schedule.VISUAL_COND_TIMESTEP))
                if audio:
                    values.append(max(t_audio,
                                      h3_schedule.AUDIO_COND_TIMESTEP))
                ts = torch.unique(torch.tensor(values, dtype=torch.float32),
                                  sorted=True)
                sets[tuple(ts.tolist())] = ts
    return list(sets.values())
