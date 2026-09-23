"""Tests for miowtion.h3.schedule and miowtion.h3.noise."""

import os

import pytest
import torch

from miowtion.h3 import geometry
from miowtion.h3 import layout
from miowtion.h3 import noise
from miowtion.h3 import schedule

_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party',
                     'MiniMax-H3')


def test_shift_scales_from_checkpoint():
    for variant in ('FL2VA', 'Ref2VA'):
        scales = schedule.ShiftScales.from_checkpoint(
            os.path.join(_REPO, variant))
        assert (scales.video, scales.audio) == (12.0, 3.0)


def test_base_and_few_step_schedules():
    base = schedule.Schedule.build(50, schedule.ShiftScales(12.0, 3.0))
    assert base.num_steps == 49
    assert base.video[0] == 1.0 and base.video[-1] == 0.0
    few = schedule.Schedule.build(9, schedule.ShiftScales(12.0, 3.0))
    assert [round(s, 3) for s in few.video] == [
        1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.8, 0.632, 0.0]
    assert min(few.video[:-1]) > 0.63


def test_euler_direction():
    x = torch.zeros(4, dtype=torch.float32)
    v = torch.ones(4)
    schedule.euler_step_(x, v, 1.0, 0.75)
    # v = x0 - noise points towards the clean sample: x moves by +0.25 v.
    assert torch.equal(x, torch.full((4,), 0.25))


def test_timestep_state_slots():
    g = geometry.resolve_geometry('16:9', 5.0)
    lay = layout.pack(torch.ones(30, dtype=torch.long), g, keyframes=(0,))
    state = schedule.build_timestep_state(lay, 0.0, 0.0)
    assert state.timesteps.tolist() == pytest.approx([0.0, 0.999])
    state = schedule.build_timestep_state(lay, 0.2, 0.5)
    assert state.timesteps.tolist() == pytest.approx([0.2, 0.5, 0.999])
    slot = state.slot
    assert (slot[:30] == 0).all()  # text uses the video timestep
    assert (slot[lay.target_img_pos] == 0).all()
    assert (slot[lay.audio_pos] == 1).all()
    assert (slot[lay.cond_img_pos] == 2).all()
    assert (slot[lay.used:] == 0).all()  # padding uses the video timestep
    tags = lay.token_tags.clamp(min=0)
    assert torch.equal(state.adaln_index, tags + 3 * slot)


def test_patchify_roundtrip_and_noise_order():
    latent = torch.randn(1, 24, 3, 4, 6)
    rows = noise.patchify(latent)
    assert rows.shape == (3 * 2 * 3, 96)
    assert torch.equal(noise.unpatchify(rows, 3, 4, 6), latent)
    g = geometry.Geometry('16:9', 128, 64, 22, 7, 4, 8, 37)
    video, audio = noise.initial_noise(g, 7)
    gen = torch.Generator().manual_seed(7)
    raw = torch.randn(1, 24, 7, 4, 8, generator=gen)
    assert torch.equal(video, noise.patchify(raw))
    gen = torch.Generator().manual_seed(7)
    assert torch.equal(audio, torch.randn(74, 32, generator=gen))


def test_visual_condition_augmentation():
    clean = torch.randn(2 * 8, 96)
    shapes = ((1, 4, 8), (1, 4, 8))
    out = noise.augment_visual_conditions(clean, shapes, target_latent_t=7,
                                          seed=3)
    gen = torch.Generator().manual_seed(3)
    ref = torch.randn(1, 24, 9, 4, 8, generator=gen)[:, :, :1]
    expected = 0.999 * clean[:8] + (1 - torch.tensor(0.999)) * noise.patchify(
        ref)
    assert torch.equal(out[:8], expected)
    # Every condition restarts the same stream.
    expected_second = 0.999 * clean[8:] + (
        1 - torch.tensor(0.999)) * noise.patchify(ref)
    assert torch.equal(out[8:], expected_second)
    with pytest.raises(ValueError):
        noise.augment_visual_conditions(clean[:5], shapes, 7, 3)
