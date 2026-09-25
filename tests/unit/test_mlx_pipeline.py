"""Tests for miowtion.mlx.pipeline against the torch denoise loop."""

import numpy as np
import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import noise as h3_noise
from miowtion.h3 import schedule as h3_schedule

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import dit as mlx_dit  # noqa: E402
from miowtion.mlx import interop  # noqa: E402
from miowtion.mlx import model as mlx_model  # noqa: E402
from miowtion.mlx import pipeline  # noqa: E402
from tests.unit.test_mlx_model import _blocks, _non_trunk, _rel_l2  # noqa
from tests.unit.test_mlx_model import _torch_model  # noqa: E402

_CONFIG = h3_config.H3Config.tiny(num_layers=3)
_GEOMETRY = h3_geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
_TEXT_LEN = 12


def _schedule(num_steps=2):
    return h3_schedule.turbo_schedule(
        num_steps, h3_schedule.ShiftScales(video=5.0, audio=2.0))


def _setup(num_steps=2):
    """A torch model and the MLX inputs of one clip."""
    model = _torch_model()
    lay = h3_layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), _GEOMETRY)
    torch.manual_seed(11)
    text_states = torch.randn(_TEXT_LEN, _CONFIG.text_dim)
    video, audio = h3_noise.initial_noise(_GEOMETRY, 0)
    weights = _non_trunk(model)
    clip = mlx_model.clip_inputs(
        weights, _CONFIG, interop.from_torch(text_states),
        lay.position_ids.numpy(), lay.img_pos.numpy(), lay.audio_pos.numpy(),
        lay.target_img_pos.numpy(), lay.target_audio_pos.numpy(),
        lay.seq_len, lay.used)
    return (model, weights, lay, text_states, video, audio, clip,
            _schedule(num_steps))


def _tables(model, weights, sets):
    """AdaLN tables of every set, straight from the torch projections."""
    tables = {}
    for timesteps in sets:
        adaln_input = mlx_dit.adaln_input(weights, timesteps)
        tables[mlx_model.timestep_key(timesteps)] = [
            mlx_dit.block_adaln_tables(
                {'weight': interop.from_torch(b.adaln_proj.linear.weight),
                 'bias': interop.from_torch(b.adaln_proj.linear.bias)},
                adaln_input, _CONFIG)
            for b in model.blocks]
    return mlx_model.AdalnTables(tables)


def test_euler_step_matches_torch_to_one_ulp():
    """torch's add_ fuses the multiply and the add; MLX cannot."""
    torch.manual_seed(3)
    x = torch.randn(17, 96)
    v = torch.randn(17, 96)
    want = x.clone()
    h3_schedule.euler_step_(want, v, 0.83, 0.41)
    got = interop.to_torch(pipeline.euler_step(
        interop.from_torch(x), interop.from_torch(v), 0.83, 0.41))
    assert got.dtype == torch.float32
    ulp = torch.finfo(torch.float32).eps
    assert ((got - want).abs() <= ulp * want.abs().clamp(min=1.0)).all()


def test_euler_step_rejects_non_fp32():
    x = mx.zeros((4, 8), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match='must be fp32'):
        pipeline.euler_step(x, x, 1.0, 0.5)


def test_timestep_inputs_match_the_torch_state():
    lay = h3_layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), _GEOMETRY)
    want = h3_schedule.build_timestep_state(lay, 0.4, 0.2)
    got = pipeline.timestep_inputs(lay, 0.4, 0.2)
    assert np.array_equal(got.timesteps, want.timesteps.numpy())
    assert np.array_equal(np.array(got.slot), want.slot.numpy())
    assert np.array_equal(np.array(got.adaln_index),
                          want.adaln_index.numpy())


def test_generate_matches_the_torch_denoise_loop():
    model, weights, lay, text_states, video, audio, clip, sched = _setup()
    sets = pipeline.schedule_timestep_sets(lay, sched)
    assert len(sets) == sched.num_steps

    # The torch reference: the same loop around h3.model.H3DiT.
    torch_clip = model.clip_inputs(lay, model.refine_text(text_states),
                                   torch.device('cpu'))
    want_video, want_audio = video.clone(), audio.clone()
    with torch.no_grad():
        for step in range(sched.num_steps):
            state = h3_schedule.build_timestep_state(
                lay, *sched.timesteps(step))
            v, a = model(torch_clip, want_video, want_audio, state)
            h3_schedule.euler_step_(want_video, v, sched.video[step],
                                    sched.video[step + 1])
            h3_schedule.euler_step_(want_audio, a, sched.audio[step],
                                    sched.audio[step + 1])

    traj = pipeline.Trajectory(lay, sched, interop.from_torch(video),
                               interop.from_torch(audio))
    got = pipeline.generate(weights, lambda: _blocks(model), clip, traj,
                            _tables(model, weights, sets), _CONFIG)
    assert traj.done and len(got.step_seconds) == sched.num_steps
    assert got.seconds > 0.0
    assert _rel_l2(interop.to_torch(got.video), want_video) < 5e-3
    assert _rel_l2(interop.to_torch(got.audio), want_audio) < 5e-3


def test_trajectory_keeps_condition_rows_fixed():
    _, _, lay, _, video, audio, _, sched = _setup()
    traj = pipeline.Trajectory(lay, sched, interop.from_torch(video),
                               interop.from_torch(audio), num_cond_video=3,
                               num_cond_audio=2)
    torch.manual_seed(5)
    v = torch.randn(video.shape[0] - 3, video.shape[1])
    a = torch.randn(audio.shape[0] - 2, audio.shape[1])
    traj.advance(interop.from_torch(v), interop.from_torch(a))
    assert traj.step == 1
    assert torch.equal(interop.to_torch(traj.video_rows[:3]), video[:3])
    assert torch.equal(interop.to_torch(traj.audio_rows[:2]), audio[:2])
    want = pipeline.euler_step(interop.from_torch(video[3:]),
                               interop.from_torch(v), sched.video[0],
                               sched.video[1])
    assert mx.array_equal(traj.video_rows[3:], want)


def test_trajectory_rejects_non_fp32_rows():
    _, _, lay, _, video, audio, _, sched = _setup()
    with pytest.raises(ValueError, match='must be fp32'):
        pipeline.Trajectory(lay, sched,
                            interop.from_torch(video.to(torch.bfloat16)),
                            interop.from_torch(audio))


def test_generate_refuses_a_finished_trajectory():
    model, weights, lay, _, video, audio, clip, sched = _setup()
    traj = pipeline.Trajectory(lay, sched, interop.from_torch(video),
                               interop.from_torch(audio))
    traj.step = sched.num_steps
    sets = pipeline.schedule_timestep_sets(lay, sched)
    with pytest.raises(ValueError, match='already at the end'):
        pipeline.generate(weights, lambda: _blocks(model), clip, traj,
                          _tables(model, weights, sets), _CONFIG)
