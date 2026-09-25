"""Tests for miowtion.mlx.model against the torch reference (h3.model)."""

import numpy as np
import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry as h3_geometry
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import noise as h3_noise
from miowtion.h3 import schedule as h3_schedule

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import convert  # noqa: E402
from miowtion.mlx import dit as mlx_dit  # noqa: E402
from miowtion.mlx import interop  # noqa: E402
from miowtion.mlx import model as mlx_model  # noqa: E402
from tests.unit.test_mlx_convert import _write_release  # noqa: E402

_CONFIG = h3_config.H3Config.tiny(num_layers=3)
_GEOMETRY = h3_geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
_TEXT_LEN = 12


def _torch_model(seed=0):
    torch.manual_seed(seed)
    model = h3_model.H3DiT(_CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.05)
    model.dense_backend = 'math'
    return model.eval()


def _non_trunk(model):
    state = model.state_dict()
    tensors = {name: interop.from_torch(state[name])
               for name in mlx_dit.NON_TRUNK_NAMES}
    refiner = [interop.block_weights_from_torch(b)
               for b in model.token_refiner.blocks]
    return mlx_dit.NonTrunkWeights.from_tensors(tensors, refiner)


def _tables(model, adaln_input):
    """The MLX AdaLN tables of every trunk block."""
    out = []
    for block in model.blocks:
        linear = block.adaln_proj.linear
        out.append(mlx_dit.block_adaln_tables(
            {'weight': interop.from_torch(linear.weight),
             'bias': interop.from_torch(linear.bias)},
            adaln_input, _CONFIG))
    return out


def _blocks(model):
    return [(i, interop.block_weights_from_torch(b))
            for i, b in enumerate(model.blocks)]


def _rel_l2(got, want):
    got, want = got.to(torch.float32), want.to(torch.float32)
    return (got - want).norm().item() / max(want.norm().item(), 1e-12)


def _setup():
    """A torch model, its MLX weights and one step's inputs on both sides."""
    model = _torch_model()
    lay = h3_layout.pack(torch.ones(_TEXT_LEN, dtype=torch.long), _GEOMETRY)
    torch.manual_seed(7)
    text_states = torch.randn(_TEXT_LEN, _CONFIG.text_dim)
    video, audio = h3_noise.initial_noise(_GEOMETRY, 0)
    state = h3_schedule.build_timestep_state(lay, 0.3, 0.1)

    weights = _non_trunk(model)
    clip = mlx_model.clip_inputs(
        weights, _CONFIG, interop.from_torch(text_states),
        lay.position_ids.numpy(), lay.img_pos.numpy(), lay.audio_pos.numpy(),
        lay.target_img_pos.numpy(), lay.target_audio_pos.numpy(),
        lay.seq_len, lay.used)
    timestep = mlx_model.TimestepInputs(
        timesteps=state.timesteps.numpy(),
        slot=interop.from_torch(state.slot.to(torch.int32)),
        adaln_index=interop.from_torch(state.adaln_index.to(torch.int32)))
    return model, weights, lay, text_states, video, audio, state, clip, \
        timestep


def test_velocity_matches_the_torch_forward():
    (model, weights, lay, text_states, video, audio, state, clip,
     timestep) = _setup()
    torch_clip = model.clip_inputs(lay, model.refine_text(text_states),
                                   torch.device('cpu'))
    with torch.no_grad():
        want_v, want_a = model(torch_clip, video, audio, state)
    adaln_input = mlx_dit.adaln_input(weights, timestep.timesteps)
    got_v, got_a = mlx_model.velocity(
        weights, _blocks(model), clip, _tables(model, adaln_input), timestep,
        interop.from_torch(video), interop.from_torch(audio), _CONFIG)
    assert got_v.shape == tuple(want_v.shape)
    assert got_a.shape == tuple(want_a.shape)
    assert _rel_l2(interop.to_torch(got_v), want_v) < 5e-3
    assert _rel_l2(interop.to_torch(got_a), want_a) < 5e-3


def test_clip_inputs_refines_the_text_and_builds_rope():
    model, weights, lay, text_states, *_, clip, _ = _setup()
    with torch.no_grad():
        want = model.refine_text(text_states)
    assert _rel_l2(interop.to_torch(clip.text), want) < 5e-3
    want_cos, want_sin = model.rope.cos_sin(lay.position_ids)
    assert torch.equal(interop.to_torch(clip.rope[0]), want_cos)
    assert torch.equal(interop.to_torch(clip.rope[1]), want_sin)
    assert clip.img_pos.dtype == mx.int32
    assert clip.used == lay.used and clip.seq_len == lay.seq_len


def test_clip_inputs_rejects_a_mismatched_position_table():
    model = _torch_model()
    weights = _non_trunk(model)
    with pytest.raises(ValueError, match='expected seq_len'):
        mlx_model.clip_inputs(
            weights, _CONFIG,
            interop.from_torch(torch.zeros(4, _CONFIG.text_dim)),
            np.zeros((8, 3)), np.zeros(1), np.zeros(1), np.zeros(1),
            np.zeros(1), 16, 8)


def test_velocity_checks_the_block_source():
    (model, weights, _, _, video, audio, _, clip, timestep) = _setup()
    adaln_input = mlx_dit.adaln_input(weights, timestep.timesteps)
    tables = _tables(model, adaln_input)
    args = (clip, tables, timestep, interop.from_torch(video),
            interop.from_torch(audio), _CONFIG)
    blocks = _blocks(model)
    with pytest.raises(ValueError, match='yielded 2, expected 0'):
        mlx_model.velocity(weights, blocks[::-1], *args)
    with pytest.raises(ValueError, match='yielded 2 blocks'):
        mlx_model.velocity(weights, blocks[:2], *args)
    with pytest.raises(ValueError, match='tables cover'):
        mlx_model.velocity(weights, blocks, clip, tables[:1], *args[2:])


def test_precompute_adaln_reads_every_block(tmp_path):
    _write_release(tmp_path, cfg=_CONFIG)
    model = _torch_model()
    weights = _non_trunk(model)
    timesteps = np.array([0.2, 0.7], dtype=np.float32)
    other = np.array([0.5], dtype=np.float32)
    with convert.ReleaseReader(str(tmp_path)) as reader:
        tables = mlx_model.precompute_adaln(reader, weights,
                                            [timesteps, other, timesteps])
        want = mlx_dit.block_adaln_tables(
            convert.adaln_tensors(reader, 1),
            mlx_dit.adaln_input(weights, timesteps), reader.config)
    assert len(tables) == 2  # the repeated set is computed once
    assert timesteps in tables and other in tables
    blocks = tables.get(timesteps)
    assert len(blocks) == _CONFIG.num_layers
    rows = len(timesteps) * h3_config.MODALITY_NUM
    for table in blocks[1]:
        assert table.shape == (rows, _CONFIG.hidden_size)
    for got, expected in zip(blocks[1], want):
        assert mx.array_equal(got, expected)
    assert tables.get(other)[0][0].shape == (h3_config.MODALITY_NUM,
                                             _CONFIG.hidden_size)
    with pytest.raises(KeyError, match='no AdaLN table'):
        tables.get(np.array([0.25], dtype=np.float32))


def test_timestep_key_does_not_round():
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.1, np.nextafter(np.float32(0.2), np.float32(1.0))],
                 dtype=np.float32)
    assert mlx_model.timestep_key(a) != mlx_model.timestep_key(b)
    assert mlx_model.timestep_key(a) == mlx_model.timestep_key(a.copy())


def test_precompute_adaln_needs_a_set(tmp_path):
    _write_release(tmp_path, cfg=_CONFIG)
    weights = _non_trunk(_torch_model())
    with convert.ReleaseReader(str(tmp_path)) as reader:
        with pytest.raises(ValueError, match='timestep_sets is empty'):
            mlx_model.precompute_adaln(reader, weights, [])
