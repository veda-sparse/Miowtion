"""Tests for miowtion.mlx.dit against the torch reference (h3.model)."""

import dataclasses

import numpy as np
import pytest
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import block as mlx_block  # noqa: E402
from miowtion.mlx import dit as mlx_dit  # noqa: E402
from miowtion.mlx import interop  # noqa: E402

_CONFIG = dataclasses.replace(h3_config.H3Config.tiny(num_layers=2),
                              hidden_size=128, num_heads=4, head_dim=32,
                              ffn_dim=64, num_refiner_layers=2,
                              rope_freqs_per_axis=4)
_TEXT_LEN = 24


def _torch_model(seed=0):
    torch.manual_seed(seed)
    model = h3_model.H3DiT(_CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.05)
    model.dense_backend = 'math'
    return model.eval()


def _non_trunk(model):
    """The torch model's non-trunk weights as mlx.dit.NonTrunkWeights."""
    state = model.state_dict()
    tensors = {name: interop.from_torch(state[name])
               for name in mlx_dit.NON_TRUNK_NAMES}
    refiner = [interop.block_weights_from_torch(b)
               for b in model.token_refiner.blocks]
    return mlx_dit.NonTrunkWeights.from_tensors(tensors, refiner)


def _rel_l2(got, want):
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    return (got - want).norm().item() / max(want.norm().item(), 1e-12)


def test_rope_tables_are_bitwise_equal_to_torch():
    rope = h3_model.Rope(_CONFIG)
    positions = torch.rand(40, 3, dtype=torch.float64) * 97.0
    want_cos, want_sin = rope.cos_sin(positions)
    got_cos, got_sin = mlx_dit.rope_cos_sin(positions.numpy(), _CONFIG)
    assert got_cos.shape == tuple(want_cos.shape)
    assert torch.equal(interop.to_torch(got_cos), want_cos)
    assert torch.equal(interop.to_torch(got_sin), want_sin)


def test_rope_rejects_bad_positions():
    with pytest.raises(ValueError, match=r'\[S, 3\]'):
        mlx_dit.rope_cos_sin(np.zeros((4, 2), dtype=np.float64), _CONFIG)


def test_time_frequencies_match_torch_to_one_ulp():
    """numpy's and torch's fp32 exp disagree on the last bit of a few
    frequencies, so the sinusoid cannot be bitwise equal; nothing else may
    differ."""
    timesteps = torch.tensor([0.0, 0.137, 0.5, 1.0])
    half = _CONFIG.freq_dim // 2
    freqs = torch.exp(-np.log(10000.0)
                      * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps[:, None] * freqs[None]
    want = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    got = torch.from_numpy(
        mlx_dit._time_frequencies(timesteps.numpy(), _CONFIG.freq_dim))
    assert got.dtype == torch.float32 and got.shape == want.shape
    ulp = torch.finfo(torch.float32).eps
    assert (got - want).abs().max().item() <= ulp
    assert torch.equal(got[0], want[0])  # t = 0: cos 0 and sin 0 exactly


def test_adaln_input_matches_torch():
    model = _torch_model()
    weights = _non_trunk(model)
    timesteps = torch.tensor([0.0, 0.3, 0.97])
    want = model.adaln_input(timesteps)
    got = interop.to_torch(mlx_dit.adaln_input(weights, timesteps.numpy()))
    assert got.shape == want.shape and got.dtype == want.dtype
    assert _rel_l2(got, want) < 5e-3


def test_adaln_tables_match_the_block_projection():
    model = _torch_model()
    timesteps = torch.tensor([0.1, 0.8])
    adaln_input = model.adaln_input(timesteps)
    want = model.blocks[1].adaln_proj(adaln_input)
    linear = model.blocks[1].adaln_proj.linear
    tensors = {'weight': interop.from_torch(linear.weight),
               'bias': interop.from_torch(linear.bias)}
    got = mlx_dit.block_adaln_tables(tensors,
                                     interop.from_torch(adaln_input), _CONFIG)
    assert len(got) == len(want) == 6
    for g, w in zip(got, want):
        assert tuple(w.shape) == g.shape
        assert _rel_l2(interop.to_torch(g), w) < 5e-3


def test_refine_text_matches_torch():
    model = _torch_model()
    weights = _non_trunk(model)
    torch.manual_seed(1)
    text = (torch.randn(_TEXT_LEN, _CONFIG.text_dim) * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        want = model.refine_text(text)
    got = interop.to_torch(mlx_dit.refine_text(weights,
                                               interop.from_torch(text),
                                               _CONFIG))
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    assert _rel_l2(got, want) < 5e-3


def test_embed_matches_torch():
    model = _torch_model()
    weights = _non_trunk(model)
    seq_len, n_video, n_audio = 64, 10, 6
    torch.manual_seed(2)
    text = (torch.randn(_TEXT_LEN, _CONFIG.hidden_size)).to(torch.bfloat16)
    video = torch.randn(n_video, _CONFIG.video_patch_dim)
    audio = torch.randn(n_audio, _CONFIG.audio_channels)
    img_pos = torch.arange(_TEXT_LEN, _TEXT_LEN + n_video)
    audio_pos = torch.arange(_TEXT_LEN + n_video,
                             _TEXT_LEN + n_video + n_audio)
    clip = h3_model.ClipInputs(
        text=text, img_pos=img_pos, audio_pos=audio_pos,
        target_img_pos=img_pos, target_audio_pos=audio_pos,
        rope=(torch.zeros(1), torch.zeros(1)), seq_len=seq_len,
        used=_TEXT_LEN + n_video + n_audio)
    with torch.no_grad():
        want = model.embed(clip, video, audio)
    got = interop.to_torch(mlx_dit.embed(
        weights, _CONFIG, seq_len, interop.from_torch(text),
        interop.from_torch(video), interop.from_torch(audio),
        interop.from_torch(img_pos.to(torch.int32)),
        interop.from_torch(audio_pos.to(torch.int32))))
    assert got.shape == want.shape
    # Text rows are copied, padding rows stay zero: both are bitwise equal.
    assert torch.equal(got[:_TEXT_LEN], want[:_TEXT_LEN])
    assert torch.equal(got[clip.used:], want[clip.used:])
    assert _rel_l2(got, want) < 5e-3


def test_final_layer_matches_torch():
    model = _torch_model()
    weights = _non_trunk(model)
    seq_len = 48
    torch.manual_seed(3)
    x = (torch.randn(seq_len, _CONFIG.hidden_size) * 0.5).to(torch.bfloat16)
    adaln_input = model.adaln_input(torch.tensor([0.2, 0.6]))
    slot = torch.randint(0, 2, (seq_len,))
    video_rows = torch.arange(4, 12)
    audio_rows = torch.arange(12, 16)
    with torch.no_grad():
        want_v, want_a = model.final_layer(x, adaln_input, slot, video_rows,
                                           audio_rows)
    got_v, got_a = mlx_dit.final_layer(
        weights, interop.from_torch(x), interop.from_torch(adaln_input),
        interop.from_torch(slot.to(torch.int32)),
        interop.from_torch(video_rows.to(torch.int32)),
        interop.from_torch(audio_rows.to(torch.int32)), _CONFIG)
    assert got_v.dtype == mx.float32 and got_a.dtype == mx.float32
    assert _rel_l2(interop.to_torch(got_v), want_v) < 5e-3
    assert _rel_l2(interop.to_torch(got_a), want_a) < 5e-3


def test_non_trunk_weights_validate_their_tensors():
    model = _torch_model()
    state = model.state_dict()
    tensors = {name: interop.from_torch(state[name])
               for name in mlx_dit.NON_TRUNK_NAMES[:-1]}
    with pytest.raises(KeyError, match='missing='):
        mlx_dit.NonTrunkWeights.from_tensors(tensors, ())


def test_non_trunk_weights_report_their_size():
    weights = _non_trunk(_torch_model())
    assert weights.nbytes == sum(
        p.numel() * p.element_size()
        for name, p in _torch_model().named_parameters()
        if not name.startswith(('blocks.', 'rope.'))
        and 'adaln_proj' not in name.replace('final_layer.adaln_proj',
                                             'final_adaln'))
