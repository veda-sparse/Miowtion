"""Tests for miowtion.h3.model and miowtion.h3.weights."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from miowtion.h3 import config as h3_config
from miowtion.h3 import geometry
from miowtion.h3 import layout
from miowtion.h3 import model as h3_model
from miowtion.h3 import noise
from miowtion.h3 import schedule
from miowtion.h3 import weights

_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party',
                     'MiniMax-H3')


def _random_model(cfg, seed=0):
    torch.manual_seed(seed)
    m = h3_model.H3DiT(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(std=0.05)
    m.dense_backend = 'math'
    return m


def _write_checkpoint(m, directory, num_files=2):
    """Writes m in the released layout: per-head interleaved fused QKV."""
    cfg = m.config
    perm = weights.qkv_row_permutation(cfg.num_heads, cfg.head_dim)
    tensors = {}
    for name, value in m.state_dict().items():
        value = value.clone()
        if name.endswith('attn.qkv_proj.weight'):
            interleaved = torch.empty_like(value)
            interleaved[perm] = value
            value = interleaved
        tensors[name] = value
    names = sorted(tensors)
    weight_map = {}
    for i in range(num_files):
        part = {n: tensors[n] for n in names[i::num_files]}
        filename = f'model-{i:05d}.safetensors'
        save_file(part, os.path.join(directory, filename))
        weight_map.update({n: filename for n in part})
    with open(os.path.join(directory, 'model.safetensors.index.json'),
              'w') as f:
        json.dump({'weight_map': weight_map}, f)


def test_config_from_checkpoints():
    for variant in ('FL2VA', 'Ref2VA'):
        cfg = h3_config.H3Config.from_pretrained(
            os.path.join(_REPO, variant, 'transformer'))
        assert cfg == h3_config.H3Config()


def test_checkpoint_dtypes_match_model():
    m = h3_model.H3DiT(h3_config.H3Config.tiny())
    for name, p in m.named_parameters():
        fp32 = name.startswith(h3_config.FP32_PARAM_PREFIXES)
        assert p.dtype == (torch.float32 if fp32 else torch.bfloat16), name


def test_qkv_permutation_pins_interleaved_layout():
    perm = weights.qkv_row_permutation(num_heads=3, head_dim=2)
    # Model rows: q_h0 q_h1 q_h2 k_h0 ... ; checkpoint rows: h0(q k v) ...
    assert perm.tolist() == [0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
                             4, 5, 10, 11, 16, 17]


def test_load_roundtrip(tmp_path):
    cfg = h3_config.H3Config.tiny()
    src = _random_model(cfg, seed=0)
    _write_checkpoint(src, tmp_path)
    dst = _random_model(cfg, seed=1)
    weights.load_dit_weights(dst, str(tmp_path))
    for (name, a), (_, b) in zip(src.state_dict().items(),
                                 dst.state_dict().items()):
        assert torch.equal(a, b), name


def test_load_is_strict(tmp_path):
    cfg = h3_config.H3Config.tiny()
    src = _random_model(cfg)
    _write_checkpoint(src, tmp_path)
    bigger = h3_model.H3DiT(h3_config.H3Config.tiny(num_layers=3))
    with pytest.raises(KeyError):
        weights.load_dit_weights(bigger, str(tmp_path))


def _clip(cfg, text_len=12):
    g = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    lay = layout.pack(torch.ones(text_len, dtype=torch.long), g)
    return g, lay


def test_forward_shapes_and_determinism():
    cfg = h3_config.H3Config.tiny()
    m = _random_model(cfg)
    g, lay = _clip(cfg)
    text = m.refine_text(torch.randn(lay.used - lay.target.num_rows
                                     - g.num_audio_rows, cfg.text_dim))
    clip = m.clip_inputs(lay, text, torch.device('cpu'))
    video, audio = noise.initial_noise(g, 0)
    state = schedule.build_timestep_state(lay, 0.3, 0.1)
    with torch.no_grad():
        v1, a1 = m(clip, video, audio, state)
        v2, a2 = m(clip, video, audio, state)
    assert v1.shape == (g.num_video_tokens, 96) and v1.dtype == torch.float32
    assert a1.shape == (g.num_audio_rows, 32)
    assert torch.equal(v1, v2) and torch.equal(a1, a2)


def test_padding_rows_do_not_leak():
    cfg = h3_config.H3Config.tiny()
    m = _random_model(cfg)
    g, lay = _clip(cfg)
    padded = layout.pack(torch.ones(12, dtype=torch.long), g,
                         seq_len=lay.seq_len + 128)
    text = m.refine_text(torch.randn(12, cfg.text_dim))
    video, audio = noise.initial_noise(g, 0)
    outs = []
    for lay_i in (lay, padded):
        clip = m.clip_inputs(lay_i, text, torch.device('cpu'))
        state = schedule.build_timestep_state(lay_i, 0.3, 0.1)
        with torch.no_grad():
            outs.append(m(clip, video, audio, state))
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-2, atol=1e-2)


def test_precomputed_adaln_is_bitwise_equal():
    cfg = h3_config.H3Config.tiny()
    m = _random_model(cfg)
    g, lay = _clip(cfg)
    text = m.refine_text(torch.randn(12, cfg.text_dim))
    clip = m.clip_inputs(lay, text, torch.device('cpu'))
    video, audio = noise.initial_noise(g, 0)
    state = schedule.build_timestep_state(lay, 0.3, 0.1)
    table = m.precompute_adaln(state.timesteps)
    with torch.no_grad():
        ref = m(clip, video, audio, state)
        cached = m(clip, video, audio, state, adaln_table=table)
    assert torch.equal(ref[0], cached[0]) and torch.equal(ref[1], cached[1])


def test_chunked_rope_is_bitwise_identical(monkeypatch):
    from miowtion.h3 import model as h3_model
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 3, 128, generator=gen).to(torch.bfloat16)
    cos = torch.randn(1000, 1, 96, generator=gen).to(torch.bfloat16)
    sin = torch.randn(1000, 1, 96, generator=gen).to(torch.bfloat16)
    whole = h3_model.apply_rope(x, cos, sin)
    monkeypatch.setattr(h3_model, '_ROPE_CHUNK_ROWS', 128)
    assert torch.equal(h3_model.apply_rope(x, cos, sin), whole)
