"""Tests for miowtion.h3.model and miowtion.h3.weights."""

import dataclasses
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
from miowtion.h3 import release
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


def test_chunked_adaln_ops_are_bitwise_identical(monkeypatch):
    from miowtion.h3 import model as h3_model
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 64, generator=gen).to(torch.bfloat16)
    h = torch.randn(1000, 64, generator=gen).to(torch.bfloat16)
    table = torch.randn(6, 64, generator=gen).to(torch.bfloat16)
    index = torch.randint(6, (1000,), generator=gen)
    mod = h3_model._modulate(x, 1.0 + table, table, index)
    res = h3_model._gated_residual(x, table, index, h)
    monkeypatch.setattr(h3_model, '_ADALN_CHUNK_ROWS', 128)
    assert torch.equal(h3_model._modulate(x, 1.0 + table, table, index), mod)
    assert torch.equal(h3_model._gated_residual(x, table, index, h), res)


_RELEASE_V2_CONFIG = {
    '_class_name': 'MiniMaxH3Transformer3DModel',
    'num_attention_heads': 56, 'attention_head_dim': 128,
    'hidden_size': 5376, 'num_layers': 50, 'num_refiner_layers': 2,
    'ffn_dim': 14336, 'in_channels': 24, 'audio_in_channels': 32,
    'patch_size': [1, 2, 2], 'text_dim': 5120, 'freq_dim': 256,
    'time_embed_hidden_dim': 5376, 'time_embed_dim': 2688,
    'rope_freq_dim': 16, 'rope_theta': 10000.0, 'norm_eps': 1e-05,
    'qk_norm_eps': 1e-05, 'final_norm_eps': 1e-05,
}


def _write_config(directory, raw):
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(raw, f)
    return str(directory)


def test_config_reads_the_diffusers_release_schema(tmp_path):
    """The diffusers port renamed most keys but describes the same model.

    Same architecture, opposite fused-mlp order: that flag is the one
    field the two releases genuinely disagree on.
    """
    cfg = h3_config.H3Config.from_pretrained(
        _write_config(tmp_path, _RELEASE_V2_CONFIG))
    assert cfg == dataclasses.replace(h3_config.H3Config(),
                                      mlp_gate_first=False)


def test_config_refuses_an_unknown_release(tmp_path):
    """A third release would have its own fused-mlp order; never guess."""
    raw = dict(_RELEASE_V2_CONFIG, _class_name='MiniMaxH4DiTModel')
    with pytest.raises(ValueError, match='unknown transformer _class_name'):
        h3_config.H3Config.from_pretrained(_write_config(tmp_path, raw))
    raw.pop('_class_name')
    with pytest.raises(ValueError, match='unknown transformer _class_name'):
        h3_config.H3Config.from_pretrained(_write_config(tmp_path, raw))


def test_config_refuses_an_unknown_schema(tmp_path):
    raw = dict(_RELEASE_V2_CONFIG)
    del raw['ffn_dim']
    with pytest.raises(KeyError, match='ffn_dim'):
        h3_config.H3Config.from_pretrained(_write_config(tmp_path, raw))


def test_swiglu_half_order_follows_the_release():
    """Which half of the fused fc1 is gated is a property of the release.

    The first release stores [gate; up], the diffusers port [up; gate].
    Swapping them leaves every shape and every norm intact, so nothing
    else in the test suite can tell the two apart - pin both here.
    """
    config = h3_config.H3Config.tiny()
    x = torch.randn(8, config.hidden_size, dtype=torch.bfloat16)
    for gate_first in (True, False):
        mlp = h3_model.Mlp(dataclasses.replace(config,
                                               mlp_gate_first=gate_first))
        torch.manual_seed(0)
        for param in mlp.parameters():
            param.data.normal_(std=0.05)
        first, second = mlp.fc1(x).chunk(2, dim=-1)
        gate, up = (first, second) if gate_first else (second, first)
        want = mlp.fc2(torch.nn.functional.silu(gate) * up)
        assert torch.equal(mlp(x), want)
        assert not torch.equal(
            mlp(x), mlp.fc2(torch.nn.functional.silu(up) * gate))


def test_release_pins_the_fused_mlp_order():
    """The weights the CUDA path loads are [gate; up]."""
    assert release.mlp_gate_first(release.SCHEMA_H3) is True
    assert release.mlp_gate_first(release.SCHEMA_DIFFUSERS) is False
    assert h3_config.H3Config.from_pretrained(
        os.path.join(_REPO, 'FL2VA', 'transformer')).mlp_gate_first is True


def test_load_refuses_a_release_the_model_does_not_expect(tmp_path):
    """Both halves have the same shape, so only this check can catch it."""
    cfg = h3_config.H3Config.tiny()
    _write_checkpoint(_random_model(cfg), tmp_path)
    swapped = h3_model.H3DiT(dataclasses.replace(cfg, mlp_gate_first=False))
    with pytest.raises(ValueError, match='fused mlp.fc1'):
        weights.load_dit_weights(swapped, str(tmp_path))
