"""Tests for miowtion.mlx.convert against a synthetic released checkpoint."""

import dataclasses
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from miowtion.h3 import config as h3_config
from miowtion.h3 import release as h3_release

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import block as mlx_block  # noqa: E402
from miowtion.mlx import convert  # noqa: E402
from miowtion.mlx import interop  # noqa: E402
from miowtion.mlx import slab as mlx_slab  # noqa: E402

_CONFIG = dataclasses.replace(h3_config.H3Config.tiny(num_layers=2),
                              hidden_size=64, num_heads=4, head_dim=8,
                              ffn_dim=32, num_refiner_layers=1)


def _config_json(cfg):
    """The diffusers-port spelling of every field."""
    return {
        '_class_name': 'MiniMaxH3Transformer3DModel',
        'hidden_size': cfg.hidden_size, 'num_layers': cfg.num_layers,
        'num_refiner_layers': cfg.num_refiner_layers,
        'num_attention_heads': cfg.num_heads,
        'attention_head_dim': cfg.head_dim, 'ffn_dim': cfg.ffn_dim,
        'in_channels': cfg.video_channels,
        'audio_in_channels': cfg.audio_channels,
        'patch_size': list(cfg.patch_size), 'text_dim': cfg.text_dim,
        'freq_dim': cfg.freq_dim,
        'time_embed_hidden_dim': cfg.time_embed_hidden,
        'time_embed_dim': cfg.time_embed_dim,
        'rope_freq_dim': cfg.rope_freqs_per_axis,
        'rope_theta': cfg.rope_theta, 'norm_eps': cfg.norm_eps,
        'qk_norm_eps': cfg.qk_norm_eps, 'final_norm_eps': cfg.final_norm_eps,
    }


def _write_release(directory, cfg=_CONFIG, seed=0, shards=2):
    """A complete diffusers-layout checkpoint with random weights."""
    torch.manual_seed(seed)
    inner = cfg.num_heads * cfg.head_dim
    tensors = {}

    def put(key, *shape, dtype=torch.bfloat16):
        tensors[key] = torch.randn(*shape).to(dtype)

    for i in range(cfg.num_layers):
        for prefix in (f'transformer_blocks.{i}',):
            put(f'{prefix}.norm1.weight', cfg.hidden_size)
            put(f'{prefix}.norm2.weight', cfg.hidden_size)
            put(f'{prefix}.attn.norm_q.weight', cfg.head_dim)
            put(f'{prefix}.attn.norm_k.weight', cfg.head_dim)
            for part in ('to_q', 'to_k', 'to_v'):
                put(f'{prefix}.attn.{part}.weight', inner, cfg.hidden_size)
            put(f'{prefix}.attn.to_out.0.weight', cfg.hidden_size, inner)
            put(f'{prefix}.ff.net.0.proj.weight', 2 * cfg.ffn_dim,
                cfg.hidden_size)
            put(f'{prefix}.ff.net.2.weight', cfg.hidden_size, cfg.ffn_dim)
            put(f'{prefix}.adaln_proj.linear.weight',
                6 * h3_config.MODALITY_NUM * cfg.hidden_size,
                cfg.time_embed_dim)
            put(f'{prefix}.adaln_proj.linear.bias',
                6 * h3_config.MODALITY_NUM * cfg.hidden_size)
    for i in range(cfg.num_refiner_layers):
        prefix = f'token_refiner.refiner_blocks.{i}'
        put(f'{prefix}.norm1.weight', cfg.hidden_size)
        put(f'{prefix}.norm2.weight', cfg.hidden_size)
        put(f'{prefix}.attn.norm_q.weight', cfg.head_dim)
        put(f'{prefix}.attn.norm_k.weight', cfg.head_dim)
        for part in ('to_q', 'to_k', 'to_v'):
            put(f'{prefix}.attn.{part}.weight', inner, cfg.hidden_size)
        put(f'{prefix}.attn.to_out.0.weight', cfg.hidden_size, inner)
        put(f'{prefix}.ff.net.0.proj.weight', 2 * cfg.ffn_dim, cfg.hidden_size)
        put(f'{prefix}.ff.net.2.weight', cfg.hidden_size, cfg.ffn_dim)
    put('token_refiner.final_norm.weight', cfg.hidden_size)
    put('context_embedder.weight', cfg.hidden_size, cfg.text_dim)
    put('context_embedder.bias', cfg.hidden_size)
    put('norm_out.norm.weight', cfg.hidden_size)
    put('norm_out.linear.weight', 2 * cfg.hidden_size, cfg.time_embed_dim)
    put('norm_out.linear.bias', 2 * cfg.hidden_size)
    fp32 = {'dtype': torch.float32}
    put('proj_in.weight', cfg.hidden_size, cfg.video_patch_dim, **fp32)
    put('proj_in.bias', cfg.hidden_size, **fp32)
    put('audio_proj_in.weight', cfg.hidden_size, cfg.audio_channels, **fp32)
    put('audio_proj_in.bias', cfg.hidden_size, **fp32)
    put('proj_out.weight', cfg.video_patch_dim, cfg.hidden_size, **fp32)
    put('proj_out.bias', cfg.video_patch_dim, **fp32)
    put('audio_proj_out.weight', cfg.audio_channels, cfg.hidden_size, **fp32)
    put('audio_proj_out.bias', cfg.audio_channels, **fp32)
    put('time_embedder.linear_1.weight', cfg.time_embed_hidden, cfg.freq_dim,
        **fp32)
    put('time_embedder.linear_1.bias', cfg.time_embed_hidden, **fp32)
    put('time_embedder.linear_2.weight', cfg.time_embed_dim,
        cfg.time_embed_hidden, **fp32)
    put('time_embedder.linear_2.bias', cfg.time_embed_dim, **fp32)

    os.makedirs(directory, exist_ok=True)
    names = sorted(tensors)
    for i in range(shards):
        part = {n: tensors[n] for n in names[i::shards]}
        save_file(part, os.path.join(
            directory, f'diffusion_pytorch_model-{i:05d}.safetensors'))
    with open(os.path.join(directory, 'config.json'), 'w') as f:
        json.dump(_config_json(cfg), f)
    return tensors


def test_fuse_qkv_pins_the_interleaved_row_order():
    heads, dim, cols = 3, 2, 1
    parts = [mx.arange(heads * dim * cols).reshape(heads * dim, cols) + off
             for off in (0.0, 100.0, 200.0)]
    fused = convert.fuse_qkv(*parts, heads, dim)
    assert fused.shape == (3 * heads * dim, cols)
    # Head h occupies rows [h * 6, h * 6 + 6): q, then k, then v.
    assert fused.reshape(-1).tolist() == [
        0, 1, 100, 101, 200, 201,
        2, 3, 102, 103, 202, 203,
        4, 5, 104, 105, 204, 205]


def test_fuse_qkv_checks_shapes():
    q = mx.zeros((8, 4))
    with pytest.raises(ValueError, match='k has shape'):
        convert.fuse_qkv(q, mx.zeros((6, 4)), q, 2, 4)


def test_reader_detects_the_release_and_is_complete(tmp_path):
    _write_release(tmp_path)
    with convert.ReleaseReader(str(tmp_path)) as reader:
        assert reader.schema == h3_release.SCHEMA_DIFFUSERS
        assert reader.config == _CONFIG
        reader.check_complete()


def test_missing_tensor_is_reported(tmp_path):
    _write_release(tmp_path)
    os.remove(os.path.join(tmp_path,
                           'diffusion_pytorch_model-00001.safetensors'))
    with convert.ReleaseReader(str(tmp_path)) as reader:
        with pytest.raises(KeyError, match='keys missing'):
            reader.check_complete()


def test_trunk_tensors_match_the_checkpoint_bitwise(tmp_path):
    tensors = _write_release(tmp_path)
    with convert.ReleaseReader(str(tmp_path)) as reader:
        got = convert.trunk_tensors(reader, 1)
        weights = mlx_block.BlockWeights.from_tensors(got)
    prefix = 'transformer_blocks.1.'
    assert torch.equal(interop.to_torch(weights.norm2),
                       tensors[prefix + 'norm2.weight'])
    assert torch.equal(interop.to_torch(weights.fc1.weight),
                       tensors[prefix + 'ff.net.0.proj.weight'])
    assert torch.equal(interop.to_torch(weights.out.weight),
                       tensors[prefix + 'attn.to_out.0.weight'])
    # The fused rows of head h are that head's q, k and v rows.
    dim = _CONFIG.head_dim
    qkv = interop.to_torch(weights.qkv.weight)
    for h in range(_CONFIG.num_heads):
        for part, name in enumerate(('to_q', 'to_k', 'to_v')):
            src = tensors[f'{prefix}attn.{name}.weight']
            row = (h * 3 + part) * dim
            assert torch.equal(qkv[row:row + dim],
                               src[h * dim:(h + 1) * dim])


def test_adaln_and_non_trunk_tensors_are_read(tmp_path):
    tensors = _write_release(tmp_path)
    with convert.ReleaseReader(str(tmp_path)) as reader:
        adaln = convert.adaln_tensors(reader, 0)
        other = convert.non_trunk_tensors(reader)
        refiner = convert.refiner_tensors(reader, 0)
    assert set(adaln) == {'weight', 'bias'}
    assert torch.equal(interop.to_torch(adaln['bias']),
                       tensors['transformer_blocks.0.adaln_proj.linear.bias'])
    assert torch.equal(interop.to_torch(other['final_layer.video_out.bias']),
                       tensors['proj_out.bias'])
    assert other['video_patch_proj.weight'].dtype == mx.float32
    assert torch.equal(interop.to_torch(refiner['mlp.fc2.weight']),
                       tensors['token_refiner.refiner_blocks.0.ff.net.2.'
                               'weight'])


def test_slabs_round_trip_the_checkpoint(tmp_path):
    source = tmp_path / 'transformer'
    _write_release(source)
    out = str(tmp_path / 'slabs')
    with convert.ReleaseReader(str(source)) as reader:
        convert.write_trunk_slabs(reader, out, [0, 1])
        expected = {i: convert.trunk_tensors(reader, i) for i in (0, 1)}
    paths = [mlx_slab.slab_path(out, i) for i in (0, 1)]
    reader = mlx_slab.SlabReader(paths, slots=1)
    try:
        for i in (0, 1):
            reader.read(i, 0)
            got = reader.weights(0).to_tensors()
            for name, value in expected[i].items():
                assert mx.array_equal(got[name], value), name
    finally:
        reader.close()
