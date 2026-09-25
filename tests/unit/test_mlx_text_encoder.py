"""The MLX text tower against transformers' Qwen3-VL text model."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip('mlx.core')
transformers = pytest.importorskip('transformers')

from miowtion.mlx import interop  # pylint: disable=wrong-import-position
from miowtion.mlx import slab as mlx_slab  # pylint: disable=wrong-import-position
from miowtion.mlx import text_encoder  # pylint: disable=wrong-import-position

# A tower with the released proportions (GQA 8:1, head_dim > hidden /
# heads, SwiGLU 5x) but small enough for a CPU reference.
_CONFIG = text_encoder.TowerConfig(
    hidden_size=64, num_hidden_layers=3, num_attention_heads=8,
    num_key_value_heads=2, head_dim=16, intermediate_size=96,
    rms_norm_eps=1e-6, rope_theta=5_000_000.0, vocab_size=97)


def _torch_config(config):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    return qwen.Qwen3VLTextConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        rms_norm_eps=config.rms_norm_eps, rope_theta=config.rope_theta,
        vocab_size=config.vocab_size, attention_bias=False,
        attn_implementation='eager')


def _torch_model(config, seed=0):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    torch.manual_seed(seed)
    model = qwen.Qwen3VLTextModel(_torch_config(config))
    return model.to(torch.bfloat16).eval()


def _mlx_layer(module):
    """LayerWeights holding the torch layer's weights (no permutation)."""
    tensors = {
        'input_layernorm.weight': module.input_layernorm.weight,
        'post_attention_layernorm.weight':
            module.post_attention_layernorm.weight,
        'self_attn.q_norm.weight': module.self_attn.q_norm.weight,
        'self_attn.k_norm.weight': module.self_attn.k_norm.weight,
        'self_attn.q_proj.weight': module.self_attn.q_proj.weight,
        'self_attn.k_proj.weight': module.self_attn.k_proj.weight,
        'self_attn.v_proj.weight': module.self_attn.v_proj.weight,
        'self_attn.o_proj.weight': module.self_attn.o_proj.weight,
        'mlp.gate_proj.weight': module.mlp.gate_proj.weight,
        'mlp.up_proj.weight': module.mlp.up_proj.weight,
        'mlp.down_proj.weight': module.mlp.down_proj.weight,
    }
    return text_encoder.LayerWeights.from_tensors(
        {k: interop.from_torch(v.detach()) for k, v in tensors.items()})


def _rel_l2(got, want):
    got, want = got.float(), want.float()
    return (got - want).norm().item() / max(want.norm().item(), 1e-12)


def _positions(seq):
    # Text-only mrope: the three axes carry the same position.
    return torch.arange(seq)[None, None, :].expand(3, 1, seq)


def test_one_layer_matches_transformers():
    model = _torch_model(_CONFIG)
    seq = 24
    torch.manual_seed(1)
    x = torch.randn(1, seq, _CONFIG.hidden_size).bfloat16()
    cos, sin = model.rotary_emb(x, _positions(seq))
    mask = torch.full((seq, seq), float('-inf')).triu(1).bfloat16()
    with torch.no_grad():
        want = model.layers[0](x, position_embeddings=(cos, sin),
                               attention_mask=mask[None, None])
    want = want[0] if isinstance(want, tuple) else want
    got = text_encoder.layer_forward(interop.from_torch(x[0]),
                                     _mlx_layer(model.layers[0]), _CONFIG)
    # bf16 GEMMs and softmax accumulate differently on the two backends;
    # one layer stays at 3e-4, a wrong RoPE base already costs 5e-3.
    assert _rel_l2(interop.to_torch(got), want[0]) < 1e-3


def test_encode_matches_the_whole_torch_tower():
    model = _torch_model(_CONFIG)
    # H3 drops the final norm and reads the raw hidden states.
    model.norm = torch.nn.Identity()
    ids = [3, 51, 7, 7, 96, 0, 42, 11]
    with torch.no_grad():
        want = model(input_ids=torch.tensor([ids]),
                     use_cache=False).last_hidden_state[0]
    embed = interop.from_torch(model.embed_tokens.weight.detach())
    layers = [(i, _mlx_layer(model.layers[i]))
              for i in range(_CONFIG.num_hidden_layers)]
    got = text_encoder.encode(embed, layers, ids, _CONFIG)
    assert got.shape == (len(ids), _CONFIG.hidden_size)
    assert _rel_l2(interop.to_torch(got), want) < 1e-2


def test_a_prefix_of_the_layers_is_what_h3_conditions_on():
    """H3 stops at TEXT_LAYERS, which is a prefix of the same tower."""
    model = _torch_model(_CONFIG)
    model.norm = torch.nn.Identity()
    model.layers = model.layers[:2]
    ids = [5, 6, 7]
    with torch.no_grad():
        want = model(input_ids=torch.tensor([ids]),
                     use_cache=False).last_hidden_state[0]
    embed = interop.from_torch(model.embed_tokens.weight.detach())
    got = text_encoder.encode(
        embed, [(i, _mlx_layer(model.layers[i])) for i in range(2)], ids,
        _CONFIG, total=2)
    assert _rel_l2(interop.to_torch(got), want) < 1e-2


def test_layer_slabs_round_trip_bitwise(tmp_path):
    model = _torch_model(_CONFIG)
    layer = _mlx_layer(model.layers[1])
    path = mlx_slab.slab_path(str(tmp_path), 1)
    mlx_slab.write_slab(path, 1, layer.to_tensors())
    reader = mlx_slab.SlabReader([path], slots=1)
    reader.read(0, 0)
    got = text_encoder.LayerWeights.from_tensors(reader.tensors(0))
    for field in ('input_norm', 'q_norm', 'q_proj', 'down_proj'):
        assert mx.array_equal(getattr(got, field), getattr(layer, field))
    reader.close()


def test_from_tensors_rejects_an_incomplete_layer():
    model = _torch_model(_CONFIG)
    tensors = _mlx_layer(model.layers[0]).to_tensors()
    del tensors['mlp.up_proj.weight']
    with pytest.raises(KeyError, match='mlp.up_proj.weight'):
        text_encoder.LayerWeights.from_tensors(tensors)


def test_encode_rejects_an_out_of_range_token():
    model = _torch_model(_CONFIG)
    embed = interop.from_torch(model.embed_tokens.weight.detach())
    with pytest.raises(ValueError, match='outside'):
        text_encoder.encode(embed, [], [_CONFIG.vocab_size], _CONFIG)


def test_tower_config_reads_the_released_shape(tmp_path):
    import json
    config = {'text_config': {
        'hidden_size': 5120, 'num_hidden_layers': 64,
        'num_attention_heads': 64, 'num_key_value_heads': 8, 'head_dim': 128,
        'intermediate_size': 25600, 'rms_norm_eps': 1e-6,
        'rope_theta': 5_000_000, 'vocab_size': 151936,
        'rope_scaling': {'rope_type': 'default',
                         'mrope_section': [24, 20, 20]}}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    got = text_encoder.TowerConfig.from_pretrained(str(tmp_path))
    assert got.hidden_size == 5120 and got.num_key_value_heads == 8
    # ~1 GB per layer is why the tower is streamed instead of resident.
    assert 0.9 < got.layer_bytes / 2**30 < 1.1


def test_tower_config_rejects_an_unknown_rope(tmp_path):
    import json
    config = {'text_config': {
        'hidden_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
        'num_key_value_heads': 1, 'head_dim': 4, 'intermediate_size': 8,
        'rms_norm_eps': 1e-6, 'rope_theta': 1e4, 'vocab_size': 4,
        'rope_scaling': {'rope_type': 'yarn'}}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='rope type'):
        text_encoder.TowerConfig.from_pretrained(str(tmp_path))
