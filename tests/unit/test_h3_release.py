"""Tests for miowtion.h3.release (names of the released checkpoints)."""

import json
import os

import pytest

from miowtion.h3 import release

_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party',
                     'MiniMax-H3')


def test_h3_schema_covers_the_pinned_release():
    """Every tensor of the pinned release is claimed by exactly one name."""
    index = os.path.join(_REPO, 'FL2VA', 'transformer',
                         'model.safetensors.index.json')
    with open(index) as f:
        keys = set(json.load(f)['weight_map'])
    schema = release.detect_schema(keys)
    assert schema == release.SCHEMA_H3
    release.check_complete(schema, keys, num_layers=50,
                           num_refiner_layers=2)
    # rope.inv_freq is recomputed from the config, never read.
    assert keys - release.expected_keys(schema, 50, 2) == {'rope.inv_freq'}


def test_detect_schema_is_not_confused_by_the_shared_suffix():
    """'blocks.0.' is a suffix of 'transformer_blocks.0.'."""
    assert release.detect_schema(
        ['transformer_blocks.0.norm1.weight']) == release.SCHEMA_DIFFUSERS
    assert release.detect_schema(
        ['blocks.0.norm1.weight']) == release.SCHEMA_H3
    with pytest.raises(ValueError, match='no transformer blocks'):
        release.detect_schema(['norm_out.norm.weight'])


def test_diffusers_block_keys_pin_the_rename():
    keys = release.block_keys(release.SCHEMA_DIFFUSERS, 7, adaln=True)
    assert keys['attn.qkv_proj.weight'] == (
        'transformer_blocks.7.attn.to_q.weight',
        'transformer_blocks.7.attn.to_k.weight',
        'transformer_blocks.7.attn.to_v.weight')
    assert keys['mlp.fc1.weight'] == (
        'transformer_blocks.7.ff.net.0.proj.weight',)
    assert keys['attn.q_norm.weight'] == (
        'transformer_blocks.7.attn.norm_q.weight',)
    assert keys['adaln_proj.linear.bias'] == (
        'transformer_blocks.7.adaln_proj.linear.bias',)
    assert 'adaln_proj.linear.weight' not in release.block_keys(
        release.SCHEMA_DIFFUSERS, 7)


def test_non_trunk_keys_map_the_output_heads():
    keys = release.non_trunk_keys(release.SCHEMA_DIFFUSERS)
    assert keys['final_layer.video_out.weight'] == ('proj_out.weight',)
    assert keys['video_patch_proj.weight'] == ('proj_in.weight',)
    assert keys['condition_proj.bias'] == ('context_embedder.bias',)
    h3_keys = release.non_trunk_keys(release.SCHEMA_H3)
    assert all(v == (k,) for k, v in h3_keys.items())
    assert set(h3_keys) == set(keys)


def test_check_complete_reports_missing_keys():
    with pytest.raises(KeyError, match='keys missing'):
        release.check_complete(release.SCHEMA_H3, ['blocks.0.norm1.weight'],
                               num_layers=1, num_refiner_layers=1)


def test_unknown_schema_raises():
    with pytest.raises(ValueError, match='schema must be one of'):
        release.block_keys('onnx', 0)
