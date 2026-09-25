"""Tests for miowtion.mlx.check and the torch side of the bridge."""

import pytest

from miowtion.h3 import config as h3_config

mx = pytest.importorskip('mlx.core', reason='requires the mlx extra')

from miowtion.mlx import check  # noqa: E402
from miowtion.mlx import convert  # noqa: E402
from miowtion.mlx import interop  # noqa: E402
from tests.unit.test_mlx_convert import _write_release  # noqa: E402


def _release_tensors(weights):
    """The release-layout tensors of a BlockWeights, by checkpoint name."""
    return {
        'norm1.weight': weights.norm1,
        'norm2.weight': weights.norm2,
        'attn.q_norm.weight': weights.q_norm,
        'attn.k_norm.weight': weights.k_norm,
        'attn.qkv_proj.weight': weights.qkv.weight,
        'attn.out_proj.weight': weights.out.weight,
        'mlp.fc1.weight': weights.fc1.weight,
        'mlp.fc2.weight': weights.fc2.weight,
    }


def test_torch_block_round_trips_the_release_layout(tmp_path):
    """Release tensors -> torch Block -> release tensors, bitwise equal.

    A permutation applied in the wrong direction survives every synthetic
    test that only ever goes one way, so the way back is pinned here.
    """
    _write_release(tmp_path)
    with convert.ReleaseReader(str(tmp_path)) as reader:
        tensors = convert.trunk_tensors(reader, 0)
        adaln = convert.adaln_tensors(reader, 0)
        block = interop.torch_block(tensors, adaln, reader.config)
    back = _release_tensors(interop.block_weights_from_torch(block))
    assert set(back) == set(tensors)
    for name, want in tensors.items():
        assert mx.array_equal(back[name], want), name
    for name in ('weight', 'bias'):
        got = interop.from_torch(getattr(block.adaln_proj.linear, name))
        assert mx.array_equal(got, adaln[name]), name


def test_compare_block_on_a_synthetic_release(tmp_path):
    # tiny(): rope_dim fits inside head_dim, as in the real config.
    _write_release(tmp_path, cfg=h3_config.H3Config.tiny())
    with convert.ReleaseReader(str(tmp_path)) as reader:
        result = check.compare_block(reader, 0, seq_len=32)
    assert result.index == 0 and result.seq_len == 32
    # Random weights are far worse conditioned than trained ones, so what
    # is asserted is the three-way relation, not an absolute error.
    assert result.as_good_as_torch
    assert result.mlx_seconds > 0.0 and result.torch_seconds > 0.0


def test_quantized_weights_are_worse_but_not_broken(tmp_path):
    _write_release(tmp_path, cfg=h3_config.H3Config.tiny())
    with convert.ReleaseReader(str(tmp_path)) as reader:
        dense = check.compare_block(reader, 0, seq_len=32)
        # group 32: the tiny ffn_dim (96) is not a multiple of 64.
        quantized = check.compare_block(reader, 0, seq_len=32, bits=8,
                                        group_size=32)
    assert quantized.mlx_vs_fp32 > dense.mlx_vs_fp32
    assert quantized.mlx_vs_fp32 < 0.2


def test_a_worse_than_torch_result_is_flagged():
    assert not check.BlockComparison(0, 32, 0.0, 3e-3, 1e-3, 0.0, 0.0,
                                     0.0).as_good_as_torch
    assert check.BlockComparison(0, 32, 0.0, 1e-3, 1e-3, 0.0, 0.0,
                                 0.0).as_good_as_torch


def test_comparison_line_mentions_every_number():
    line = check.BlockComparison(3, 128, 1e-3, 2e-3, 2e-3, 0.5, 0.1,
                                 1.25).line()
    for part in ('block 3', 'seq 128', '1.000e-03', '2.000e-03', '0.50 s',
                 '0.10 s', '1.25 GB'):
        assert part in line
