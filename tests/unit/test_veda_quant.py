"""Fake quantization of the block-scoring inputs (miowtion.veda.quant)."""

import math

import pytest
import torch

from miowtion.veda import quant


def _rows(n=256, heads=4, dim=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, heads, dim, generator=gen).bfloat16()


def test_bf16_scheme_is_the_identity():
    x = _rows()
    assert torch.equal(quant.fake_quantize(x, 'bf16'), x)


@pytest.mark.parametrize('scheme', sorted(quant.SCHEMES))
def test_shape_and_dtype_survive(scheme):
    x = _rows()
    out = quant.fake_quantize(x, scheme)
    assert out.shape == x.shape and out.dtype == x.dtype


@pytest.mark.parametrize('scheme,bound', [
    ('fp8_e4m3_head', 2.0 ** -4),   # 3 mantissa bits
    ('fp8_e4m3_row', 2.0 ** -4),
    ('fp8_e5m2_row', 2.0 ** -3),    # 2 mantissa bits
    ('mxfp8_e4m3', 2.0 ** -4),
    ('nvfp4', 1.0 / 3.0),           # e2m1: worst step is 4 -> 6
    ('mxfp4', 1.0 / 3.0),
])
def test_relative_error_stays_inside_the_format(scheme, bound):
    """Every element is within the scheme's own rounding step of the input.

    The comparison is against the largest magnitude of its scaling group,
    since that is what the scale is pinned to: an element far below the
    group's amax is allowed to be wrong by a fraction of the *group's*
    step, not of its own value.
    """
    x = _rows().float()
    out = quant.fake_quantize(x.bfloat16(), scheme).float()
    group_max = x.abs().amax()
    assert (out - x).abs().max() <= bound * group_max


def test_e2m1_values_land_on_the_eight_levels():
    # One row, so the whole tensor is a single scaling block after the
    # per-head amax: 6.0 maps to the top level and the rest follow.
    x = torch.tensor([[[0.0, 0.2, 0.5, 0.9, 1.6, 2.6, 4.9, 6.0] * 4]])
    out = quant.fake_quantize(x.bfloat16(), 'mxfp4').float()[0, 0, :8]
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 6.0])
    assert torch.equal(out, expected)


def test_e8m0_scales_round_up_to_a_power_of_two():
    x = torch.tensor([0.3, 1.0, 3.0, 8.0])
    assert torch.equal(quant._round_e8m0(x),
                       torch.tensor([0.5, 1.0, 4.0, 8.0]))


def test_mxfp8_is_exact_on_a_block_it_can_represent():
    """A power-of-two scale and e4m3 levels round-trip exactly.

    The block's amax is 2, so the e8m0 scale is 2^-7 and every value
    becomes a multiple of 16 below 256 -- all e4m3 numbers. Anything but a
    power-of-two scale or a different level set would perturb this.
    """
    x = (0.125 * torch.arange(1, 17, dtype=torch.float32)).repeat(2)
    x = x.view(1, 1, 32).bfloat16()
    assert torch.equal(quant.fake_quantize(x, 'mxfp8_e4m3'), x)


@pytest.mark.parametrize('scheme', sorted(quant.SCHEMES))
def test_all_zero_input_stays_zero(scheme):
    x = torch.zeros(128, 2, 128, dtype=torch.bfloat16)
    out = quant.fake_quantize(x, scheme)
    assert torch.equal(out, x) and not out.isnan().any()


def test_finer_scaling_is_not_worse_than_coarser():
    """Per-row e4m3 must beat per-head e4m3 on a badly scaled tensor."""
    x = _rows().float()
    x[:64] *= 1000.0  # one group of rows dominates the per-head amax
    ref = x.bfloat16().float()
    err = {name: (quant.fake_quantize(x.bfloat16(), name).float()
                  - ref).abs().max().item()
           for name in ('fp8_e4m3_head', 'fp8_e4m3_row')}
    assert err['fp8_e4m3_row'] < err['fp8_e4m3_head']


def test_unknown_scheme_and_wrong_rank_raise():
    with pytest.raises(ValueError, match='scheme must be one of'):
        quant.fake_quantize(_rows(), 'int4')
    with pytest.raises(ValueError, match='tile-ordered rows'):
        quant.fake_quantize(torch.zeros(4, 4), 'nvfp4')


def test_head_dim_must_divide_the_scaling_block():
    with pytest.raises(ValueError, match='not a multiple of'):
        quant.fake_quantize(torch.zeros(4, 2, 24, dtype=torch.bfloat16),
                            'mxfp4')


def test_summarize_weighs_groups_by_head_count():
    records = [
        quant.QuantRecord(0, 0, '8x4x4', 30, 'nvfp4', 1.0, 0.5, 0.5,
                          1e-3, 1e-2),
        quant.QuantRecord(0, 0, '2x8x8', 10, 'nvfp4', 0.0, 0.1, 0.5,
                          3e-3, 5e-2),
    ]
    out = quant.summarize(records)['nvfp4']
    assert out['recall'] == pytest.approx(0.75)
    assert out['max_abs'] == pytest.approx(5e-2)
    assert out['kept_vs_ceiling'] == pytest.approx(0.8)


def test_summarize_skips_nan_rows():
    records = [
        quant.QuantRecord(0, 0, '8x4x4', 1, 'mxfp4', math.nan, 0.5, 0.5,
                          1e-3, 1e-2),
        quant.QuantRecord(0, 1, '8x4x4', 1, 'mxfp4', 0.8, 0.5, 0.5,
                          1e-3, 1e-2),
    ]
    out = quant.summarize(records)['mxfp4']
    assert out['recall'] == pytest.approx(0.4)  # 0.8 over both weights


def test_probe_rejects_bad_settings():
    with pytest.raises(ValueError, match='unknown scheme'):
        quant.QuantHeatProbe(None, None, ['fp6'])
    with pytest.raises(ValueError, match='q_tile_fraction'):
        quant.QuantHeatProbe(None, None, ['bf16'], q_tile_fraction=0.0)
    with pytest.raises(ValueError, match='layer_every'):
        quant.QuantHeatProbe(None, None, ['bf16'], layer_every=0)


def _tiny_probe_case(schemes, layer_every=1):
    from miowtion.h3 import geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import mask as veda_mask
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import tiling
    geo = geometry.Geometry('16:9', 512, 256, 39, 12, 16, 32, 20)
    lay = h3_layout.pack(torch.ones(30, dtype=torch.long), geo)
    gen = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(lay.seq_len, 4, 32, generator=gen).bfloat16()
               for _ in range(3))
    plan = veda_plan.TilePlan(geo.name, geo.video_grid,
                              [tiling.TileShape(4, 4, 8)], [[0, 0, 0, 0]])
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.3), teacher_q_tiles=1.0)
    clip = veda_attention.ClipTiling(lay, config, torch.device('cpu'))
    probe = quant.QuantHeatProbe(clip, plan, schemes, layer_every=layer_every,
                                 dense_backend='math')
    return lay, (q, k, v), probe


def test_probe_returns_the_dense_output_and_measures_every_scheme():
    from miowtion.h3 import attention as h3_attention
    lay, (q, k, v), probe = _tiny_probe_case(['bf16', 'fp8_e4m3_row',
                                              'mxfp4'])
    out = probe(q, k, v, 0)
    dense = h3_attention.dense_attention(q, k, v, lay.used, backend='math')[0]
    assert torch.equal(out, dense)
    by_scheme = {r.scheme: r for r in probe.records}
    assert sorted(by_scheme) == ['bf16', 'fp8_e4m3_row', 'mxfp4']
    # The control: quantizing with the identity cannot move the selection.
    assert by_scheme['bf16'].recall == 1.0
    assert by_scheme['bf16'].rel_l2 == 0.0
    assert by_scheme['bf16'].max_abs == 0.0
    for record in probe.records:
        assert 0.0 <= record.heat_kept <= record.heat_ceiling <= 1.0


def test_probe_skips_layers_outside_the_stride():
    _, (q, k, v), probe = _tiny_probe_case(['bf16'], layer_every=2)
    probe(q, k, v, 1)
    assert not probe.records
    probe(q, k, v, 0)
    assert [r.layer for r in probe.records] == [0]
