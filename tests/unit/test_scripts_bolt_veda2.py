"""Tests for scripts/bolt_veda2.py and the bundle fields it needs."""

import importlib.util
import os

import pytest
import torch

from miowtion.h3 import geometry
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _load_script():
    path = os.path.join(_ROOT, 'scripts', 'bolt_veda2.py')
    spec = importlib.util.spec_from_file_location('bolt_veda2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plans():
    geo = geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    plan = veda_plan.TilePlan.uniform(
        geo, tiling.least_padding_shape(geo.video_grid), 2, 4)
    return veda_plan.PlanTable([plan])


def _veda1(tmp_path, dtype=torch.bfloat16):
    torch.manual_seed(0)
    pred = veda_predictor.TileScorePredictor(2, 4, 16)
    path = str(tmp_path / 'veda1.safetensors')
    veda_bundle.save(path, pred.state_dict(), _plans(), num_layers=2,
                     num_heads=4, head_dim=16, keep_ratio=0.1,
                     source='unit', source_weights='ema', step=600,
                     dtype=dtype)
    return path, pred


def test_bundle_round_trips_the_new_predictor_flags(tmp_path):
    pred = veda_predictor.TileScorePredictor(2, 4, 16, second_order_rank=16,
                                             count_term=True)
    pred.init_exact_second_order_()
    path = str(tmp_path / 'veda2.safetensors')
    veda_bundle.save(path, pred.state_dict(), _plans(), num_layers=2,
                     num_heads=4, head_dim=16, keep_ratio=0.1,
                     source='unit', source_weights='ema', step=0,
                     second_order_rank=16, count_term=True)
    back = veda_bundle.load(path)
    assert back.predictor.second_order_rank == 16
    assert back.predictor.count_term is True
    for name, param in pred.state_dict().items():
        # The bundle stores bf16, so compare in fp32 at bf16 tolerance.
        torch.testing.assert_close(
            back.predictor.state_dict()[name].float(), param.float(),
            rtol=1e-2, atol=1e-2)


def test_a_bundle_without_the_fields_loads_as_the_plain_predictor(tmp_path):
    path, _ = _veda1(tmp_path)
    meta = veda_bundle.read_metadata(path)
    assert meta['second_order_rank'] == '0' and meta['count_term'] == '0'
    back = veda_bundle.load(path)
    assert back.predictor.second_order_rank == 0
    assert back.predictor.count_term is False


def test_bolting_keeps_the_trained_projections_and_adds_the_terms(tmp_path):
    path, pred = _veda1(tmp_path)
    out = str(tmp_path / 'veda2.safetensors')
    script = _load_script()
    import sys
    argv = sys.argv
    try:
        sys.argv = ['bolt_veda2.py', '--in', path, '--out', out]
        script.main()
    finally:
        sys.argv = argv
    back = veda_bundle.load(out)
    assert back.predictor.second_order_rank == 16
    assert back.predictor.count_term is True
    # The trained projections survive, bf16 round-trip aside.
    for i in range(2):
        for which in ('proj_q', 'proj_k'):
            want = getattr(pred.layers[i], which).float()
            got = getattr(back.predictor.layers[i], which).float()
            torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
        # And the new terms land on their exact coefficients, per head.
        eye = (torch.eye(16) * (0.5 / 16) ** 0.5).expand(4, 16, 16)
        torch.testing.assert_close(
            back.predictor.layers[i].so_q.float(), eye, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(
            back.predictor.layers[i].count_gain.float(), torch.ones(4),
            rtol=1e-2, atol=1e-3)
    assert back.plans.plans and back.keep_ratio == pytest.approx(0.1)


def test_bolting_refuses_an_already_bolted_bundle(tmp_path):
    path, _ = _veda1(tmp_path)
    out = str(tmp_path / 'once.safetensors')
    script = _load_script()
    import sys
    argv = sys.argv
    try:
        sys.argv = ['bolt_veda2.py', '--in', path, '--out', out]
        script.main()
        sys.argv = ['bolt_veda2.py', '--in', out, '--out',
                    str(tmp_path / 'twice.safetensors')]
        with pytest.raises(ValueError, match='already carries'):
            script.main()
    finally:
        sys.argv = argv


def test_bolting_can_leave_either_term_out(tmp_path):
    path, _ = _veda1(tmp_path)
    script = _load_script()
    import sys
    argv = sys.argv
    try:
        out = str(tmp_path / 'count_only.safetensors')
        sys.argv = ['bolt_veda2.py', '--in', path, '--out', out,
                    '--no-second-order']
        script.main()
        back = veda_bundle.load(out)
        assert back.predictor.second_order_rank == 0
        assert back.predictor.count_term is True
    finally:
        sys.argv = argv


def test_scales_multiply_the_term_not_each_factor(tmp_path):
    """The term is a product of two factors, so each takes the sqrt."""
    path, _ = _veda1(tmp_path)
    out = str(tmp_path / 'scaled.safetensors')
    script = _load_script()
    import sys
    argv = sys.argv
    try:
        sys.argv = ['bolt_veda2.py', '--in', path, '--out', out,
                    '--second-order-scale', '4.0', '--count-scale', '3.0']
        script.main()
    finally:
        sys.argv = argv
    back = veda_bundle.load(out)
    layer = back.predictor.layers[0]
    exact = (0.5 / 16) ** 0.5
    # so_q . so_k now carries 4x the exact coefficient, so each factor 2x.
    assert float(layer.so_q.float()[0, 0, 0]) == pytest.approx(exact * 2.0,
                                                               rel=2e-2)
    assert float(layer.so_k.float()[0, 0, 0]) == pytest.approx(exact * 2.0,
                                                               rel=2e-2)
    torch.testing.assert_close(layer.count_gain.float(),
                               torch.full((4,), 3.0), rtol=2e-2, atol=1e-2)


def test_reset_base_draws_fresh_projections(tmp_path):
    path, pred = _veda1(tmp_path)
    # Stand in for a trained predictor: projections far above init scale.
    with torch.no_grad():
        for layer in pred.layers:
            layer.proj_q.normal_(0.0, 0.5)
            layer.proj_k.normal_(0.0, 0.5)
    veda_bundle.save(path, pred.state_dict(), _plans(), num_layers=2,
                     num_heads=4, head_dim=16, keep_ratio=0.1,
                     source='unit', source_weights='ema', step=600)
    out = str(tmp_path / 'reset.safetensors')
    script = _load_script()
    import sys
    argv = sys.argv
    try:
        sys.argv = ['bolt_veda2.py', '--in', path, '--out', out,
                    '--base', 'reset']
        script.main()
    finally:
        sys.argv = argv
    back = veda_bundle.load(out)
    # Fresh N(0, 1e-4) projections, so nowhere near the trained ones, but
    # the plans and the new terms come through.
    got = back.predictor.layers[0].proj_q.float()
    assert float(got.abs().max()) < 1e-2
    assert float(pred.layers[0].proj_q.abs().max()) > 1e-2
    assert back.predictor.second_order_rank == 16
    assert back.predictor.count_term is True
    assert back.plans.plans
