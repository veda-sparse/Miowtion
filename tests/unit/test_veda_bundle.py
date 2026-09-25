"""Tests for miowtion.veda.bundle."""

import pytest
import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

_LAYERS, _HEADS, _DIM = 2, 3, 8


def _plans() -> veda_plan.PlanTable:
    geo = h3_geometry.geometry_from_latent_t('16:9', 7)
    plan = veda_plan.TilePlan.uniform(geo, tiling.TileShape(2, 4, 16),
                                      _LAYERS, _HEADS)
    return veda_plan.PlanTable([plan, plan.mirrored()])


def _write(tmp_path, **overrides) -> tuple[str, dict]:
    model = veda_predictor.TileScorePredictor(_LAYERS, _HEADS, _DIM)
    weights = {f'predictor.{k}': v for k, v in model.state_dict().items()}
    path = str(tmp_path / 'bundle.safetensors')
    kwargs = dict(num_layers=_LAYERS, num_heads=_HEADS, head_dim=_DIM,
                  keep_ratio=0.1, source='runs/x/ckpt/step_0000600',
                  source_weights='live', step=600)
    kwargs.update(overrides)
    veda_bundle.save(path, weights, _plans(), **kwargs)
    return path, model.state_dict()


def test_round_trip_is_bit_wise_exact(tmp_path):
    path, state = _write(tmp_path)
    loaded = veda_bundle.load(path)
    got = loaded.predictor.state_dict()
    assert sorted(got) == sorted(state)
    for name, value in state.items():
        assert torch.equal(got[name], value)
    assert loaded.keep_ratio == 0.1
    assert loaded.metadata['source_weights'] == 'live'
    assert loaded.metadata['step'] == '600'


def test_plans_travel_with_the_weights(tmp_path):
    """The point of the bundle: tile shapes cannot be lost or mispaired."""
    path, _ = _write(tmp_path)
    plans = veda_bundle.load(path).plans
    assert sorted(plans.plans) == ['16x9_t7', '9x16_t7']
    for name, want in _plans().plans.items():
        got = plans.plans[name]
        assert got.grid == want.grid
        assert got.shapes == want.shapes
        assert got.head_shape == want.head_shape
    # And they are selectable by geometry, like a plan directory.
    geo = h3_geometry.geometry_from_latent_t('9:16', 7)
    assert plans.select(geo).geometry == '9x16_t7'


def test_read_metadata_does_not_need_the_tensors(tmp_path):
    path, _ = _write(tmp_path)
    assert veda_bundle.read_metadata(path)['keep_ratio'] == '0.1'


def test_a_foreign_safetensors_file_is_rejected(tmp_path):
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
    path = str(tmp_path / 'other.safetensors')
    save_file({'x': torch.zeros(2)}, path)
    with pytest.raises(ValueError, match='not a miowtion-veda-predictor'):
        veda_bundle.load(path)


def test_mismatched_shape_metadata_is_rejected(tmp_path):
    path, _ = _write(tmp_path, num_heads=_HEADS + 1)
    with pytest.raises(ValueError, match='do not match the'):
        veda_bundle.load(path)


def test_empty_weights_and_empty_plans_are_rejected(tmp_path):
    path = str(tmp_path / 'b.safetensors')
    with pytest.raises(ValueError, match='no predictor weights'):
        veda_bundle.save(path, {}, _plans(), num_layers=1, num_heads=1,
                         head_dim=1, keep_ratio=0.1, source='x',
                         source_weights='live', step=0)
    model = veda_predictor.TileScorePredictor(_LAYERS, _HEADS, _DIM)
    with pytest.raises(ValueError, match='without plans'):
        veda_bundle.save(path, dict(model.state_dict()),
                         veda_plan.PlanTable([]), num_layers=_LAYERS,
                         num_heads=_HEADS, head_dim=_DIM, keep_ratio=0.1,
                         source='x', source_weights='live', step=0)
