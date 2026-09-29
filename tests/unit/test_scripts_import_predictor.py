"""Tests for scripts/import_predictor.py (the inverse of the export)."""

import json
import os

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.train import checkpoint
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

from scripts import import_predictor


def _bundle(path, dtype=torch.bfloat16, num_layers=2,
            num_heads=4, head_dim=32):
    torch.manual_seed(0)
    model = veda_predictor.TileScorePredictor(num_layers, num_heads, head_dim)
    weights = {f'predictor.{k}': v for k, v in model.state_dict().items()}
    geos = [h3_geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37),
            h3_geometry.Geometry('1:1', 128, 128, 22, 7, 8, 8, 37)]
    plans = veda_plan.PlanTable([
        veda_plan.TilePlan.uniform(g, tiling.TileShape.parse('4x8x4'),
                                   num_layers, num_heads) for g in geos])
    veda_bundle.save(path, weights, plans, num_layers=num_layers,
                     num_heads=num_heads, head_dim=head_dim, keep_ratio=0.1,
                     source='runs/fake/ckpt/step_0000600',
                     source_weights='live', step=600, dtype=dtype)
    return weights, plans


def _run(bundle, ckpt, plan_dir, monkeypatch):
    monkeypatch.setattr('sys.argv', [
        'import_predictor.py', '--bundle', bundle, '--out-checkpoint', ckpt,
        '--out-plan-dir', plan_dir])
    import_predictor.main()


def test_round_trip_seeds_a_trainable_predictor(tmp_path, monkeypatch):
    path = str(tmp_path / 'b.safetensors')
    weights, plans = _bundle(path)
    ckpt, plan_dir = str(tmp_path / 'ckpt'), str(tmp_path / 'plans')
    _run(path, ckpt, plan_dir, monkeypatch)
    # The trainer's init path reads the EMA shadow and is strict both ways.
    fresh = veda_predictor.TileScorePredictor(2, 4, 32)
    named = [(f'predictor.{n}', p) for n, p in fresh.named_parameters()]
    checkpoint.init_weights(named, checkpoint.load(ckpt), use_ema=True)
    # Exactly the bundle's storage rounding, nothing else: the file is the
    # only source, so what comes back is the rounded weights (see the
    # script's docstring).
    for name, value in weights.items():
        got = dict(named)[name]
        assert torch.equal(got, value.to(torch.bfloat16).float()), name
    # Plans come back one file per geometry, loadable as a table.
    assert sorted(os.listdir(plan_dir)) == sorted(
        f'{name}.json' for name in plans.plans)
    back = veda_plan.PlanTable.load_dir(plan_dir)
    assert sorted(back.plans) == sorted(plans.plans)


def test_payload_records_provenance_and_is_not_a_resume(tmp_path,
                                                       monkeypatch):
    path = str(tmp_path / 'b.safetensors')
    _bundle(path, dtype=torch.float8_e4m3fn)
    ckpt = str(tmp_path / 'ckpt')
    _run(path, ckpt, str(tmp_path / 'plans'), monkeypatch)
    payload = checkpoint.load(ckpt)
    assert payload['config']['bundle_dtype'] == 'float8_e4m3fn'
    assert payload['config']['keep_ratio'] == 0.1
    # No moments, no sampler states: this seeds a stage, it cannot resume.
    assert payload['optimizer'] == {} and payload['rank_states'] == []
    with open(os.path.join(ckpt, 'done.json')) as f:
        assert json.load(f)['init_only'] is True
