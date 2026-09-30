"""Tests for scripts/export_predictor.py, mainly the --leap mix."""

import os

import pytest
import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.train import checkpoint
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import tiling

from scripts import export_predictor


def _checkpoint(tmp_path, layers=2, heads=4, dim=32):
    """A checkpoint whose live and EMA weights differ by a known amount."""
    torch.manual_seed(0)
    model = veda_predictor.TileScorePredictor(layers, heads, dim)
    live = {f'predictor.{k}': v.float().clone()
            for k, v in model.state_dict().items()}
    ema = {k: v - 1.0 for k, v in live.items()}   # shadow is live minus one
    directory = str(tmp_path / 'ckpt')
    os.makedirs(directory)
    torch.save({'step': 100, 'weights': live, 'ema': ema,
                'param_groups': [], 'rank_states': [],
                'shared_generator': None, 'config': {}},
               os.path.join(directory, 'state.pt'))
    with open(os.path.join(directory, 'done.json'), 'w') as f:
        f.write('{"step": 100}')
    plan_dir = str(tmp_path / 'plans')
    os.makedirs(plan_dir)
    geo = h3_geometry.Geometry('16:9', 256, 128, 22, 7, 8, 16, 37)
    veda_plan.TilePlan.uniform(geo, tiling.TileShape.parse('4x8x4'), layers,
                               heads).save(os.path.join(plan_dir,
                                                        f'{geo.name}.json'))
    return directory, plan_dir, live, ema


def _run(argv, monkeypatch):
    monkeypatch.setattr('sys.argv', ['export_predictor.py'] + argv)
    export_predictor.main()


@pytest.mark.parametrize('delta', [0.0, 0.2, 0.8, 1.0])
def test_leap_mixes_live_and_ema_in_fp32(tmp_path, monkeypatch, delta):
    ckpt, plans, live, ema = _checkpoint(tmp_path)
    out = str(tmp_path / f'leap{delta}.safetensors')
    _run(['--checkpoint', ckpt, '--plan-dir', plans, '--out', out,
          '--dtype', 'float32', '--leap', str(delta)], monkeypatch)
    got = veda_bundle.load(out).predictor.state_dict()
    for name, value in live.items():
        bare = name[len('predictor.'):]
        want = ema[name] * (1 - delta) + value * delta
        torch.testing.assert_close(got[bare].float(), want, rtol=1e-6,
                                   atol=1e-6)
    assert veda_bundle.read_metadata(out)['source_weights'] == (
        f'leap{delta:g}')


def test_leap_endpoints_equal_the_plain_exports(tmp_path, monkeypatch):
    ckpt, plans, live, ema = _checkpoint(tmp_path)
    paths = {}
    for name, extra in (('live', []), ('ema', ['--ema']),
                        ('leap1', ['--leap', '1.0']),
                        ('leap0', ['--leap', '0.0'])):
        paths[name] = str(tmp_path / f'{name}.safetensors')
        _run(['--checkpoint', ckpt, '--plan-dir', plans, '--out',
              paths[name], '--dtype', 'float32'] + extra, monkeypatch)
    load = lambda p: veda_bundle.load(p).predictor.state_dict()
    for a, b in (('live', 'leap1'), ('ema', 'leap0')):
        for k, v in load(paths[a]).items():
            assert torch.equal(v, load(paths[b])[k]), (a, b, k)


def test_leap_is_exclusive_with_ema_and_bounded(tmp_path, monkeypatch):
    ckpt, plans, _, _ = _checkpoint(tmp_path)
    out = str(tmp_path / 'x.safetensors')
    base = ['--checkpoint', ckpt, '--plan-dir', plans, '--out', out]
    with pytest.raises(ValueError, match='exclusive'):
        _run(base + ['--leap', '0.8', '--ema'], monkeypatch)
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        _run(base + ['--leap', '1.5'], monkeypatch)


def test_leap_quantizes_after_mixing_not_before(tmp_path, monkeypatch):
    """fp8 rounding is applied once, to the mix -- not to each input."""
    ckpt, plans, live, ema = _checkpoint(tmp_path)
    out = str(tmp_path / 'leap_fp8.safetensors')
    _run(['--checkpoint', ckpt, '--plan-dir', plans, '--out', out,
          '--dtype', 'float8_e4m3fn', '--leap', '0.8'], monkeypatch)
    got = veda_bundle.load(out).predictor.state_dict()
    for name, value in live.items():
        bare = name[len('predictor.'):]
        want = ema[name] * 0.2 + value * 0.8
        # fp8 e4m3 with a per-head scale: a few percent, not a different mix.
        torch.testing.assert_close(got[bare].float(), want, rtol=0.1,
                                   atol=1e-3)
