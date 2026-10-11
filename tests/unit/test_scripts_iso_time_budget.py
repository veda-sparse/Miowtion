"""Tests for scripts/iso_time_budget.py."""

import json
import os

import pytest

from scripts import iso_time_budget


def _write(root, arm, meta):
    path = os.path.join(root, arm)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'summary.json'), 'w') as handle:
        json.dump(meta, handle)


def _run(ratio, modes, geometry='16x9_t37'):
    return {'sample': 's', 'geometry': geometry, 'keep_ratio': ratio,
            'modes': {m: {'attention_seconds_excl_step0': s}
                      for m, s in modes.items()}}


def test_reads_both_summary_layouts(tmp_path):
    """A multi-clip run writes a list; a single-clip run writes a dict.

    Reading only one of them drops whole arms from the cost table and
    says nothing, which is how a comparison silently loses an arm.
    """
    root = str(tmp_path)
    _write(root, 'a', [_run(0.05, {'veda': 9.0}), _run(0.05, {'veda': 9.0})])
    _write(root, 'b', _run(0.20, {'veda': 14.0}))
    points = iso_time_budget._points(root, '16x9_t37')
    assert len(points['veda']) == 3


def test_sol_keep_ratio_is_not_taken_for_a_density(tmp_path):
    """Sol routes by a threshold; its keep ratio is not what it ran at."""
    root = str(tmp_path)
    _write(root, 'a', _run(0.05, {'sol': 17.0, 'veda': 9.0}))
    points = iso_time_budget._points(root, '16x9_t37')
    assert 'sol' not in points
    assert points['veda'] == [(0.05, 9.0)]


def test_fit_recovers_an_affine_curve_and_its_residual():
    a, b, residual = iso_time_budget._fit([(0.05, 9.0), (0.20, 16.5)])
    assert a == pytest.approx(6.5)
    assert b == pytest.approx(50.0)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_one_density_cannot_determine_a_slope():
    with pytest.raises(ValueError, match='at least two densities'):
        iso_time_budget._fit([(0.05, 9.0)])
    with pytest.raises(ValueError, match='slope is not determined'):
        iso_time_budget._fit([(0.05, 9.0), (0.05, 9.2)])
