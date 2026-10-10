"""Clip discovery of scripts/quality_vs_dense.py (no ffmpeg needed)."""

import importlib.util
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _module():
    path = os.path.join(_ROOT, 'scripts', 'quality_vs_dense.py')
    spec = importlib.util.spec_from_file_location('quality_vs_dense', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(b'')


def test_both_output_layouts_are_found(tmp_path):
    """generate.py collapses the clip directory for a single-sample run.

    A glob for the per-clip layout alone dropped whole geometries without
    saying so: the original-duration demo set reported 10 clips of 12,
    leaving 16:9@72 and 9:16@102 out of every table.
    """
    root = str(tmp_path)
    _write(os.path.join(root, 'veda2_16x9', 'demo00_x_16x9_t37', 'veda.mp4'))
    _write(os.path.join(root, 'veda2_9x16', 'veda.mp4'))
    with open(os.path.join(root, 'veda2_9x16', 'summary.json'), 'w') as handle:
        json.dump({'sample': 'demo01_y', 'geometry': '9x16_t102'}, handle)

    found = _module()._clips(root)
    assert set(found) == {('demo00_x', '16x9_t37'), ('demo01_y', '9x16_t102')}


def test_a_collapsed_arm_without_a_summary_raises(tmp_path):
    """Stopping beats silently dropping the clip."""
    root = str(tmp_path)
    _write(os.path.join(root, 'veda2_9x16', 'veda.mp4'))
    with pytest.raises(FileNotFoundError, match='summary.json'):
        _module()._clips(root)


def test_one_mode_from_two_arms_carries_its_arm(tmp_path):
    """sol07 and dense_sol both write sol.mp4.

    Labelling by mode alone silently kept whichever arm sorted last.
    """
    root = str(tmp_path)
    clip = 'demo00_x_16x9_t37'
    _write(os.path.join(root, 'dense_sol_16x9', clip, 'sol.mp4'))
    _write(os.path.join(root, 'dense_sol_16x9', clip, 'dense.mp4'))
    _write(os.path.join(root, 'sol07_16x9', clip, 'sol.mp4'))

    labels = _module()._clips(root)[('demo00_x', '16x9_t37')]
    assert sorted(labels) == ['dense', 'sol@dense', 'sol@sol07']
