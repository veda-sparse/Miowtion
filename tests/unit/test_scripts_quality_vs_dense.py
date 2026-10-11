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


def test_tile_phase_separates_aligned_flashes_from_spread_ones():
    """The test that decides whether the routing's time grid is to blame.

    A Veda query tile shares one key selection across its whole span,
    so a routing-caused artifact can only appear at a tile boundary. If
    flashes are spread uniformly instead, the time structure of the
    routing is not what produces them.
    """
    from scripts import luma_flash

    aligned = [0, 32, 64, 96, 128]
    histogram, peak = luma_flash.tile_phase(aligned, 32, 160)
    assert peak == 1.0
    assert histogram.index(max(histogram)) == 0

    spread = [1, 9, 17, 25, 33, 41, 49, 57]
    _histogram, peak = luma_flash.tile_phase(spread, 32, 160)
    assert peak <= 0.25


def test_tile_phase_is_empty_without_a_period_or_flashes():
    from scripts import luma_flash

    assert luma_flash.tile_phase([4, 8], 1, 100) == ([], 0.0)
    assert luma_flash.tile_phase([], 32, 100) == ([], 0.0)


def test_join_arms_rejects_an_arm_spec_without_a_title(tmp_path, monkeypatch):
    """A clip missing an arm must not be joined with fewer panes.

    A five-pane comparison that looks like a six-pane one is worse than
    no comparison, so the script reports and skips rather than joining
    whatever it found.
    """
    import subprocess
    import sys

    from scripts import join_arms

    monkeypatch.setattr(sys, 'argv',
                        ['join_arms.py', '--root', str(tmp_path),
                         '--arm', 'nodelimiter'])
    with pytest.raises(SystemExit, match='Title=label'):
        join_arms.main()
    del subprocess


def test_flash_grid_of_the_reproduction_is_17_latent_frames():
    """The measured flash grid, pinned so a refactor cannot lose it.

    Every sparse clip in the issue #3 reproduction flashes at frames
    51, 119 and 187 and nowhere else systematically. Those are 68
    frames apart, and 68 frames is 17 latent frames at the VAE's 4x
    temporal ratio -- which is `vae_clip_length` in the video VAE's own
    config. The arithmetic is the whole argument, so it is a test.
    """
    from scripts import luma_flash

    observed = [51, 119, 187]
    assert [f // 17 for f in observed] == [3, 7, 11]
    assert all(f % 17 == 0 for f in observed)
    # One event per 17 latent frames, i.e. 68 RGB frames.
    assert [b - a for a, b in zip(observed, observed[1:])] == [68, 68]
    histogram, peak = luma_flash.tile_phase(observed, 17, 243)
    assert peak == 1.0
    assert histogram.index(max(histogram)) == 0
