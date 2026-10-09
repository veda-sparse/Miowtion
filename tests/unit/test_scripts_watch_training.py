"""Tests for scripts/watch_training.py."""

import importlib.util
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _watcher():
    path = os.path.join(_ROOT, 'scripts', 'watch_training.py')
    spec = importlib.util.spec_from_file_location('watch_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verdict_catches_the_measured_decline():
    """The run this exists for: 0.919 -> 0.882 over 48 updates."""
    w = _watcher()
    values = [0.9185 - 0.00076 * i for i in range(48)]
    assert w.verdict(values[:6], 8, 10, 0.05) is None, 'needs a full window'
    reason = w.verdict(values, 8, 10, 0.05)
    assert reason and ('no improvement' in reason or 'diverging' in reason)
    # It should have fired long before 48 updates.
    first = next(i for i in range(8, 49)
                 if w.verdict(values[:i], 8, 10, 0.05))
    assert first <= 20, f'caught only at update {first}'


def test_verdict_tolerates_geometry_noise():
    w = _watcher()
    levels = [0.91, 0.93, 0.88, 0.90]
    values = [levels[i % 4] + 0.0005 * i for i in range(60)]
    assert w.verdict(values, 8, 10, 0.05) is None


def test_verdict_flags_divergence_on_its_own():
    w = _watcher()
    values = [0.90] * 8 + [0.70] * 8
    reason = w.verdict(values, 8, 100, 0.05)
    assert reason and 'diverging' in reason
    assert '0.05' in reason


def test_read_records_skips_everything_but_update_records(tmp_path):
    w = _watcher()
    path = str(tmp_path / 'train.log')
    with open(path, 'w') as f:
        f.write('[12:00:00] some progress line\n')
        f.write(json.dumps({'event': 'initialized'}) + '\n')
        f.write('{"step": 1, "kept_over_ceiling": 0.9}\n')
        f.write('{"step": 2, "kl": 1.0}\n')          # metric absent
        f.write('{"step": 3, "kept_over_ceiling"\n')  # truncated
        f.write('{"step": 4, "kept_over_ceiling": 0.8}\n')
    assert w.read_records(path, 'kept_over_ceiling') == [0.9, 0.8]
    assert w.read_records(str(tmp_path / 'missing.log'), 'x') == []


def test_parse_args_defaults_match_the_documented_policy():
    w = _watcher()
    args = w.parse_args(['--log', 'x'])
    assert args.patience == 10 and args.window == 8
    assert args.metric == 'kept_over_ceiling'
    assert args.kill is False, 'killing must be opt-in'


def test_patience_zero_disables_that_criterion():
    """It did not, and that killed a healthy run at update 9.

    `since >= patience` is true for every `since` when patience is 0, so
    the watcher fired on the first update that was not a new best.
    """
    w = _watcher()
    values = [0.9343, 0.9346, 0.9239, 0.9294, 0.9091, 0.9100, 0.9037,
              0.9092, 0.8899]
    assert w.verdict(values, 8, 10, 0.05) is None, 'patience 10 tolerates it'
    assert w.verdict(values, 8, 0, 0.12) is None, 'patience 0 must be off'
    # max_drop still fires on its own.
    assert w.verdict([0.93] * 8 + [0.70] * 8, 8, 0, 0.05)


def test_verdict_rejects_having_no_criterion_at_all():
    w = _watcher()
    with pytest.raises(ValueError, match='at least one of'):
        w.verdict([0.9] * 10, 8, 0, 0.0)
