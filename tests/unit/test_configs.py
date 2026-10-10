"""Every config under configs/ is a launchable set of parameters.

A run that dies on a typo'd or dropped key an hour in costs more than the
whole test suite, and twice already a config has been launched with a key
the loader silently did not know. So: every `infer_*.yaml` has to parse
into scripts/generate.py's parser and leave nothing required unset,
every `search_*.yaml` into a SearchConfig, every `ablate_*.yaml` into a
solattn.RunConfig, every `offline_*.yaml` into a ProbeConfig, and every
other one into a TrainConfig.
"""

import dataclasses
import glob
import importlib.util
import os

import pytest

from miowtion.train import trainer
from miowtion.veda import search
from miowtion.veda import solattn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_CONFIGS = sorted(glob.glob(os.path.join(_ROOT, 'configs', '*.yaml')))


def _load_script(name: str):
    path = os.path.join(_ROOT, 'scripts', name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_configs():
    assert _CONFIGS, 'configs/ is empty; the globbing is wrong'


@pytest.mark.parametrize('path', _CONFIGS, ids=os.path.basename)
def test_config_parses(path):
    name = os.path.basename(path)
    if name.startswith('search_'):
        _check_dataclass(path, search.SearchConfig)
    elif name.startswith('ablate_'):
        _check_dataclass(path, solattn.RunConfig)
    elif name.startswith('offline_'):
        probe = _load_script('offline_probe.py')
        config = probe.ProbeConfig.from_yaml(path)
        assert config.sample_id and config.geometry, \
            'a probe needs samples and a geometry'
    elif name.startswith('infer_'):
        generate = _load_script('generate.py')
        args = generate.parse_args(['--config', path])
        assert args.attention, 'no attention mode requested'
        if 'veda' in args.attention:
            assert args.predictor or (args.checkpoint and args.plan_dir), \
                'veda needs a bundle, or a checkpoint with its plans'
        if args.sample_id and len(args.geometry) > 1:
            assert len(args.geometry) == len(args.sample_id), \
                'give one geometry, or one per sample'
    else:
        config = trainer.TrainConfig.from_yaml(path)
        assert config.run_name, 'a run needs a name'


def _check_dataclass(path: str, config_type) -> None:
    """search_tiles.py and ablate_sol.py keep their own config types."""
    import yaml  # pylint: disable=import-outside-toplevel
    with open(path) as f:
        raw = yaml.safe_load(f)
    unknown = sorted(set(raw) - {f.name for f in
                                 dataclasses.fields(config_type)})
    assert not unknown, f'unknown config keys {unknown}'
    assert config_type(**raw).run_name


def test_offline_probe_builds_a_valid_veda_config():
    """The probe's config has to produce a VedaConfig that exists.

    VedaConfig takes a Budget, not a keep_ratio: the probe passed the
    float and died after eight minutes of model load, twice. Constructing
    it here costs nothing and fails in the suite instead.
    """
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import mask as veda_mask

    probe = _load_script('offline_probe.py')
    config = probe.ProbeConfig.from_yaml(
        os.path.join(_ROOT, 'configs', 'offline_probe_t37.yaml'))
    built = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=config.keep_ratio))
    assert built.target_budget.ratio == config.keep_ratio


def test_generate_does_not_silently_drop_the_plan_override():
    """--plan-dir with --predictor used to be ignored.

    The bundle carries the plan it was searched with, and that branch
    returned before ever reading --plan-dir, so four runs with four
    different plan directories produced pixel-identical video. The flag
    must take effect; an explicit option that is silently dropped is
    worse than one that raises.
    """
    source = open(os.path.join(_ROOT, 'scripts', 'generate.py')).read()
    branch = source[source.index('if args.predictor:'):
                    source.index('elif args.plan_dir and args.checkpoint:')]
    assert 'args.plan_dir' in branch, (
        'the --predictor branch never reads --plan-dir, so the override '
        'is dropped again')
    assert 'PlanTable.load_dir(args.plan_dir)' in branch
