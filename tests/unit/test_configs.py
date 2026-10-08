"""Every config under configs/ is a launchable set of parameters.

A run that dies on a typo'd or dropped key an hour in costs more than the
whole test suite, and twice already a config has been launched with a key
the loader silently did not know. So: every `infer_*.yaml` has to parse
into scripts/generate.py's parser and leave nothing required unset,
every `search_*.yaml` into a SearchConfig, every `ablate_*.yaml` into a
solattn.RunConfig, and every other one into a TrainConfig.
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
