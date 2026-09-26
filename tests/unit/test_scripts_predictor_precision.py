"""Argument handling of scripts/predictor_precision.py."""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _module():
    path = os.path.join(_ROOT, 'scripts', 'predictor_precision.py')
    spec = importlib.util.spec_from_file_location('predictor_precision', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundles_keep_their_order_so_the_first_is_the_reference():
    parse = _module()._parse_bundles
    assert list(parse(['bf16=a.st', 'fp8=b.st', 'fp32=c.st'])) == [
        'bf16', 'fp8', 'fp32']
    assert parse(['bf16=weights/a=b.st'])['bf16'] == 'weights/a=b.st'


@pytest.mark.parametrize('specs,match', [
    (['bf16'], 'NAME=PATH'),
    (['bf16=a.st', 'bf16=b.st'], 'duplicate'),
])
def test_bad_bundle_specs_raise(specs, match):
    with pytest.raises(ValueError, match=match):
        _module()._parse_bundles(specs)
