"""Shared pytest configuration: GPU tests are skipped without CUDA."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    del config
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason='requires CUDA')
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip)
