"""The vendored Sol+Veda kernel is upstream plus its patch series.

AGENTS.md 3 lets us vendor a patched copy of an external module, on four
conditions: the original licence travels with it, there is a patch
series, the regeneration is reproducible, and the baseline is checked at
run time. The run-time hash check lives in `miowtion/kernels/sol_veda.py`
and the first two are files; this pins the third, which is the one that
rots silently -- an edit to the vendored file that nobody mirrored into
`patches/` leaves the copy underivable from its stated source.
"""

import os

import pytest

from miowtion.kernels import compensation_triton
from miowtion.kernels import sol_veda

import scripts.regen_sol_veda_patch as regen


def _submodule_present() -> bool:
    return os.path.exists(sol_veda._UPSTREAM)              # noqa: SLF001


@pytest.mark.skipif(not _submodule_present(),
                    reason='third_party/Veda-on-ComfyUI is not checked out')
def test_vendored_kernel_is_upstream_plus_patches():
    with open(regen._VENDORED, encoding='utf-8') as handle:  # noqa: SLF001
        vendored = handle.read()
    assert regen.rebuild() == vendored


@pytest.mark.skipif(not _submodule_present(),
                    reason='third_party/Veda-on-ComfyUI is not checked out')
def test_patch_series_is_numbered_and_non_empty():
    names = sorted(name for name in os.listdir(regen._PATCHES)  # noqa: SLF001
                   if name.endswith('.patch'))
    assert names, 'the vendored kernel claims a patch series'
    assert [name[:4] for name in names] == [
        f'{i + 1:04d}' for i in range(len(names))]


class _Layout:
    """The two fields `compensate` reads before it touches the device."""

    def __init__(self, n_tiles, counts):
        self.n_tiles = n_tiles
        self.valid_count = counts


def test_compensation_rejects_a_mismatched_lse():
    import torch
    rows, heads, dim, tiles = 256, 2, 128, 2
    zeros = torch.zeros(rows, heads, dim)
    layout = _Layout(tiles, torch.full((tiles,), 128, dtype=torch.int32))
    mask = torch.ones(heads, tiles, tiles, dtype=torch.bool)
    with pytest.raises(ValueError, match='lse_kept'):
        compensation_triton.compensate(zeros, torch.zeros(rows, heads),
                                       zeros, zeros, zeros, mask, layout)


def test_compensation_rejects_a_query_tile_subset():
    import torch
    rows, heads, dim, tiles = 256, 2, 128, 4
    zeros = torch.zeros(rows, heads, dim)
    layout = _Layout(tiles, torch.full((tiles,), 128, dtype=torch.int32))
    mask = torch.ones(heads, rows // 128, tiles, dtype=torch.bool)
    # One program reads mask row r for query tile r, so a subset of the
    # query tiles would silently pair row r with key tile r.
    with pytest.raises(ValueError, match='same id'):
        compensation_triton.compensate(zeros, torch.zeros(heads, rows),
                                       zeros, zeros, zeros, mask, layout)
