"""Wan2.1's packed layout, which is all Veda needs from a second model.

docs/dependencies.md used to say the layout interfaces would have to be
redone for Wan because it has no audio quadrant. They do not: Veda reads
twelve layout fields and every one is geometry-agnostic, so a correct
set of spans is the whole job.
"""

import pytest
import torch

from miowtion.veda import tiling
from miowtion.wan import layout as wan_layout


def test_token_grid_at_480p():
    """832x480x81 is Wan2.1-T2V-1.3B's documented configuration."""
    assert wan_layout.token_grid(832, 480, 81) == (21, 30, 52)


@pytest.mark.parametrize('args,match', [
    ((833, 480, 81), 'must divide'),
    ((832, 481, 81), 'must divide'),
    ((832, 480, 80), '4k \\+ 1'),
])
def test_bad_sizes_raise_instead_of_dropping_tokens(args, match):
    """A remainder would silently lose a row or a column."""
    with pytest.raises(ValueError, match=match):
        wan_layout.token_grid(*args)


def test_veda_tiling_runs_on_the_wan_layout():
    """The point of the whole file: tile it and lose nothing.

    Every real row must land in exactly one tile slot, so the valid
    counts have to sum to `used` -- that is what catches an off-by-one
    in the span or a grid that does not match the row count.
    """
    layout = wan_layout.packed_layout(832, 480, 81)
    assert layout.used == 512 + 21 * 30 * 52
    assert len(layout.spans) == 1 and layout.target.role == 'target'
    assert layout.audio_pos.numel() == 0

    tile_layout = tiling.build_tile_layout(
        [tiling.TiledSpan(layout.target.start, layout.target.grid,
                          tiling.TileShape.parse('2x16x4'))],
        used=layout.used, seq_len=layout.seq_len)
    assert tile_layout.n_ref_tiles == 0
    assert int(tile_layout.valid_count.sum()) == layout.used
    # And every real row appears exactly once in the permutation.
    rows = tile_layout.perm[tile_layout.perm >= 0]
    assert torch.equal(rows.sort().values, torch.arange(layout.used))


def test_the_least_padding_shape_is_the_one_the_static_analysis_picked():
    """2x16x4 at 11.75% padding, from docs/dependencies.md."""
    grid = wan_layout.token_grid(832, 480, 81)
    best = tiling.least_padding_shape(grid)
    assert str(best) == '2x16x4'
    assert best.padding_ratio(grid) == pytest.approx(0.1175, abs=1e-4)
