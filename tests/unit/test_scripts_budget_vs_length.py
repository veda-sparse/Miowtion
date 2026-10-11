"""Tests for scripts/budget_vs_length.py."""

import math

from miowtion.veda import plan as veda_plan
from scripts import budget_vs_length


class _Plan:
    """The three fields head_weighted_tiles reads."""

    def __init__(self, grid, shapes_per_layer):
        self.grid = grid
        self._shapes = shapes_per_layer
        self.num_layers = len(shapes_per_layer)

    def layer_shapes(self, layer):
        return self._shapes[layer]


def test_tiles_are_weighted_by_how_many_heads_use_a_shape():
    """A plan's shapes do not all make the same number of tiles.

    An unweighted mean would report a tile count no head ever sees, and
    every density derived from it would be wrong by the spread between
    the shapes.
    """
    from miowtion.veda import tiling

    wide = tiling.TileShape.parse('8x4x4')
    tall = tiling.TileShape.parse('2x8x8')
    grid = (37, 24, 42)
    plan = _Plan(grid, [[wide, wide, wide, tall]])
    got = budget_vs_length.head_weighted_tiles(plan)

    def count(shape):
        return (math.ceil(grid[0] / shape.t) * math.ceil(grid[1] / shape.h)
                * math.ceil(grid[2] / shape.w))

    assert got == 0.75 * count(wide) + 0.25 * count(tall)


def test_an_absolute_budget_gets_sparser_as_the_clip_grows():
    """The claim behind Veda-on-ComfyUI issue #3, on our own plans."""
    table = veda_plan.PlanTable.load_dir('plans/released')
    short = budget_vs_length.head_weighted_tiles(table.plans['16x9_t37'])
    long = budget_vs_length.head_weighted_tiles(table.plans['16x9_t102'])
    # Same 32 tiles: 9.7% of the keys at t37, 3.7% at t102.
    assert 32 / short > 2.5 * (32 / long)
