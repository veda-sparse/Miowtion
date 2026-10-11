"""What a fixed tile budget actually means at each clip length.

Veda-on-ComfyUI issue #3 reports that the node's default budget -- an
absolute number of tiles -- does not scale with the clip, so the same
setting is roughly 89% sparse on a 5 s clip and 96% on a 14 s one, and
the long clip flashes. That claim is checkable without a GPU: the tile
count of a geometry is fixed by the plan table, so the density an
absolute budget reaches is pure arithmetic.

It also checks the proposed fix. The issue suggests scaling the budget
linearly in latent T; tiles do not grow quite linearly in T, because
every shape rounds up on all three axes, so the right scale factor is
the tile count itself.

Example:
    python scripts/budget_vs_length.py --plans plans/released --tiles 32 10
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.veda import plan as veda_plan                # noqa: E402
from miowtion.veda import tiling                           # noqa: E402


def head_weighted_tiles(plan) -> float:
    """Video tile count of a plan, weighted by how many heads use a shape.

    A plan assigns different tile shapes to different layers and heads,
    and they do not all produce the same number of tiles, so a single
    number only means anything when weighted by actual use.

    Args:
        plan: A TilePlan.

    Returns:
        The expected number of video tiles a head sees.
    """
    counts: dict[str, int] = {}
    for layer in range(plan.num_layers):
        for shape in plan.layer_shapes(layer):
            counts[str(shape)] = counts.get(str(shape), 0) + 1
    total = sum(counts.values())
    grid = plan.grid
    out = 0.0
    for text, weight in counts.items():
        shape = tiling.TileShape.parse(text)
        tiles = (math.ceil(grid[0] / shape.t) * math.ceil(grid[1] / shape.h)
                 * math.ceil(grid[2] / shape.w))
        out += (weight / total) * tiles
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', default='plans/released')
    parser.add_argument('--tiles', type=int, nargs='+', default=[32, 10])
    parser.add_argument('--anchor', default='16x9_t37',
                        help='the geometry an absolute budget was chosen on')
    args = parser.parse_args()

    table = veda_plan.PlanTable.load_dir(args.plans)
    tiles = {name: head_weighted_tiles(plan)
             for name, plan in table.plans.items()}
    columns = ''.join(f'{f"{n} tiles":>12}' for n in args.tiles)
    print(f'{"geometry":<12}{"grid":>16}{"video tiles":>13}{columns}')
    for name in sorted(tiles):
        cells = ''.join(f'{n / tiles[name]:>11.2%}' for n in args.tiles)
        print(f'{name:<12}{str(table.plans[name].grid):>16}'
              f'{tiles[name]:>13.0f}{cells}')

    if args.anchor not in tiles:
        raise SystemExit(f'--anchor {args.anchor} is not in {args.plans}')
    base = tiles[args.anchor]
    base_t = table.plans[args.anchor].grid[0]
    print(f'\nto hold the density of {args.tiles[0]} tiles at '
          f'{args.anchor}, scale the budget by the tile count, not by '
          f'latent T:')
    print(f'{"geometry":<12}{"by tile count":>15}{"by latent T":>13}'
          f'{"T overshoots by":>17}')
    for name in sorted(tiles):
        exact = args.tiles[0] * tiles[name] / base
        linear = args.tiles[0] * table.plans[name].grid[0] / base_t
        print(f'{name:<12}{exact:>15.1f}{linear:>13.1f}'
              f'{linear / exact - 1:>16.1%}')


if __name__ == '__main__':
    main()
