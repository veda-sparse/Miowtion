"""Head-weighted leverage of the log B_j term, for every released plan.

`log B_j` is a per-key-tile constant, so it reorders a row's top-k only
in proportion to how much those constants differ across columns. That
makes its value a property of the *layout*, not of the method, and a
property that can be computed before any GPU is touched.

The number that matters is not the padding fraction and not the leverage
of one hand-picked tile shape: it is the leverage of the shapes that are
actually deployed, weighted by how many of the 50 x 56 (layer, head)
slots each one occupies in the plan. Quoting a single shape overstated
16:9@37 by about a factor of two (see docs/features/veda2.md 2.9).

Example:
    python scripts/count_term_report.py --plan-dir plans/released \\
        --out artifacts/count_term_leverage.json
"""

import argparse
import collections
import glob
import json
import os

from miowtion.veda import solattn
from miowtion.veda import tiling
from miowtion.utils import progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', default='plans/released')
    parser.add_argument('--base-spread', type=float, default=3.9,
                        help='spread of the base score the term competes '
                        'with; the default is the measured untrained value')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    plans = sorted(glob.glob(os.path.join(args.plan_dir, '*.json')))
    if not plans:
        raise SystemExit(f'no plans in {args.plan_dir}')
    progress.log(f'{len(plans)} plans in {args.plan_dir}')
    rows, per_geometry = [], {}
    for path in plans:
        plan = json.load(open(path))
        grid = tuple(plan['grid'])
        flat = [i for layer in plan['head_shape'] for i in layer]
        counts = collections.Counter(flat)
        total = sum(counts.values())
        weighted_leverage = weighted_std = 0.0
        for index, name in enumerate(plan['shapes']):
            t, h, w = (int(x) for x in name.split('x'))
            got = solattn.count_term_leverage(
                grid, tiling.TileShape(t, h, w), args.base_spread)
            share = counts.get(index, 0) / total
            rows.append({'geometry': plan['geometry'], 'shape': name,
                         'head_share': share, **got})
            weighted_leverage += share * got['leverage']
            weighted_std += share * got['log_count_std']
        per_geometry[plan['geometry']] = {
            'leverage': weighted_leverage, 'log_count_std': weighted_std,
            'all_tiles_full': max(
                r['padding'] for r in rows
                if r['geometry'] == plan['geometry']) < 1e-12}
        progress.log(f"  {plan['geometry']:<11} head-weighted leverage "
                     f'{weighted_leverage:.1%}, std(log B) '
                     f'{weighted_std:.4f}'
                     + ('  [every tile full: the term is identically 0]'
                        if per_geometry[plan['geometry']]['all_tiles_full']
                        else ''))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'base_spread': args.base_spread, 'per_shape': rows,
                   'per_geometry': per_geometry}, open(args.out, 'w'),
                  indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
