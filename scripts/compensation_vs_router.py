"""Compensation's error reduction against the router's distance to the oracle.

Plan item S-2 asked whether the zero-order compensation's benefit shrinks
as the router approaches the oracle, which the error identity appears to
suggest: compensation replaces each dropped term by the residual of its
estimate, so a router that drops nothing important should leave nothing
to repair.

It does not shrink. The reason is in docs/features/veda2.md 2.10: at 5%
density the routing decision accounts for only 16.3% of the attention
output error and the long tail for 84%. Compensation addresses the tail,
not the selection, so a perfect router still leaves it most of its work.

Reads the per-(clip, step, layer, head, density) records that
scripts/ablate_sol.py writes, which already carry `errors[mask|c]` for
c = 0, 1, 2 and `recall_vs_mass[mask]`, so this needs no GPU.

Example:
    python scripts/compensation_vs_router.py \\
        --records runs/sol_ceiling_t37/sol_ablation_turbo8_t37/16x9_t37.json
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402


def _rows(records: list[dict], density: float, mask: str) -> dict:
    """Mean error at c=0 and c=1, the relative gain, and the win rate."""
    e0 = [r['errors'][f'{mask}|0'] for r in records]
    e1 = [r['errors'][f'{mask}|1'] for r in records]
    recall = [r['recall_vs_mass'][mask] for r in records
              if mask in r['recall_vs_mass']]
    # Per row, so a few huge errors cannot dominate the summary; a row
    # with zero error has nothing to gain and is dropped rather than
    # divided by zero.
    gains = [(a - b) / a for a, b in zip(e0, e1) if a > 0]
    return {
        'density': density,
        'mask': mask,
        'n': len(e0),
        'recall_vs_mass': statistics.mean(recall) if recall else None,
        'eps_c0': statistics.mean(e0),
        'eps_c1': statistics.mean(e1),
        'gain': statistics.mean(gains),
        'win_rate': sum(1 for a, b in zip(e0, e1) if b < a) / len(e0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True,
                        help='json written by scripts/ablate_sol.py')
    parser.add_argument('--out', default=None, help='write the table as json')
    args = parser.parse_args()

    with progress.Timer(f'load {args.records}'):
        with open(args.records) as handle:
            records = json.load(handle)['records']
    masks = sorted({key.split('|')[0] for key in records[0]['errors']})
    densities = sorted({r['density'] for r in records})
    progress.log(f'{len(records)} rows, {len(masks)} masks, '
                 f'densities {densities}')

    table = []
    for density in densities:
        subset = [r for r in records if r['density'] == density]
        rows = [_rows(subset, density, mask) for mask in masks]
        rows.sort(key=lambda r: (r['recall_vs_mass'] is None,
                                 r['recall_vs_mass'] or 0.0))
        table += rows
        print(f'\n=== density {density:.0%} (n={len(subset)} rows) ===')
        print(f'{"router":22s} {"recall":>8s} {"eps c=0":>9s} '
              f'{"eps c=1":>9s} {"gain":>8s} {"helps":>7s}')
        for row in rows:
            recall = ('      --' if row['recall_vs_mass'] is None
                      else f'{row["recall_vs_mass"]:8.4f}')
            print(f'{row["mask"]:22s} {recall} {row["eps_c0"]:9.4f} '
                  f'{row["eps_c1"]:9.4f} {row["gain"]:+7.1%} '
                  f'{row["win_rate"]:6.1%}')
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(table, handle, indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
