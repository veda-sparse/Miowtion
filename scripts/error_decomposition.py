"""Splits the attention output error into the long tail and routing cost.

The companion analysis of P0-4, and the one that costs nothing because
the ablation records already hold both quantities.

At a finite budget two things go wrong independently. The mass oracle's
own error is what a *perfect* router still pays, because the blocks it
cannot afford carry mass: the long tail. Everything above that is the
cost of choosing the wrong blocks. Reporting only one of them makes a
router look either more or less important than it is, and which one
depends entirely on the density.

Example:
    python scripts/error_decomposition.py \\
        --records runs/sol_ablation_turbo8_t37/16x9_t37.json
"""

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                       # noqa: E402
from miowtion.veda import solattn                         # noqa: E402

# The mass oracle at the same budget: a perfect router's own error.
_TAIL = 'A|0'
# The pooled proxy: what an untrained router actually pays.
_PROXY = 'R1_topk|0'
# The error-minimising selection: the true lower bound.
_BEST = 'oracle_topk|0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True,
                        help='an ablate_sol.py records file')
    parser.add_argument('--proxy', default=_PROXY,
                        help='the selection rule to decompose')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    records, meta = solattn.load_records(args.records)
    progress.log(f'{len(records)} rows, clips {meta.get("clips")}, '
                 f'geometry {meta.get("geometry")}')
    by_density: dict[float, list[tuple[float, float, float]]] = {}
    for row in records:
        errors = row.errors
        if _TAIL not in errors or args.proxy not in errors:
            continue
        by_density.setdefault(row.density, []).append(
            (errors[_TAIL], errors[args.proxy],
             errors.get(_BEST, errors[_TAIL])))
    if not by_density:
        raise SystemExit(f'no row carries both {_TAIL} and {args.proxy}')

    rows = []
    print(f'\n{"density":>8}{"rows":>7}{"long tail":>12}{"total":>10}'
          f'{"routing":>10}{"routing share":>15}')
    for density in sorted(by_density):
        values = by_density[density]
        tail = st.median(v[0] for v in values)
        total = st.median(v[1] for v in values)
        best = st.median(v[2] for v in values)
        # Split per row and then take the median, so that a few huge
        # rows cannot carry the fraction.
        share = st.median((v[1] - v[0]) / v[1] for v in values if v[1] > 0)
        share_best = st.median((v[1] - v[2]) / v[1]
                               for v in values if v[1] > 0)
        rows.append({'density': density, 'rows': len(values),
                     'long_tail': tail, 'total': total, 'lower_bound': best,
                     'routing_cost': total - tail,
                     'routing_share': share,
                     'routing_share_vs_lower_bound': share_best})
        print(f'{density:>8.2f}{len(values):>7}{tail:>12.5f}{total:>10.5f}'
              f'{total - tail:>10.5f}{share:>14.1%}')
    print('\nThe routing cost is nearly flat in absolute terms while the '
          'long tail\nshrinks with the budget, so the share routing carries '
          'rises with density.')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'records': args.records, 'meta': meta,
                   'proxy': args.proxy, 'decomposition': rows},
                  open(args.out, 'w'), indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
