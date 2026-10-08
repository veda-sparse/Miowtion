"""Ablates block-selection targets and per-row budgets on a dense teacher.

Checks, offline and without training, whether Veda should select blocks by
output-error contribution instead of attention quality (H1), and whether a
per-row variable budget beats a fixed top-k (H2). See
docs/features/sol_ablation.md.

Example:
    python scripts/ablate_sol.py \
        --config configs/ablate_sol_turbo8_16x9_t37.yaml
    python scripts/ablate_sol.py \
        --summarize runs/sol_ablation_turbo8_t37/16x9_t37.json
"""

import argparse
import dataclasses
import json

import yaml

from miowtion.veda import solattn


def _print_summary(path: str, eta: float, g2_margin: float) -> None:
    records, meta = solattn.load_records(path)
    print(json.dumps(meta, indent=1))
    for row in solattn.summarize(records, eta, g2_margin):
        print(f"\ndensity {row['density']:.3f}  ({row['heads']} rows)")
        print(f"  G1 selection target: "
              f"{'pass' if row['g1_pass'] else 'FAIL'} "
              f"({row['g1_pass_fraction']:.1%} of heads beat Mx by "
              f"{eta:.0%})")
        print(f"  G2 variable budget:  "
              f"{'pass' if row['g2_pass'] else 'FAIL'} "
              f"(oracle top-k vs global threshold "
              f"{row['g2_oracle_gap']:+.2%})")
        print('  median relative error:')
        for name, value in sorted(row['median_error'].items()):
            print(f'    {name:28s} {value:.5f}')
        print('  median load imbalance (max/mean kept per row):')
        for name, value in sorted(row['median_load_imbalance'].items()):
            print(f'    {name:28s} {value:.3f}')
        print(f"  proxy vs Omega Spearman: {row['median_spearman']}")
        print(f"  proxy z-score skew {row['median_skew']:+.3f}, "
              f"excess kurtosis {row['median_excess_kurtosis']:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--summarize', help='re-print a saved run instead '
                        'of measuring')
    parser.add_argument('--geometries', nargs='+', default=None,
                        help="measure these specs instead of the config's")
    parser.add_argument('--eta', type=float, default=0.15)
    parser.add_argument('--g2-margin', type=float, default=0.05)
    args = parser.parse_args()
    if args.summarize:
        _print_summary(args.summarize, args.eta, args.g2_margin)
        return
    if not args.config:
        parser.error('one of --config / --summarize is required')
    with open(args.config) as handle:
        raw = yaml.safe_load(handle)
    unknown = set(raw) - {f.name for f in dataclasses.fields(
        solattn.RunConfig)}
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    if args.geometries:
        raw['geometries'] = args.geometries
    solattn.run_ablation(solattn.RunConfig(**raw))


if __name__ == '__main__':
    main()
