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
        _print_scorers(records, row['density'])
    _print_allocation(records)


def _print_scorers(records, density: float) -> None:
    """Each score's distance from the oracle, at one density."""
    try:
        rows = solattn.scorer_report(records, density)
    except ValueError as exc:
        print(f'  scorer report unavailable: {exc}')
        return
    print('  scores, ranked by error (gap closed: 0 = untrained pooled '
          'score, 1 = oracle):')
    print(f"    {'mask':24s} {'eps':>8s} {'gap':>8s} {'rec/Om':>8s} "
          f"{'rec/A':>8s}")
    for row in rows:
        print(f"    {row['mask']:24s} {row['median_error']:8.5f} "
              f"{row['gap_closed']:+8.1%} {row['recall_vs_omega']:8.3f} "
              f"{row['recall_vs_mass']:8.3f}")


def _print_allocation(records) -> None:
    """What a static per-head budget buys at the same total cost."""
    for mask in ('proxy_topk', 'R1_topk'):
        try:
            curves = solattn.error_curve(records, mask=mask)
        except ValueError:
            continue
        break
    else:
        print('\nno per-head curves available')
        return
    densities = sorted({r.density for r in records})
    if len(densities) < 2:
        print('\nper-head allocation needs at least two densities')
        return
    print(f'\nstatic per-head budget vs uniform, same total cost '
          f'({len(curves)} heads, {mask}):')
    lo, hi = densities[0], densities[-1]
    for i in range(5):
        target = lo * (hi / lo) ** (i / 4.0)
        try:
            rep = solattn.allocation_report(curves, target)
        except ValueError as exc:
            print(f'  rho={target:.3f}: {exc}')
            continue
        print(f"  rho={target:.3f}: total eps "
              f"{rep['relative_saving']:+.2%}   per-head rho "
              f"p10 {rep['density_p10']:.3f} med {rep['density_median']:.3f} "
              f"p90 {rep['density_p90']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--summarize', help='re-print a saved run instead '
                        'of measuring')
    parser.add_argument('--second-moments', action='store_true',
                        help='collect the calibration moments the low-rank '
                        'second-order head is initialized from')
    parser.add_argument('--init-probe', action='store_true',
                        help="score predictor initializations in the "
                        "training loop's own metrics instead of running "
                        'the ablation')
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
    config = solattn.RunConfig(**raw)
    if args.second_moments:
        solattn.run_second_moments(config)
    elif args.init_probe:
        solattn.run_init_probe(config)
    else:
        solattn.run_ablation(config)


if __name__ == '__main__':
    main()
