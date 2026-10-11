"""Solve each arm's density for an equal attention-time budget.

Comparing routers at a matched *density* answers a question nobody has:
a user picks a runtime, not a sparsity. Each arm has its own cost per
unit density -- a compensated kernel does strictly more work per kept
block than a plain block-sparse one -- so an equal-time comparison puts
them at different densities, and that spread is the whole point.

The cost model is the one the measurements support: attention time is
affine in density, `a + b * d`, with `a` the geometry's fixed cost
(gather, pool, score, select, scatter: O(n)) and `b` the kernel
(O(n^2)). Two densities per arm determine it; more are used when the
runs exist, and the residual is printed because an arm whose points do
not lie on a line has something else going on and must not be
extrapolated.

Reads the `summary.json` that generate.py already writes, so the curve
is measured rather than assumed.

Example:
    python scripts/iso_time_budget.py --root artifacts/demo12_compare \\
        --geometry 16x9_t37 --target-arm sol
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402


def _points(root: str, geometry: str) -> dict[str, list[tuple[float, float]]]:
    """{arm: [(density, attention seconds)]} over every run under `root`.

    The density is the configured keep ratio, which is what the caller
    can set; Sol has no budget at all, so its arm is keyed by the
    density it actually reached when the run recorded one.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(glob.glob(os.path.join(root, '*', '*',
                                              'summary.json'))
                       + glob.glob(os.path.join(root, '*',
                                                'summary.json'))):
        with open(path) as handle:
            loaded = json.load(handle)
        # A run over several clips writes one list at the arm root; a run
        # over one clip writes a dict per clip directory. Reading only
        # the dict form drops whole arms without saying so.
        for meta in (loaded if isinstance(loaded, list) else [loaded]):
            if meta.get('geometry') != geometry:
                continue
            ratio = meta.get('keep_ratio')
            for mode, record in meta.get('modes', {}).items():
                seconds = record.get('attention_seconds_excl_step0')
                if seconds is None:
                    continue
                density = record.get('density')
                if density is None and mode.startswith('sol'):
                    # Sol routes by a threshold, so its keep ratio is not
                    # its density and substituting one for the other
                    # would put it on the curve at a density it never
                    # ran. A run that did not record the reached density
                    # cannot be used for a cost curve.
                    progress.log(f'  {path}: {mode} has no measured '
                                 'density; skipped (its keep_ratio is not '
                                 'a density)')
                    continue
                if density is None:
                    density = ratio
                if density is None:
                    continue
                out.setdefault(mode, []).append(
                    (float(density), float(seconds)))
    return out


def _fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares `a + b * d`; returns (a, b, max residual)."""
    n = len(points)
    if n < 2:
        raise ValueError('an affine cost curve needs at least two densities')
    mean_d = sum(p[0] for p in points) / n
    mean_s = sum(p[1] for p in points) / n
    var = sum((p[0] - mean_d) ** 2 for p in points)
    if var <= 0:
        raise ValueError('every point is at the same density; the slope is '
                         'not determined, so nothing can be extrapolated')
    slope = sum((p[0] - mean_d) * (p[1] - mean_s) for p in points) / var
    intercept = mean_s - slope * mean_d
    residual = max(abs(s - (intercept + slope * d)) for d, s in points)
    return intercept, slope, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--geometry', default='16x9_t37')
    parser.add_argument('--target-arm', default=None,
                        help='take the time budget from this arm')
    parser.add_argument('--target-seconds', type=float, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    points = _points(args.root, args.geometry)
    if not points:
        raise SystemExit(f'no summary.json with geometry {args.geometry}')

    # One clip per (arm, density) is enough for the curve only because
    # attention time barely varies between clips of one geometry; the
    # spread is printed so that assumption is visible rather than held.
    print(f'\n{"arm":<12}{"points":>7}{"densities":>34}'
          f'{"fixed s":>10}{"per-unit s":>12}{"max resid":>11}')
    curves = {}
    for arm, pts in sorted(points.items()):
        merged: dict[float, list[float]] = {}
        for density, seconds in pts:
            merged.setdefault(round(density, 4), []).append(seconds)
        reduced = [(d, sum(v) / len(v)) for d, v in sorted(merged.items())]
        shown = ' '.join(f'{d:.3f}' for d, _ in reduced)[:33]
        try:
            a, b, residual = _fit(reduced)
        except ValueError as error:
            print(f'{arm:<12}{len(reduced):>7}{shown:>34}'
                  f'  {error}'[:90])
            continue
        curves[arm] = {'fixed': a, 'per_unit': b, 'residual': residual,
                       'points': reduced}
        print(f'{arm:<12}{len(reduced):>7}{shown:>34}'
              f'{a:>10.2f}{b:>12.2f}{residual:>11.3f}')

    target = args.target_seconds
    if target is None and args.target_arm:
        pts = points.get(args.target_arm)
        if not pts:
            raise SystemExit(f'no runs for target arm {args.target_arm!r}')
        target = sum(s for _d, s in pts) / len(pts)
        progress.log(f'time budget {target:.2f} s, the mean attention time '
                     f'of {args.target_arm!r} over {len(pts)} run(s)')
    if target is None:
        return

    print(f'\ndensity each arm can buy for {target:.2f} s of attention:')
    print(f'{"arm":<12}{"density":>10}{"note":>44}')
    rows = []
    for arm, curve in sorted(curves.items()):
        if curve['per_unit'] <= 0:
            note = 'cost does not rise with density; curve unusable'
            density = float('nan')
        else:
            density = (target - curve['fixed']) / curve['per_unit']
            lo = min(d for d, _ in curve['points'])
            hi = max(d for d, _ in curve['points'])
            note = ('within the measured range' if lo <= density <= hi else
                    f'EXTRAPOLATED past [{lo:.3f}, {hi:.3f}] -- measure it')
            if density <= 0:
                note = 'fixed cost alone already exceeds the budget'
        rows.append({'arm': arm, 'density': density, 'note': note})
        print(f'{arm:<12}{density:>10.4f}{note:>44}')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'geometry': args.geometry, 'target_seconds': target,
                   'curves': curves, 'solution': rows},
                  open(args.out, 'w'), indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
