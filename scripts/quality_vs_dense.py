"""Per-clip PSNR / SSIM of every sparse arm against the dense output.

The routing metrics this project reports (tile recall, retained mass,
attention output error) are all upstream of the video. A reviewer's first
question is whether a better router makes a better clip, and that
question is only answered by comparing pixels to the dense run at the
same seed, prompt and geometry -- which is the standard every
sparse-attention paper reports.

Dense is the reference, so this needs the dense arm to have finished. It
walks a directory of per-arm output trees written by scripts/generate.py,
pairs clips by (sample, geometry), and reports both the per-clip numbers
and a bootstrap confidence interval over clips, because five clips with
no interval is not evidence.

Example:
    python scripts/quality_vs_dense.py \\
        --root artifacts/demo12_compare --reference dense \\
        --out artifacts/demo12_compare/quality.json
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402

_METRICS = ('psnr_db', 'ssim')


def _clips(root: str) -> dict[tuple[str, str], dict[str, str]]:
    """{(sample, geometry): {arm_or_mode: mp4}} over an output root.

    generate.py writes `<root>/<arm>/<sample>_<geometry>/<mode>.mp4`, and
    one run can hold several modes (a dense+sol run writes both), so the
    key is the mode rather than the directory it came from.
    """
    found: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.mp4'))):
        mode = os.path.splitext(os.path.basename(path))[0]
        clip = os.path.basename(os.path.dirname(path))
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # `<sample>_<geometry>`; the geometry is the last two underscore
        # fields (`16x9_t37`), the sample is everything before them.
        parts = clip.split('_')
        sample, geometry = '_'.join(parts[:-2]), '_'.join(parts[-2:])
        label = mode if mode != 'veda' else arm.split('_')[0]
        found.setdefault((sample, geometry), {})[label] = path
    return found


def _measure(reference: str, other: str, ffmpeg: str) -> dict[str, float]:
    """PSNR and SSIM of `other` against `reference`, via ffmpeg filters."""
    argv = [ffmpeg, '-hide_banner', '-i', other, '-i', reference,
            '-filter_complex', '[0:v][1:v]psnr;[0:v][1:v]ssim',
            '-f', 'null', '-']
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    from scripts import visual_check                        # noqa: PLC0415
    return visual_check.parse_metrics(done.stderr)


def _bootstrap(values: list[float], draws: int, seed: int
               ) -> tuple[float, float]:
    """A 95% percentile interval of the mean, resampling over clips."""
    if len(values) < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (lo, hi)


def _paired(per_clip: list[dict], metric: str, draws: int, seed: int
            ) -> list[dict]:
    """Paired comparisons between arms on the clips they share.

    Clip difficulty dominates the variance here: at 16:9@37 the marginal
    intervals of three routers overlap almost completely, which reads as
    "no difference", while the same data paired shows one of them ahead
    on 9 of 10 clips. Independent intervals therefore hide real effects
    in this design, and a table that reports "no significant difference"
    from them is not entitled to that conclusion.

    Args:
        per_clip: Rows as produced by the per-clip pass.
        metric: Which metric to pair on.
        draws: Bootstrap resamples over clips.
        seed: Bootstrap seed.

    Returns:
        One row per (arm_a, arm_b, geometry) with the mean paired
        difference, its percentile interval, and how many clips it won.
    """
    index: dict[tuple, dict[str, float]] = {}
    for row in per_clip:
        if metric in row:
            index.setdefault((row['sample'], row['geometry']),
                             {})[row['arm']] = row[metric]
    out = []
    arms = sorted({a for v in index.values() for a in v})
    geometries = sorted({k[1] for k in index})
    for geometry in geometries:
        for i, first in enumerate(arms):
            for second in arms[i + 1:]:
                diffs = [v[first] - v[second] for k, v in index.items()
                         if k[1] == geometry and first in v and second in v]
                if len(diffs) < 3:
                    continue
                rng = random.Random(seed)
                means = sorted(
                    sum(diffs[rng.randrange(len(diffs))]
                        for _ in diffs) / len(diffs)
                    for _ in range(draws))
                lo = means[int(0.025 * draws)]
                hi = means[min(draws - 1, int(0.975 * draws))]
                out.append({
                    'metric': metric, 'geometry': geometry,
                    'arm_a': first, 'arm_b': second, 'clips': len(diffs),
                    'mean_difference': sum(diffs) / len(diffs),
                    'ci': [lo, hi], 'significant': lo * hi > 0,
                    'wins': sum(1 for d in diffs if d > 0)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True,
                        help='directory of per-arm output trees')
    parser.add_argument('--reference', default='dense',
                        help='the mode every other arm is scored against')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    clips = _clips(args.root)
    if not clips:
        raise SystemExit(f'no mp4 under {args.root}/*/*/')
    usable = {k: v for k, v in clips.items() if args.reference in v}
    progress.log(f'{len(clips)} clips, {len(usable)} with a '
                 f'{args.reference!r} reference')
    if not usable:
        raise SystemExit(f'no clip has a {args.reference!r} arm yet; the '
                         'dense run has to finish first')

    per_clip, steps = [], progress.Progress('quality', len(usable), every=1)
    for (sample, geometry), arms in sorted(usable.items()):
        reference = arms[args.reference]
        for label, path in sorted(arms.items()):
            if label == args.reference:
                continue
            try:
                got = _measure(reference, path, args.ffmpeg)
            except ValueError as error:
                progress.log(f'  {sample} {label}: {error}')
                continue
            per_clip.append({'sample': sample, 'geometry': geometry,
                             'arm': label, **got})
        steps.update(f'{sample} {geometry}')

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in per_clip:
        groups.setdefault((row['arm'], row['geometry']), []).append(row)
    summary = []
    print(f'\n{"arm":<12}{"geometry":<11}{"n":>4}'
          f'{"PSNR dB":>10}{"95% CI":>18}{"SSIM":>9}{"95% CI":>18}')
    for (arm, geometry), rows in sorted(groups.items()):
        entry = {'arm': arm, 'geometry': geometry, 'clips': len(rows)}
        cells = []
        for metric in _METRICS:
            values = [r[metric] for r in rows if metric in r]
            if not values:
                cells.append(('-', '-'))
                continue
            mean = sum(values) / len(values)
            lo, hi = _bootstrap(values, args.bootstrap, args.seed)
            entry[metric] = mean
            entry[f'{metric}_ci'] = [lo, hi]
            cells.append((f'{mean:.3f}', f'[{lo:.3f}, {hi:.3f}]'))
        summary.append(entry)
        print(f'{arm:<12}{geometry:<11}{len(rows):>4}'
              f'{cells[0][0]:>10}{cells[0][1]:>18}'
              f'{cells[1][0]:>9}{cells[1][1]:>18}')

    paired = []
    for metric in _METRICS:
        paired += _paired(per_clip, metric, args.bootstrap, args.seed)
    if paired:
        print(f'\npaired over the clips each pair shares '
              f'(clip difficulty dominates the variance, so this is the '
              f'test with power):')
        print(f'{"metric":<9}{"geometry":<11}{"comparison":<22}{"n":>4}'
              f'{"mean diff":>11}{"95% CI":>22}{"wins":>7}')
        for row in paired:
            mark = ' *' if row['significant'] else '  '
            comparison = f'{row["arm_a"]} - {row["arm_b"]}'
            print(f'{row["metric"]:<9}{row["geometry"]:<11}'
                  f'{comparison:<22}{row["clips"]:>4}'
                  f'{row["mean_difference"]:>+11.4f}'
                  f'  [{row["ci"][0]:+.4f}, {row["ci"][1]:+.4f}]'
                  f'{row["wins"]:>5}/{row["clips"]}{mark}')
        print('  * the paired interval excludes zero')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'reference': args.reference, 'per_clip': per_clip,
                   'summary': summary, 'paired': paired},
                  open(args.out, 'w'), indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
