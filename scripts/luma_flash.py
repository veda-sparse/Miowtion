"""Single-frame brightness flashes: the artifact users actually report.

Measured against the dense run of the same clip, because the raw count
is not specific: scanning every arm on the demo set, *dense itself*
trips a bare single-frame detector on 8 of 11 clips (2.7 "flashes" per
clip), since these captions produce hard cuts and the detector cannot
tell a cut from a flash on one clip alone. The dense arm is the control
that removes them -- a flash that is in both is the content.

Veda-on-ComfyUI issue #3 reports "a one-frame brightness flash mid-clip",
and two more users confirm it. That is not what `flicker_profile.py`
measures: a frame-difference profile is dominated by motion and by shot
cuts, and a flash lasting one frame moves it barely at all while being
the most visible artifact in the clip.

So this measures the statistic the artifact is actually in: the mean
luminance of each frame. A flash is a frame whose luma departs from
*both* neighbours in the *same* direction -- a shot cut or a lighting
change moves luma and keeps it moved, a flash comes back. The score is
that departure in units of the clip's own frame-to-frame luma noise, so
clips with different motion are comparable.

Example:
    python scripts/luma_flash.py --root artifacts/demo12_compare \\
        --out artifacts/demo12_compare/luma_flash.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402

# A flash has to clear the clip's own frame-to-frame noise by this much
# before it is called one. 4 sigma on a robust scale: shot cuts and fast
# motion sit well under it because they do not come back.
_FLASH_SIGMA = 4.0


def frame_luma(path: str, ffmpeg: str = 'ffmpeg') -> list[float]:
    """Mean luma of every frame, via ffmpeg's signalstats.

    Returns:
        One value per frame, in the file's order.
    """
    argv = [ffmpeg, '-hide_banner', '-nostats', '-i', path,
            '-vf', 'signalstats,metadata=print:key=lavfi.signalstats.YAVG',
            '-f', 'null', '-']
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    out = []
    for line in done.stderr.splitlines():
        if 'lavfi.signalstats.YAVG' in line:
            out.append(float(line.rsplit('=', 1)[1]))
    if not out:
        raise ValueError(f'no luma read from {path}; is ffmpeg built with '
                         'signalstats?')
    return out


def flashes(luma: list[float]) -> tuple[list[dict], float]:
    """Frames that depart from both neighbours the same way, and the scale.

    Args:
        luma: Per-frame mean luma.

    Returns:
        The flashes found, and the robust frame-to-frame luma scale they
        were measured against.
    """
    if len(luma) < 3:
        return [], 0.0
    steps = sorted(abs(luma[i + 1] - luma[i]) for i in range(len(luma) - 1))
    # Median absolute step, not the standard deviation: one shot cut in a
    # 100-frame clip moves the sd enough to hide every flash behind it.
    scale = max(steps[len(steps) // 2], 1e-6)
    found = []
    for i in range(1, len(luma) - 1):
        left, right = luma[i] - luma[i - 1], luma[i] - luma[i + 1]
        if left * right <= 0:          # not a return: a step, not a flash
            continue
        excess = min(abs(left), abs(right)) / scale
        if excess >= _FLASH_SIGMA:
            found.append({'frame': i, 'luma': luma[i],
                          'excess_sigma': excess,
                          'delta': (left + right) / 2.0})
    return found, scale


def tile_phase(frames: list[int], period: int, total: int
               ) -> tuple[list[int], float]:
    """Where flashes fall inside one query tile's span in time.

    A Veda query tile shares one key selection across every row it
    holds, so the selection is piecewise constant in time and can only
    change at a tile boundary. If the flash is caused by the routing,
    flash frames must pile up at one phase of that period; if they are
    spread uniformly, the routing's time structure is not what is
    producing them. This is the direct version of the periodicity test
    in docs/features/veda2.md 2.22, which looked at frame differences
    rather than at flashes.

    Args:
        frames: Frame indices of the flashes found.
        period: RGB frames per query tile (4 per latent frame times the
            tile's temporal extent).
        total: Frames per clip, for the uniform null.

    Returns:
        The histogram over phase, and the fraction of flashes in the
        fullest bin -- 1 / period under the null.
    """
    if period <= 1 or not frames:
        return [], 0.0
    histogram = [0] * period
    for frame in frames:
        histogram[frame % period] += 1
    del total
    return histogram, max(histogram) / len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--reference', default='dense',
                        help='the arm every other one is differenced against')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--tile-frames', type=int, default=0,
                        help='RGB frames spanned by one query tile; '
                        'reports where flashes fall inside that period')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    from scripts import quality_vs_dense                   # noqa: PLC0415
    clips = quality_vs_dense._clips(args.root)             # noqa: SLF001
    usable = {k: v for k, v in clips.items() if args.reference in v}
    if not usable:
        raise SystemExit(f'no clip has a {args.reference!r} arm')
    paths = sorted({p for arms in usable.values() for p in arms.values()})
    progress.log(f'{len(usable)} clips with a {args.reference!r} reference, '
                 f'{len(paths)} videos')

    luma: dict[str, list[float]] = {}
    steps = progress.Progress('luma', len(paths), every=10)
    for path in paths:
        try:
            luma[path] = frame_luma(path, args.ffmpeg)
        except ValueError as error:
            progress.log(f'  {path}: {error}')
        steps.update(os.path.basename(path))

    rows = []
    for (sample, geometry), arms in sorted(usable.items()):
        base = luma.get(arms[args.reference])
        if base is None:
            continue
        for label, path in sorted(arms.items()):
            if label == args.reference or path not in luma:
                continue
            series = luma[path]
            n = min(len(series), len(base))
            # The difference against dense cancels the content: a cut is
            # in both arms and leaves the difference flat, while a flash
            # that only one arm has survives.
            found, scale = flashes([series[i] - base[i] for i in range(n)])
            rows.append({'sample': sample, 'geometry': geometry,
                         'arm': label, 'frames': n, 'scale': scale,
                         'flashes': found,
                         'worst': max((f['excess_sigma'] for f in found),
                                      default=0.0)})

    by_arm: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_arm.setdefault((row['arm'], row['geometry']), []).append(row)
    print(f'\n{"arm":<14}{"geometry":<11}{"clips":>6}{"with a flash":>14}'
          f'{"flashes/clip":>14}{"worst sigma":>13}')
    for (arm, geometry), group in sorted(by_arm.items()):
        hit = [r for r in group if r['flashes']]
        total = sum(len(r['flashes']) for r in group)
        worst = max(r['worst'] for r in group)
        print(f'{arm:<14}{geometry:<11}{len(group):>6}{len(hit):>14}'
              f'{total / len(group):>14.2f}{worst:>13.1f}')
    if args.tile_frames > 1:
        print(f'\nwhere the flashes fall inside one {args.tile_frames}-frame '
              f'query tile (uniform would be '
              f'{1 / args.tile_frames:.3f} in the fullest bin):')
        print(f'{"arm":<14}{"flashes":>9}{"fullest bin":>13}{"phase":>8}')
        for (arm, _geometry), group in sorted(by_arm.items()):
            frames = [f['frame'] for r in group for f in r['flashes']]
            histogram, peak = tile_phase(frames, args.tile_frames, 0)
            if not histogram:
                continue
            print(f'{arm:<14}{len(frames):>9}{peak:>13.3f}'
                  f'{histogram.index(max(histogram)):>8}')
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'reference': args.reference, 'per_clip': rows,
                   'luma': luma}, open(args.out, 'w'), indent=1)
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
