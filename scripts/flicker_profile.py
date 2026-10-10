"""Temporal flicker of a sparse arm, as the excess over the dense run.

A per-frame metric cannot see flicker (docs/features/veda2.md 2.11), so
this measures the thing that can: the frame-to-frame change. Real motion
also changes frames, so the signal is the arm's frame difference *minus
the dense run's*, which leaves what the sparsity added.

A Veda query tile shares one key selection across all its rows, so the
attention pattern is piecewise constant in time and steps at every tile
boundary. If that is what the eye sees, the excess should spike every
`tt * vae_temporal` frames, and flattening the tile (tt = 1) should
remove it.

Example:
    python scripts/flicker_profile.py --dense a/dense.mp4 \\
        --arm "released"=b/veda.mp4 --arm "tt1"=c/veda.mp4 --period 32
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402


def _frames(path: str, width: int = 192) -> np.ndarray:
    """[T, H, W] float32 grey frames, downscaled; ffmpeg does the work."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
        capture_output=True, text=True, check=True)
    src_w, src_h = (int(x) for x in probe.stdout.strip().split(',')[:2])
    height = max(2, int(round(src_h * width / src_w)) // 2 * 2)
    out = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', path, '-vf',
         f'scale={width}:{height},format=gray', '-f', 'rawvideo', '-'],
        capture_output=True, check=True)
    data = np.frombuffer(out.stdout, dtype=np.uint8)
    return data.reshape(-1, height, width).astype(np.float32)


def _profile(path: str) -> np.ndarray:
    """Mean |frame[t] - frame[t-1]| over the clip."""
    frames = _frames(path)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense', required=True, help='the reference mp4')
    parser.add_argument('--arm', action='append', required=True,
                        metavar='LABEL=PATH')
    parser.add_argument('--periods', type=int, nargs='+',
                        default=[4, 8, 16, 32],
                        help='candidate flicker periods in frames')
    args = parser.parse_args()

    with progress.Timer('dense profile'):
        base = _profile(args.dense)
    print(f'{"arm":20s} {"median":>8s} {"p90":>8s} {"trimmed":>8s} ' +
          ' '.join(f'p={p:<4d}' for p in args.periods))
    for spec in args.arm:
        if '=' not in spec:
            raise ValueError(f'--arm wants LABEL=PATH, got {spec!r}')
        label, path = spec.split('=', 1)
        arm = _profile(path)
        n = min(len(arm), len(base))
        excess = arm[:n] - base[:n]
        # The mean and the max are dominated by shot cuts: if a sparse arm
        # places a cut one frame early, that single transition swamps
        # everything. Sol scores a peak of 72 here and does not flicker,
        # which is how the first version of this gave a misleading answer.
        # So report robust statistics and drop the largest 5% of frames
        # before looking for periodicity.
        magnitude = np.abs(excess)
        keep = magnitude <= np.quantile(magnitude, 0.95)
        scores = []
        for period in args.periods:
            idx = np.arange(len(excess))
            on = magnitude[keep & (idx % period == 0)]
            off = magnitude[keep & (idx % period != 0)]
            scores.append(on.mean() / max(off.mean(), 1e-9)
                          if on.size else float('nan'))
        print(f'{label:20s} {np.median(magnitude):8.3f} '
              f'{np.quantile(magnitude, 0.90):8.3f} '
              f'{magnitude[keep].mean():8.3f} '
              + ' '.join(f'{s:6.2f}' for s in scores))


if __name__ == '__main__':
    main()
