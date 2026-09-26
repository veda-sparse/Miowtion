"""Builds the side-by-side comparison AGENTS.md 1.5 asks for.

When a path cannot be bit-aligned against its reference, the evidence is a
human looking at the frames. This script makes what that person needs, in
one place and the same way every time: a copy of every input and one titled
N-up video. Scalar metrics (--metrics) and difference heatmaps (--heatmap)
are opt-in; quality is judged by eye (AGENTS.md 1.5.1).

Panes keep the order given on the command line, baseline first, and the
first one is the reference unless --reference names another. Output goes to
artifacts/visual_checks/<feature>/<date>/ by convention.

    python scripts/visual_check.py --feature veda_predictor \\
        --video "Step 600 (t37)"=runs/a/x.mp4 \\
        --video "Step 200 bf16"=runs/b/x.mp4 \\
        --video "Step 200 fp8"=runs/c/x.mp4 \\
        --out-dir artifacts/visual_checks/veda_predictor/2026-09-26
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess

from miowtion.utils import progress

# Height of the title bar drawn above each pane, in pixels. Big enough to
# read in a 3-up of a 720p clip without eating into the picture.
_TITLE_HEIGHT = 48
_TITLE_FONT_SIZE = 28

# ffmpeg writes the metric summary to stderr; these pick the mean out of
# lines like 'PSNR y:.. u:.. v:.. average:31.94 min:.. max:..' and
# 'SSIM Y:.. U:.. V:.. All:0.912345 (10.6)'.
_PSNR_RE = re.compile(r'average:([0-9.]+|inf)')
_SSIM_RE = re.compile(r'All:([0-9.]+)')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='append', required=True,
                        metavar='LABEL=PATH',
                        help='a pane, repeatable; order is kept')
    parser.add_argument('--reference', default=None,
                        help='label the others are measured against '
                        '(default: the first --video)')
    parser.add_argument('--feature', default=None,
                        help='feature name, for the default --out-dir')
    parser.add_argument('--out-dir', default=None,
                        help='default artifacts/visual_checks/<feature>/'
                        '<today>')
    parser.add_argument('--heatmap', action='store_true',
                        help='also write per-frame difference videos '
                        '(off by default: only useful when chasing a '
                        'specific defect)')
    parser.add_argument('--metrics', action='store_true',
                        help='also compute PSNR / SSIM against the '
                        'reference (off by default: quality is judged by '
                        'eye, see AGENTS.md 1.5.1)')
    parser.add_argument('--heatmap-gain', type=float, default=8.0,
                        help='multiplies the difference before it is '
                        'colored; 1.0 shows the raw difference, which is '
                        'usually invisible')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    return parser.parse_args(argv)


def parse_videos(specs: list[str]) -> dict[str, str]:
    """`['Dense=a.mp4', ...]` -> ordered {label: path}."""
    out: dict[str, str] = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'--video wants LABEL=PATH, got {spec!r}')
        label, path = spec.split('=', 1)
        if not label:
            raise ValueError(f'empty label in {spec!r}')
        if label in out:
            raise ValueError(f'duplicate label {label!r}')
        out[label] = path
    return out


def escape_text(label: str) -> str:
    """Escapes a label for ffmpeg's drawtext, which parses its own syntax."""
    for char in ('\\', ':', "'", '%', ',', '[', ']', ';'):
        label = label.replace(char, '\\' + char)
    return label


def stack_command(videos: dict[str, str], out_path: str,
                  ffmpeg: str = 'ffmpeg') -> list[str]:
    """ffmpeg argv for the titled 1xN stack.

    Every pane is padded with a title bar of its own and then hstacked, so
    the labels stay attached to their pane whatever the aspect ratio is.
    """
    args = [ffmpeg, '-y']
    for path in videos.values():
        args += ['-i', path]
    parts = []
    for index, label in enumerate(videos):
        parts.append(
            f'[{index}:v]pad=iw:ih+{_TITLE_HEIGHT}:0:{_TITLE_HEIGHT}:black,'
            f"drawtext=text='{escape_text(label)}':fontcolor=white:"
            f'fontsize={_TITLE_FONT_SIZE}:x=(w-text_w)/2:'
            f'y={(_TITLE_HEIGHT - _TITLE_FONT_SIZE) // 2}[p{index}]')
    inputs = ''.join(f'[p{i}]' for i in range(len(videos)))
    parts.append(f'{inputs}hstack=inputs={len(videos)}[v]')
    args += ['-filter_complex', ';'.join(parts), '-map', '[v]']
    # The audio of the reference pane: the panes are the same clip, so N
    # copies of one track would only phase against each other.
    args += ['-map', '0:a?', '-c:v', 'libx264', '-crf', '18',
             '-pix_fmt', 'yuv420p', '-c:a', 'aac', out_path]
    return args


def heatmap_command(reference: str, other: str, out_path: str, gain: float,
                    ffmpeg: str = 'ffmpeg') -> list[str]:
    """ffmpeg argv for a per-frame |a - b| heatmap video."""
    chain = (f'[0:v][1:v]blend=all_mode=difference[d];'
             f'[d]format=gray,eq=contrast={gain},'
             f'pseudocolor=preset=turbo[v]')
    return [ffmpeg, '-y', '-i', reference, '-i', other,
            '-filter_complex', chain, '-map', '[v]', '-an',
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            out_path]


def metrics_command(reference: str, other: str,
                    ffmpeg: str = 'ffmpeg') -> list[str]:
    """ffmpeg argv computing PSNR and SSIM of `other` against `reference`."""
    return [ffmpeg, '-hide_banner', '-i', other, '-i', reference,
            '-filter_complex',
            '[0:v][1:v]psnr;[0:v][1:v]ssim', '-f', 'null', '-']


def parse_metrics(stderr: str) -> dict[str, float]:
    """Pulls the mean PSNR / SSIM out of ffmpeg's report.

    Raises:
        ValueError: If neither metric is present, which means the filters
            did not run (mismatched sizes or frame counts, usually).
    """
    psnr = _PSNR_RE.search(stderr)
    ssim = _SSIM_RE.search(stderr)
    if not psnr and not ssim:
        raise ValueError('ffmpeg reported neither PSNR nor SSIM')
    out = {}
    if psnr:
        out['psnr_db'] = float('inf') if psnr.group(1) == 'inf' else float(
            psnr.group(1))
    if ssim:
        out['ssim'] = float(ssim.group(1))
    return out


def _run(args: list[str]) -> str:
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f'{args[0]} failed ({done.returncode}):\n'
                           f'{done.stderr[-2000:]}')
    return done.stderr


def main():
    args = parse_args()
    videos = parse_videos(args.video)
    if len(videos) < 2:
        raise ValueError('a comparison needs at least two --video panes')
    reference = args.reference or next(iter(videos))
    if reference not in videos:
        raise ValueError(f'--reference {reference!r} is not one of '
                         f'{sorted(videos)}')
    missing = [p for p in videos.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'no such video: {missing}')
    out_dir = args.out_dir
    if not out_dir:
        if not args.feature:
            raise ValueError('pass --out-dir or --feature')
        out_dir = os.path.join('artifacts', 'visual_checks', args.feature,
                               datetime.date.today().isoformat())
    os.makedirs(out_dir, exist_ok=True)

    # The inputs travel with the comparison: a stacked video alone cannot
    # be re-cropped or stepped through, and the originals move or expire.
    for label, path in videos.items():
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', label).strip('_')
        copy = os.path.join(out_dir, f'{name}.mp4')
        if os.path.abspath(copy) != os.path.abspath(path):
            shutil.copyfile(path, copy)

    stack = os.path.join(out_dir, 'side_by_side.mp4')
    with progress.Timer(f'stack {len(videos)} panes'):
        _run(stack_command(videos, stack, args.ffmpeg))

    report = {'reference': reference, 'panes': videos, 'metrics': {}}
    for label, path in videos.items():
        if label == reference:
            continue
        if args.metrics:
            with progress.Timer(f'psnr / ssim: {label}'):
                values = parse_metrics(_run(metrics_command(
                    videos[reference], path, args.ffmpeg)))
            report['metrics'][label] = values
            progress.log(
                f'  {label}: PSNR {values.get("psnr_db", float("nan")):.2f}'
                f' dB  SSIM {values.get("ssim", float("nan")):.4f}')
        if args.heatmap:
            name = re.sub(r'[^A-Za-z0-9._-]+', '_', label).strip('_')
            with progress.Timer(f'heatmap: {label}'):
                _run(heatmap_command(videos[reference], path,
                                     os.path.join(out_dir,
                                                  f'diff_{name}.mp4'),
                                     args.heatmap_gain, args.ffmpeg))
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    progress.log(f'wrote {out_dir}/side_by_side.mp4 and report.json')


if __name__ == '__main__':
    main()
