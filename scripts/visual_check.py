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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.infer import decode                           # noqa: E402
from miowtion.utils import progress                         # noqa: E402

# Height of the title bar drawn above each pane, in pixels. Big enough to
# read in a 3-up of a 720p clip without eating into the picture.
_TITLE_HEIGHT = 48

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
    parser.add_argument('--stack', default='auto',
                        help="pane layout: 'RxC' a row-major grid (2x2, "
                        "3x2, ...), 'h' left to right, 'v' top to bottom. "
                        "'auto' (default) reads the reference pane with "
                        'ffprobe: a portrait clip goes in a row, anything '
                        'else in a two-column grid from four panes up, else '
                        'a column')
    parser.add_argument('--ffprobe', default='ffprobe',
                        help='ffprobe binary for --stack auto')
    parser.add_argument('--joined-dir', default=None,
                        help='also copy the stacked video here as '
                        '<clip>.mp4, where <clip> is the output directory '
                        'name -- a flat folder of just the comparisons, '
                        'without the per-pane inputs beside them')
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


def write_title(label: str, path: str,
                bar_height: int = _TITLE_HEIGHT) -> str:
    """Renders `label` as a tight title-bar PNG; returns `path`.

    The bar is drawn here rather than by ffmpeg's drawtext, which needs an
    ffmpeg built against libfreetype -- neither the workstation nor the GPU
    box in use has one, and a comparison that cannot be labelled is not a
    comparison (AGENTS.md 1.5.1). It also shares its font policy with
    miowtion.infer.decode, so a stacked video from this script and the
    dense-vs-veda video generate.py writes carry identical-looking titles.
    """
    from PIL import Image  # pylint: disable=import-outside-toplevel
    Image.fromarray(decode.title_bar(label, bar_height)).save(path)
    return path


def stack_command(videos: dict[str, str], titles: dict[str, str],
                  out_path: str, ffmpeg: str = 'ffmpeg',
                  stack: str = 'h') -> list[str]:
    """ffmpeg argv for the titled 1xN stack.

    Every pane is padded with a bar of its own and the pane's title image is
    overlaid centered on it, then the panes are hstacked, so the labels stay
    attached to their pane whatever the aspect ratio is. The title is
    composited at its natural size (never scaled), which keeps the text
    crisp and needs no probing of the pane's dimensions; a label wider than
    its pane is cropped.

    Args:
        videos: Ordered {label: video path}.
        titles: {label: PNG path} from write_title, one per pane.
        out_path: Destination.
        ffmpeg: Binary to call.
        stack: 'h' places the panes left to right, 'v' top to bottom, and
            'RxC' (e.g. '2x2', '3x2') fills an R-row by C-column grid
            row-major. The useful choice depends on the clip's own
            orientation: four 16:9 panes in a row are 5376 px wide and
            unwatchable, in a column 3264 px tall, but a 2x2 grid is
            2688x1632 -- close to a screen's own shape. Four 9:16 panes
            already fit side by side. A grid's last row may be partly
            empty (five panes in 3x2); the gap stays black.

    Raises:
        ValueError: On an unknown `stack`, on a grid too small for the
            panes, or on a grid whose last row would be entirely empty
            (which would mean the caller asked for the wrong shape).
    """
    grid = None
    if stack not in ('h', 'v'):
        match = re.fullmatch(r'(\d+)x(\d+)', stack)
        if not match:
            raise ValueError(
                f"stack must be 'h', 'v' or 'RxC', got {stack!r}")
        grid = (int(match.group(1)), int(match.group(2)))
        rows, cols = grid
        if rows < 1 or cols < 1:
            raise ValueError(f'grid must be positive, got {stack!r}')
        if not (rows - 1) * cols < len(videos) <= rows * cols:
            raise ValueError(
                f'{stack} holds {rows * cols} panes with no empty row for '
                f'{(rows - 1) * cols + 1}..{rows * cols}, got '
                f'{len(videos)}')
    args = [ffmpeg, '-y']
    for path in videos.values():
        args += ['-i', path]
    for label in videos:
        args += ['-i', titles[label]]
    parts = []
    count = len(videos)
    for index, label in enumerate(videos):
        parts.append(
            f'[{index}:v]pad=iw:ih+{_TITLE_HEIGHT}:0:{_TITLE_HEIGHT}:black'
            f'[b{index}]')
        parts.append(f'[b{index}][{count + index}:v]'
                     f'overlay=(W-w)/2:0[p{index}]')
    inputs = ''.join(f'[p{i}]' for i in range(count))
    if grid is not None:
        # Row-major. Every pane is the same clip and therefore the same
        # size, so an offset can be written as a multiple of pane 0's own
        # width and height rather than a sum over the preceding panes.
        cells = []
        for index in range(count):
            row, col = divmod(index, grid[1])
            x = '+'.join(['w0'] * col) or '0'
            y = '+'.join(['h0'] * row) or '0'
            cells.append(f'{x}_{y}')
        parts.append(f'{inputs}xstack=inputs={count}:'
                     f'layout={"|".join(cells)}[v]')
    else:
        parts.append(f'{inputs}{stack}stack=inputs={count}[v]')
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


def probe_stack(path: str, ffprobe: str = 'ffprobe',
                panes: int = 0) -> str:
    """The layout that keeps a comparison watchable: a grid, 'v' or 'h'.

    A portrait clip gets 'h' (the panes are narrow, so a row is fine).
    Anything else goes in a two-column grid once there are four or more
    panes -- '2x2' for four, '3x2' for five or six -- since a row of wide
    panes is unusably wide and a column of more than three is unusably
    tall (see stack_command). Fewer than four wide panes stay a column.

    Raises:
        RuntimeError: If ffprobe is missing or reports no video stream. It
            is not guessed: the wrong choice makes a comparison nobody can
            watch, and `--stack` states it explicitly.
    """
    try:
        done = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
            capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(
            f'{ffprobe} not found, so the stack direction cannot be probed; '
            "pass --stack h or --stack v") from error
    fields = done.stdout.strip().split(',')
    if done.returncode or len(fields) < 2:
        raise RuntimeError(f'{ffprobe} found no video stream in {path}: '
                           f'{done.stderr[-500:]}')
    width, height = int(fields[0]), int(fields[1])
    if width < height:
        return 'h'
    if panes >= 4:
        return f'{-(-panes // 2)}x2'
    return 'v'


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
    titles = {label: write_title(
        label, os.path.join(out_dir, f'.title_{index}.png'))
        for index, label in enumerate(videos)}
    direction = (probe_stack(videos[reference], args.ffprobe, len(videos))
                 if args.stack == 'auto' else args.stack)
    with progress.Timer(f'stack {len(videos)} panes ({direction})'):
        _run(stack_command(videos, titles, stack, args.ffmpeg, direction))
    for path in titles.values():
        os.remove(path)

    if args.joined_dir:
        os.makedirs(args.joined_dir, exist_ok=True)
        joined = os.path.join(args.joined_dir,
                              f'{os.path.basename(out_dir.rstrip("/"))}.mp4')
        shutil.copyfile(stack, joined)
        progress.log(f'joined copy: {joined}')

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
