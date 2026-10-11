"""One titled side-by-side per clip, over a whole comparison root.

Judging video quality is done by eye, so AGENTS.md 1.5.1 asks for two
things on every clip of the test set: each arm as its own video, and one
joined comparison with a title bar per pane in a fixed order. Doing that
by hand for twenty clips and six arms is where it silently turns into
"here is one representative sample", which is the thing the rule exists
to prevent.

So this walks the per-arm output trees scripts/generate.py writes, pairs
the clips, and calls scripts/visual_check.py once per clip with the
panes in the order given. A clip that is missing an arm is reported and
skipped rather than quietly joined with fewer panes, because a
five-pane comparison that looks like a six-pane one is worse than none.

Example:
    python scripts/join_arms.py --root artifacts/isotime20 \\
        --arm 'Dense=dense' --arm 'Sol 4.8%=sol' \\
        --arm 'Veda2 20%=veda' --feature isotime20
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True,
                        help='directory of per-arm output trees')
    parser.add_argument('--arm', action='append', required=True,
                        help="'Title=mode' or 'Title=mode@arm', repeated; "
                        'the order given is the order of the panes')
    parser.add_argument('--feature', default=None,
                        help='artifacts/visual_checks/<feature>/<date>')
    parser.add_argument('--out-root', default=None)
    parser.add_argument('--stack', default='auto')
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many clips (0 = the whole '
                        'set, which is what the rule asks for)')
    args = parser.parse_args()

    from scripts import quality_vs_dense                   # noqa: PLC0415
    clips = quality_vs_dense._clips(args.root)             # noqa: SLF001
    wanted = []
    for spec in args.arm:
        if '=' not in spec:
            raise SystemExit(f'--arm needs Title=label, got {spec!r}')
        title, label = spec.split('=', 1)
        wanted.append((title, label))

    done, skipped = 0, []
    for (sample, geometry), arms in sorted(clips.items()):
        missing = [label for _t, label in wanted if label not in arms]
        if missing:
            skipped.append((sample, geometry, missing))
            continue
        if args.limit and done >= args.limit:
            break
        name = f'{sample}_{geometry}'
        out_dir = (os.path.join(args.out_root, name) if args.out_root
                   else None)
        argv = [sys.executable, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'visual_check.py')]
        for title, label in wanted:
            argv += ['--video', f'{title}={arms[label]}']
        argv += ['--stack', args.stack]
        if out_dir:
            argv += ['--out-dir', out_dir]
        else:
            argv += ['--feature', os.path.join(args.feature or 'arms', name)]
        progress.log(f'join {name}: {len(wanted)} panes')
        subprocess.run(argv, check=True)
        done += 1

    progress.log(f'joined {done} clips')
    for sample, geometry, missing in skipped:
        progress.log(f'  skipped {sample} {geometry}: missing {missing}')
    if skipped and not done:
        raise SystemExit('every clip was missing an arm; check --arm labels '
                         f'against {sorted({a for v in clips.values() for a in v})}')


if __name__ == '__main__':
    main()
