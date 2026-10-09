"""Watches a training log and stops a run that is not improving.

The trainer has its own early abort (train.monitor.EarlyAbort), but it only
fires on a metric the trainer itself computes and only while the trainer is
alive. This watches from outside, so it also catches a run that has stopped
logging altogether -- hung on a kernel, stuck on a disk, or killed without
a traceback, all of which happened while developing Veda2.

Three reasons to stop, each reported with its numbers:

* no improvement: the trailing mean of the watched metric has not set a
  new best for `--patience` updates. Trailing means, not single updates,
  because cycled geometries move the metric more than progress does.
* diverging: the metric has fallen below its own starting value by more
  than `--max-drop`, which is what a runaway looks like from outside.
* stalled: nothing new in the log for `--stall` seconds.

It prints what it decided and exits non-zero when it stops a run, so a
shell can tell the difference. With --kill it sends SIGTERM to the matching
process; without, it only reports.

Example:
    python scripts/watch_training.py --log runs/logs/train.log \\
        --patience 10 --kill
"""

import argparse
import json
import os
import re
import signal
import subprocess
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True,
                        help="the run's log file, as written by scripts/train.py")
    parser.add_argument('--metric', default='kept_over_ceiling',
                        help='record field to watch; larger is better')
    parser.add_argument('--window', type=int, default=8,
                        help='updates per trailing mean')
    parser.add_argument('--patience', type=int, default=10,
                        help='updates without a new best before stopping')
    parser.add_argument('--max-drop', type=float, default=0.05,
                        help='stop if the metric falls this far below its '
                        'first trailing mean')
    parser.add_argument('--stall', type=float, default=1800.0,
                        help='stop if the log has not grown for this long, '
                        'in seconds')
    parser.add_argument('--poll', type=float, default=60.0,
                        help='seconds between checks')
    parser.add_argument('--pattern', default='scripts/train.py',
                        help='pgrep -f pattern of the process to signal')
    parser.add_argument('--kill', action='store_true',
                        help='actually signal the process; off by default')
    return parser.parse_args(argv)


def read_records(path: str, metric: str) -> list[float]:
    """The watched metric from every per-update record in the log."""
    values = []
    if not os.path.exists(path):
        return values
    with open(path, errors='replace') as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith('{"step"'):
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if metric in record:
                values.append(float(record[metric]))
    return values


def verdict(values: list[float], window: int, patience: int,
            max_drop: float) -> str | None:
    """Why to stop, or None to carry on.

    Args:
        values: The watched metric, one per update, in order.
        window: Updates per trailing mean.
        patience: Updates without a new best before stopping.
        max_drop: Absolute fall below the first trailing mean that counts
            as divergence on its own.

    Returns:
        A reason, with the numbers in it, or None.
    """
    if len(values) < window:
        return None
    means = [sum(values[i:i + window]) / window
             for i in range(len(values) - window + 1)]
    best = max(means)
    best_at = means.index(best)
    latest = means[-1]
    if means[0] - latest > max_drop:
        return (f'diverging: trailing mean {latest:.4f} is '
                f'{means[0] - latest:.4f} below its starting value '
                f'{means[0]:.4f} (limit {max_drop})')
    since = len(means) - 1 - best_at
    if since >= patience:
        return (f'no improvement for {since} updates: trailing mean '
                f'{latest:.4f} against a best of {best:.4f} at update '
                f'{best_at + window}')
    return None


def stop(pattern: str) -> list[int]:
    """SIGTERMs every process matching `pattern`; returns the pids."""
    found = subprocess.run(['pgrep', '-f', pattern], capture_output=True,
                           text=True, check=False)
    pids = [int(p) for p in found.stdout.split() if p.isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return pids


def main() -> int:
    args = parse_args()
    last_size, last_change = -1, time.time()
    while True:
        size = os.path.getsize(args.log) if os.path.exists(args.log) else 0
        if size != last_size:
            last_size, last_change = size, time.time()
        values = read_records(args.log, args.metric)
        reason = verdict(values, args.window, args.patience, args.max_drop)
        if reason is None and time.time() - last_change > args.stall:
            reason = (f'stalled: {args.log} has not grown for '
                      f'{time.time() - last_change:.0f}s')
        if reason:
            print(f'[watcher] stopping: {reason}', flush=True)
            if args.kill:
                pids = stop(args.pattern)
                print(f'[watcher] signalled {pids or "nothing"}', flush=True)
            else:
                print('[watcher] --kill not given, reporting only',
                      flush=True)
            return 1
        done = re.search(r'EXIT=(\d+)', open(args.log, errors='replace').read()
                         ) if os.path.exists(args.log) else None
        if done:
            print(f'[watcher] run finished with {done.group(0)}, '
                  f'{len(values)} updates logged', flush=True)
            return 0
        print(f'[watcher] {len(values)} updates, '
              f'{args.metric} '
              f'{values[-1]:.4f}' if values else '[watcher] waiting',
              flush=True)
        time.sleep(args.poll)


if __name__ == '__main__':
    raise SystemExit(main())
