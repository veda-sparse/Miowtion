"""Immediate, rank-aware progress logging.

Every long-running entry point reports what it is doing as it happens: one
timestamped line per event, flushed immediately, printed by rank 0 only
(other ranks run the same schedule in lockstep). Long loops use `Progress`,
which adds done/total, elapsed time and an ETA.
"""

from __future__ import annotations

import datetime
import os
import sys
import time


def _rank() -> int:
    return int(os.environ.get('RANK', '0'))


def log(message: str, all_ranks: bool = False) -> None:
    """Prints `[HH:MM:SS rank] message` immediately."""
    if not all_ranks and _rank() != 0:
        return
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    prefix = f'[{stamp}]' if not all_ranks else f'[{stamp} r{_rank()}]'
    print(f'{prefix} {message}', file=sys.stdout, flush=True)


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours}h{minutes:02d}m'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


class Progress:
    """done/total counter with elapsed time and ETA.

    Example:
        progress = Progress('layers', total=50, every=10)
        for i in range(50):
            ...
            progress.update(f'layer {i}')
    """

    def __init__(self, name: str, total: int, every: int = 1):
        self.name = name
        self.total = total
        self.every = max(1, every)
        self.done = 0
        self.start = time.time()

    def update(self, detail: str = '', count: int = 1) -> None:
        self.done += count
        if self.done % self.every and self.done != self.total:
            return
        elapsed = time.time() - self.start
        rate = elapsed / max(1, self.done)
        eta = rate * (self.total - self.done)
        log(f'{self.name} {self.done}/{self.total} '
            f'({100 * self.done / max(1, self.total):.0f}%) '
            f'elapsed {_fmt(elapsed)} eta {_fmt(eta)}'
            + (f' | {detail}' if detail else ''))


class Timer:
    """Context manager logging the start and duration of a phase.

    After a successful exit `seconds` holds the duration, so a caller that
    reports stage times (scripts/benchmark.py) reads the same clock the log
    line was printed from instead of timing the phase a second time.
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.start = 0.0
        self.seconds = 0.0

    def __enter__(self):
        self.start = time.time()
        log(f'{self.phase} ...')
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.seconds = time.time() - self.start
            log(f'{self.phase} done in {_fmt(self.seconds)}')
        return False
