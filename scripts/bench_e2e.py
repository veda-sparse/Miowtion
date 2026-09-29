"""Run repeatable end-to-end attention benchmark sweeps.

The benchmark includes per-call metadata preparation in the ``+prep``
measurements and writes all raw rounds alongside aggregate statistics.  This
makes results from a new GPU auditable without relying on copied terminal
tables.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_e2e.py \
        --seq 4096 8192 16384 --heads 8 --density 0.1 \
        --repeat 3 --calls 16 --out runs/bench/e2e.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
from collections.abc import Iterable

import torch

from miowtion.kernels import bench


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq', type=int, nargs='+', default=[16384, 32768],
                        help='sequence lengths; each must be divisible by 128')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--patterns', type=int, default=8)
    parser.add_argument('--calls', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=3,
                        help='independent rounds per sequence length')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True, help='JSON output path')
    return parser


def _finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(value)]


def _aggregate(rounds: list[dict]) -> dict:
    """Aggregates one result field over independent rounds.

    Missing values are kept missing rather than converted to zero; unavailable
    backends therefore cannot look faster than working backends.
    """
    fields = ('kernel_ms', 'prep_ms', 'max_err', 'efficiency',
              'efficiency_prep', 'tflops', 'mfu', 'gemm_frac')
    aggregate = {}
    for field in fields:
        values = _finite(round_[field] for round_ in rounds)
        if not values:
            aggregate[field] = None
            continue
        aggregate[field] = {
            'median': statistics.median(values),
            'p95': statistics.quantiles(values, n=20,
                                        method='inclusive')[18]
            if len(values) >= 2 else values[0],
            'min': min(values),
            'max': max(values),
        }
    return aggregate


def _result_dict(result: bench.Result) -> dict:
    payload = dataclasses.asdict(result)
    # JSON has no NaN value in the portable interchange format.
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
    return payload


def run(args: argparse.Namespace) -> dict:
    """Runs the configured sweep and returns a JSON-serializable payload."""
    if torch.cuda.device_count() != 1:
        raise ValueError('benchmark one GPU at a time (CUDA_VISIBLE_DEVICES)')
    if args.repeat < 1:
        raise ValueError('--repeat must be at least 1')
    if args.calls < 1 or args.warmup < 0:
        raise ValueError('--calls must be positive and --warmup non-negative')
    device_name = torch.cuda.get_device_name(0)
    rounds = {}
    for seq in args.seq:
        seq_rounds = []
        for repeat in range(args.repeat):
            results = bench.run(
                seq=seq, heads=args.heads, head_dim=args.head_dim,
                density=args.density, seed=args.seed + repeat,
                patterns=args.patterns, calls=args.calls,
                warmup=args.warmup)
            seq_rounds.append({r.name: _result_dict(r) for r in results})
        by_kernel = {}
        for name in sorted({name for round_ in seq_rounds
                            for name in round_}):
            samples = [round_[name] for round_ in seq_rounds
                       if name in round_]
            by_kernel[name] = {
                'rounds': samples,
                'aggregate': _aggregate(samples),
            }
        rounds[str(seq)] = by_kernel
    return {
        'device': {'name': device_name, 'torch': torch.__version__},
        'config': {
            'seq': args.seq, 'heads': args.heads,
            'head_dim': args.head_dim, 'density': args.density,
            'patterns': args.patterns, 'calls': args.calls,
            'warmup': args.warmup, 'repeat': args.repeat,
            'seed': args.seed,
        },
        'rounds': rounds,
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as output:
        json.dump(payload, output, indent=1, allow_nan=False)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
