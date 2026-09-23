"""Benchmarks block-sparse attention kernels against dense attention.

A new random pattern is used on every call (see miowtion.kernels.bench).
Set MIOWTION_FASTVIDEO_KERNEL to FastVideo's python/fastvideo_kernel directory
to include its Triton kernel.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_sparse_attention.py \
        --seq 32768 --heads 8 --density 0.1
"""

import argparse

import torch

from miowtion.kernels import bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq', type=int, nargs='+', default=[16384, 32768])
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--patterns', type=int, default=8,
                        help='distinct random patterns cycled per call')
    parser.add_argument('--calls', type=int, default=16)
    args = parser.parse_args()
    name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    for seq in args.seq:
        results = bench.run(seq=seq, heads=args.heads,
                            head_dim=args.head_dim, density=args.density,
                            patterns=args.patterns, calls=args.calls)
        print(bench.format_results(
            results, f'\n== {name} sm{cap[0]}{cap[1]} seq={seq} '
            f'heads={args.heads} d={args.head_dim} density={args.density}'),
              flush=True)


if __name__ == '__main__':
    main()
