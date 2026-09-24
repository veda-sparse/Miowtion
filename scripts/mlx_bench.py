"""Benchmarks of MLX inference with NVMe offloading (synthetic weights).

Run every measurement in its own process so that peak memory is per option.
Results are printed as one JSON line and appended to --out.

Examples:
    python scripts/mlx_bench.py slabs --dir artifacts/mlx/slabs_bf16 \
        --blocks 8 --bits 0
    python scripts/mlx_bench.py ssd --paths artifacts/mlx/slabs_bf16/*.slab
    python scripts/mlx_bench.py load --source slab \
        --dir artifacts/mlx/slabs_bf16 --blocks 8
    python scripts/mlx_bench.py compute --seq-len 4096 16384 --bits 0
    python scripts/mlx_bench.py attention --seq-len 38912 --density 1.0 0.1
    python scripts/mlx_bench.py stream --dir artifacts/mlx/slabs_bf16 \
        --blocks 8 --passes 1 --seq-len 4096
"""

import argparse
import json

import mlx.core as mx

from miowtion.h3 import config as h3_config
from miowtion.mlx import bench
from miowtion.mlx import block as mlx_block
from miowtion.utils import progress


def _emit(result: dict, out: str | None) -> None:
    line = json.dumps(result)
    print(line, flush=True)
    if out:
        with open(out, 'a') as f:
            f.write(line + '\n')


def _options(args, seq_len: int) -> mlx_block.BlockOptions:
    plan = None
    if args.density < 1.0:
        plan = bench.sparse_plan(seq_len, args.density, args.q_block,
                                 args.k_block)
    return mlx_block.BlockOptions(head_chunk=args.head_chunk,
                                  row_chunk=args.row_chunk,
                                  eval_chunks=args.eval_chunks, sparse=plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', help='JSON-lines file to append results to')
    parser.add_argument('--min-available-gb', type=float, default=6.0,
                        help='refuse to start below this available memory')
    parser.add_argument('--cache-limit-gb', type=float, default=1.0,
                        help='MLX buffer cache limit')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('slabs', help='write synthetic slabs')
    p.add_argument('--dir', required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--bits', type=int, default=0, choices=(0, 4, 8))

    p = sub.add_parser('safetensors', help='write a synthetic checkpoint')
    p.add_argument('--dir', required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--per-shard', type=int, default=4)

    p = sub.add_parser('ssd', help='raw sequential read throughput')
    p.add_argument('--paths', nargs='+', required=True)
    p.add_argument('--cache', action='store_true',
                   help='read through the page cache (default: F_NOCACHE)')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--piece-mb', type=int, default=16)

    p = sub.add_parser('load', help='per-block load latency')
    p.add_argument('--source', choices=('slab', 'mx_load', 'mmap'),
                   required=True)
    p.add_argument('--dir', required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--cache', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--dequantize', action='store_true')

    p = sub.add_parser('residency', help='page-cache residency of files')
    p.add_argument('--paths', nargs='+', required=True)

    p = sub.add_parser('attention', help='attention alone, dense vs sparse')
    p.add_argument('--seq-len', type=int, nargs='+', required=True)
    p.add_argument('--density', type=float, nargs='+', default=[1.0])
    p.add_argument('--q-block', type=int, nargs='+', default=[2048])
    p.add_argument('--k-block', type=int, default=128)
    p.add_argument('--head-chunk', type=int, default=None)
    p.add_argument('--reps', type=int, default=3)

    for name in ('compute', 'stream'):
        p = sub.add_parser(name)
        p.add_argument('--head-chunk', type=int, default=None)
        p.add_argument('--density', type=float, default=1.0,
                       help='Veda block-sparse density (1.0 = dense)')
        p.add_argument('--q-block', type=int, default=2048)
        p.add_argument('--k-block', type=int, default=128)
        p.add_argument('--row-chunk', type=int,
                       default=mlx_block.DEFAULT_ROW_CHUNK)
        p.add_argument('--no-eval-chunks', dest='eval_chunks',
                       action='store_false',
                       help='keep every chunk lazy (raises peak memory)')
        if name == 'compute':
            p.add_argument('--seq-len', type=int, nargs='+', required=True)
            p.add_argument('--bits', type=int, default=0, choices=(0, 4, 8))
            p.add_argument('--qmm', action='store_true',
                           help='quantized matmuls instead of dequantizing')
            p.add_argument('--reps', type=int, default=1)
            p.add_argument('--no-profile', action='store_true')
        else:
            p.add_argument('--dir', required=True)
            p.add_argument('--blocks', type=int, required=True)
            p.add_argument('--passes', type=int, default=1)
            p.add_argument('--seq-len', type=int, required=True)
            p.add_argument('--depth', type=int, default=1)
            p.add_argument('--cache', action='store_true')
            p.add_argument('--dequantize', action='store_true')
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * bench.GB))
    config = h3_config.H3Config()
    before = bench.require_headroom(args.min_available_gb)
    progress.log(f'{args.cmd}: available {before["available"]:.1f} GB, '
                 f'swap used {before["swap_used"]:.1f} GB')
    if args.cmd == 'slabs':
        bench.write_synthetic_slabs(args.dir, config, args.blocks, args.bits)
        return
    if args.cmd == 'safetensors':
        bench.write_synthetic_safetensors(args.dir, config, args.blocks,
                                          args.per_shard)
        return
    if args.cmd == 'ssd':
        result = bench.ssd_read(args.paths, not args.cache, args.threads,
                                args.piece_mb << 20)
        result.update(cmd='ssd', cache=args.cache, threads=args.threads,
                      piece_mb=args.piece_mb)
    elif args.cmd == 'load':
        result = bench.load_latency(args.source, args.dir,
                                    list(range(args.blocks)),
                                    nocache=not args.cache,
                                    threads=args.threads,
                                    dequantize=args.dequantize)
        result['cmd'] = 'load'
    elif args.cmd == 'residency':
        result = {'cmd': 'residency',
                  'resident': [bench.file_resident_fraction(p)
                               for p in args.paths]}
    elif args.cmd == 'attention':
        for seq_len in args.seq_len:
            for density in args.density:
                blocks = args.q_block if density < 1.0 else [args.q_block[0]]
                for q_block in blocks:
                    bench.require_headroom(args.min_available_gb)
                    plan = (None if density >= 1.0 else
                            bench.sparse_plan(seq_len, density, q_block,
                                              args.k_block))
                    result = bench.attention_only(config, seq_len, plan,
                                                  head_chunk=args.head_chunk,
                                                  reps=args.reps)
                    result['cmd'] = 'attention'
                    _emit(result, args.out)
                    mx.clear_cache()
        return
    elif args.cmd == 'compute':
        for seq_len in args.seq_len:
            bench.require_headroom(args.min_available_gb)
            result = bench.compute_block(config, seq_len, args.bits,
                                         args.qmm, _options(args, seq_len),
                                         reps=args.reps,
                                         profile=not args.no_profile)
            result['cmd'] = 'compute'
            _emit(result, args.out)
            mx.clear_cache()
        return
    else:
        result = bench.streamed_run(args.dir, args.blocks, args.passes,
                                    args.seq_len, config,
                                    _options(args, args.seq_len),
                                    depth=args.depth, nocache=not args.cache,
                                    dequantize=args.dequantize)
        result['cmd'] = 'stream'
    after = bench.system_memory()
    result['system_before'] = before
    result['system_after'] = after
    _emit(result, args.out)


if __name__ == '__main__':
    main()
