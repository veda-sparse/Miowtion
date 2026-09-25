"""Compares a released trunk block on MLX against the torch reference.

Prints one line per sequence length (and one JSON line per comparison when
--out is given). The torch side is CPU eager and quadratic in the sequence
length, so keep the lengths small; the point is the numbers, not the speed.

Examples:
    python scripts/mlx_check.py --transformer weights/h3/transformer
    python scripts/mlx_check.py --transformer weights/h3/transformer \
        --block 7 --seq-len 512 4096 --out runs/mlx_check/blocks.jsonl
"""

import argparse
import dataclasses
import json

from miowtion.mlx import check
from miowtion.mlx import convert
from miowtion.utils import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', required=True,
                        help='released <variant>/transformer directory')
    parser.add_argument('--block', type=int, default=0,
                        help='trunk block to compare')
    parser.add_argument('--seq-len', type=int, nargs='+', default=[512],
                        help='packed sequence lengths to compare at')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bits', type=int, default=0, choices=(0, 4, 8),
                        help='quantize the MLX linears; the torch sides keep '
                             'the released weights, so this measures what '
                             'quantization costs')
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--out', help='append one JSON line per comparison')
    args = parser.parse_args()

    with convert.ReleaseReader(args.transformer) as reader:
        for seq_len in args.seq_len:
            result = check.compare_block(reader, args.block, seq_len,
                                         args.seed, bits=args.bits,
                                         group_size=args.group_size)
            flag = ('' if args.bits or result.as_good_as_torch
                    else '  [WORSE THAN TORCH BF16]')
            progress.log(f'{result.line()}{flag}')
            if args.out:
                with open(args.out, 'a') as f:
                    f.write(json.dumps(dataclasses.asdict(result)) + '\n')


if __name__ == '__main__':
    main()
