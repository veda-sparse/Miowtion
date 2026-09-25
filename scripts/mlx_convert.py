"""Converts a released H3 checkpoint into per-block MLX slab files.

The transformer directory is the released `<variant>/transformer` (either
release layout, see miowtion.h3.release). Slabs hold only the trunk
tensors; the AdaLN projections and the non-trunk weights are read from the
checkpoint at run time.

Examples:
    python scripts/mlx_convert.py --transformer weights/h3/transformer \
        --out artifacts/mlx/slabs_bf16
    python scripts/mlx_convert.py --transformer weights/h3/transformer \
        --out artifacts/mlx/slabs_q8 --bits 8 --blocks 0 8
"""

import argparse

from miowtion.mlx import convert
from miowtion.utils import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', required=True,
                        help='released <variant>/transformer directory')
    parser.add_argument('--out', required=True, help='slab directory')
    parser.add_argument('--blocks', type=int, nargs=2, metavar=('FIRST',
                                                                'LAST'),
                        help='block range to convert (default: all)')
    parser.add_argument('--bits', type=int, default=0, choices=(0, 4, 8),
                        help='0 keeps bf16; 4 or 8 quantize the linears')
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='convert what is there instead of requiring '
                             'every tensor the DiT needs')
    args = parser.parse_args()

    with convert.ReleaseReader(args.transformer) as reader:
        progress.log(f'schema {reader.schema}, {reader.config.num_layers} '
                     f'blocks, hidden {reader.config.hidden_size}')
        if not args.allow_incomplete:
            reader.check_complete()
        first, last = args.blocks or (0, reader.config.num_layers - 1)
        blocks = range(first, last + 1)
        convert.write_trunk_slabs(reader, args.out, list(blocks), args.bits,
                                  args.group_size)
    progress.log(f'wrote blocks {first}..{last} to {args.out}')


if __name__ == '__main__':
    main()
