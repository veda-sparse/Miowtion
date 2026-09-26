"""Converts a released H3 checkpoint into per-block MLX slab files.

The transformer directory is the released `<variant>/transformer` (either
release layout, see miowtion.h3.release). Slabs hold only the trunk
tensors; the AdaLN projections and the non-trunk weights are read from the
checkpoint at run time.

The text encoder is converted the same way (one slab per Qwen3-VL layer),
because it does not fit in memory either; there quantization does not
apply, the layers are copied verbatim.

Examples:
    python scripts/mlx_convert.py --transformer weights/h3/transformer \
        --out artifacts/mlx/slabs_bf16
    python scripts/mlx_convert.py --transformer weights/h3/transformer \
        --out artifacts/mlx/slabs_q8 --bits 8 --blocks 0 8
    python scripts/mlx_convert.py --text-encoder weights/h3/text_encoder \
        --out artifacts/mlx/text_slabs --blocks 0 49
"""

import argparse

from miowtion.mlx import convert
from miowtion.mlx import text_encoder as mlx_text
from miowtion.utils import progress


def _convert_text_encoder(args: argparse.Namespace) -> None:
    with convert.ShardedSafetensors(args.text_encoder) as reader:
        config = mlx_text.TowerConfig.from_pretrained(args.text_encoder)
        progress.log(f'text encoder: {config.num_hidden_layers} layers, '
                     f'hidden {config.hidden_size}, '
                     f'{config.layer_bytes / 2**30:.2f} GB per layer')
        first, last = args.blocks or (0, config.num_hidden_layers - 1)
        mlx_text.write_tower_slabs(reader, args.out,
                                   list(range(first, last + 1)))
    progress.log(f'wrote layers {first}..{last} to {args.out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--transformer',
                        help='released <variant>/transformer directory')
    source.add_argument('--text-encoder',
                        help='released <variant>/text_encoder directory; '
                             'writes one slab per Qwen3-VL layer')
    parser.add_argument('--out', required=True, help='slab directory')
    parser.add_argument('--blocks', type=int, nargs=2, metavar=('FIRST',
                                                                'LAST'),
                        help='block (or text encoder layer) range to '
                             'convert (default: all)')
    parser.add_argument('--bits', type=int, default=0, choices=(0, 4, 8),
                        help='0 keeps bf16; 4 or 8 quantize the linears')
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='convert what is there instead of requiring '
                             'every tensor the DiT needs')
    args = parser.parse_args()

    if args.text_encoder:
        _convert_text_encoder(args)
        return
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
