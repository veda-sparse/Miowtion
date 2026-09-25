"""Generates one clip with the H3 DiT on MLX, trunk streamed from slabs.

Writes the denoised latent rows (video and audio) as .npy; decoding them
into a video is a separate step (the VAE stays on torch/MPS). The text
states come either from a prompt (the tower is run here, layer by layer)
or from a file written earlier by scripts/mlx_encode_text.py.

Examples:
    python scripts/mlx_generate.py --transformer weights/h3/transformer \
        --slabs artifacts/mlx/slabs --text-encoder weights/h3/text_encoder \
        --tokenizer weights/h3/tokenizer --prompt 'a cat on a skateboard' \
        --steps 8 --out runs/mlx_clip
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np

from miowtion.h3 import geometry
from miowtion.mlx import convert
from miowtion.mlx import pipeline
from miowtion.mlx import slab as mlx_slab
from miowtion.mlx import text_encoder as mlx_text
from miowtion.train import encode as train_encode
from miowtion.utils import progress


def _text_states(args: argparse.Namespace) -> mx.array:
    """The [L, text_dim] bf16 rows the DiT conditions on."""
    if args.text_states:
        states = np.load(args.text_states)
        progress.log(f'text states {states.shape} from {args.text_states}')
        return mx.array(states).astype(mx.bfloat16)

    ids = mlx_text.token_ids(args.tokenizer, args.prompt)
    config = mlx_text.TowerConfig.from_pretrained(args.text_encoder)
    progress.log(f'{len(ids)} tokens, {args.text_layers} layers')
    reader = convert.ShardedSafetensors(args.text_encoder)
    embed = mlx_text.embed_tokens(reader)
    slabs = None
    if args.text_slabs:
        paths = [mlx_slab.slab_path(args.text_slabs, i)
                 for i in range(args.text_layers)]
        slabs = mlx_slab.SlabReader(paths, slots=2)
        layers = mlx_text.slab_layers(slabs, range(args.text_layers))
    else:
        layers = ((i, mlx_text.LayerWeights.from_tensors(
            mlx_text.layer_tensors(reader, i)))
            for i in range(args.text_layers))
    start = time.time()
    hidden = mlx_text.encode(embed, layers, ids, config,
                             total=args.text_layers)
    mx.eval(hidden)
    progress.log(f'text hidden {tuple(hidden.shape)} in '
                 f'{time.time() - start:.1f} s')
    if slabs is not None:
        slabs.close()
    reader.close()
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', required=True,
                        help='released <variant>/transformer directory')
    parser.add_argument('--slabs', required=True,
                        help='trunk slab directory (scripts/mlx_convert.py)')
    parser.add_argument('--text-encoder',
                        help='released <variant>/text_encoder directory')
    parser.add_argument('--tokenizer',
                        help='released <variant>/tokenizer directory')
    parser.add_argument('--text-slabs', help='text encoder slab directory')
    parser.add_argument('--text-layers', type=int,
                        default=train_encode.TEXT_LAYERS,
                        help='text layers to run')
    parser.add_argument('--prompt', help='prompt text')
    parser.add_argument('--text-states',
                        help='.npy written by scripts/mlx_encode_text.py')
    parser.add_argument('--aspect', default='16:9')
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--short-edge', type=int,
                        default=geometry.BASE_SHORT_EDGE,
                        help='only lower it for smoke tests')
    parser.add_argument('--out', required=True,
                        help='directory for video.npy / audio.npy')
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.text_states):
        parser.error('pass exactly one of --prompt, --text-states')
    if args.prompt and not (args.text_encoder and args.tokenizer):
        parser.error('--prompt needs --text-encoder and --tokenizer')

    text = _text_states(args)
    reader = convert.ReleaseReader(args.transformer)
    out = pipeline.run_clip(
        reader, args.slabs, text,
        pipeline.ClipRequest(aspect=args.aspect, seconds=args.seconds,
                             steps=args.steps, seed=args.seed,
                             short_edge=args.short_edge))
    reader.close()
    progress.log(pipeline.steps_summary(out.step_seconds))
    progress.log(f'video {tuple(out.video.shape)} '
                 f'audio {tuple(out.audio.shape)}, '
                 f'peak {mx.get_peak_memory() / 2**30:.2f} GB')

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (('video', out.video), ('audio', out.audio)):
        values = np.array(rows.astype(mx.float32))
        np.save(os.path.join(args.out, f'{name}.npy'), values)
        progress.log(f'{name}: mean {values.mean():.4f}, '
                     f'std {values.std():.4f}')


if __name__ == '__main__':
    main()
