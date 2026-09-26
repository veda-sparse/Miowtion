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
import dataclasses
import json
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
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
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


def _sparse_request(args: argparse.Namespace
                    ) -> pipeline.SparseRequest | None:
    """The Veda request, or None for dense attention.

    The default scorer pools every layer's own q and k and scores the tiles
    with mean-pooled QK -- what an untrained Veda predictor computes -- so
    the selection follows the content. `--sparse-scorer random` keeps the
    tile shapes, the budget and the per-head spread but picks arbitrary
    tiles: that run is a speed measurement, not a clip anyone should look
    at.
    """
    if args.sparse_ratio is None:
        return None
    dense = frozenset(int(i) for i in args.dense_layers.split(',')
                      if i.strip())
    veda = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=args.sparse_ratio),
        dense_layers=dense)
    progress.log(f'veda: keep {args.sparse_ratio}, {len(dense)} dense '
                 f'layers, scorer {args.sparse_scorer}')
    return pipeline.SparseRequest(veda=veda, plan_path=args.sparse_plan,
                                  seed=args.sparse_seed,
                                  scorer=args.sparse_scorer)


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
    parser.add_argument('--prompt-file', help='file holding the prompt')
    parser.add_argument('--text-states',
                        help='.npy written by scripts/mlx_encode_text.py')
    parser.add_argument('--aspect', default='16:9')
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--short-edge', type=int,
                        default=geometry.BASE_SHORT_EDGE,
                        help='only lower it for smoke tests')
    parser.add_argument('--sparse-ratio', type=float,
                        help='Veda keep ratio; omit to run dense attention')
    parser.add_argument('--sparse-plan',
                        help='searched tile plan (plans/*.json); the '
                             'least-padding shape is used without one')
    parser.add_argument('--sparse-seed', type=int, default=0,
                        help='seed of the stand-in tile scorer')
    parser.add_argument('--sparse-scorer', default='pooled',
                        choices=('pooled', 'random'),
                        help='pooled: score each layer from its own q/k; '
                             'random: stand-in scorer, speed only')
    parser.add_argument('--dense-layers', default='',
                        help='comma-separated trunk layers to keep dense')
    parser.add_argument('--out', required=True,
                        help='directory for video.npy / audio.npy')
    args = parser.parse_args()
    if args.prompt_file:
        with open(args.prompt_file) as f:
            args.prompt = f.read().strip()
    if bool(args.prompt) == bool(args.text_states):
        parser.error('pass exactly one of --prompt, --prompt-file, '
                     '--text-states')
    if args.prompt and not (args.text_encoder and args.tokenizer):
        parser.error('--prompt needs --text-encoder and --tokenizer')

    sparse = _sparse_request(args)
    text = _text_states(args)
    reader = convert.ReleaseReader(args.transformer)
    out = pipeline.run_clip(
        reader, args.slabs, text,
        pipeline.ClipRequest(aspect=args.aspect, seconds=args.seconds,
                             steps=args.steps, seed=args.seed,
                             short_edge=args.short_edge, sparse=sparse))
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
    # The rows alone do not say what shape they fold back into, so the
    # geometry the decoder needs is written next to them.
    meta = {'geometry': dataclasses.asdict(
        geometry.resolve_geometry(args.aspect, args.seconds,
                                  args.short_edge)),
            'aspect': args.aspect, 'seconds': args.seconds,
            'short_edge': args.short_edge, 'steps': args.steps,
            'seed': args.seed, 'prompt': args.prompt,
            'sparse_ratio': args.sparse_ratio,
            'step_seconds': list(out.step_seconds)}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
