"""Encodes a prompt with the H3 text tower on MLX.

Writes the [L, 5120] bf16 hidden states the DiT conditions on. Reads the
layers from slabs when `--slabs` is given (the fast path, see
scripts/mlx_convert.py --text-encoder) and straight from the released
checkpoint otherwise.

Examples:
    python scripts/mlx_encode_text.py --text-encoder weights/h3/text_encoder \
        --tokenizer weights/h3/tokenizer --slabs artifacts/mlx/text_slabs \
        --prompt 'a cat on a skateboard' --out artifacts/mlx/prompt.npy
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from miowtion.mlx import convert
from miowtion.mlx import slab as mlx_slab
from miowtion.mlx import text_encoder as mlx_text
from miowtion.train import encode as train_encode
from miowtion.utils import progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-encoder', required=True,
                        help='released <variant>/text_encoder directory')
    parser.add_argument('--tokenizer', required=True,
                        help='released <variant>/tokenizer directory')
    parser.add_argument('--slabs', help='text encoder slab directory')
    parser.add_argument('--prompt', help='prompt text')
    parser.add_argument('--prompt-file', help='file holding the prompt')
    parser.add_argument('--layers', type=int,
                        default=train_encode.TEXT_LAYERS,
                        help='layers to run (H3 stops before the rest)')
    parser.add_argument('--out', help='write the hidden states as .npy')
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.prompt_file):
        parser.error('pass exactly one of --prompt, --prompt-file')

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
    ids = mlx_text.token_ids(args.tokenizer, prompt)
    config = mlx_text.TowerConfig.from_pretrained(args.text_encoder)
    progress.log(f'{len(ids)} tokens, {args.layers} layers, '
                 f'{config.layer_bytes / 2**30:.2f} GB per layer')

    reader = convert.ShardedSafetensors(args.text_encoder)
    start = time.time()
    embed = mlx_text.embed_tokens(reader)
    mx.eval(embed)
    progress.log(f'embedding {embed.nbytes / 2**30:.2f} GB in '
                 f'{time.time() - start:.1f} s')

    slabs = None
    if args.slabs:
        paths = [mlx_slab.slab_path(args.slabs, i) for i in range(args.layers)]
        slabs = mlx_slab.SlabReader(paths, slots=2)
        layers = mlx_text.slab_layers(slabs, range(args.layers))
    else:
        layers = ((i, mlx_text.LayerWeights.from_tensors(
            mlx_text.layer_tensors(reader, i))) for i in range(args.layers))

    start = time.time()
    hidden = mlx_text.encode(embed, layers, ids, config, total=args.layers)
    mx.eval(hidden)
    progress.log(f'hidden {tuple(hidden.shape)} in {time.time() - start:.1f} s'
                 f', peak {mx.get_peak_memory() / 2**30:.2f} GB')
    if slabs is not None:
        slabs.close()
    reader.close()
    if args.out:
        np.save(args.out, np.array(hidden.astype(mx.float32)))
        progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
