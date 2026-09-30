"""Encodes prompts (and keyframes) into a training sample cache.

Manifest: .jsonl, one object per sample:
    {"id": "p0001", "task": "t2va", "prompt": "..."}
    {"id": "k0001", "task": "fl2va", "prompt": "...", "aspect": "16:9",
     "keyframes": [0, -1], "images": ["first.png", "last.png"]}

Example:
    python scripts/encode_samples.py --root /path/minimaxh3 \
        --manifest prompts.jsonl --out artifacts/samples/v1
"""

import argparse
import json

from miowtion.h3 import geometry as h3_geometry
from miowtion.train import data
from miowtion.train import encode
from miowtion.utils import progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='checkpoint root')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-test', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-memory', default=None,
                        help='per-device budget, e.g. "0=21GiB,1=21GiB,'
                        'cpu=60GiB" (layers beyond GPU budget run on CPU)')
    args = parser.parse_args()
    with open(args.manifest) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        data.validate_prompt(r['prompt'], r['task'])
        # A quoted latent_t survives encoding intact and then fails the
        # geometry check in generate.py, hours later, against a value that
        # prints identically. Reject it here rather than coerce it: a
        # manifest that quotes one field usually quotes others.
        if not isinstance(r.get('latent_t'), (int, type(None))):
            raise ValueError(
                f"{r['id']}: latent_t must be an int or absent, got "
                f"{r['latent_t']!r}")
    splits = data.split_ids([r['id'] for r in records], args.num_test,
                            args.seed)
    variant_dir = f'{args.root}/FL2VA'
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(','):
            key, value = item.split('=')
            max_memory[int(key) if key.isdigit() else key] = value
    with progress.Timer(f'load text encoder ({variant_dir})'):
        text_encoder = encode.TextEncoder(variant_dir, max_memory=max_memory)
    counter = progress.Progress('encode samples', len(records))
    conditions = None
    writer = data.SampleCacheWriter(args.out)
    for r in records:
        task = r['task']
        if task == 't2va':
            hidden, tags = text_encoder.encode_t2va(r['prompt'])
            writer.add(r['id'], task, splits[r['id']], hidden, tags,
                       latent_t=r.get('latent_t'))
        elif task == 'fl2va':
            from PIL import Image  # pylint: disable=import-outside-toplevel
            if conditions is None:
                conditions = encode.ConditionEncoder(
                    variant_dir, text_encoder.model.device)
            aspect_w, aspect_h = (int(v) for v in r['aspect'].split(':'))
            width, height = h3_geometry.resolve_canvas(aspect_w, aspect_h)
            frames = [encode.cover_crop(Image.open(p), width, height)
                      for p in r['images']]
            hidden, tags = text_encoder.encode_fl2va(r['prompt'], frames)
            import torch  # pylint: disable=import-outside-toplevel
            rows = torch.cat([conditions.encode_image(f) for f in frames])
            writer.add(r['id'], task, splits[r['id']], hidden, tags,
                       keyframes=r['keyframes'], aspect=r['aspect'],
                       latent_t=r.get('latent_t'), cond_video=rows)
        else:
            raise NotImplementedError(
                f'{task} encoding is not implemented yet (see '
                'docs/features/training.md)')
        counter.update(f'{r["id"]}: {task} text_len={hidden.shape[0]}')
    with progress.Timer(f'write sample cache {args.out}'):
        writer.finalize()


if __name__ == '__main__':
    main()
