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
from pathlib import Path

import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.train import data
from miowtion.train import encode
from miowtion.utils import progress


def _variant_for_records(records: list[dict]) -> str:
    """Selects the H3 variant whose processor/encoders own the tasks."""
    variants = {'Ref2VA' if r.get('task') == 'ref2va' else 'FL2VA'
                for r in records}
    if len(variants) != 1:
        raise ValueError('one cache cannot mix Ref2VA with FL2VA/T2VA; '
                         'encode them into separate cache directories')
    return variants.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='checkpoint root')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-test', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-root', default=None,
                        help='base for relative media paths; defaults to the '
                        'parent of the manifest directory')
    parser.add_argument('--ref2va-latent-t', type=int, default=37,
                        help='target latent duration recorded for Ref2VA '
                        'prompts (default: 37, about 5.17 seconds)')
    parser.add_argument('--max-memory', default=None,
                        help='per-device budget, e.g. "0=21GiB,1=21GiB,'
                        'cpu=60GiB" (layers beyond GPU budget run on CPU)')
    args = parser.parse_args()
    with open(args.manifest) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        r['prompt'] = (r.get('prompt') or r.get('h3_prompt') or
                       r.get('instruction'))
        if not r['prompt']:
            raise ValueError(f'{r.get("id")}: missing prompt')
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
    variant_dir = f'{args.root}/{_variant_for_records(records)}'
    dataset_root = Path(args.dataset_root) if args.dataset_root else Path(
        args.manifest).resolve().parent.parent
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
            rows = torch.cat([conditions.encode_image(f) for f in frames])
            writer.add(r['id'], task, splits[r['id']], hidden, tags,
                       keyframes=r['keyframes'], aspect=r['aspect'],
                       latent_t=r.get('latent_t'), cond_video=rows)
        elif task == 'ref2va':
            from PIL import Image  # pylint: disable=import-outside-toplevel
            if conditions is None:
                conditions = encode.ConditionEncoder(
                    variant_dir, text_encoder.model.device)
            images = []
            qwen_videos = []
            video_timestamps = []
            labels = []
            ref_blocks = []
            video_rows = []
            audio_rows = []
            modality_counts = {'image': 0, 'video': 0, 'audio': 0}
            for ref in r.get('references', []):
                modality = ref['modality']
                modality_counts[modality] += 1
                ordinal = modality_counts[modality]
                expected_kind = {'image': 'Picture', 'video': 'Video',
                                 'audio': 'Audio'}[modality]
                expected_label = f'<{expected_kind} {ordinal}>'
                if ref.get('label') != expected_label:
                    raise ValueError(f'{r["id"]}: expected reference label '
                                     f'{expected_label}, got {ref.get("label")}')
                labels.append((modality, ordinal))
                path = Path(ref['path'])
                path = path if path.is_absolute() else dataset_root / path
                if modality == 'image':
                    with Image.open(path) as source:
                        image = source.convert('RGB')
                    size = encode.reference_image_size(*image.size)
                    if image.size != size:
                        image = image.resize(size, Image.Resampling.LANCZOS)
                    images.append(image)
                    rows = conditions.encode_image(image)
                    video_rows.append(rows)
                    ref_blocks.append({
                        'kind': 'image', 'latent_h': image.height // 16,
                        'latent_w': image.width // 16,
                    })
                elif modality == 'video':
                    frames = encode.load_reference_video(str(path))
                    sampled, timestamps = encode.sample_qwen_video(frames)
                    qwen_videos.append(sampled)
                    video_timestamps.append(timestamps)
                    rows, shape = conditions.encode_reference_video(frames)
                    video_rows.append(rows)
                    ref_blocks.append({
                        'kind': 'video', 'audio_t': 0,
                        'latent_t': shape[0], 'latent_h': shape[1],
                        'latent_w': shape[2],
                    })
                elif modality == 'audio':
                    waveform = encode.load_reference_audio(str(path))
                    rows, audio_t = conditions.encode_audio(waveform)
                    audio_rows.append(rows)
                    ref_blocks.append({'kind': 'audio',
                                       'audio_t': audio_t})
                else:
                    raise ValueError(f'{r["id"]}: unsupported reference '
                                     f'modality {modality!r}')
            hidden, tags = text_encoder.encode_ref2va(
                r['prompt'], images, qwen_videos, labels,
                video_timestamps)
            writer.add(
                r['id'], task, splits[r['id']], hidden, tags,
                references=ref_blocks,
                latent_t=r.get('latent_t', args.ref2va_latent_t),
                cond_video=(torch.cat(video_rows) if video_rows else None),
                cond_audio=(torch.cat(audio_rows) if audio_rows else None))
        else:
            raise NotImplementedError(
                f'{task} encoding is not implemented yet (see '
                'docs/features/r2va.md)')
        counter.update(f'{r["id"]}: {task} text_len={hidden.shape[0]}')
    with progress.Timer(f'write sample cache {args.out}'):
        writer.finalize()


if __name__ == '__main__':
    main()
