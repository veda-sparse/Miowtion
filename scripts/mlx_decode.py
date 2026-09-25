"""Decodes latent rows written by scripts/mlx_generate.py into an mp4.

The VAEs stay on torch: they are released as torch packages and the video
decoder is a 2.4B ViT3D, so there is nothing to gain from porting them
before the DiT itself is settled. On Apple silicon `--device mps` runs
them on the same GPU MLX uses; `--device cpu` is the fallback when a VAE
op has no MPS kernel.

Examples:
    python scripts/mlx_decode.py --variant weights/h3 --latents runs/clip \
        --device mps --out runs/clip/clip.mp4
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.infer import decode
from miowtion.utils import progress

_DTYPES = {'bf16': torch.bfloat16, 'fp32': torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True,
                        help='release variant directory (holds video_vae/)')
    parser.add_argument('--latents', required=True,
                        help='directory with video.npy, audio.npy, meta.json')
    parser.add_argument('--device', default='mps',
                        help='torch device for both VAEs')
    parser.add_argument('--dtype', default='bf16', choices=sorted(_DTYPES),
                        help='video VAE weight dtype (audio stays fp32)')
    parser.add_argument('--layout', default='auto',
                        choices=('auto', 'diffusers', 'package'),
                        help='which release layout the VAEs come in')
    parser.add_argument('--out', required=True, help='output .mp4')
    args = parser.parse_args()

    with open(os.path.join(args.latents, 'meta.json')) as f:
        meta = json.load(f)
    geometry = h3_geometry.Geometry(**meta['geometry'])
    video_rows = torch.from_numpy(
        np.load(os.path.join(args.latents, 'video.npy')))
    audio_rows = torch.from_numpy(
        np.load(os.path.join(args.latents, 'audio.npy')))
    progress.log(f'{geometry.name}: video {tuple(video_rows.shape)}, '
                 f'audio {tuple(audio_rows.shape)}')

    layout = args.layout
    if layout == 'auto':
        layout = ('diffusers' if os.path.isdir(os.path.join(args.variant,
                                                            'vae'))
                  else 'package')
    decoder_class = (decode.DiffusersDecoder if layout == 'diffusers'
                     else decode.Decoder)
    device = torch.device(args.device)
    start = time.time()
    decoder = decoder_class(args.variant, device, _DTYPES[args.dtype])
    progress.log(f'{layout} VAEs on {device} in {time.time() - start:.1f} s')

    start = time.time()
    frames = decoder.video(video_rows, geometry)
    progress.log(f'video {frames.shape} in {time.time() - start:.1f} s')
    start = time.time()
    waveform = decoder.audio(audio_rows)
    progress.log(f'audio {tuple(waveform.shape)} in '
                 f'{time.time() - start:.1f} s')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    decode.write_mp4(args.out, frames, [(waveform, 'generated')],
                     decoder.sample_rate)
    progress.log(f'wrote {args.out}')
    meta['decode'] = {'device': args.device, 'dtype': args.dtype,
                      'layout': layout,
                      'frames': list(frames.shape)}
    with open(os.path.join(args.latents, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
