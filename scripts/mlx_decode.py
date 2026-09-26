"""Decodes latent rows written by scripts/mlx_generate.py into an mp4.

`--video-backend mlx` runs the 2.4B ViT3D video decoder through
miowtion.mlx.video_vae, which decodes several tiles per forward; `torch`
runs the released decoder instead, on `--device` (`mps` uses the same GPU
MLX does, `cpu` is the fallback when a VAE op has no MPS kernel). The audio
VAE is small and stays on torch either way.

Examples:
    python scripts/mlx_decode.py --variant weights/h3 --latents runs/clip \
        --video-backend mlx --out runs/clip/clip.mp4
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np
import torch

from miowtion.h3 import geometry as h3_geometry
from miowtion.infer import decode
from miowtion.mlx import video_vae as mlx_video_vae
from miowtion.utils import progress

_DTYPES = {'bf16': torch.bfloat16, 'fp32': torch.float32}
_MLX_DTYPES = {'bf16': mx.bfloat16, 'fp32': mx.float32}


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
    parser.add_argument('--video-backend', default='torch',
                        choices=('torch', 'mlx'),
                        help='where the video decoder runs')
    parser.add_argument('--tile-batch', type=int,
                        default=mlx_video_vae.DEFAULT_TILE_BATCH,
                        help='tiles per forward (mlx backend)')
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
    use_mlx = args.video_backend == 'mlx'
    if use_mlx and layout != 'diffusers':
        parser.error('--video-backend mlx needs the diffusers-port vae/')
    device = torch.device(args.device)
    start = time.time()
    if layout == 'diffusers':
        decoder = decode.DiffusersDecoder(args.variant, device,
                                          _DTYPES[args.dtype],
                                          load_video=not use_mlx)
    else:
        decoder = decode.Decoder(args.variant, device, _DTYPES[args.dtype])
    progress.log(f'{layout} VAEs on {device} in {time.time() - start:.1f} s')

    start = time.time()
    if use_mlx:
        mlx_decoder = mlx_video_vae.VideoDecoder(
            os.path.join(args.variant, 'vae'), _MLX_DTYPES[args.dtype],
            args.tile_batch)
        progress.log(f'mlx video vae in {time.time() - start:.1f} s, '
                     f'tile batch {args.tile_batch}')
        start = time.time()
        frames = np.array(mlx_decoder.video(mx.array(video_rows.numpy()),
                                            geometry))
        progress.log(f'video {frames.shape} in {time.time() - start:.1f} s, '
                     f'peak {mx.get_peak_memory() / 2**30:.2f} GB')
    else:
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
                      'layout': layout, 'video_backend': args.video_backend,
                      'frames': list(frames.shape)}
    with open(os.path.join(args.latents, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
