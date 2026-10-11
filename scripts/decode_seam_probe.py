"""Is the brightness flash made by the DiT or by the VAE's clip seams?

The flash sits at a fixed grid: every sparse clip flashes at frames 51,
119 and 187, which is one event every 17 latent frames, and the video
VAE's config carries `vae_clip_length: 17`. That is suggestive and not
proof -- plenty of things could have that period.

The test that settles it costs no denoising at all. generate.py saves
the latents of every arm, so the same latents can be decoded twice with
nothing changed but the decoder's temporal clip length. If the flash
moves with that setting, it is made at the seam between decode clips
and the DiT only supplies latents inconsistent enough to show it; if it
stays put, the decoder is innocent and the flash is in the latents.

Example:
    python scripts/decode_seam_probe.py \\
        --latents artifacts/flash20/main/<clip>/veda_t10_latents.pt \\
        --root weights/MiniMax-H3 --variant FL2VA \\
        --clip-lengths 17 33 1000 --out-dir artifacts/seam_probe
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.h3 import geometry as h3_geometry             # noqa: E402  # noqa: F401
from miowtion.infer import decode as infer_decode           # noqa: E402
from miowtion.utils import progress                         # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latents', required=True, nargs='+',
                        help='<mode>_latents.pt files written by generate.py')
    parser.add_argument('--root', required=True)
    parser.add_argument('--variant', default='FL2VA')
    parser.add_argument('--clip-lengths', type=int, nargs='+',
                        default=[17, 33, 1000],
                        help='decoder temporal clip lengths to try; one '
                        'larger than the clip disables chunking')
    parser.add_argument('--geometry', required=True,
                        help="e.g. '16:9@37'; the saved latents hold the "
                        'rows but not the geometry that shaped them')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    import json                                            # noqa: PLC0415
    with open(os.path.join(args.root, args.variant, 'video_vae',
                           'config.json')) as handle:
        config = json.load(handle)
    kwargs = {key[len('vae_'):]: int(value) for key, value in config.items()
              if key.startswith('vae_') and isinstance(value, int)}
    aspect, latent_t = args.geometry.split('@')
    geometry = h3_geometry.geometry_from_latent_t(aspect, int(latent_t))
    decoder = infer_decode.Decoder(os.path.join(args.root, args.variant),
                                   device)
    vae = decoder.video_vae
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.latents:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        rows = saved['video_rows']
        stem = os.path.basename(path).replace('_latents.pt', '')
        for length in args.clip_lengths:
            # setup_forward recomputes every constant from its kwargs
            # and defaults the ones it is not given, so passing only
            # clip_length would also silently turn tiling off and reset
            # the parallel schema -- the first attempt did exactly that
            # and produced a seam at every 17 frames that the original
            # decode does not have. Every key the release package sets
            # from config.json is therefore passed through, with only
            # clip_length changed.
            vae.setup_forward(clip_length=length, **{
                k: v for k, v in kwargs.items() if k != 'clip_length'})
            progress.log(f'{stem}: clip_length {length} -> '
                         f'chunk {vae.tokens_chunk_size} tokens, overlap '
                         f'{vae.token_overlap}')
            frames = decoder.video(rows.to(torch.float32), geometry)
            out = os.path.join(args.out_dir, f'{stem}_clip{length}.mp4')
            infer_decode.write_mp4(out, frames, [], decoder.sample_rate)
            progress.log(f'  wrote {out}: {len(frames)} frames')


if __name__ == '__main__':
    main()
