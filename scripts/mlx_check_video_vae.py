"""Compares miowtion.mlx.video_vae with the diffusers reference.

The reference is `diffusers.AutoencoderKLMiniMaxH3`, which needs a diffusers
version that cannot live in the same environment as the release's own pin
(see infer.decode.DiffusersDecoder), so this check is a script rather than a
unit test: run it with that diffusers on the path and paste the numbers into
docs/features/mlx_inference.md.

What it compares is a small randomly initialized decoder, both sides in
fp32, so the only differences left are accumulation order. The tile layout,
the overlap blend and the stitch are pure data transforms and are required
to be bitwise equal (AGENTS 1.5).

Examples:
    PYTHONPATH=.:<diffusers> python scripts/mlx_check_video_vae.py
"""

import argparse

import mlx.core as mx
import numpy as np
import torch

from miowtion.mlx import video_vae
from miowtion.utils import progress

# Small enough to run on CPU in a minute, wide enough that every op is
# exercised (two heads, two blocks, two register tokens).
_LATENT_CHANNELS = 6
_LAYERS = 2
_HEADS = 2
_HEAD_DIM = 16
_REGISTER_TOKENS = 2
_FFN_MULT = 2
# Tiling of the check: the released 256/64 would need a 256 px canvas per
# tile, which is 30x more tokens than the geometry needs to be exercised.
_TILE_SIZE = 64
_TILE_OVERLAP = 16
# Std of the random weights. Large enough that the residual gates (which
# the release initializes to zero) actually mix the blocks in.
_WEIGHT_STD = 0.2


def _rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    got, want = got.astype(np.float64), want.astype(np.float64)
    return float(np.sqrt(((got - want) ** 2).sum() / (want ** 2).sum()))


def _reference(seed: int):
    """A small randomly initialized diffusers VAE, and its config."""
    import diffusers  # pylint: disable=import-outside-toplevel

    torch.manual_seed(seed)
    vae = diffusers.AutoencoderKLMiniMaxH3(
        latent_channels=_LATENT_CHANNELS, decoder_num_layers=_LAYERS,
        decoder_num_attention_heads=_HEADS,
        decoder_attention_head_dim=_HEAD_DIM,
        decoder_num_register_tokens=_REGISTER_TOKENS,
        decoder_ffn_mult=_FFN_MULT, block_out_channels=(16,) * 6,
        layers_per_block=1, norm_num_groups=8).eval()
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.normal_(0, _WEIGHT_STD)
    vae.tile_sample_min_height = _TILE_SIZE
    vae.tile_sample_min_width = _TILE_SIZE
    vae.tile_sample_min_overlap_height = _TILE_OVERLAP
    vae.tile_sample_min_overlap_width = _TILE_OVERLAP
    config = video_vae.DecoderConfig(
        latent_channels=_LATENT_CHANNELS, num_layers=_LAYERS,
        num_heads=_HEADS, head_dim=_HEAD_DIM,
        num_register_tokens=_REGISTER_TOKENS, ffn_mult=_FFN_MULT,
        spatial_ratio=vae.spatial_compression_ratio,
        temporal_ratio=vae.temporal_compression_ratio,
        clip_length=vae.config.clip_length,
        token_drop=vae.config.token_drop)
    return vae, config


def _mirror(state, config: video_vae.DecoderConfig
            ) -> video_vae.DecoderWeights:
    """The reference's parameters as MLX decoder weights, in fp32."""
    def arr(name: str) -> mx.array:
        return mx.array(state[name].detach().float().numpy())

    def linear(prefix: str) -> video_vae.Linear:
        return video_vae.Linear(arr(f'{prefix}.weight'), arr(f'{prefix}.bias'))

    blocks = []
    for index in range(config.num_layers):
        prefix = f'decoder.transformer_blocks.{index}'
        blocks.append(video_vae.BlockWeights(
            norm1=arr(f'{prefix}.norm1.weight'),
            qkv=video_vae.Linear(
                mx.concatenate([arr(f'{prefix}.attn.to_{n}.weight')
                                for n in 'qkv'], axis=0),
                mx.concatenate([arr(f'{prefix}.attn.to_{n}.bias')
                                for n in 'qkv'], axis=0)),
            out=linear(f'{prefix}.attn.to_out.0'),
            scale1=arr(f'{prefix}.scale1'),
            norm2=arr(f'{prefix}.norm2.weight'),
            ff_in=linear(f'{prefix}.ff.net.0.proj'),
            ff_out=linear(f'{prefix}.ff.net.2'),
            scale2=arr(f'{prefix}.scale2')))
    channels = config.latent_channels
    return video_vae.DecoderWeights(
        post_quant=video_vae.Linear(
            arr('post_quant_conv.weight').reshape(channels, channels),
            arr('post_quant_conv.bias')),
        proj_in=linear('decoder.proj_in'),
        register_tokens=arr('decoder.register_tokens').reshape(
            config.num_register_tokens, config.dim),
        blocks=tuple(blocks),
        norm_out=(arr('decoder.norm_out.weight'),
                  arr('decoder.norm_out.bias')),
        proj_out=linear('decoder.proj_out'))


def _check_geometry(vae) -> None:
    """The pure data transforms, which have to be bitwise equal."""
    ratio = vae.spatial_compression_ratio
    for length in (768, 1344, 512, 272, 1024):
        got = video_vae.split_tiles(length, _TILE_SIZE, _TILE_OVERLAP, ratio)
        want = vae._split_tiles(length, _TILE_SIZE, _TILE_OVERLAP)  # pylint: disable=protected-access
        if [list(part) for part in got] != [list(p) for p in want]:
            raise AssertionError(f'split_tiles({length}): {got} != {want}')
    progress.log('split_tiles: identical')

    left = torch.randn(1, 3, 4, 20, 30)
    right = torch.randn(1, 3, 4, 20, 30)
    for axis in (-1, -2, -3):
        want = vae._blend(left, right, 7, axis).numpy()  # pylint: disable=protected-access
        got = np.array(video_vae.blend(mx.array(left.numpy()),
                                       mx.array(right.numpy()), 7, axis))
        if not np.array_equal(got, want):
            raise AssertionError(f'blend along {axis} differs')
    progress.log('blend: bitwise equal')

    tiles = [[torch.randn(1, 3, 2, 16, 16) for _ in range(3)]
             for _ in range(2)]
    want = vae._stitch_tiles(tiles, [4], [4, 4]).numpy()  # pylint: disable=protected-access
    got = np.array(video_vae.stitch_tiles(
        [[mx.array(tile.numpy()) for tile in row] for row in tiles],
        [4], [4, 4]))
    if not np.array_equal(got, want):
        raise AssertionError('stitch_tiles differs')
    progress.log('stitch_tiles: bitwise equal')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile-batch', type=int, default=3,
                        help='tiles per forward; the result must not depend '
                             'on it')
    args = parser.parse_args()

    vae, config = _reference(args.seed)
    weights = _mirror(vae.state_dict(), config)
    # The module's tiling is the release's; the check shrinks it so that a
    # few latent frames already produce a grid of tiles.
    video_vae.TILE_SIZE = _TILE_SIZE
    video_vae.TILE_MIN_OVERLAP = _TILE_OVERLAP
    _check_geometry(vae)

    torch.manual_seed(args.seed)
    for shape in ((1, _LATENT_CHANNELS, 7, 6, 10),
                  (1, _LATENT_CHANNELS, 9, 8, 8),
                  (1, _LATENT_CHANNELS, 12, 4, 4)):
        latent = torch.randn(*shape)
        with torch.no_grad():
            want = vae._decode(latent).float().numpy()  # pylint: disable=protected-access
        got = video_vae.decode(mx.array(latent.numpy()), weights, config,
                               args.tile_batch)
        mx.eval(got)
        got = np.array(got.astype(mx.float32))
        if got.shape != want.shape:
            raise AssertionError(f'{shape}: {got.shape} != {want.shape}')
        progress.log(f'{shape} -> {got.shape}: max abs '
                     f'{np.abs(got - want).max():.3e}, rel-L2 '
                     f'{_rel_l2(got, want):.3e}')


if __name__ == '__main__':
    main()
