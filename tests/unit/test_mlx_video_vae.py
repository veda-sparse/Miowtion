"""The MLX video VAE decoder: tiling geometry and the batched forward.

The reference these mirror is `diffusers.AutoencoderKLMiniMaxH3`, which is
not installable next to the release's own diffusers pin, so the checks that
need it live in scripts/mlx_check_video_vae.py and are recorded in
docs/features/mlx_inference.md. What is checked here is everything that can
be stated without it: the tile layout's invariants, the blend and the stitch
as pure data transforms, and the property the batching rests on -- decoding
B tiles at once is bitwise the same as decoding them one at a time.
"""

import math

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core', reason='MLX requires Apple silicon')

from miowtion.mlx import video_vae

_CONFIG = video_vae.DecoderConfig(
    latent_channels=4, out_channels=3, num_layers=2, num_heads=2,
    head_dim=8, num_register_tokens=2, ffn_mult=2, spatial_ratio=4,
    temporal_ratio=2, clip_length=5, token_drop=1)


def _linear(out_dim: int, in_dim: int) -> video_vae.Linear:
    return video_vae.Linear(
        mx.random.normal((out_dim, in_dim)).astype(mx.bfloat16) * 0.1,
        mx.random.normal((out_dim,)).astype(mx.bfloat16) * 0.1)


def _weights(config: video_vae.DecoderConfig) -> video_vae.DecoderWeights:
    dim, ffn = config.dim, config.ffn_mult * config.dim
    blocks = tuple(video_vae.BlockWeights(
        norm1=mx.random.normal((dim,)),
        qkv=_linear(3 * dim, dim), out=_linear(dim, dim),
        scale1=mx.random.normal((dim,)).astype(mx.bfloat16),
        norm2=mx.random.normal((dim,)),
        ff_in=_linear(2 * ffn, dim), ff_out=_linear(dim, ffn),
        scale2=mx.random.normal((dim,)).astype(mx.bfloat16))
        for _ in range(config.num_layers))
    patch = (config.out_channels * config.temporal_ratio *
             config.spatial_ratio ** 2)
    return video_vae.DecoderWeights(
        post_quant=_linear(config.latent_channels, config.latent_channels),
        proj_in=_linear(dim, config.latent_channels),
        register_tokens=mx.random.normal(
            (config.num_register_tokens, dim)).astype(mx.bfloat16),
        blocks=blocks,
        norm_out=(mx.random.normal((dim,)).astype(mx.bfloat16),
                  mx.random.normal((dim,)).astype(mx.bfloat16)),
        proj_out=_linear(patch, dim))


def test_split_tiles_covers_the_length_with_aligned_boundaries():
    # Pixel sizes are whole latent cells; the slack is only spread in
    # `ratio` steps, so a length that is not a multiple of it would leave
    # the last tile hanging over the edge.
    for length in (768, 1344, 512, 272, 400, 1024):
        starts, lengths, overlaps = video_vae.split_tiles(
            length, video_vae.TILE_SIZE, video_vae.TILE_MIN_OVERLAP, 16)
        assert starts[0] == 0
        assert starts[-1] + lengths[-1] == length
        assert len(overlaps) == len(starts) - 1
        for i, overlap in enumerate(overlaps):
            # Every boundary has to land on a latent cell, or the tile
            # cannot be cut out of the latent at all.
            assert overlap % 16 == 0
            assert overlap >= video_vae.TILE_MIN_OVERLAP
            assert starts[i] + lengths[i] - overlap == starts[i + 1]


def test_split_tiles_returns_one_tile_when_it_fits():
    assert video_vae.split_tiles(200, 256, 64, 16) == ([0], [200], [])


def test_blend_is_a_linear_cross_fade():
    a = mx.random.normal((1, 3, 2, 8, 10))
    b = mx.random.normal((1, 3, 2, 8, 10))
    out = video_vae.blend(a, b, 4, -1)
    assert out.shape == b.shape
    # The fade starts at a's last column and ends just short of b's.
    assert mx.array_equal(out[..., 0], a[..., -4])
    assert mx.array_equal(out[..., 4:], b[..., 4:])
    weights = mx.arange(4, dtype=mx.float32) / 4
    expected = a[..., -4:] * (1 - weights) + b[..., :4] * weights
    assert mx.array_equal(out[..., :4], expected)


def test_blend_of_a_shorter_tail_keeps_the_whole_tile():
    a = mx.random.normal((1, 1, 6))
    b = mx.random.normal((1, 1, 6))
    assert video_vae.blend(a, b, 6, -1).shape == (1, 1, 6)


def test_stitch_tiles_writes_every_pixel_once():
    # Constant tiles: any convex blend of equal values is that value, so a
    # stitch that double-counts or drops a row shows up immediately.
    tiles = [[mx.full((1, 1, 2, 8, 8), 3.0) for _ in range(3)]
             for _ in range(2)]
    out = video_vae.stitch_tiles(tiles, [2], [2, 2])
    assert out.shape == (1, 1, 2, 8 * 2 - 2, 8 * 3 - 4)
    assert mx.array_equal(out, mx.full(out.shape, 3.0))


def test_rope_tables_match_the_reference_formula():
    config = video_vae.DecoderConfig(head_dim=8, rope_dim_ratio=0.75)
    frames, height, width = 2, 3, 4
    cos, sin = video_vae.rope_tables(frames, height, width, config,
                                     mx.float32)
    tokens = frames * height * width + config.suffix_tokens
    assert cos.shape == (tokens, 1, config.rope_dim)
    # Rotate-half duplicates the angle block; the suffix tokens sit at
    # position 0, i.e. angle 0.
    assert mx.array_equal(cos[:, :, :config.rope_dim // 2],
                          cos[:, :, config.rope_dim // 2:])
    assert mx.array_equal(sin[-config.suffix_tokens:],
                          mx.zeros((config.suffix_tokens, 1,
                                    config.rope_dim)))
    inv = 1.0 / config.rope_theta ** np.arange(0.0, 1.0,
                                               6 / config.rope_dim)
    grids = [2.0 * (np.arange(0.5, size) / size) - 1.0
             for size in (frames, height, width)]
    first = np.array([grids[0][0], grids[1][0], grids[2][0]])
    expected = np.cos(2 * math.pi * np.outer(first, inv).reshape(-1))
    got = np.array(cos[0, 0, :config.rope_dim // 2])
    assert np.allclose(got, expected, atol=1e-6)


def test_decoding_a_batch_of_tiles_equals_decoding_them_one_by_one():
    mx.random.seed(0)
    weights = _weights(_CONFIG)
    tiles = [mx.random.normal((1, _CONFIG.latent_channels, 2, 3, 3)
                              ).astype(mx.bfloat16) for _ in range(3)]
    batched = video_vae.decode_tiles(mx.concatenate(tiles, axis=0), weights,
                                     _CONFIG)
    for index, tile in enumerate(tiles):
        alone = video_vae.decode_tiles(tile, weights, _CONFIG)
        assert mx.array_equal(batched[index:index + 1], alone)


def test_decode_tiles_rejects_the_wrong_latent_channels():
    weights = _weights(_CONFIG)
    with pytest.raises(ValueError, match='channels'):
        video_vae.decode_tiles(mx.zeros((1, 5, 2, 3, 3), mx.bfloat16),
                               weights, _CONFIG)


def test_decode_rejects_an_empty_tile_batch():
    weights = _weights(_CONFIG)
    with pytest.raises(ValueError, match='tile_batch'):
        video_vae.decode_clip(mx.zeros((1, 4, 3, 3, 3), mx.bfloat16),
                              weights, _CONFIG, tile_batch=0)


def test_decode_emits_the_frames_the_chunking_implies():
    mx.random.seed(0)
    weights = _weights(_CONFIG)
    latent = mx.random.normal((1, _CONFIG.latent_channels, 5, 3, 3)
                              ).astype(mx.bfloat16)
    pixels = video_vae.decode(latent, weights, _CONFIG, tile_batch=2)
    # One latent frame covers temporal_ratio pixel frames except the last
    # of a chunk, which covers clip_length % temporal_ratio.
    assert pixels.shape[0] == 1
    assert pixels.shape[1] == _CONFIG.out_channels
    assert pixels.shape[-2:] == (3 * _CONFIG.spatial_ratio,
                                 3 * _CONFIG.spatial_ratio)
    assert pixels.shape[2] == 8


def test_unpatchify_is_bitwise_equal_to_the_torch_one():
    import torch  # pylint: disable=import-outside-toplevel

    from miowtion.h3 import noise  # pylint: disable=import-outside-toplevel

    latent_t, latent_h, latent_w = 3, 6, 8
    rows = torch.randn(latent_t * latent_h * latent_w // 4, 96)
    want = noise.unpatchify(rows, latent_t, latent_h, latent_w)
    got = video_vae.unpatchify(mx.array(rows.numpy()), latent_t, latent_h,
                               latent_w)
    assert np.array_equal(np.array(got), want.numpy())


def test_derived_chunking_matches_the_release_geometry():
    config = video_vae.DecoderConfig()
    assert config.frame_pre_padding == 3
    assert config.tokens_chunk_size == 5
    assert config.token_overlap == 2
    assert config.frame_overlap == 5
    assert config.rope_dim == 48
    assert config.suffix_tokens == 5
