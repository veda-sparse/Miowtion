"""The H3 video VAE decoder on MLX, with the release's tiling untouched.

The decoder is a 2.4B ViT3D: every latent voxel is one token, four learned
register tokens plus one all-zero token are appended at position 0, 36
transformer blocks run full self-attention over them, and a linear layer
expands each token back into a 4x16x16 pixel block. Ported from
`diffusers.AutoencoderKLMiniMaxH3` (Apache-2.0), whose geometry this module
reproduces exactly -- the tile layout, the overlap blending and the temporal
chunking are part of what the released weights were trained to decode, so
none of them is a tuning knob.

What the port is for is utilization: on the same clip it decodes in 105 s
where the released torch decoder on MPS takes 155 s, at a 5.8 GB peak.
Tiles of a clip are all the same shape and independent, so this module can
decode several per forward (`tile_batch`) without changing a single value
each token sees -- attention stays within a tile because the batch axis
never mixes. That turned out to be worth only a few percent (one tile is
already 1797 rows per GEMM), so it stays a knob and the default is small:
what actually decides the time here is the peak, because the machine this
targets has 18 GB of unified memory shared with the OS and starts paging
the 4.8 GB of trunk weights the moment the peak approaches it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Sequence

import mlx.core as mx

from miowtion.h3 import geometry as h3_geometry
from miowtion.mlx import block as mlx_block
from miowtion.mlx import convert
from miowtion.utils import progress

# Tile geometry of the release (`AutoencoderKLMiniMaxH3.__init__`): tiles are
# laid out in pixel space, at most 256 px per side, overlapping by at least
# 64 px. Tiling is on by default there, so these are the numbers the
# published samples were decoded with.
TILE_SIZE = 256
TILE_MIN_OVERLAP = 64

# Tiles decoded in one batch. One tile is already 1797 rows per GEMM,
# which reaches 4.56 TFLOP/s on an M3 Pro, so batching buys little:
# measured 1.91 s/tile at 1, 1.86 at 2, 1.84 at 4, 1.83 at 8, against a
# peak of 4.78 / 5.03 / 5.55 / 6.11 GB. Two is that curve's knee, and on
# 18 GB of unified memory the peak is what matters -- see the pitfall on
# memory pressure, where the same batch
# is 20x slower with the machine under pressure.
DEFAULT_TILE_BATCH = 2

# Release names of the tensors of one transformer block, relative to
# `decoder.transformer_blocks.{i}.`.
_BLOCK_LINEARS = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0',
                  'ff.net.0.proj', 'ff.net.2')


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    """Shape of the ViT decoder and of the chunking around it.

    Attributes:
        latent_channels: Channels of the latent the decoder consumes.
        out_channels: Pixel channels.
        num_layers: Transformer blocks.
        num_heads: Attention heads.
        head_dim: Channels per head.
        num_register_tokens: Learned tokens appended before the cls token.
        ffn_mult: SwiGLU inner width as a multiple of the model width.
        rope_theta: Base of the 3-axis rotary embedding.
        rope_dim_ratio: Fraction of a head's channels that is rotated.
        norm_eps: Epsilon of every norm in the decoder.
        spatial_ratio: Pixels per latent cell, per spatial axis.
        temporal_ratio: Pixel frames per latent frame.
        clip_length: Pixel frames the encoder consumed at a time.
        token_drop: Latent frames the encoder dropped per clip.
    """

    latent_channels: int = 24
    out_channels: int = 3
    num_layers: int = 36
    num_heads: int = 32
    head_dim: int = 64
    num_register_tokens: int = 4
    ffn_mult: int = 4
    rope_theta: float = 100.0
    rope_dim_ratio: float = 0.75
    norm_eps: float = 1e-5
    spatial_ratio: int = 16
    temporal_ratio: int = 4
    clip_length: int = 17
    token_drop: int = 3

    @property
    def dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def rope_dim(self) -> int:
        """Rotated channels per head; see MiniMaxH3VideoRotaryPosEmbed."""
        return int(self.head_dim * self.rope_dim_ratio)

    @property
    def suffix_tokens(self) -> int:
        """Register tokens plus the single all-zero cls token."""
        return self.num_register_tokens + 1

    @property
    def frame_pre_padding(self) -> int:
        """Implicit leading pad: clip_length is not a whole latent frame."""
        return (-self.clip_length) % self.temporal_ratio

    @property
    def tokens_chunk_size(self) -> int:
        return math.ceil(self.clip_length / self.temporal_ratio)

    @property
    def token_overlap(self) -> int:
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_overlap(self) -> int:
        return max(self.token_overlap * self.temporal_ratio -
                   self.frame_pre_padding, 0)

    @classmethod
    def from_pretrained(cls, directory: str) -> DecoderConfig:
        """Reads `config.json` of a diffusers-port `vae/` directory."""
        with open(os.path.join(directory, 'config.json')) as f:
            raw = json.load(f)
        return cls(
            latent_channels=raw['latent_channels'],
            out_channels=raw['out_channels'],
            num_layers=raw['decoder_num_layers'],
            num_heads=raw['decoder_num_attention_heads'],
            head_dim=raw['decoder_attention_head_dim'],
            num_register_tokens=raw['decoder_num_register_tokens'],
            ffn_mult=raw['decoder_ffn_mult'],
            rope_theta=raw['decoder_rope_theta'],
            rope_dim_ratio=raw['decoder_rope_dim_ratio'],
            norm_eps=raw['decoder_norm_eps'],
            spatial_ratio=math.prod(raw['spatial_downsample_factors']),
            temporal_ratio=math.prod(raw['temporal_downsample_factors']),
            clip_length=raw['clip_length'],
            token_drop=raw['token_drop'])


@dataclasses.dataclass(frozen=True)
class Linear:
    """An affine layer in the release layout.

    Attributes:
        weight: [out, in].
        bias: [out].
    """

    weight: mx.array
    bias: mx.array

    def __call__(self, x: mx.array) -> mx.array:
        """x: [..., in] -> [..., out] in x's dtype."""
        return mx.addmm(self.bias, x, self.weight.T)


@dataclasses.dataclass(frozen=True)
class BlockWeights:
    """One transformer block.

    The three attention projections are stored fused: they read the same
    rows and are the same shape, so one [3 * dim, dim] GEMM replaces three,
    which matters at the short sequence lengths a tile has.

    Attributes:
        norm1: [dim] RMSNorm weight before attention.
        qkv: Fused q/k/v projection, [3 * dim, dim].
        out: Attention output projection.
        scale1: [dim] per-channel gate on the attention residual.
        norm2: [dim] RMSNorm weight before the MLP.
        ff_in: SwiGLU projection, [2 * ffn_mult * dim, dim].
        ff_out: SwiGLU down projection.
        scale2: [dim] per-channel gate on the MLP residual.
    """

    norm1: mx.array
    qkv: Linear
    out: Linear
    scale1: mx.array
    norm2: mx.array
    ff_in: Linear
    ff_out: Linear
    scale2: mx.array


@dataclasses.dataclass(frozen=True)
class DecoderWeights:
    """Everything the decoder needs, including the 1x1x1 post-quant conv.

    Attributes:
        post_quant: The 1x1x1 conv, which is a per-channel affine map.
        proj_in: Latent channels -> model width.
        register_tokens: [R, dim].
        blocks: One entry per transformer block.
        norm_out: LayerNorm weight and bias, each [dim].
        proj_out: Model width -> one pixel patch.
    """

    post_quant: Linear
    proj_in: Linear
    register_tokens: mx.array
    blocks: tuple[BlockWeights, ...]
    norm_out: tuple[mx.array, mx.array]
    proj_out: Linear


def _cast(reader: convert.ShardedSafetensors, key: str,
          dtype: mx.Dtype) -> mx.array:
    """Reads one tensor and materializes it in `dtype`.

    The cast is evaluated here on purpose. The checkpoint is fp32, and a
    lazy cast keeps its fp32 source alive until whoever holds the result
    is evaluated -- with one eval per block that is the fp32 copy of the
    whole decoder, so the load peaks at 9.0 GB instead of the 4.8 GB the
    bf16 weights occupy. Evaluating per tensor bounds the overshoot at
    one tensor (128 MB).

    Args:
        reader: Open checkpoint.
        key: Tensor name.
        dtype: Dtype to keep the tensor in.

    Returns:
        The tensor, resident in `dtype`.
    """
    source = reader.read(key)
    value = source.astype(dtype)
    mx.eval(value)
    return value


def _linear(reader: convert.ShardedSafetensors, prefix: str,
            dtype: mx.Dtype) -> Linear:
    return Linear(_cast(reader, f'{prefix}.weight', dtype),
                  _cast(reader, f'{prefix}.bias', dtype))


def load_weights(directory: str, config: DecoderConfig,
                 dtype: mx.Dtype = mx.bfloat16) -> DecoderWeights:
    """Loads the decoder from a diffusers-port `vae/` directory.

    Args:
        directory: Holds `config.json` and the safetensors shards.
        config: Shape the checkpoint is expected to have.
        dtype: Compute dtype of every weight.

    Returns:
        The decoder weights, resident (4.8 GB in bf16).

    Raises:
        ValueError: When a block's fused QKV does not have the shape
            `config` describes, i.e. the checkpoint is not this decoder.
    """
    reader = convert.ShardedSafetensors(directory)
    try:
        bar = progress.Progress('video vae', config.num_layers, every=6)
        blocks = []
        for index in range(config.num_layers):
            prefix = f'decoder.transformer_blocks.{index}'
            parts = [_linear(reader, f'{prefix}.attn.to_{n}', dtype)
                     for n in 'qkv']
            qkv = Linear(mx.concatenate([p.weight for p in parts], axis=0),
                         mx.concatenate([p.bias for p in parts], axis=0))
            if qkv.weight.shape != (3 * config.dim, config.dim):
                raise ValueError(
                    f'block {index} has qkv {qkv.weight.shape}, expected '
                    f'{(3 * config.dim, config.dim)}')
            blocks.append(BlockWeights(
                norm1=_cast(reader, f'{prefix}.norm1.weight', mx.float32),
                qkv=qkv,
                out=_linear(reader, f'{prefix}.attn.to_out.0', dtype),
                scale1=_cast(reader, f'{prefix}.scale1', dtype),
                norm2=_cast(reader, f'{prefix}.norm2.weight', mx.float32),
                ff_in=_linear(reader, f'{prefix}.ff.net.0.proj', dtype),
                ff_out=_linear(reader, f'{prefix}.ff.net.2', dtype),
                scale2=_cast(reader, f'{prefix}.scale2', dtype)))
            mx.eval(blocks[-1])
            # The fused qkv is the one tensor built here rather than in
            # _cast, so its three bf16 sources are only freed now.
            mx.clear_cache()
            bar.update()
        post = Linear(
            _cast(reader, 'post_quant_conv.weight', dtype).reshape(
                config.latent_channels, config.latent_channels),
            _cast(reader, 'post_quant_conv.bias', dtype))
        weights = DecoderWeights(
            post_quant=post,
            proj_in=_linear(reader, 'decoder.proj_in', dtype),
            register_tokens=_cast(
                reader, 'decoder.register_tokens', dtype).reshape(
                    config.num_register_tokens, config.dim),
            blocks=tuple(blocks),
            norm_out=(_cast(reader, 'decoder.norm_out.weight', dtype),
                      _cast(reader, 'decoder.norm_out.bias', dtype)),
            proj_out=_linear(reader, 'decoder.proj_out', dtype))
        mx.eval(weights)
        mx.clear_cache()
        return weights
    finally:
        reader.close()


def rope_tables(frames: int, height: int, width: int, config: DecoderConfig,
                dtype: mx.Dtype = mx.bfloat16) -> tuple[mx.array, mx.array]:
    """Cos/sin of the 3-axis rotary embedding of one tile's token grid.

    Coordinates are the cell centres of each axis mapped onto [-1, 1), the
    three angles are concatenated and then duplicated (rotate-half), and the
    suffix tokens sit at position 0 on every axis.

    Args:
        frames: Latent frames of the tile.
        height: Latent rows.
        width: Latent columns.
        config: Decoder shape.
        dtype: Dtype of the returned tables.

    Returns:
        Two [frames * height * width + suffix, 1, rope_dim] arrays.
    """
    axes = 3
    step = 2.0 * axes / config.rope_dim
    inv_freq = 1.0 / config.rope_theta ** mx.arange(0.0, 1.0, step,
                                                    dtype=mx.float32)
    grids = [2.0 * (mx.arange(0.5, size, dtype=mx.float32) / size) - 1.0
             for size in (frames, height, width)]
    positions = mx.stack(
        [grids[0][:, None, None] + mx.zeros((frames, height, width)),
         grids[1][None, :, None] + mx.zeros((frames, height, width)),
         grids[2][None, None, :] + mx.zeros((frames, height, width))],
        axis=-1).reshape(-1, axes)
    positions = mx.concatenate(
        [positions, mx.zeros((config.suffix_tokens, axes))], axis=0)
    angles = (2.0 * math.pi * positions[:, :, None] *
              inv_freq[None, None, :]).reshape(positions.shape[0], -1)
    angles = mx.concatenate([angles, angles], axis=-1)[:, None, :]
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def _attention(x: mx.array, weights: BlockWeights, cos: mx.array,
               sin: mx.array, config: DecoderConfig) -> mx.array:
    """[B, N, dim] -> [B, N, dim]; q/k are RMS-normalized in fp32."""
    batch, tokens, _ = x.shape
    qkv = weights.qkv(x).reshape(batch, tokens, 3, config.num_heads,
                                 config.head_dim)
    query, key, value = (qkv[:, :, i] for i in range(3))
    # The reference normalizes q/k in fp32 whatever the compute dtype, with
    # no learned weight (elementwise_affine=False).
    query = mx.fast.rms_norm(query.astype(mx.float32), None,
                             config.norm_eps).astype(x.dtype)
    key = mx.fast.rms_norm(key.astype(mx.float32), None,
                           config.norm_eps).astype(x.dtype)
    query = mlx_block.apply_rope(query, cos, sin)
    key = mlx_block.apply_rope(key, cos, sin)
    out = mx.fast.scaled_dot_product_attention(
        query.transpose(0, 2, 1, 3), key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3), scale=config.head_dim ** -0.5)
    return weights.out(out.transpose(0, 2, 1, 3).reshape(batch, tokens, -1))


def block_forward(x: mx.array, weights: BlockWeights, cos: mx.array,
                  sin: mx.array, config: DecoderConfig) -> mx.array:
    """One transformer block.

    Args:
        x: [B, N, dim] bf16 tokens of B tiles.
        weights: The block's weights.
        cos: [N, 1, rope_dim] in x's dtype.
        sin: [N, 1, rope_dim].
        config: Decoder shape.

    Returns:
        [B, N, dim] in x's dtype.
    """
    h = mlx_block.rms_norm(x, weights.norm1, config.norm_eps)
    x = x + _attention(h, weights, cos, sin, config) * weights.scale1
    h = mlx_block.rms_norm(x, weights.norm2, config.norm_eps)
    value, gate = mx.split(weights.ff_in(h), 2, axis=-1)
    return x + weights.ff_out(mlx_block.swiglu(gate, value)) * weights.scale2


def _unpatchify(tokens: mx.array, frames: int, height: int, width: int,
                config: DecoderConfig) -> mx.array:
    """[B, N, C * pt * p * p] -> [B, C, frames * pt, height * p, width * p]."""
    patch, patch_t = config.spatial_ratio, config.temporal_ratio
    batch = tokens.shape[0]
    x = tokens.reshape(batch, frames, height, width, config.out_channels,
                       patch_t, patch, patch)
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(batch, config.out_channels, frames * patch_t,
                     height * patch, width * patch)


def decode_tiles(latents: mx.array, weights: DecoderWeights,
                 config: DecoderConfig) -> mx.array:
    """Runs the ViT over a batch of equally shaped latent tiles.

    Attention never crosses the batch axis, so a batch of B tiles is
    bitwise the same work as B separate forwards, only wider.

    Args:
        latents: [B, C, T, H, W] latent tiles in the compute dtype.
        weights: Decoder weights.
        config: Decoder shape.

    Returns:
        [B, 3, T * 4, H * 16, W * 16] pixels in the compute dtype.

    Raises:
        ValueError: When the latent does not have `config.latent_channels`.
    """
    batch, channels, frames, height, width = latents.shape
    if channels != config.latent_channels:
        raise ValueError(f'latent has {channels} channels, expected '
                         f'{config.latent_channels}')
    x = latents.transpose(0, 2, 3, 4, 1).reshape(batch, -1, channels)
    x = weights.proj_in(weights.post_quant(x))
    num_patches = x.shape[1]
    suffix = mx.concatenate(
        [weights.register_tokens,
         mx.zeros((1, config.dim), dtype=x.dtype)], axis=0)
    x = mx.concatenate(
        [x, mx.broadcast_to(suffix, (batch,) + suffix.shape)], axis=1)
    cos, sin = rope_tables(frames, height, width, config, x.dtype)
    for block in weights.blocks:
        x = block_forward(x, block, cos, sin, config)
        # Bounds the working set: without it every block's activations of
        # the whole tile batch stay alive until the last one is needed.
        mx.eval(x)
    x = mx.fast.layer_norm(x, *weights.norm_out, config.norm_eps)
    x = weights.proj_out(x)[:, :num_patches]
    return _unpatchify(x, frames, height, width, config)


def unpatchify(rows: mx.array, latent_t: int, latent_h: int,
               latent_w: int) -> mx.array:
    """h3.noise.unpatchify in MLX: [N, 4 * C] rows -> [1, C, T, H, W].

    Args:
        rows: [latent_t * latent_h * latent_w / 4, 4 * C].
        latent_t: Latent frames.
        latent_h: Latent rows (even).
        latent_w: Latent columns (even).

    Returns:
        The latent the VAE decoder consumes, in the rows' dtype.
    """
    channels = rows.shape[-1] // 4
    x = rows.reshape(1, latent_t, latent_h // 2, latent_w // 2, channels,
                     1, 2, 2)
    # 'nthwcrpq->nctrhpwq'
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, channels, latent_t, latent_h, latent_w)


def split_tiles(length: int, tile_size: int, min_overlap: int, ratio: int
                ) -> tuple[list[int], list[int], list[int]]:
    """Lays `tile_size`-wide tiles over `length` pixels.

    The smallest tile count whose union covers `length` with every overlap
    at least `min_overlap`; the slack is spread round-robin over the
    overlaps in whole `ratio` steps so that every boundary stays aligned to
    the latent grid. Ported from `AutoencoderKLMiniMaxH3._split_tiles`.

    Args:
        length: Pixels to cover.
        tile_size: Tile width.
        min_overlap: Smallest allowed overlap.
        ratio: Pixels per latent cell.

    Returns:
        Start offsets, tile lengths, and the overlaps between neighbours.
    """
    if tile_size >= length:
        return [0], [length], []
    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (num_tiles - 1)] += ratio
    starts = [0]
    for i in range(num_tiles - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * num_tiles, overlaps


def blend(a: mx.array, b: mx.array, extent: int, axis: int) -> mx.array:
    """Linear cross-fade of `a`'s tail into `b`'s head along `axis`.

    Ported from `AutoencoderKLMiniMaxH3._blend`; the weights are computed in
    `b`'s dtype there too, so a bf16 decode blends in bf16.

    Args:
        a: The earlier tile.
        b: The later tile, returned with its head replaced.
        extent: Overlap length, clipped to what both sides have.
        axis: Axis to blend along.

    Returns:
        `b` with its first `extent` entries faded in from `a`.
    """
    extent = min(a.shape[axis], b.shape[axis], extent)
    positions = mx.arange(extent, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[axis] = extent
    weight_b = (positions / extent).reshape(shape)
    weight_a = (1 - positions / extent).reshape(shape)
    head_a = mx.take(a, mx.arange(a.shape[axis] - extent, a.shape[axis]),
                     axis=axis)
    head_b = mx.take(b, mx.arange(extent), axis=axis)
    blended = head_a * weight_a + head_b * weight_b
    if extent == b.shape[axis]:
        return blended
    rest = mx.take(b, mx.arange(extent, b.shape[axis]), axis=axis)
    return mx.concatenate([blended, rest], axis=axis)


def _drop_tail(x: mx.array, count: int, axis: int) -> mx.array:
    return mx.take(x, mx.arange(x.shape[axis] - count), axis=axis)


def stitch_tiles(tiles: Sequence[Sequence[mx.array]],
                 row_overlaps: Sequence[int],
                 col_overlaps: Sequence[int]) -> mx.array:
    """Blends a grid of decoded tiles into one frame stack.

    Ported from `AutoencoderKLMiniMaxH3._stitch_tiles`: each tile fades into
    its upper and left neighbour and then gives up the overlap it shares
    with its lower and right one, so every pixel is written once.

    Args:
        tiles: `rows x cols` decoded tiles, each [..., H, W].
        row_overlaps: Pixel overlaps between consecutive rows.
        col_overlaps: Pixel overlaps between consecutive columns.

    Returns:
        The stitched array.
    """
    rows = []
    for i, row in enumerate(tiles):
        pieces = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(tiles[i - 1][j], tile, row_overlaps[i - 1], -2)
            if j > 0:
                tile = blend(row[j - 1], tile, col_overlaps[j - 1], -1)
            if i < len(tiles) - 1:
                tile = _drop_tail(tile, row_overlaps[i], -2)
            if j < len(row) - 1:
                tile = _drop_tail(tile, col_overlaps[j], -1)
            pieces.append(tile)
        rows.append(mx.concatenate(pieces, axis=-1))
    return mx.concatenate(rows, axis=-2)


def decode_clip(latent: mx.array, weights: DecoderWeights,
                config: DecoderConfig, tile_batch: int = DEFAULT_TILE_BATCH,
                bar: progress.Progress | None = None) -> mx.array:
    """Decodes one temporal chunk, spatially tiled as the release tiles it.

    Args:
        latent: [1, C, T, H, W] latent of the chunk.
        weights: Decoder weights.
        config: Decoder shape.
        tile_batch: Tiles decoded in one forward.
        bar: Progress bar advanced once per tile.

    Returns:
        [1, 3, T * 4, H * 16, W * 16] pixels.

    Raises:
        ValueError: When `tile_batch` is not positive.
    """
    if tile_batch < 1:
        raise ValueError(f'tile_batch {tile_batch} must be positive')
    ratio = config.spatial_ratio
    height, width = latent.shape[-2] * ratio, latent.shape[-1] * ratio
    y_starts, y_lengths, y_overlaps = split_tiles(
        height, TILE_SIZE, TILE_MIN_OVERLAP, ratio)
    x_starts, x_lengths, x_overlaps = split_tiles(
        width, TILE_SIZE, TILE_MIN_OVERLAP, ratio)

    cuts = [(y // ratio, y_len // ratio, x // ratio, x_len // ratio)
            for y, y_len in zip(y_starts, y_lengths)
            for x, x_len in zip(x_starts, x_lengths)]
    decoded: list[mx.array] = []
    for start in range(0, len(cuts), tile_batch):
        group = cuts[start:start + tile_batch]
        batch = mx.concatenate(
            [latent[:, :, :, y:y + h, x:x + w] for y, h, x, w in group],
            axis=0)
        pixels = decode_tiles(batch, weights, config)
        mx.eval(pixels)
        # Attention over a tile batch is the widest allocation in the
        # decode; holding its buffers in the cache costs more than
        # re-allocating them for the next batch does.
        mx.clear_cache()
        decoded.extend(pixels[i:i + 1] for i in range(pixels.shape[0]))
        if bar is not None:
            for _ in group:
                bar.update()
    grid = [decoded[i * len(x_starts):(i + 1) * len(x_starts)]
            for i in range(len(y_starts))]
    return stitch_tiles(grid, y_overlaps, x_overlaps)


def decode(latent: mx.array, weights: DecoderWeights, config: DecoderConfig,
           tile_batch: int = DEFAULT_TILE_BATCH) -> mx.array:
    """Decodes a latent video, mirroring the encoder's temporal chunking.

    `token_drop` removed the tail of every encoded chunk, so consecutive
    decoded chunks overlap by `frame_overlap` pixel frames and are
    cross-faded; latent frames are repeated when the length is not a whole
    number of chunks and the extra pixel frames are cut off again. Ported
    from `AutoencoderKLMiniMaxH3._decode`.

    Args:
        latent: [1, C, T, H, W] latent in the compute dtype.
        weights: Decoder weights.
        config: Decoder shape.
        tile_batch: Tiles decoded in one forward.

    Returns:
        [1, 3, frames, H * 16, W * 16] pixels in the compute dtype.
    """
    chunk_tokens = config.tokens_chunk_size
    drop = config.token_drop
    chunk_frames = chunk_tokens * config.temporal_ratio

    num_tokens = latent.shape[2] + drop
    pad_tokens = (-num_tokens) % chunk_tokens
    num_chunks = (num_tokens + pad_tokens) // chunk_tokens - int(drop > 0)
    if pad_tokens > 0:
        tail = mx.broadcast_to(
            latent[:, :, -1:],
            latent.shape[:2] + (pad_tokens,) + latent.shape[3:])
        latent = mx.concatenate([latent, tail], axis=2)

    tiles = _tiles_per_clip(latent, config)
    bar = progress.Progress('video vae tiles', num_chunks * tiles, every=4)
    chunks: list[mx.array] = []
    overlap = None
    for i in range(num_chunks):
        start = i * chunk_tokens
        clip = decode_clip(
            latent[:, :, start:start + chunk_tokens + config.token_overlap],
            weights, config, tile_batch, bar)
        for j in range(int(drop > 0) + 1):
            first = j * chunk_frames
            chunk = clip[:, :, first + config.frame_pre_padding:
                         first + chunk_frames]
            if j == 0:
                if overlap is not None:
                    chunk = blend(overlap, chunk, config.frame_overlap, -3)
                chunks.append(chunk)
            else:
                overlap = chunk
        mx.eval(chunks[-1])
    if overlap is not None:
        chunks.append(overlap)
    pixels = mx.concatenate(chunks, axis=2)

    # The repeated latent frames produced pixel frames nobody asked for. A
    # chunk's last latent frame only covers `clip_length % temporal_ratio`
    # pixel frames; the others cover `temporal_ratio`.
    if pad_tokens > 0:
        intra_tail = config.clip_length % config.temporal_ratio
        before_pad = latent.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and
            (before_pad + k) % chunk_tokens == 0 else config.temporal_ratio
            for k in range(pad_tokens))
        pixels = _drop_tail(pixels, pad_frames, 2)
    return pixels


class VideoDecoder:
    """Latent rows -> uint8 frames, the same interface as infer.decode.

    Holds the trunk resident (4.8 GB in bf16), so build it once per process
    and keep it only while decoding: on an 18 GB machine it does not
    coexist with a DiT generation.
    """

    def __init__(self, directory: str, dtype: mx.Dtype = mx.bfloat16,
                 tile_batch: int = DEFAULT_TILE_BATCH):
        """Loads the decoder of a diffusers-port `vae/` directory.

        Args:
            directory: Holds `config.json` and the safetensors shards.
            dtype: Compute dtype; bf16 is what the torch path uses.
            tile_batch: Tiles decoded in one forward.
        """
        self.config = DecoderConfig.from_pretrained(directory)
        self.dtype = dtype
        self.tile_batch = tile_batch
        with open(os.path.join(directory, 'config.json')) as f:
            stats = json.load(f)
        self._mean = mx.array(stats['latents_mean'], mx.float32)
        self._std = mx.array(stats['latents_std'], mx.float32)
        self.weights = load_weights(directory, self.config, dtype)

    def video(self, rows: mx.array,
              geometry: h3_geometry.Geometry) -> mx.array:
        """De-normalized decode of one clip's video rows.

        Args:
            rows: [N_video, 96] fp32 rows as the DiT wrote them.
            geometry: The clip's geometry. The decoder emits
                `4 * (T - 1) + 1` frames; a geometry that asked for fewer
                (an unaligned duration) keeps the leading ones.

        Returns:
            [frames, H * 16, W * 16, 3] uint8.
        """
        latent = unpatchify(rows.astype(mx.float32), geometry.latent_t,
                            geometry.latent_h, geometry.latent_w)
        shape = (1, -1, 1, 1, 1)
        z = latent * self._std.reshape(shape) + self._mean.reshape(shape)
        frame_count = geometry.frame_count
        pixels = decode(z.astype(self.dtype), self.weights, self.config,
                        self.tile_batch)
        pixels = pixels[0, :, :frame_count].transpose(1, 2, 3, 0)
        frames = mx.round((pixels.astype(mx.float32) + 1.0) * 127.5)
        return mx.clip(frames, 0.0, 255.0).astype(mx.uint8)


def _tiles_per_clip(latent: mx.array, config: DecoderConfig) -> int:
    """Tiles one temporal chunk is split into, for the progress bar."""
    ratio = config.spatial_ratio
    rows = split_tiles(latent.shape[-2] * ratio, TILE_SIZE,
                       TILE_MIN_OVERLAP, ratio)[0]
    cols = split_tiles(latent.shape[-1] * ratio, TILE_SIZE,
                       TILE_MIN_OVERLAP, ratio)[0]
    return len(rows) * len(cols)
