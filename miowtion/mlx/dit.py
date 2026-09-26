"""The H3 DiT layers outside the trunk, in MLX.

`miowtion.mlx.block` runs one trunk block; a generation also needs the
embedding, time, text-refiner and output layers. They are small (about
1.7 GB in bf16, against 38 GB of trunk), so they stay resident while the
trunk streams block by block.

The op chain follows miowtion.h3.model exactly, including its
mixed-precision islands: the patch projections, the time embedder and the
output heads run in fp32, everything else in bf16, and AdaLN modulation
rounds where torch rounds.

Two transcendental tables — the RoPE cos/sin and the timestep sinusoid —
are built on the host in numpy rather than with MLX ops. Both are tiny
(seq_len x 96 and M x 256) and both are pure elementwise math, so computing
them on the host costs nothing and keeps them as close to the torch
reference as the platform allows: the RoPE tables come out bitwise equal,
and the timestep sinusoid is within one ulp (numpy's and torch's fp32 exp
disagree on the last bit of 14 of the 128 frequencies). Everything
downstream of them is a GEMM, whose accumulation order differs from torch
anyway.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.mlx import block as mlx_block

_BF16 = mx.bfloat16
_FP32 = mx.float32

# Names of the non-trunk tensors, as produced by
# miowtion.mlx.convert.non_trunk_tensors.
NON_TRUNK_NAMES = (
    'video_patch_proj.weight', 'video_patch_proj.bias',
    'audio_patch_proj.weight', 'audio_patch_proj.bias',
    'condition_proj.weight', 'condition_proj.bias',
    'time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
    'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias',
    'token_refiner.final_norm.weight',
    'final_layer.norm.weight',
    'final_layer.adaln_proj.linear.weight',
    'final_layer.adaln_proj.linear.bias',
    'final_layer.video_out.weight', 'final_layer.video_out.bias',
    'final_layer.audio_out.weight', 'final_layer.audio_out.bias',
)


@dataclasses.dataclass(frozen=True)
class Affine:
    """A linear layer with a bias, in the weight's dtype."""

    weight: mx.array
    bias: mx.array

    def __call__(self, x: mx.array) -> mx.array:
        """x: [..., in] -> [..., out], computed in the weight's dtype."""
        return (x.astype(self.weight.dtype) @ self.weight.T) + self.bias


@dataclasses.dataclass(frozen=True)
class NonTrunkWeights:
    """Everything outside the trunk blocks.

    Attributes:
        video_patch_proj: fp32, [hidden, video_patch_dim].
        audio_patch_proj: fp32, [hidden, audio_channels].
        condition_proj: bf16, text encoder states -> hidden.
        time_proj_in: fp32 time embedder, [time_embed_hidden, freq_dim].
        time_proj_out: fp32, [time_embed_dim, time_embed_hidden].
        refiner: One BlockWeights per token refiner block.
        refiner_final_norm: bf16 [hidden].
        final_norm: bf16 [hidden] of the output layer.
        final_adaln: bf16, [2 * hidden, time_embed_dim].
        video_out: fp32, [video_patch_dim, hidden].
        audio_out: fp32, [audio_channels, hidden].
    """

    video_patch_proj: Affine
    audio_patch_proj: Affine
    condition_proj: Affine
    time_proj_in: Affine
    time_proj_out: Affine
    refiner: tuple[mlx_block.BlockWeights, ...]
    refiner_final_norm: mx.array
    final_norm: mx.array
    final_adaln: Affine
    video_out: Affine
    audio_out: Affine

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, mx.array],
                     refiner: Sequence[mlx_block.BlockWeights]
                     ) -> NonTrunkWeights:
        """Builds from convert.non_trunk_tensors plus the refiner blocks.

        Raises:
            KeyError: On missing or unexpected tensors.
        """
        if set(tensors) != set(NON_TRUNK_NAMES):
            raise KeyError(
                f'missing={sorted(set(NON_TRUNK_NAMES) - set(tensors))}, '
                f'unexpected={sorted(set(tensors) - set(NON_TRUNK_NAMES))}')

        def affine(prefix):
            return Affine(tensors[f'{prefix}.weight'],
                          tensors[f'{prefix}.bias'])

        return cls(
            video_patch_proj=affine('video_patch_proj'),
            audio_patch_proj=affine('audio_patch_proj'),
            condition_proj=affine('condition_proj'),
            time_proj_in=affine('time_embedder.proj_in'),
            time_proj_out=affine('time_embedder.proj_out'),
            refiner=tuple(refiner),
            refiner_final_norm=tensors['token_refiner.final_norm.weight'],
            final_norm=tensors['final_layer.norm.weight'],
            final_adaln=affine('final_layer.adaln_proj.linear'),
            video_out=affine('final_layer.video_out'),
            audio_out=affine('final_layer.audio_out'),
        )

    @property
    def nbytes(self) -> int:
        total = self.refiner_final_norm.nbytes + self.final_norm.nbytes
        for affine in (self.video_patch_proj, self.audio_patch_proj,
                       self.condition_proj, self.time_proj_in,
                       self.time_proj_out, self.final_adaln, self.video_out,
                       self.audio_out):
            total += affine.weight.nbytes + affine.bias.nbytes
        return total + sum(b.nbytes for b in self.refiner)


def rope_cos_sin(position_ids: np.ndarray, config: h3_config.H3Config
                 ) -> tuple[mx.array, mx.array]:
    """3-axis RoPE tables, bitwise equal to h3.model.Rope.cos_sin.

    Args:
        position_ids: [S, 3] fp64 (t, h, w) coordinates.
        config: Architecture (rope_theta, rope_freqs_per_axis).

    Returns:
        (cos, sin), each [S, 1, rope_dim] bf16.

    Raises:
        ValueError: When position_ids is not [S, 3].
    """
    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError(f'position_ids must be [S, 3], got '
                         f'{position_ids.shape}')
    n = config.rope_freqs_per_axis
    inv_freq = (1.0 / (config.rope_theta ** (
        np.arange(0, 2 * n, 2, dtype=np.float32) / (2 * n))))
    pos = position_ids.astype(np.float32)
    per_axis = pos[:, :, None] * inv_freq[None, None, :]
    half = np.concatenate(list(per_axis.transpose(1, 0, 2)), axis=-1)
    cos = np.concatenate([np.cos(half)] * 2, axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(half)] * 2, axis=-1)[:, None, :]
    return (mx.array(cos).astype(_BF16), mx.array(sin).astype(_BF16))


def _time_frequencies(timesteps: np.ndarray, freq_dim: int) -> np.ndarray:
    """Sinusoidal timestep features, cos first then sin (torch's order)."""
    half = freq_dim // 2
    freqs = np.exp(-math.log(10000.0)
                   * np.arange(half, dtype=np.float32) / half)
    args = timesteps.astype(np.float32)[:, None] * freqs[None]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1)


def _silu_f32(x: mx.array) -> mx.array:
    """torch F.silu on fp32 (the time embedder and AdaLN input are fp32)."""
    return x * mx.sigmoid(x)


def adaln_input(weights: NonTrunkWeights, timesteps: np.ndarray) -> mx.array:
    """[M] timesteps -> SiLU(time embedding) [M, time_embed_dim] bf16.

    The embedder is an fp32 island; only the final SiLU output is rounded to
    bf16, as in h3.model.H3DiT.adaln_input.
    """
    features = mx.array(_time_frequencies(np.asarray(timesteps),
                                          weights.time_proj_in.weight.shape[1]))
    hidden = _silu_f32(weights.time_proj_in(features))
    return _silu_f32(weights.time_proj_out(hidden)).astype(_BF16)


def adaln_tables(linear: Affine, adaln_input_: mx.array, hidden_size: int,
                 expand: int, modalities: int) -> tuple[mx.array, ...]:
    """One AdaLN projection -> `expand` tables of [M * modalities, hidden].

    Args:
        linear: The projection (bf16).
        adaln_input_: [M, time_embed_dim] bf16.
        hidden_size: Model width.
        expand: 6 for a trunk block, 2 for the output layer.
        modalities: 3 for a trunk block, 1 for the output layer.
    """
    m = adaln_input_.shape[0]
    x = linear(adaln_input_).reshape(m * modalities, expand * hidden_size)
    return tuple(mx.split(x, expand, axis=-1))


def block_adaln_tables(adaln: Mapping[str, mx.array], adaln_input_: mx.array,
                       config: h3_config.H3Config) -> tuple[mx.array, ...]:
    """The six tables of one trunk block, from convert.adaln_tensors."""
    return adaln_tables(Affine(adaln['weight'], adaln['bias']), adaln_input_,
                        config.hidden_size, expand=6,
                        modalities=h3_config.MODALITY_NUM)


def refine_text(weights: NonTrunkWeights, text: mx.array,
                config: h3_config.H3Config) -> mx.array:
    """Text encoder states -> refined text rows, once per clip.

    Args:
        weights: The non-trunk weights (condition_proj and the refiner).
        text: [L, text_dim] bf16 encoder states.
        config: Architecture.

    Returns:
        [L, hidden] bf16.
    """
    x = weights.condition_proj(text.astype(_BF16))
    for block in weights.refiner:
        x = _refiner_block(block, x, config)
    return mlx_block.rms_norm(x, weights.refiner_final_norm,
                              config.final_norm_eps)


def _refiner_block(block: mlx_block.BlockWeights, x: mx.array,
                   config: h3_config.H3Config) -> mx.array:
    """Pre-norm text block: no AdaLN, no RoPE, dense attention."""
    seq = x.shape[0]
    heads, dim = config.num_heads, config.head_dim
    h = mlx_block.rms_norm(x, block.norm1, config.norm_eps)
    qkv = block.qkv(h).reshape(seq, heads, 3, dim)
    q = mlx_block.rms_norm(qkv[:, :, 0], block.q_norm, config.qk_norm_eps)
    k = mlx_block.rms_norm(qkv[:, :, 1], block.k_norm, config.qk_norm_eps)
    v = qkv[:, :, 2]
    del qkv
    out = mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None], scale=1.0 / math.sqrt(dim))[0]
    x = x + block.out(out.transpose(1, 0, 2).reshape(seq, heads * dim))
    h = mlx_block.rms_norm(x, block.norm2, config.norm_eps)
    halves = mx.split(block.fc1(h), 2, axis=-1)
    gate, up = halves if config.mlp_gate_first else halves[::-1]
    return x + block.fc2(mlx_block.swiglu(gate, up))


def embed(weights: NonTrunkWeights, config: h3_config.H3Config, seq_len: int,
          text: mx.array, video_rows: mx.array, audio_rows: mx.array,
          img_pos: mx.array, audio_pos: mx.array) -> mx.array:
    """Builds the packed sequence: text rows, video rows, audio rows.

    Args:
        weights: Non-trunk weights.
        config: Architecture.
        seq_len: Packed length (padding rows stay zero).
        text: [L, hidden] bf16 refined text.
        video_rows: [Nv, video_patch_dim] video latent rows in img_pos order.
        audio_rows: [Na, audio_channels] audio rows in audio_pos order.
        img_pos: [Nv] int32 packed row of every video row.
        audio_pos: [Na] int32 packed row of every audio row.

    Returns:
        [seq_len, hidden] bf16.
    """
    x = mx.zeros((seq_len, config.hidden_size), dtype=_BF16)
    x[:text.shape[0]] = text
    x[img_pos] = weights.video_patch_proj(
        video_rows.astype(_FP32)).astype(_BF16)
    x[audio_pos] = weights.audio_patch_proj(
        audio_rows.astype(_FP32)).astype(_BF16)
    return x


def final_layer(weights: NonTrunkWeights, x: mx.array, adaln_input_: mx.array,
                slot: mx.array, video_rows: mx.array, audio_rows: mx.array,
                config: h3_config.H3Config) -> tuple[mx.array, mx.array]:
    """Output layer: AdaLN modulation then the two fp32 heads.

    Args:
        weights: Non-trunk weights.
        x: [S, hidden] bf16 trunk output.
        adaln_input_: [M, time_embed_dim] bf16.
        slot: [S] int32 timestep slot of every packed row.
        video_rows: [Nt] int32 packed rows whose video velocity is wanted.
        audio_rows: [Nta] int32 packed rows whose audio velocity is wanted.
        config: Architecture.

    Returns:
        (video_v [Nt, video_patch_dim] fp32, audio_v [Nta, channels] fp32).
    """
    shift, scale = adaln_tables(weights.final_adaln, adaln_input_,
                                config.hidden_size, expand=2, modalities=1)
    h = mlx_block.rms_norm(x, weights.final_norm, config.final_norm_eps)
    h = mlx_block.modulate(h, 1.0 + scale, shift, slot)
    video = weights.video_out(mx.take(h, video_rows, axis=0).astype(_FP32))
    audio = weights.audio_out(mx.take(h, audio_rows, axis=0).astype(_FP32))
    return video, audio
