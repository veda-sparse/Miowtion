"""One H3 trunk block in MLX, following miowtion.h3.model.Block.

The op chain mirrors the torch teacher so that every elementwise step rounds
to bf16 at the same place: RMSNorm computes the statistics and the weight
product in fp32 and rounds once (torch's nn.RMSNorm on bf16), AdaLN
modulation and the gated residuals run in bf16, SiLU is evaluated in fp32 and
rounded once. With these choices modulation, RoPE and SwiGLU are bitwise
equal to torch; only reductions (RMSNorm statistics, GEMM, attention) differ
in accumulation order.

Weights stay in the release layout: the fused QKV rows are interleaved per
head ([h0: q k v, h1: q k v, ...]), so a block can be streamed from the
checkpoint without a permutation, and a group of consecutive heads is one
contiguous row range (used for head-chunked attention).

Everything except attention is row-local, so the forward is chunked twice to
bound unified memory on long clips: heads for QKV + attention, rows for the
normalization, output projection, residuals and the MLP.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence

import mlx.core as mx

from miowtion.h3 import config as h3_config
from miowtion.mlx import sparse_attention

# Release names of the trunk tensors of one block, relative to `blocks.{i}.`.
# The AdaLN projection is absent: inference uses precomputed tables.
LINEAR_NAMES = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
NORM_NAMES = ('norm1', 'norm2', 'attn.q_norm', 'attn.k_norm')

# Rows per chunk of the row-local work; the same default as the torch
# reference's _ADALN_CHUNK_ROWS.
DEFAULT_ROW_CHUNK = 16384

# BlockWeights field -> release name.
_FIELD_NAMES = {'norm1': 'norm1', 'norm2': 'norm2', 'q_norm': 'attn.q_norm',
                'k_norm': 'attn.k_norm', 'qkv': 'attn.qkv_proj',
                'out': 'attn.out_proj', 'fc1': 'mlp.fc1', 'fc2': 'mlp.fc2'}


@dataclasses.dataclass(frozen=True)
class Linear:
    """Bias-free linear layer, dense or MLX affine-quantized.

    Attributes:
        weight: [out, in] bf16 (dense) or [out, in * bits / 32] uint32.
        scales: [out, in / group_size] bf16, quantized only.
        biases: [out, in / group_size] bf16, quantized only.
        group_size: Quantization group (0 when dense).
        bits: Quantization bits (0 when dense).
    """

    weight: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None
    group_size: int = 0
    bits: int = 0

    @property
    def quantized(self) -> bool:
        return self.scales is not None

    def __call__(self, x: mx.array) -> mx.array:
        """x: [..., in] -> [..., out] in x's dtype."""
        if not self.quantized:
            return x @ self.weight.T
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                   transpose=True, group_size=self.group_size,
                                   bits=self.bits)

    def rows(self, start: int, stop: int) -> Linear:
        """The output rows [start, stop) (a view, no copy)."""
        if not self.quantized:
            return Linear(self.weight[start:stop])
        return Linear(self.weight[start:stop], self.scales[start:stop],
                      self.biases[start:stop], self.group_size, self.bits)

    def quantize(self, bits: int, group_size: int) -> Linear:
        if self.quantized:
            raise ValueError('already quantized')
        weight, scales, biases = mx.quantize(self.weight,
                                             group_size=group_size, bits=bits)
        return Linear(weight, scales, biases, group_size, bits)

    def dequantize(self) -> Linear:
        """bf16 weight reconstructed from the quantized one."""
        if not self.quantized:
            raise ValueError('not quantized')
        return Linear(mx.dequantize(self.weight, self.scales, self.biases,
                                    group_size=self.group_size,
                                    bits=self.bits))


@dataclasses.dataclass(frozen=True)
class BlockWeights:
    """Trunk weights of one block (release layout, see module docstring).

    Norm weights are [hidden] or [head_dim] in the activation dtype.
    """

    norm1: mx.array
    norm2: mx.array
    q_norm: mx.array
    k_norm: mx.array
    qkv: Linear
    out: Linear
    fc1: Linear
    fc2: Linear

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, mx.array], group_size: int = 0,
                     bits: int = 0) -> BlockWeights:
        """Builds from `<name>.weight` (+ `.scales`, `.biases`) tensors.

        Args:
            tensors: Names relative to `blocks.{i}.`; exactly the trunk
                tensors of one block.
            group_size: Quantization group of the linear weights (0: dense).
            bits: Quantization bits (0: dense).

        Raises:
            KeyError: On missing or unexpected tensors.
        """
        quantized = bits > 0
        expected = {f'{n}.weight' for n in NORM_NAMES + LINEAR_NAMES}
        if quantized:
            expected |= {f'{n}.{s}' for n in LINEAR_NAMES
                         for s in ('scales', 'biases')}
        if set(tensors) != expected:
            raise KeyError(f'missing={sorted(expected - set(tensors))}, '
                           f'unexpected={sorted(set(tensors) - expected)}')
        kwargs = {}
        for field, name in _FIELD_NAMES.items():
            if name in NORM_NAMES:
                kwargs[field] = tensors[f'{name}.weight']
            elif quantized:
                kwargs[field] = Linear(tensors[f'{name}.weight'],
                                       tensors[f'{name}.scales'],
                                       tensors[f'{name}.biases'], group_size,
                                       bits)
            else:
                kwargs[field] = Linear(tensors[f'{name}.weight'])
        return cls(**kwargs)

    def to_tensors(self) -> dict[str, mx.array]:
        """Inverse of from_tensors."""
        out = {}
        for field, name in _FIELD_NAMES.items():
            value = getattr(self, field)
            if isinstance(value, Linear):
                out[f'{name}.weight'] = value.weight
                if value.quantized:
                    out[f'{name}.scales'] = value.scales
                    out[f'{name}.biases'] = value.biases
            else:
                out[f'{name}.weight'] = value
        return out

    def _map_linears(self, fn) -> BlockWeights:
        return dataclasses.replace(self, qkv=fn(self.qkv), out=fn(self.out),
                                   fc1=fn(self.fc1), fc2=fn(self.fc2))

    def quantize(self, bits: int, group_size: int) -> BlockWeights:
        """Affine-quantizes the four linear weights (norms stay dense)."""
        return self._map_linears(lambda lin: lin.quantize(bits, group_size))

    def dequantize(self) -> BlockWeights:
        return self._map_linears(Linear.dequantize)

    @property
    def quantization(self) -> tuple[int, int]:
        """(bits, group_size), (0, 0) when dense."""
        return self.qkv.bits, self.qkv.group_size

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.to_tensors().values())


@dataclasses.dataclass(frozen=True)
class BlockOptions:
    """Chunking of one block forward (results do not depend on eval points).

    Attributes:
        head_chunk: Heads per QKV + attention group (must divide num_heads);
            None processes all heads at once.
        row_chunk: Rows per chunk of the row-local work.
        eval_chunks: Evaluate every chunk before building the next one, so
            that at most one chunk's temporaries are alive (bounds peak
            unified memory on long clips; costs one GPU sync per chunk).
        sparse: Veda block-sparse attention plan; None runs dense attention.
            Requires `used` to be a multiple of both tile sizes.
    """

    head_chunk: int | None = None
    row_chunk: int = DEFAULT_ROW_CHUNK
    eval_chunks: bool = False
    sparse: sparse_attention.SparsePlan | None = None


class _Stages:
    """Optional per-stage wall time; evaluates at every stage boundary."""

    def __init__(self, profile: dict[str, float] | None):
        self.profile = profile
        self.last = time.perf_counter()

    def mark(self, name: str, *arrays: mx.array) -> None:
        if self.profile is None:
            return
        mx.eval(*arrays)
        now = time.perf_counter()
        self.profile[name] = self.profile.get(name, 0.0) + now - self.last
        self.last = now


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """torch nn.RMSNorm: fp32 statistics and weight product, one rounding.

    Args:
        x: [..., C] bf16 or fp32.
        weight: [C].
        eps: Epsilon inside the square root.
    """
    return mx.fast.rms_norm(x, weight.astype(mx.float32), eps).astype(x.dtype)


# The elementwise chains below are compiled (fused into one kernel each):
# a fused kernel keeps every intermediate in its declared dtype, so the
# rounding points and results are unchanged (the unit tests compare them
# bitwise with torch), while the fp32 temporaries of eager SiLU (three
# [rows, ffn] fp32 arrays per MLP chunk) are never materialized.


@mx.compile
def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """torch F.silu(gate) * up: SiLU in fp32 rounded once, then the product.

    Args:
        gate: [..., F] bf16.
        up: [..., F] bf16.
    """
    gf = gate.astype(mx.float32)
    return (gf * mx.sigmoid(gf)).astype(gate.dtype) * up


@mx.compile
def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotates the leading rope_dim channels of every head (bf16 chain).

    Args:
        x: [S, H, D].
        cos: [S, 1, rope_dim] in x's dtype.
        sin: [S, 1, rope_dim].
    """
    rope_dim = cos.shape[-1]
    half = rope_dim // 2
    x_rot = x[..., :rope_dim]
    rotated = mx.concatenate([-x_rot[..., half:], x_rot[..., :half]], axis=-1)
    return mx.concatenate([x_rot * cos + rotated * sin, x[..., rope_dim:]],
                          axis=-1)


@mx.compile
def _fma(x: mx.array, a: mx.array, b: mx.array) -> mx.array:
    """x * a + b with the product rounded before the sum (torch eager)."""
    return x * a + b


def _modulate(x: mx.array, one_plus_scale: mx.array, shift: mx.array,
              index: mx.array) -> mx.array:
    return _fma(x, mx.take(one_plus_scale, index, axis=0),
                mx.take(shift, index, axis=0))


def _gated_residual(x: mx.array, gate: mx.array, index: mx.array,
                    h: mx.array) -> mx.array:
    """x + gate[index] * h (torch: product rounded, then the sum)."""
    return _fma(mx.take(gate, index, axis=0), h, x)


def _row_chunks(num_rows: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, num_rows)) for s in range(0, num_rows, chunk)]


def _check_inputs(x, adaln, adaln_index, rope, used, config, options):
    seq, hidden = x.shape
    if hidden != config.hidden_size:
        raise ValueError(f'x hidden {hidden} != {config.hidden_size}')
    if len(adaln) != 6:
        raise ValueError(f'expected 6 AdaLN tables, got {len(adaln)}')
    if adaln_index.shape != (seq,):
        raise ValueError(f'adaln_index shape {adaln_index.shape} != ({seq},)')
    if rope[0].shape != (seq, 1, config.rope_dim):
        raise ValueError(f'rope shape {rope[0].shape}')
    if not 0 < used <= seq:
        raise ValueError(f'used {used} not in (0, {seq}]')
    heads = options.head_chunk or config.num_heads
    if config.num_heads % heads:
        raise ValueError(f'head_chunk {heads} does not divide '
                         f'{config.num_heads}')
    if options.row_chunk < 1:
        raise ValueError(f'row_chunk must be >= 1, got {options.row_chunk}')
    plan = options.sparse
    if plan is not None:
        # Padding rows must stay out of attention, and the gathered problem
        # must be rectangular, so the real rows have to tile exactly. Veda's
        # geometries are tile-aligned by construction; refuse rather than
        # silently attend to padding.
        if used % plan.q_block or used % plan.k_block:
            raise ValueError(
                f'used {used} must be a multiple of q_block {plan.q_block} '
                f'and k_block {plan.k_block} for sparse attention')
        if plan.index.shape[0] != used // plan.q_block:
            raise ValueError(
                f'sparse index has {plan.index.shape[0]} query tiles, '
                f'expected {used // plan.q_block}')


def block_forward(x: mx.array, weights: BlockWeights,
                  adaln: Sequence[mx.array], adaln_index: mx.array,
                  rope: tuple[mx.array, mx.array], used: int,
                  config: h3_config.H3Config,
                  options: BlockOptions = BlockOptions(),
                  profile: dict[str, float] | None = None) -> mx.array:
    """One trunk block with dense attention over rows [0, used).

    Args:
        x: [S, hidden] bf16 packed sequence.
        weights: The block's trunk weights.
        adaln: The block's six AdaLN tables (shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp), each [M * 3, hidden] bf16.
        adaln_index: [S] int32 AdaLN row of every packed row.
        rope: (cos, sin), each [S, 1, rope_dim] bf16.
        used: Real rows; padding rows attend to nothing (attention output
            zero), as in miowtion.h3.attention.dense_attention.
        config: Shape hyper-parameters.
        options: Chunking.
        profile: When given, receives wall seconds per stage ('norm_mod',
            'qkv', 'qk_norm_rope', 'attention', 'out_proj', 'mlp',
            'residual'); every stage is then evaluated separately, so the
            total is slightly slower than an unprofiled call.

    Returns:
        [S, hidden] block output.
    """
    _check_inputs(x, adaln, adaln_index, rope, used, config, options)
    stages = _Stages(profile)
    seq = x.shape[0]
    heads, head_dim = config.num_heads, config.head_dim
    head_chunk = options.head_chunk or heads
    rows = _row_chunks(seq, options.row_chunk)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln
    one_plus_msa = 1.0 + scale_msa
    one_plus_mlp = 1.0 + scale_mlp
    cos, sin = rope

    def maybe_eval(*arrays):
        if options.eval_chunks:
            mx.eval(*arrays)

    # norm1 + modulation, row-local.
    parts = []
    for start, stop in rows:
        idx = adaln_index[start:stop]
        parts.append(_modulate(rms_norm(x[start:stop], weights.norm1,
                                        config.norm_eps),
                               one_plus_msa, shift_msa, idx))
        maybe_eval(parts[-1])
    h = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    del parts
    stages.mark('norm_mod', h)

    # QKV + attention, one group of consecutive heads at a time.
    scale = 1.0 / math.sqrt(head_dim)
    outs = []
    for h0 in range(0, heads, head_chunk):
        qkv = weights.qkv.rows(3 * head_dim * h0,
                               3 * head_dim * (h0 + head_chunk))(h)
        qkv = qkv.reshape(seq, head_chunk, 3, head_dim)
        stages.mark('qkv', qkv)
        q = apply_rope(rms_norm(qkv[:, :, 0], weights.q_norm,
                                config.qk_norm_eps), cos, sin)
        k = apply_rope(rms_norm(qkv[:, :, 1], weights.k_norm,
                                config.qk_norm_eps), cos, sin)
        v = qkv[:, :, 2]
        del qkv
        stages.mark('qk_norm_rope', q, k)
        if options.sparse is None:
            out = mx.fast.scaled_dot_product_attention(
                q[:used].transpose(1, 0, 2)[None],
                k[:used].transpose(1, 0, 2)[None],
                v[:used].transpose(1, 0, 2)[None], scale=scale)[0]
        else:
            out = sparse_attention.block_sparse_attention(
                q[:used].transpose(1, 0, 2), k[:used].transpose(1, 0, 2),
                v[:used].transpose(1, 0, 2), options.sparse.index,
                q_block=options.sparse.q_block,
                k_block=options.sparse.k_block, scale=scale)
        out = out.transpose(1, 0, 2)  # [used, head_chunk, D]
        del q, k, v
        outs.append(out)
        maybe_eval(out)
        stages.mark('attention', out)
    del h
    attn = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
    del outs
    if used < seq:
        attn = mx.concatenate(
            [attn, mx.zeros((seq - used, heads, head_dim), dtype=attn.dtype)])
    attn = attn.reshape(seq, heads * head_dim)

    # Output projection, residuals, norm2 and MLP, row-local.
    parts = []
    for start, stop in rows:
        idx = adaln_index[start:stop]
        o = weights.out(attn[start:stop])
        stages.mark('out_proj', o)
        xr = _gated_residual(x[start:stop], gate_msa, idx, o)
        h2 = _modulate(rms_norm(xr, weights.norm2, config.norm_eps),
                       one_plus_mlp, shift_mlp, idx)
        stages.mark('residual', h2)
        gate, up = mx.split(weights.fc1(h2), 2, axis=-1)
        m = weights.fc2(swiglu(gate, up))
        stages.mark('mlp', m)
        parts.append(_gated_residual(xr, gate_mlp, idx, m))
        maybe_eval(parts[-1])
        stages.mark('residual', parts[-1])
    return parts[0] if len(parts) == 1 else mx.concatenate(parts)
