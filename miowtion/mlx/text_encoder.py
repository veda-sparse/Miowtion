"""The Qwen3-VL text tower of H3 on MLX, streamed one layer at a time.

H3 conditions the DiT on the hidden states of the first `TEXT_LAYERS`
layers of a 32B Qwen3-VL (see miowtion.train.encode). That is ~1 GB of
bf16 weights per layer and ~50 GB in total, three times the memory of the
machines this port targets, so the tower is streamed exactly like the DiT
trunk. Unlike the trunk it is read once per clip instead of once per
denoise step, and the prompt is a few hundred tokens, so the whole pass is
I/O: what matters is the read path, not the arithmetic.

Only the text path is ported, because H3's T2VA prompt is plain text: the
vision tower, its deepstack merge and the image rope sections never run.
The three mrope axes then carry the same position, and transformers'
Qwen3VLTextRotaryEmbedding recomposition picks the same frequency on every
axis, so the rotation reduces to the ordinary rotate-half RoPE that
mx.fast.rope implements (pinned against transformers in the unit tests).

The layer weights keep the released names and shapes; nothing is fused or
permuted, so a slab written from the checkpoint is a byte-for-byte copy of
the released tensors.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence

import mlx.core as mx

from miowtion.mlx import convert as mlx_convert
from miowtion.mlx import slab as mlx_slab
from miowtion.utils import progress

# Released key prefix of the text tower inside the Qwen3-VL checkpoint.
PREFIX = 'model.language_model'
EMBED_KEY = f'{PREFIX}.embed_tokens.weight'

# Per-layer tensors, relative to `{PREFIX}.layers.{i}.`. RMSNorm weights
# and linear weights are separated because only the latter are large.
NORM_NAMES = ('input_layernorm', 'post_attention_layernorm',
              'self_attn.q_norm', 'self_attn.k_norm')
LINEAR_NAMES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj',
                'mlp.down_proj')

# LayerWeights field -> released name.
_FIELD_NAMES = {
    'input_norm': 'input_layernorm', 'post_norm': 'post_attention_layernorm',
    'q_norm': 'self_attn.q_norm', 'k_norm': 'self_attn.k_norm',
    'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
    'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj',
    'gate_proj': 'mlp.gate_proj', 'up_proj': 'mlp.up_proj',
    'down_proj': 'mlp.down_proj',
}


@dataclasses.dataclass(frozen=True)
class TowerConfig:
    """The text tower's architecture.

    Attributes:
        hidden_size: Model width.
        num_hidden_layers: Layers in the released checkpoint (H3 uses only
            the first miowtion.train.encode.TEXT_LAYERS of them).
        num_attention_heads: Query heads.
        num_key_value_heads: Key/value heads (GQA).
        head_dim: Channels per head.
        intermediate_size: SwiGLU width.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base.
        vocab_size: Embedding rows.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int

    @classmethod
    def from_pretrained(cls, directory: str) -> TowerConfig:
        """Reads `<directory>/config.json` (the released Qwen3-VL config).

        Raises:
            ValueError: When the config carries no `text_config`, or when
                it asks for a rope type this port does not implement.
        """
        with open(os.path.join(directory, 'config.json')) as f:
            config = json.load(f)
        text = config.get('text_config')
        if text is None:
            raise ValueError(f'{directory}/config.json has no text_config')
        # 'rope_scaling' in the release, 'rope_parameters' in transformers 5.
        rope = text.get('rope_scaling') or text.get('rope_parameters') or {}
        if rope.get('rope_type', 'default') != 'default':
            raise ValueError(f'unsupported rope type {rope["rope_type"]}')
        return cls(hidden_size=text['hidden_size'],
                   num_hidden_layers=text['num_hidden_layers'],
                   num_attention_heads=text['num_attention_heads'],
                   num_key_value_heads=text['num_key_value_heads'],
                   head_dim=text['head_dim'],
                   intermediate_size=text['intermediate_size'],
                   rms_norm_eps=text['rms_norm_eps'],
                   rope_theta=text['rope_theta'],
                   vocab_size=text['vocab_size'])

    @property
    def layer_bytes(self) -> int:
        """Bytes of one bf16 layer."""
        heads = (self.num_attention_heads + 2 * self.num_key_value_heads)
        attn = self.hidden_size * heads * self.head_dim
        attn += self.num_attention_heads * self.head_dim * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size + 2 * self.head_dim
        return 2 * (attn + mlp + norms)


@dataclasses.dataclass(frozen=True)
class LayerWeights:
    """One decoder layer, in the released layout.

    Attributes:
        input_norm: [hidden] bf16.
        post_norm: [hidden] bf16.
        q_norm: [head_dim] bf16, applied per head.
        k_norm: [head_dim] bf16.
        q_proj: [heads * head_dim, hidden] bf16.
        k_proj: [kv_heads * head_dim, hidden] bf16.
        v_proj: [kv_heads * head_dim, hidden] bf16.
        o_proj: [hidden, heads * head_dim] bf16.
        gate_proj: [intermediate, hidden] bf16.
        up_proj: [intermediate, hidden] bf16.
        down_proj: [hidden, intermediate] bf16.
    """

    input_norm: mx.array
    post_norm: mx.array
    q_norm: mx.array
    k_norm: mx.array
    q_proj: mx.array
    k_proj: mx.array
    v_proj: mx.array
    o_proj: mx.array
    gate_proj: mx.array
    up_proj: mx.array
    down_proj: mx.array

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, mx.array]) -> LayerWeights:
        """Builds from `<released name>.weight` tensors of one layer.

        Raises:
            KeyError: When the set of names is not exactly one layer.
        """
        expected = {f'{n}.weight' for n in NORM_NAMES + LINEAR_NAMES}
        if set(tensors) != expected:
            raise KeyError(f'missing={sorted(expected - set(tensors))}, '
                           f'unexpected={sorted(set(tensors) - expected)}')
        return cls(**{field: tensors[f'{name}.weight']
                      for field, name in _FIELD_NAMES.items()})

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f).nbytes for f in _FIELD_NAMES)

    def to_tensors(self) -> dict[str, mx.array]:
        """The inverse of from_tensors (slab writing order)."""
        return {f'{name}.weight': getattr(self, field)
                for field, name in _FIELD_NAMES.items()}


def layer_keys(index: int) -> dict[str, str]:
    """Released checkpoint keys of layer `index`, keyed for from_tensors."""
    return {f'{n}.weight': f'{PREFIX}.layers.{index}.{n}.weight'
            for n in NORM_NAMES + LINEAR_NAMES}


def layer_tensors(reader: mlx_convert.ShardedSafetensors,
                  index: int) -> dict[str, mx.array]:
    """Reads one layer out of the released text encoder."""
    return {name: reader.read(key)
            for name, key in layer_keys(index).items()}


def embed_tokens(reader: mlx_convert.ShardedSafetensors) -> mx.array:
    """The token embedding table, [vocab, hidden] bf16 (resident)."""
    return reader.read(EMBED_KEY)


def layer_forward(x: mx.array, weights: LayerWeights,
                  config: TowerConfig) -> mx.array:
    """One decoder layer on a single causal sequence.

    The op chain follows transformers' Qwen3VLTextDecoderLayer: RMSNorm on
    the rows, per-head RMSNorm on q and k before the rotation, causal GQA
    attention, SwiGLU, both residuals in bf16.

    Args:
        x: [seq, hidden] bf16 hidden states.
        weights: The layer.
        config: The tower.

    Returns:
        [seq, hidden] bf16.
    """
    seq = x.shape[0]
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = config.head_dim
    h = mx.fast.rms_norm(x, weights.input_norm, config.rms_norm_eps)
    q = (h @ weights.q_proj.T).reshape(seq, heads, dim)
    k = (h @ weights.k_proj.T).reshape(seq, kv_heads, dim)
    v = (h @ weights.v_proj.T).reshape(seq, kv_heads, dim)
    q = mx.fast.rms_norm(q, weights.q_norm, config.rms_norm_eps)
    k = mx.fast.rms_norm(k, weights.k_norm, config.rms_norm_eps)
    # [heads, seq, dim]: mx.fast.rope rotates the last axis and indexes
    # positions along the one before it.
    q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))
    rope = {'dims': dim, 'traditional': False, 'base': config.rope_theta,
            'scale': 1.0, 'offset': 0}
    q, k = mx.fast.rope(q, **rope), mx.fast.rope(k, **rope)
    attn = mx.fast.scaled_dot_product_attention(
        q[None], k[None], v[None], scale=dim**-0.5, mask='causal')
    attn = attn[0].transpose(1, 0, 2).reshape(seq, heads * dim)
    x = x + attn @ weights.o_proj.T
    h = mx.fast.rms_norm(x, weights.post_norm, config.rms_norm_eps)
    gate = h @ weights.gate_proj.T
    # SiLU in fp32, rounded once, as in the torch reference.
    gate = (gate.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32)))
    hidden = gate.astype(mx.bfloat16) * (h @ weights.up_proj.T)
    return x + hidden @ weights.down_proj.T


def encode(embed: mx.array, layers: Iterable[tuple[int, LayerWeights]],
           ids: Sequence[int], config: TowerConfig,
           total: int | None = None) -> mx.array:
    """Runs the tower over one prompt.

    The final RMSNorm is not applied: H3 replaces it with an identity and
    conditions on the unnormalized hidden states (miowtion.train.encode).

    Args:
        embed: [vocab, hidden] bf16 embedding table.
        layers: (index, weights) in order; a generator so that the caller
            decides how the layers are streamed.
        ids: Token ids of the prompt.
        config: The tower.
        total: Layers the caller streams, for the progress line; the
            checkpoint's layer count by default.

    Returns:
        [len(ids), hidden] bf16 hidden states.

    Raises:
        ValueError: On an empty prompt or an out-of-range token id.
    """
    if not len(ids):
        raise ValueError('ids is empty')
    index = mx.array(list(ids), dtype=mx.uint32)
    if int(mx.max(index)) >= embed.shape[0]:
        raise ValueError(f'token id {int(mx.max(index))} is outside the '
                         f'{embed.shape[0]} row embedding table')
    x = embed[index]
    counter = progress.Progress(
        'text encoder',
        config.num_hidden_layers if total is None else total)
    for number, weights in layers:
        x = layer_forward(x, weights, config)
        mx.eval(x)
        counter.update(f'layer {number}')
    return x


def write_tower_slabs(reader: mlx_convert.ShardedSafetensors, out_dir: str,
                      layers: Sequence[int]) -> None:
    """Writes one slab per text encoder layer.

    Streaming the released shards with mmap is fault driven and reaches
    only a fraction of the SSD's bandwidth (see the feature doc), so the
    layers are laid out the same way as the DiT trunk.

    Args:
        reader: The released `<variant>/text_encoder`.
        out_dir: Output directory (block_XXX.slab, one per layer).
        layers: Layer indices to write.
    """
    os.makedirs(out_dir, exist_ok=True)
    counter = progress.Progress('write text encoder slabs', len(layers))
    for index in layers:
        tensors = layer_tensors(reader, index)
        mx.eval(*tensors.values())
        mlx_slab.write_slab(mlx_slab.slab_path(out_dir, index), index,
                            LayerWeights.from_tensors(tensors).to_tensors())
        del tensors
        counter.update(f'layer {index}')


def slab_layers(reader: mlx_slab.SlabReader,
                layers: Sequence[int]) -> Iterator[tuple[int, LayerWeights]]:
    """Streams text encoder layers from slabs, one read ahead."""
    prefetcher = mlx_slab.BlockPrefetcher(reader, layers,
                                          unpack=reader.tensors)
    for index, tensors in prefetcher:
        yield index, LayerWeights.from_tensors(tensors)
