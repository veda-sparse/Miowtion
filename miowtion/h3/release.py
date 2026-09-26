"""Tensor names of the released MiniMax-H3 transformer checkpoints.

Two releases of the same 33B DiT are in circulation and they name their
tensors differently:

- `'h3'`: the first release (`MiniMaxH3DiTModel`, diffusers 0.32). Names
  equal the H3DiT module names, and the fused QKV weight is one tensor whose
  rows are interleaved per head ([h0: q k v, h1: q k v, ...]).
- `'diffusers'`: the current diffusers port (`MiniMaxH3Transformer3DModel`,
  diffusers 0.36). Blocks live under `transformer_blocks.`, attention is
  three separate projections (`attn.to_q/to_k/to_v`), and the embedding and
  output layers are renamed (`proj_in`, `context_embedder`, `norm_out`, ...).

This module only maps names; reading and fusing the tensors is left to the
caller (`miowtion.mlx.convert` for the MLX path). A mapping value is a tuple
of release keys: one key is read as-is, three keys are the q, k and v
projections and must be fused into the per-head interleaved row order
(row h * 3 * head_dim + part * head_dim + d), which is the layout both the
MLX block and the first release use.
"""

from __future__ import annotations

from collections.abc import Iterable

SCHEMA_H3 = 'h3'
SCHEMA_DIFFUSERS = 'diffusers'
SCHEMAS = (SCHEMA_H3, SCHEMA_DIFFUSERS)

# `_class_name` of the release's transformer/config.json. Detecting the
# schema from the config is the only option before the tensor index is
# open (H3Config.from_pretrained); detect_schema() is used once it is.
_CLASS_NAME_SCHEMA = {
    'MiniMaxH3DiTModel': SCHEMA_H3,
    'MiniMaxH3Transformer3DModel': SCHEMA_DIFFUSERS,
}

# Which half of the fused mlp.fc1 output is the gated one. fc1 emits both
# SwiGLU projections in one GEMM, and the two releases fused them in
# opposite orders: the first release stores [gate; up] (H3DiT's own
# `gate, up = fc1(x).chunk(2)`), the diffusers port stores [up; gate]
# (diffusers' SwiGLU: `up, gate = proj(x).chunk(2)`).
#
# Nothing but the samples can catch a wrong choice here: swapping the
# halves keeps every shape, dtype and norm identical, so shape checks,
# completeness checks and synthetic-weight tests all stay green while the
# real model degenerates into noise. It is therefore pinned per schema and
# cross-checked against the checkpoint at load time
# (miowtion.h3.weights.load_dit_weights).
MLP_GATE_FIRST = {SCHEMA_H3: True, SCHEMA_DIFFUSERS: False}

_BLOCK_PREFIX = {SCHEMA_H3: 'blocks', SCHEMA_DIFFUSERS: 'transformer_blocks'}

# H3DiT name (relative to the block) -> release names, per schema.
_BLOCK_NAMES = {
    SCHEMA_H3: {
        'norm1.weight': ('norm1.weight',),
        'norm2.weight': ('norm2.weight',),
        'attn.q_norm.weight': ('attn.q_norm.weight',),
        'attn.k_norm.weight': ('attn.k_norm.weight',),
        'attn.qkv_proj.weight': ('attn.qkv_proj.weight',),
        'attn.out_proj.weight': ('attn.out_proj.weight',),
        'mlp.fc1.weight': ('mlp.fc1.weight',),
        'mlp.fc2.weight': ('mlp.fc2.weight',),
        'adaln_proj.linear.weight': ('adaln_proj.linear.weight',),
        'adaln_proj.linear.bias': ('adaln_proj.linear.bias',),
    },
    SCHEMA_DIFFUSERS: {
        'norm1.weight': ('norm1.weight',),
        'norm2.weight': ('norm2.weight',),
        'attn.q_norm.weight': ('attn.norm_q.weight',),
        'attn.k_norm.weight': ('attn.norm_k.weight',),
        'attn.qkv_proj.weight': ('attn.to_q.weight', 'attn.to_k.weight',
                                 'attn.to_v.weight'),
        'attn.out_proj.weight': ('attn.to_out.0.weight',),
        'mlp.fc1.weight': ('ff.net.0.proj.weight',),
        'mlp.fc2.weight': ('ff.net.2.weight',),
        'adaln_proj.linear.weight': ('adaln_proj.linear.weight',),
        'adaln_proj.linear.bias': ('adaln_proj.linear.bias',),
    },
}

# The trunk tensors of a block: everything but the AdaLN projection, which
# inference consumes as precomputed tables (see miowtion.mlx.block).
TRUNK_NAMES = tuple(n for n in _BLOCK_NAMES[SCHEMA_H3]
                    if not n.startswith('adaln_proj.'))
ADALN_NAMES = ('adaln_proj.linear.weight', 'adaln_proj.linear.bias')

# H3DiT name -> release names for everything outside the trunk blocks. The
# token refiner blocks are handled by refiner_keys().
_NON_TRUNK_NAMES = {
    SCHEMA_DIFFUSERS: {
        'video_patch_proj.weight': ('proj_in.weight',),
        'video_patch_proj.bias': ('proj_in.bias',),
        'audio_patch_proj.weight': ('audio_proj_in.weight',),
        'audio_patch_proj.bias': ('audio_proj_in.bias',),
        'condition_proj.weight': ('context_embedder.weight',),
        'condition_proj.bias': ('context_embedder.bias',),
        'time_embedder.proj_in.weight': ('time_embedder.linear_1.weight',),
        'time_embedder.proj_in.bias': ('time_embedder.linear_1.bias',),
        'time_embedder.proj_out.weight': ('time_embedder.linear_2.weight',),
        'time_embedder.proj_out.bias': ('time_embedder.linear_2.bias',),
        'token_refiner.final_norm.weight': ('token_refiner.final_norm.weight',),
        'final_layer.norm.weight': ('norm_out.norm.weight',),
        'final_layer.adaln_proj.linear.weight': ('norm_out.linear.weight',),
        'final_layer.adaln_proj.linear.bias': ('norm_out.linear.bias',),
        'final_layer.video_out.weight': ('proj_out.weight',),
        'final_layer.video_out.bias': ('proj_out.bias',),
        'final_layer.audio_out.weight': ('audio_proj_out.weight',),
        'final_layer.audio_out.bias': ('audio_proj_out.bias',),
    },
}
_NON_TRUNK_NAMES[SCHEMA_H3] = {name: (name,)
                               for name in _NON_TRUNK_NAMES[SCHEMA_DIFFUSERS]}

# Where the token refiner blocks live. The refiner has no AdaLN and no RoPE,
# so it needs the trunk names minus the AdaLN projection.
_REFINER_PREFIX = {SCHEMA_H3: 'token_refiner.blocks',
                   SCHEMA_DIFFUSERS: 'token_refiner.refiner_blocks'}


def detect_schema(keys: Iterable[str]) -> str:
    """Which release layout a checkpoint's tensor names belong to.

    Args:
        keys: Every tensor name of the checkpoint (the safetensors index).

    Returns:
        One of SCHEMAS.

    Raises:
        ValueError: When the names match neither layout, or both.
    """
    keys = set(keys)
    found = [schema for schema in SCHEMAS
             if any(k.startswith(_BLOCK_PREFIX[schema] + '.0.') for k in keys)]
    # 'blocks.0.' is a suffix of 'transformer_blocks.0.', so the H3 layout
    # only wins when the diffusers prefix is absent.
    if len(found) == 2:
        found = [SCHEMA_DIFFUSERS]
    if not found:
        raise ValueError('no transformer blocks found; tried prefixes '
                         f'{sorted(_BLOCK_PREFIX.values())}')
    return found[0]


def schema_from_class_name(class_name: str | None) -> str:
    """Which release a `transformer/config.json` describes.

    Args:
        class_name: The config's `_class_name` field (None when absent).

    Returns:
        One of SCHEMAS.

    Raises:
        ValueError: On a missing or unknown class name. A third release
            would have its own fused-mlp order (MLP_GATE_FIRST) and its own
            key spellings, so guessing one is never right.
    """
    if class_name not in _CLASS_NAME_SCHEMA:
        raise ValueError(
            f'unknown transformer _class_name {class_name!r}; known '
            f'releases: {sorted(_CLASS_NAME_SCHEMA)}. Add it to '
            'miowtion.h3.release together with its MLP_GATE_FIRST order')
    return _CLASS_NAME_SCHEMA[class_name]


def mlp_gate_first(schema: str) -> bool:
    """Whether `schema` fuses mlp.fc1 as [gate; up] (see MLP_GATE_FIRST)."""
    return MLP_GATE_FIRST[_checked(schema)]


def _prefixed(prefix: str, names: dict[str, tuple[str, ...]]
              ) -> dict[str, tuple[str, ...]]:
    return {name: tuple(f'{prefix}.{k}' for k in keys)
            for name, keys in names.items()}


def block_keys(schema: str, index: int, adaln: bool = False
               ) -> dict[str, tuple[str, ...]]:
    """Release keys of trunk block `index`, keyed by H3DiT block name.

    Args:
        schema: One of SCHEMAS.
        index: Block index.
        adaln: Include the AdaLN projection (weight and bias).
    """
    names = dict(_BLOCK_NAMES[_checked(schema)])
    if not adaln:
        names = {n: k for n, k in names.items() if n in TRUNK_NAMES}
    return _prefixed(f'{_BLOCK_PREFIX[schema]}.{index}', names)


def refiner_keys(schema: str, index: int) -> dict[str, tuple[str, ...]]:
    """Release keys of token refiner block `index`."""
    names = {n: k for n, k in _BLOCK_NAMES[_checked(schema)].items()
             if n in TRUNK_NAMES}
    return _prefixed(f'{_REFINER_PREFIX[schema]}.{index}', names)


def non_trunk_keys(schema: str) -> dict[str, tuple[str, ...]]:
    """Release keys of the embedding, time and output layers."""
    return dict(_NON_TRUNK_NAMES[_checked(schema)])


def expected_keys(schema: str, num_layers: int,
                  num_refiner_layers: int) -> set[str]:
    """Every release key the H3 DiT needs, for a completeness check."""
    keys: set[str] = set()
    for mapping in ([block_keys(schema, i, adaln=True)
                     for i in range(num_layers)]
                    + [refiner_keys(schema, i)
                       for i in range(num_refiner_layers)]
                    + [non_trunk_keys(schema)]):
        for names in mapping.values():
            keys.update(names)
    return keys


def check_complete(schema: str, available: Iterable[str], num_layers: int,
                   num_refiner_layers: int) -> None:
    """Raises when the checkpoint misses a key the DiT needs.

    A renamed layer would otherwise only surface as a shape error deep in
    the conversion, or not at all.

    Raises:
        KeyError: Listing the missing keys.
    """
    missing = expected_keys(schema, num_layers,
                            num_refiner_layers) - set(available)
    if missing:
        raise KeyError(f'{len(missing)} keys missing from the checkpoint, '
                       f'e.g. {sorted(missing)[:5]}')


def _checked(schema: str) -> str:
    if schema not in SCHEMAS:
        raise ValueError(f'schema must be one of {SCHEMAS}, got {schema!r}')
    return schema
