"""Strict loading of the released H3 DiT checkpoints into H3DiT.

Checkpoint: `<root>/<FL2VA|Ref2VA>/transformer/model.safetensors.index.json`.

The only non-trivial mapping is the fused QKV weight. The checkpoint stores
it interleaved per head ([h0: q k v, h1: q k v, ...]); H3DiT keeps
[q_all; k_all; v_all]. Without the reorder the model still runs and silently
produces garbage, so tests/unit/test_h3_weights.py pins the permutation.

Parameters that are FSDP2 DTensors are filled shard-locally: each rank reads
only the checkpoint rows of its own dim-0 shard.
"""

from __future__ import annotations

import collections
import json
import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
from torch.distributed import tensor as dtensor
from torch.distributed.tensor import _utils as dtensor_utils

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.h3 import release as h3_release
from miowtion.utils import progress

_QKV_SUFFIX = 'attn.qkv_proj.weight'


def qkv_row_permutation(num_heads: int, head_dim: int) -> torch.Tensor:
    """perm[r] = checkpoint row feeding H3DiT qkv row r.

    H3DiT row r = (part p, head h, d) with r = p*H*D + h*D + d;
    checkpoint row = h*3*D + p*D + d.
    """
    part, head, dim = torch.meshgrid(torch.arange(3), torch.arange(num_heads),
                                     torch.arange(head_dim), indexing='ij')
    return (head * 3 * head_dim + part * head_dim + dim).reshape(-1)


def _row_runs(rows: torch.Tensor) -> list[tuple[int, int, int]]:
    """Splits a row index vector into (dst_offset, src_start, length) runs."""
    runs = []
    rows = rows.tolist()
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i] != rows[i - 1] + 1:
            runs.append((start, rows[start], i - start))
            start = i
    return runs


class Checkpoint:
    """Lazy, sliceable view over a sharded safetensors checkpoint."""

    def __init__(self, transformer_dir: str):
        index_path = os.path.join(transformer_dir,
                                  'model.safetensors.index.json')
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        self._dir = transformer_dir
        self._key_to_file = weight_map
        self._handles = {}

    def keys(self) -> set[str]:
        return set(self._key_to_file)

    def _handle(self, key: str):
        filename = self._key_to_file[key]
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                os.path.join(self._dir, filename), framework='pt',
                device='cpu')
        return self._handles[filename]

    def read_rows(self, key: str, rows: torch.Tensor | None) -> torch.Tensor:
        """Reads `key`, restricted to dim-0 `rows` (None = everything)."""
        tensor_slice = self._handle(key).get_slice(key)
        if rows is None:
            return tensor_slice[:]
        shape = tensor_slice.get_shape()
        if rows.numel() == 0:
            return tensor_slice[0:0]
        out = None
        for dst, src, length in _row_runs(rows):
            chunk = tensor_slice[src:src + length]
            if out is None:
                out = chunk.new_empty((rows.numel(), *shape[1:]))
            out[dst:dst + length] = chunk
        return out


def _local_rows(param: torch.Tensor) -> torch.Tensor | None:
    """Global dim-0 rows owned by this rank, or None for a plain tensor."""
    if not isinstance(param, dtensor.DTensor):
        return None
    for placement in param.placements:
        if placement.is_shard() and placement.dim != 0:
            raise ValueError(f'only dim-0 sharding is supported: {placement}')
    local_shape, offset = dtensor_utils.compute_local_shape_and_global_offset(
        param.shape, param.device_mesh, param.placements)
    return torch.arange(offset[0], offset[0] + local_shape[0])


def _expected_dtype(name: str) -> torch.dtype:
    if name.startswith(h3_config.FP32_PARAM_PREFIXES) or (
            name in h3_config.FP32_BUFFER_NAMES):
        return torch.float32
    return torch.bfloat16


def _check_mlp_order(config: h3_config.H3Config,
                     ckpt_keys: Iterable[str]) -> None:
    """Refuses weights whose fused SwiGLU order the model does not expect.

    The two releases fuse mlp.fc1 as [gate; up] and [up; gate]. Both halves
    have the same shape and the same norm, so loading one release into a
    model configured for the other is silently accepted everywhere else and
    only shows as noise in generated video. This is the one place where the
    tensor names (the release) and the model's assumption meet.

    Raises:
        ValueError: On a mismatch, or on an unknown release layout.
    """
    schema = h3_release.detect_schema(ckpt_keys)
    expected = h3_release.mlp_gate_first(schema)
    if expected != config.mlp_gate_first:
        raise ValueError(
            f'checkpoint is the {schema!r} release, whose fused mlp.fc1 is '
            f'{"[gate; up]" if expected else "[up; gate]"}, but the model '
            f'was built with mlp_gate_first={config.mlp_gate_first}. Build '
            'the config with H3Config.from_pretrained(transformer_dir) so '
            'the order comes from the release itself')


def load_dit_weights(model: h3_model.H3DiT, transformer_dir: str,
                     skip_prefixes: Iterable[str] = ()) -> None:
    """Loads a released checkpoint into `model` (plain or FSDP2-sharded).

    Every model parameter/buffer must be found in the checkpoint and every
    checkpoint tensor must be consumed, except names under `skip_prefixes`
    (e.g. adaln projections that were replaced by precomputed tables).

    Raises:
        KeyError: On missing or unexpected tensors.
        ValueError: On shape or dtype mismatches, or when the checkpoint's
            release disagrees with `model.config.mlp_gate_first`.
    """
    ckpt = Checkpoint(transformer_dir)
    _check_mlp_order(model.config, ckpt.keys())
    skip_prefixes = tuple(skip_prefixes)
    # LoRA parameters (miowtion.train.lora) are not part of the release.
    state = collections.OrderedDict(
        (n, p) for n, p in model.named_parameters() if '.lora_' not in n)
    state.update((name, buf) for name, buf in model.named_buffers()
                 if name in h3_config.FP32_BUFFER_NAMES)
    ckpt_keys = {k for k in ckpt.keys() if not k.startswith(skip_prefixes)}
    missing = sorted(set(state) - ckpt_keys)
    unexpected = sorted(ckpt_keys - set(state))
    if missing or unexpected:
        raise KeyError(f'missing={missing[:5]} ({len(missing)}), '
                       f'unexpected={unexpected[:5]} ({len(unexpected)})')
    cfg = model.config
    qkv_perm = qkv_row_permutation(cfg.num_heads, cfg.head_dim)
    counter = progress.Progress('load weights: tensors', len(state),
                                every=max(1, len(state) // 10))
    for name, param in state.items():
        local_rows = _local_rows(param)
        if name.endswith(_QKV_SUFFIX):
            rows = qkv_perm if local_rows is None else qkv_perm[local_rows]
        else:
            rows = local_rows
        value = ckpt.read_rows(name, rows)
        if value.dtype != _expected_dtype(name):
            raise ValueError(f'{name}: checkpoint dtype {value.dtype}, '
                             f'expected {_expected_dtype(name)}')
        target = param.to_local() if isinstance(
            param, dtensor.DTensor) else param
        if target.dtype != value.dtype:
            raise ValueError(f'{name}: model dtype {target.dtype} != '
                             f'checkpoint dtype {value.dtype}')
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f'{name}: checkpoint {tuple(value.shape)} vs '
                             f'model {tuple(target.shape)}')
        with torch.no_grad():
            target.copy_(value)
        counter.update(name)
