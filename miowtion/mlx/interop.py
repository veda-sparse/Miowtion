"""Bit-exact conversion between torch tensors and MLX arrays.

numpy has no bfloat16, so bf16 crosses the boundary as its raw 16-bit
pattern (int16 views on both sides); other dtypes go through numpy directly.
"""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx
import numpy as np
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.h3 import weights as h3_weights
from miowtion.mlx import block as mlx_block

_TORCH_TO_MX = {
    torch.float32: mx.float32,
    torch.float16: mx.float16,
    torch.int32: mx.int32,
    torch.int64: mx.int64,
    torch.uint8: mx.uint8,
    torch.bool: mx.bool_,
}


def from_torch(t: torch.Tensor) -> mx.array:
    """CPU tensor -> MLX array with the same dtype and bits.

    Raises:
        ValueError: On an unsupported dtype.
    """
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    if t.dtype not in _TORCH_TO_MX:
        raise ValueError(f'unsupported dtype {t.dtype}')
    return mx.array(t.numpy())


def to_torch(a: mx.array) -> torch.Tensor:
    """MLX array -> CPU tensor with the same dtype and bits."""
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.view(mx.int16))).view(
            torch.bfloat16)
    return torch.from_numpy(np.array(a))


def block_weights_from_torch(block: h3_model.Block) -> mlx_block.BlockWeights:
    """Trunk weights of a torch Block in the release layout.

    The torch model keeps the fused QKV as [q_all; k_all; v_all]; the MLX
    block reads the release's per-head interleaved rows, so the rows are
    permuted back (the inverse of the load-time reorder in h3.weights).
    """
    attn = block.attn
    perm = h3_weights.qkv_row_permutation(attn.num_heads, attn.head_dim)
    qkv = torch.empty_like(attn.qkv_proj.weight)
    qkv[perm] = attn.qkv_proj.weight.detach()
    tensors = {
        'norm1.weight': block.norm1.weight,
        'norm2.weight': block.norm2.weight,
        'attn.q_norm.weight': attn.q_norm.weight,
        'attn.k_norm.weight': attn.k_norm.weight,
        'attn.qkv_proj.weight': qkv,
        'attn.out_proj.weight': attn.out_proj.weight,
        'mlp.fc1.weight': block.mlp.fc1.weight,
        'mlp.fc2.weight': block.mlp.fc2.weight,
    }
    return mlx_block.BlockWeights.from_tensors(
        {k: from_torch(v) for k, v in tensors.items()})


def torch_block(tensors: Mapping[str, mx.array],
                adaln: Mapping[str, mx.array],
                config: h3_config.H3Config) -> h3_model.Block:
    """A torch Block holding the release-layout weights of one trunk block.

    The inverse of block_weights_from_torch: it takes what
    convert.trunk_tensors and convert.adaln_tensors read out of a released
    checkpoint and gives the torch reference the same numbers, so the two
    backends can be compared on real weights (see miowtion.mlx.check).

    Args:
        tensors: Release-layout trunk tensors of one block (the keys of
            mlx_block.BlockWeights.from_tensors), bf16.
        adaln: The block's AdaLN projection ('weight', 'bias'), bf16.
        config: Architecture.

    Returns:
        An eval-mode bf16 Block.

    Raises:
        KeyError: When a tensor is missing.
    """
    block = h3_model.Block(config).to(torch.bfloat16)
    perm = h3_weights.qkv_row_permutation(config.num_heads, config.head_dim)
    same = ('norm1.weight', 'norm2.weight', 'attn.q_norm.weight',
            'attn.k_norm.weight', 'attn.out_proj.weight', 'mlp.fc1.weight',
            'mlp.fc2.weight')
    with torch.no_grad():
        for name in same:
            block.get_parameter(name).copy_(to_torch(tensors[name]))
        # mlx[perm[r]] == torch[r], so the way back gathers with perm.
        block.attn.qkv_proj.weight.copy_(
            to_torch(tensors['attn.qkv_proj.weight'])[perm])
        block.adaln_proj.linear.weight.copy_(to_torch(adaln['weight']))
        block.adaln_proj.linear.bias.copy_(to_torch(adaln['bias']))
    return block.eval()
