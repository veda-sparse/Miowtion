"""Numerical checks of the MLX path against the torch reference.

The unit tests compare the two backends on tiny synthetic weights; this
module does it on a real released block, where the condition number of the
GEMMs is whatever the training left behind. Neither backend can be bitwise
equal to the other (MLX's matmul and softmax accumulate in a different
order), so the check that means something is a three-way one: both paths
are also run against an fp32 torch gold built from the same weights. As
long as `mlx_vs_fp32` is not worse than `torch_vs_fp32`, the MLX block is
as good as the reference's own bf16 path and the rest of the difference is
bf16 rounding, not a porting bug (AGENTS 1.5).

Only one block is compared: a block is where every ported op lives, and
running 50 of them on the torch CPU reference would take a quarter of an
hour per sequence length.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping

import mlx.core as mx
import numpy as np
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.mlx import block as mlx_block
from miowtion.mlx import convert as mlx_convert
from miowtion.mlx import dit
from miowtion.mlx import interop
from miowtion.mlx import text_encoder as mlx_text

# Std of the synthetic activations fed to the block. The released blocks
# see RMS-normed rows, so the scale only has to be O(1).
_INPUT_STD = 0.5
# Std of the synthetic AdaLN input (a time embedding, which is O(0.1)).
_ADALN_STD = 0.1
# Distinct timesteps in the synthetic AdaLN input; the packed sequence of a
# clip carries two (video and audio).
_TIMESTEPS = 2


@dataclasses.dataclass(frozen=True)
class BlockComparison:
    """One block compared on both backends.

    Attributes:
        index: Trunk block index.
        seq_len: Rows in the packed sequence.
        mlx_vs_torch: Relative L2 of the MLX output against bf16 torch.
        mlx_vs_fp32: Relative L2 of the MLX output against the fp32 gold.
        torch_vs_fp32: The same for the bf16 torch output; the bar that
            mlx_vs_fp32 has to meet.
        torch_seconds: Wall seconds of the torch (CPU) block.
        mlx_seconds: Wall seconds of the MLX block, second run.
        peak_gb: MLX peak memory after the comparison.
    """

    index: int
    seq_len: int
    mlx_vs_torch: float
    mlx_vs_fp32: float
    torch_vs_fp32: float
    torch_seconds: float
    mlx_seconds: float
    peak_gb: float

    @property
    def as_good_as_torch(self) -> bool:
        """Whether MLX is no further from fp32 than torch's own bf16 path."""
        return self.mlx_vs_fp32 <= self.torch_vs_fp32 * 1.1

    def line(self) -> str:
        return (f'block {self.index} seq {self.seq_len}: '
                f'mlx-vs-torch {self.mlx_vs_torch:.3e}, '
                f'mlx-vs-fp32 {self.mlx_vs_fp32:.3e}, '
                f'torch-vs-fp32 {self.torch_vs_fp32:.3e}, '
                f'torch {self.torch_seconds:.2f} s, '
                f'mlx {self.mlx_seconds:.2f} s, '
                f'peak {self.peak_gb:.2f} GB')


def _rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.float(), want.float()
    return (got - want).norm().item() / max(want.norm().item(), 1e-12)


def compare_block(reader: mlx_convert.ReleaseReader, index: int,
                  seq_len: int, seed: int = 0,
                  options: mlx_block.BlockOptions = mlx_block.BlockOptions(),
                  bits: int = 0, group_size: int = 64) -> BlockComparison:
    """Runs one released trunk block on both backends.

    Args:
        reader: An open ReleaseReader over `<variant>/transformer`.
        index: Trunk block to compare.
        seq_len: Rows of the synthetic packed sequence. The torch reference
            is a CPU eager implementation, so this is quadratic and slow.
        seed: Seed of the synthetic inputs.
        options: MLX block chunking.
        bits: Quantizes the MLX linears (0 keeps bf16). The torch sides
            stay at the released weights, so the comparison then measures
            what quantization costs, not a porting difference, and
            `as_good_as_torch` is expected to be False.
        group_size: Quantization group.

    Returns:
        The comparison.
    """
    config = reader.config
    tensors = mlx_convert.trunk_tensors(reader, index)
    adaln = mlx_convert.adaln_tensors(reader, index)
    mx.eval(*tensors.values())
    weights = mlx_block.BlockWeights.from_tensors(tensors)
    if bits:
        weights = weights.quantize(bits, group_size)
    block = interop.torch_block(tensors, adaln, config)
    del tensors, adaln

    torch.manual_seed(seed)
    x = (torch.randn(seq_len, config.hidden_size) * _INPUT_STD).bfloat16()
    adaln_input = (torch.randn(_TIMESTEPS, config.time_embed_dim)
                   * _ADALN_STD).bfloat16()
    rows = _TIMESTEPS * h3_config.MODALITY_NUM
    adaln_index = torch.randint(0, rows, (seq_len,))
    # Real RoPE tables: consecutive positions on all three axes, so the
    # rotation is the one the block was trained with.
    positions = np.tile(np.arange(seq_len, dtype=np.float64)[:, None], (1, 3))
    cos, sin = (interop.to_torch(t)
                for t in dit.rope_cos_sin(positions, config))
    args = (adaln_index, (cos, sin), h3_model.DenseAttention(seq_len, 'math'),
            index)

    with torch.no_grad():
        start = time.time()
        want = block(x, None, adaln_input, *args)
        torch_seconds = time.time() - start
        # .float() and .bfloat16() convert in place; the round trip is
        # exact, so the bf16 weights survive the fp32 gold.
        gold = block.float()(x.float(), None, adaln_input.float(),
                             adaln_index, (cos.float(), sin.float()),
                             *args[2:])
        tables = [interop.from_torch(t)
                  for t in block.bfloat16().adaln_proj(adaln_input)]

    mx_args = (weights, tables, interop.from_torch(adaln_index.int()),
               (interop.from_torch(cos), interop.from_torch(sin)), seq_len,
               config, options)
    got = mlx_block.block_forward(interop.from_torch(x), *mx_args)
    mx.eval(got)  # The first run pays for compilation and page faults.
    start = time.time()
    got = mlx_block.block_forward(interop.from_torch(x), *mx_args)
    mx.eval(got)
    mlx_seconds = time.time() - start

    got = interop.to_torch(got)
    return BlockComparison(
        index=index, seq_len=seq_len,
        mlx_vs_torch=_rel_l2(got, want), mlx_vs_fp32=_rel_l2(got, gold),
        torch_vs_fp32=_rel_l2(want, gold), torch_seconds=torch_seconds,
        mlx_seconds=mlx_seconds, peak_gb=mx.get_peak_memory() / 2**30)


def torch_text_layer(tensors: Mapping[str, mx.array], directory: str,
                     index: int):
    """A transformers decoder layer holding one released text layer.

    Args:
        tensors: The layer, keyed as in miowtion.mlx.text_encoder.
        directory: The released `<variant>/text_encoder` (for the config).
        index: Layer index (transformers uses it for the layer type).

    Returns:
        An evaluated Qwen3VLTextDecoderLayer in bf16.
    """
    from transformers import AutoConfig  # pylint: disable=import-outside-toplevel
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen  # pylint: disable=import-outside-toplevel

    config = AutoConfig.from_pretrained(directory).text_config
    layer = qwen.Qwen3VLTextDecoderLayer(config, index).to(torch.bfloat16)
    with torch.no_grad():
        for name in mlx_text.NORM_NAMES + mlx_text.LINEAR_NAMES:
            key = f'{name}.weight'
            layer.get_parameter(key).copy_(interop.to_torch(tensors[key]))
    return layer.eval()


def compare_text_layer(reader, directory: str, index: int, seq_len: int,
                       seed: int = 0) -> BlockComparison:
    """Runs one released text encoder layer on both backends.

    The same three-way check as compare_block: what matters is whether
    `mlx_vs_fp32` beats `torch_vs_fp32`, not the distance between the two
    bf16 paths.

    Args:
        reader: An open ShardedSafetensors over `<variant>/text_encoder`.
        directory: That same directory (the config is read from it).
        index: Layer to compare.
        seq_len: Prompt length of the synthetic input.
        seed: Seed of the synthetic input.

    Returns:
        The comparison.
    """
    from transformers import AutoConfig  # pylint: disable=import-outside-toplevel
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen  # pylint: disable=import-outside-toplevel

    config = mlx_text.TowerConfig.from_pretrained(directory)
    tensors = mlx_text.layer_tensors(reader, index)
    mx.eval(*tensors.values())
    weights = mlx_text.LayerWeights.from_tensors(tensors)
    layer = torch_text_layer(tensors, directory, index)
    del tensors

    torch.manual_seed(seed)
    x = (torch.randn(1, seq_len, config.hidden_size) * _INPUT_STD).bfloat16()
    # Text-only mrope: one position on all three axes.
    positions = torch.arange(seq_len)[None, None, :].expand(3, 1, seq_len)
    rotary = qwen.Qwen3VLTextRotaryEmbedding(
        AutoConfig.from_pretrained(directory).text_config)
    mask = torch.full((seq_len, seq_len), float('-inf')).triu(1)
    with torch.no_grad():
        cos, sin = rotary(x, positions)
        start = time.time()
        want = layer(x, position_embeddings=(cos, sin),
                     attention_mask=mask.bfloat16()[None, None])
        torch_seconds = time.time() - start
        want = (want[0] if isinstance(want, tuple) else want)[0]
        gold = layer.float()(x.float(),
                             position_embeddings=(cos.float(), sin.float()),
                             attention_mask=mask[None, None])
        gold = (gold[0] if isinstance(gold, tuple) else gold)[0]

    mx_x = interop.from_torch(x[0])
    got = mlx_text.layer_forward(mx_x, weights, config)
    mx.eval(got)  # The first run pays for compilation and page faults.
    start = time.time()
    got = mlx_text.layer_forward(mx_x, weights, config)
    mx.eval(got)
    mlx_seconds = time.time() - start

    got = interop.to_torch(got)
    return BlockComparison(
        index=index, seq_len=seq_len,
        mlx_vs_torch=_rel_l2(got, want), mlx_vs_fp32=_rel_l2(got, gold),
        torch_vs_fp32=_rel_l2(want, gold), torch_seconds=torch_seconds,
        mlx_seconds=mlx_seconds, peak_gb=mx.get_peak_memory() / 2**30)
