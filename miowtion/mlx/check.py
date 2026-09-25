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

import mlx.core as mx
import numpy as np
import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import model as h3_model
from miowtion.mlx import block as mlx_block
from miowtion.mlx import convert as mlx_convert
from miowtion.mlx import dit
from miowtion.mlx import interop

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
                  ) -> BlockComparison:
    """Runs one released trunk block on both backends.

    Args:
        reader: An open ReleaseReader over `<variant>/transformer`.
        index: Trunk block to compare.
        seq_len: Rows of the synthetic packed sequence. The torch reference
            is a CPU eager implementation, so this is quadratic and slow.
        seed: Seed of the synthetic inputs.
        options: MLX block chunking.

    Returns:
        The comparison.
    """
    config = reader.config
    tensors = mlx_convert.trunk_tensors(reader, index)
    adaln = mlx_convert.adaln_tensors(reader, index)
    mx.eval(*tensors.values())
    weights = mlx_block.BlockWeights.from_tensors(tensors)
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
