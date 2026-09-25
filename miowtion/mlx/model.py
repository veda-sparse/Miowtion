"""One velocity evaluation of the whole DiT, with the trunk streamed.

The trunk never fits in unified memory (33B parameters), so a forward is a
loop over blocks that arrive one at a time from a slab reader or prefetcher
(miowtion.mlx.slab) and are dropped again. Two things make that work:

- the packed sequence `x` is evaluated after every block, so MLX's lazy
  graph cannot keep block N's weights alive until the end of the trunk
  (the same reason BlockOptions.eval_chunks exists);
- the per-block AdaLN projections, 520 MB each and 26 GB over the trunk,
  are never part of that loop. They depend only on the timesteps, so a
  trajectory precomputes its tables in one pass over the checkpoint
  (precompute_adaln) and the denoise loop consumes the tables.

Everything that stays fixed over a trajectory -- the refined text, the
packed positions, the RoPE tables -- is built once by clip_inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.mlx import block as mlx_block
from miowtion.mlx import convert as mlx_convert
from miowtion.mlx import dit
from miowtion.utils import progress

BlockSource = Iterable[tuple[int, mlx_block.BlockWeights]]


@dataclasses.dataclass(frozen=True)
class ClipInputs:
    """Trajectory-static inputs (mirrors h3.model.ClipInputs).

    Attributes:
        text: [L, hidden] bf16 refined text rows.
        img_pos: [Nv] int32 packed row of every video-latent row.
        audio_pos: [Na] int32 packed row of every audio row.
        target_img_pos: [Nt] int32 packed rows whose video velocity is
            predicted.
        target_audio_pos: [Nta] int32.
        rope: (cos, sin), each [S, 1, rope_dim] bf16.
        seq_len: Packed length.
        used: Real rows; the rest is padding.
    """

    text: mx.array
    img_pos: mx.array
    audio_pos: mx.array
    target_img_pos: mx.array
    target_audio_pos: mx.array
    rope: tuple[mx.array, mx.array]
    seq_len: int
    used: int


@dataclasses.dataclass(frozen=True)
class TimestepInputs:
    """Per-step timestep inputs (mirrors h3.schedule.TimestepState).

    Attributes:
        timesteps: [M] fp32 distinct timesteps of this step. M varies, so it
            must never be assumed to be 2.
        slot: [S] int32 index into `timesteps` of every packed row.
        adaln_index: [S] int32 AdaLN table row of every packed row.
    """

    timesteps: np.ndarray
    slot: mx.array
    adaln_index: mx.array


def clip_inputs(weights: dit.NonTrunkWeights, config: h3_config.H3Config,
                text_states: mx.array, position_ids: np.ndarray,
                img_pos: np.ndarray, audio_pos: np.ndarray,
                target_img_pos: np.ndarray, target_audio_pos: np.ndarray,
                seq_len: int, used: int) -> ClipInputs:
    """Refines the text and builds the RoPE tables, once per clip.

    Args:
        weights: The resident non-trunk weights.
        config: Architecture.
        text_states: [L, text_dim] bf16 text encoder states.
        position_ids: [S, 3] fp64 packed positions (h3.layout).
        img_pos: [Nv] int packed rows of the video-latent rows.
        audio_pos: [Na] int.
        target_img_pos: [Nt] int.
        target_audio_pos: [Nta] int.
        seq_len: Packed length.
        used: Real rows.

    Returns:
        The trajectory-static inputs.
    """
    if position_ids.shape[0] != seq_len:
        raise ValueError(f'position_ids has {position_ids.shape[0]} rows, '
                         f'expected seq_len {seq_len}')

    def rows(name: str, values: np.ndarray) -> mx.array:
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f'{name} must be 1-D, got shape {values.shape}')
        return mx.array(values.astype(np.int32))

    return ClipInputs(
        text=dit.refine_text(weights, text_states, config),
        img_pos=rows('img_pos', img_pos),
        audio_pos=rows('audio_pos', audio_pos),
        target_img_pos=rows('target_img_pos', target_img_pos),
        target_audio_pos=rows('target_audio_pos', target_audio_pos),
        rope=dit.rope_cos_sin(position_ids, config),
        seq_len=seq_len,
        used=used,
    )


def precompute_adaln(reader: mlx_convert.ReleaseReader,
                     weights: dit.NonTrunkWeights, timesteps: np.ndarray,
                     ) -> list[tuple[mx.array, ...]]:
    """The per-block AdaLN tables of one distinct-timestep set.

    One pass over the 26 GB of AdaLN projections, holding one 520 MB
    projection at a time; what stays is 6 tables of [M * 3, hidden] per
    block, a few megabytes for the whole trunk.

    Args:
        reader: An open ReleaseReader over `<variant>/transformer`.
        weights: The non-trunk weights (for the time embedder).
        timesteps: [M] fp32 distinct timesteps.

    Returns:
        One 6-tuple of [M * MODALITY_NUM, hidden] bf16 tables per block.
    """
    config = reader.config
    adaln_input = dit.adaln_input(weights, timesteps)
    counter = progress.Progress('precompute adaln', config.num_layers)
    tables = []
    for index in range(config.num_layers):
        tensors = mlx_convert.adaln_tensors(reader, index)
        block_tables = dit.block_adaln_tables(tensors, adaln_input, config)
        mx.eval(*block_tables)
        del tensors
        tables.append(block_tables)
        counter.update(f'block {index}')
    return tables


def velocity(weights: dit.NonTrunkWeights, blocks: BlockSource,
             clip: ClipInputs, tables: Sequence[Sequence[mx.array]],
             timestep: TimestepInputs, video_rows: mx.array,
             audio_rows: mx.array, config: h3_config.H3Config,
             options: mlx_block.BlockOptions = mlx_block.BlockOptions(),
             ) -> tuple[mx.array, mx.array]:
    """One velocity evaluation over a streamed trunk.

    Args:
        weights: The resident non-trunk weights.
        blocks: Yields (index, weights) for blocks 0 .. num_layers - 1, in
            that order; a slab BlockPrefetcher is one such source.
        clip: Trajectory-static inputs.
        tables: Per-block AdaLN tables (precompute_adaln) for
            `timestep.timesteps`.
        timestep: This step's timestep inputs.
        video_rows: [Nv, video_patch_dim] fp32 rows in img_pos order.
        audio_rows: [Na, audio_channels] fp32 rows in audio_pos order.
        config: Architecture.
        options: Block chunking (head_chunk, row_chunk, sparse plan).

    Returns:
        (video_v [Nt, video_patch_dim] fp32, audio_v [Nta, channels] fp32).

    Raises:
        ValueError: When the source does not yield every block in order, or
            when `tables` does not cover the trunk.
    """
    if len(tables) != config.num_layers:
        raise ValueError(f'tables cover {len(tables)} blocks, expected '
                         f'{config.num_layers}')
    adaln_input = dit.adaln_input(weights, timestep.timesteps)
    x = dit.embed(weights, config, clip.seq_len, clip.text, video_rows,
                  audio_rows, clip.img_pos, clip.audio_pos)
    expected = 0
    for index, block in blocks:
        if index != expected:
            raise ValueError(f'block source yielded {index}, expected '
                             f'{expected}')
        x = mlx_block.block_forward(x, block, tables[index],
                                    timestep.adaln_index, clip.rope,
                                    clip.used, config, options)
        # Streaming only works if the block's weights die with the block:
        # without this the lazy graph holds all 50 of them.
        mx.eval(x)
        expected += 1
    if expected != config.num_layers:
        raise ValueError(f'block source yielded {expected} blocks, expected '
                         f'{config.num_layers}')
    return dit.final_layer(weights, x, adaln_input, timestep.slot,
                           clip.target_img_pos, clip.target_audio_pos, config)
