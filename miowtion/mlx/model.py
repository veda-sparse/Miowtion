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
from collections.abc import Callable, Iterable, Mapping, Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.mlx import block as mlx_block
from miowtion.mlx import convert as mlx_convert
from miowtion.mlx import dit
from miowtion.mlx import sparse_attention
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


def timestep_key(timesteps: np.ndarray) -> tuple[int, ...]:
    """Exact (bit-level) key of an fp32 timestep set.

    Mirrors miowtion.train.adaln.timestep_key without pulling in torch: a
    table is only valid for the exact timesteps it was built from, so the
    key must not round.
    """
    values = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    return tuple(int(v) for v in values.view(np.uint32))


class AdalnTables:
    """Per-block AdaLN tables, keyed by timestep set.

    A schedule uses a handful of distinct timestep sets, and the AdaLN
    projections are 26 GB, so every set a run can need is computed in one
    pass over the checkpoint (see precompute_adaln) and looked up by its
    exact fp32 key afterwards.
    """

    def __init__(self, tables: Mapping[tuple[int, ...],
                                       list[tuple[mx.array, ...]]]):
        self._tables = dict(tables)

    def __contains__(self, timesteps: np.ndarray) -> bool:
        return timestep_key(timesteps) in self._tables

    def __len__(self) -> int:
        return len(self._tables)

    def get(self, timesteps: np.ndarray) -> list[tuple[mx.array, ...]]:
        """The per-block tables of one set.

        Raises:
            KeyError: When the set was not precomputed.
        """
        key = timestep_key(timesteps)
        if key not in self._tables:
            raise KeyError(f'no AdaLN table for timesteps '
                           f'{np.asarray(timesteps).tolist()}; precompute '
                           'every set of the schedule')
        return self._tables[key]

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for tables in self._tables.values()
                   for block in tables for t in block)


def precompute_adaln(reader: mlx_convert.ReleaseReader,
                     weights: dit.NonTrunkWeights,
                     timestep_sets: Iterable[np.ndarray]) -> AdalnTables:
    """The per-block AdaLN tables of every distinct timestep set.

    One pass over the 26 GB of AdaLN projections, holding one 520 MB
    projection at a time; what stays is 6 tables of [M * 3, hidden] per
    block and per set, a few megabytes for the whole trunk. Doing one pass
    per set instead would re-read those 26 GB for every step of the
    schedule.

    Args:
        reader: An open ReleaseReader over `<variant>/transformer`.
        weights: The non-trunk weights (for the time embedder).
        timestep_sets: Distinct sorted fp32 timestep sets (duplicates are
            dropped by their exact key).

    Returns:
        The tables of every set.

    Raises:
        ValueError: When no set is given.
    """
    config = reader.config
    sets = {timestep_key(t): np.asarray(t, dtype=np.float32)
            for t in timestep_sets}
    if not sets:
        raise ValueError('timestep_sets is empty')
    inputs = {key: dit.adaln_input(weights, t) for key, t in sets.items()}
    tables = {key: [] for key in sets}
    counter = progress.Progress(f'precompute adaln ({len(sets)} sets)',
                                config.num_layers)
    for index in range(config.num_layers):
        tensors = mlx_convert.adaln_tensors(reader, index)
        for key, adaln_input in inputs.items():
            block_tables = dit.block_adaln_tables(tensors, adaln_input,
                                                  config)
            mx.eval(*block_tables)
            tables[key].append(block_tables)
        del tensors
        counter.update(f'block {index}')
    return AdalnTables(tables)


def velocity(weights: dit.NonTrunkWeights, blocks: BlockSource,
             clip: ClipInputs, tables: Sequence[Sequence[mx.array]],
             timestep: TimestepInputs, video_rows: mx.array,
             audio_rows: mx.array, config: h3_config.H3Config,
             options: mlx_block.BlockOptions = mlx_block.BlockOptions(),
             plans: Sequence[sparse_attention.LayerPlan | Callable
                             | None] | None = None,
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
        plans: One Veda plan -- or one planner, see BlockOptions.sparse --
            per trunk layer, None for the layers that stay dense;
            `plans=None` runs `options` unchanged everywhere. A per-layer
            entry overrides `options.sparse`, which cannot express that
            different layers select different tiles.

    Returns:
        (video_v [Nt, video_patch_dim] fp32, audio_v [Nta, channels] fp32).

    Raises:
        ValueError: When the source does not yield every block in order, or
            when `tables` or `plans` does not cover the trunk.
    """
    if len(tables) != config.num_layers:
        raise ValueError(f'tables cover {len(tables)} blocks, expected '
                         f'{config.num_layers}')
    if plans is not None and len(plans) != config.num_layers:
        raise ValueError(f'plans cover {len(plans)} blocks, expected '
                         f'{config.num_layers}')
    adaln_input = dit.adaln_input(weights, timestep.timesteps)
    x = dit.embed(weights, config, clip.seq_len, clip.text, video_rows,
                  audio_rows, clip.img_pos, clip.audio_pos)
    expected = 0
    for index, block in blocks:
        if index != expected:
            raise ValueError(f'block source yielded {index}, expected '
                             f'{expected}')
        block_options = (options if plans is None else
                         dataclasses.replace(options, sparse=plans[index]))
        x = mlx_block.block_forward(x, block, tables[index],
                                    timestep.adaln_index, clip.rope,
                                    clip.used, config, block_options)
        # Streaming only works if the block's weights die with the block:
        # without this the lazy graph holds all 50 of them.
        mx.eval(x)
        expected += 1
    if expected != config.num_layers:
        raise ValueError(f'block source yielded {expected} blocks, expected '
                         f'{config.num_layers}')
    return dit.final_layer(weights, x, adaln_input, timestep.slot,
                           clip.target_img_pos, clip.target_audio_pos, config)
