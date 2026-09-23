"""Precomputed per-block AdaLN tables for a frozen trunk.

The 50 block AdaLN projections hold 13B of the 33B parameters but only map
the timestep embedding to modulation vectors. With a frozen trunk and a
fixed sampling schedule, every forward uses one of a few known timestep
sets, so their outputs are computed once from the checkpoint (one block at a
time, never holding all 13B parameters) and the projections are dropped from
the model. Each table is computed with exactly the [M, t_dim] GEMM the model
would run, so cached and live results are bitwise equal.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from miowtion.h3 import model as h3_model
from miowtion.h3 import weights as h3_weights
from miowtion.train import lora
from miowtion.utils import progress

Table = list[tuple[torch.Tensor, ...]]


def timestep_key(timesteps: torch.Tensor) -> tuple[int, ...]:
    """Exact (bit-level) key of a sorted fp32 timestep set."""
    return tuple(struct.unpack('<I', struct.pack('<f', float(t)))[0]
                 for t in timesteps.tolist())


class AdalnTables:
    """Maps a TimestepState.timesteps set to its per-block tables."""

    def __init__(self):
        self._tables: dict[tuple[int, ...], Table] = {}

    def __contains__(self, timesteps: torch.Tensor) -> bool:
        return timestep_key(timesteps) in self._tables

    def get(self, timesteps: torch.Tensor) -> Table:
        key = timestep_key(timesteps)
        if key not in self._tables:
            raise KeyError(f'no AdaLN table for timesteps '
                           f'{timesteps.tolist()}; add it to the schedule '
                           'sets before dropping the projections')
        return self._tables[key]

    @torch.no_grad()
    def build(self, model: h3_model.H3DiT, transformer_dir: str,
              timestep_sets: Iterable[torch.Tensor], device: torch.device,
              adapter: dict | None = None) -> set[str]:
        """Computes tables for every set, reading one block at a time.

        Args:
            model: The DiT; only its (replicated) time embedder is used.
            transformer_dir: Checkpoint directory.
            timestep_sets: Distinct sorted fp32 timestep sets.
            device: Device of the tables (where the model runs).
            adapter: Optional merged LoRA (miowtion.train.lora.load_adapter);
                its `blocks.{i}.adaln_proj.linear` deltas are merged into the
                projection weights before the tables are computed.

        Returns:
            Adapter entries consumed here.
        """
        sets = {timestep_key(t): t.to(device) for t in timestep_sets}
        sets = {k: v for k, v in sets.items() if k not in self._tables}
        adapter = adapter or {}
        consumed = {name for name in adapter
                    if name.startswith('blocks.')
                    and name.endswith('.adaln_proj.linear')}
        if not sets:
            return consumed
        inputs = {k: F.silu(model.time_embedder(t)).to(torch.bfloat16)
                  for k, t in sets.items()}
        ckpt = h3_weights.Checkpoint(transformer_dir)
        tables = {k: [] for k in sets}
        hidden = model.config.hidden_size
        counter = progress.Progress(
            f'AdaLN tables ({len(sets)} timestep sets): blocks',
            model.config.num_layers, every=10)
        for i in range(model.config.num_layers):
            prefix = f'blocks.{i}.adaln_proj.linear.'
            weight = ckpt.read_rows(prefix + 'weight', None).to(device)
            bias = ckpt.read_rows(prefix + 'bias', None).to(device)
            name = f'blocks.{i}.adaln_proj.linear'
            if name in adapter:
                weight = lora.merged_weight(weight, *adapter[name])
            for key, adaln_input in inputs.items():
                m = adaln_input.shape[0]
                out = F.linear(adaln_input, weight, bias).view(m * 3,
                                                               6 * hidden)
                tables[key].append(tuple(out.chunk(6, dim=-1)))
            del weight, bias
            counter.update()
        self._tables.update(tables)
        return consumed

    def num_bytes(self) -> int:
        """Device memory held by the tables (6 chunks share one storage)."""
        return sum(block[0].untyped_storage().nbytes()
                   for table in self._tables.values() for block in table)
