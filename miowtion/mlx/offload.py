"""Streaming block weights straight from the released safetensors shards.

The alternative to slab files (miowtion.mlx.slab) that needs no conversion.
Two modes:

- 'mx_load': `mx.load` on the shard. MLX loads safetensors lazily (only the
  evaluated tensors are read), through the page cache. The returned dict is
  dropped after every block: an evaluated array keeps its data, so caching
  the dict would accumulate every block read so far.
- 'mmap': the shard is memory-mapped once and each tensor is copied into an
  MLX buffer. Touched pages stay mapped (resident, file-backed) until the
  kernel evicts them.

Both go through the page cache, which cannot hold 50 blocks; see
the module docstring for the measured behaviour.
"""

from __future__ import annotations

import json
import mmap
import os
import struct

import mlx.core as mx
import numpy as np

from miowtion.mlx import block as mlx_block

MODES = ('mx_load', 'mmap')

_ST_DTYPES = {'BF16': (np.uint16, mx.bfloat16), 'F32': (np.float32, None),
              'F16': (np.float16, None)}


def trunk_names(index: int) -> list[str]:
    """Release keys of block `index`'s trunk tensors."""
    return [f'blocks.{index}.{n}.weight'
            for n in mlx_block.NORM_NAMES + mlx_block.LINEAR_NAMES]


class SafetensorsSource:
    """Reads one block at a time from `<variant>/transformer`."""

    def __init__(self, transformer_dir: str, mode: str = 'mx_load'):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
        with open(os.path.join(transformer_dir,
                               'model.safetensors.index.json')) as f:
            self._key_to_file = json.load(f)['weight_map']
        self._dir = transformer_dir
        self.mode = mode
        self._maps: dict[str, tuple[mmap.mmap, dict, int]] = {}

    def _mapped(self, filename: str) -> tuple[mmap.mmap, dict, int]:
        if filename not in self._maps:
            with open(os.path.join(self._dir, filename), 'rb') as f:
                (length,) = struct.unpack('<Q', f.read(8))
                header = json.loads(f.read(length))
                mapped = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            self._maps[filename] = (mapped, header, 8 + length)
        return self._maps[filename]

    def _read_mmap(self, key: str) -> mx.array:
        mapped, header, base = self._mapped(self._key_to_file[key])
        meta = header[key]
        np_dtype, mx_view = _ST_DTYPES[meta['dtype']]
        start, stop = meta['data_offsets']
        host = np.frombuffer(mapped, dtype=np_dtype,
                             count=(stop - start) // np.dtype(np_dtype).itemsize,
                             offset=base + start).reshape(meta['shape'])
        value = mx.array(host)
        return value.view(mx_view) if mx_view is not None else value

    def load(self, index: int) -> mlx_block.BlockWeights:
        """Block `index`'s trunk weights, evaluated (read from disk now)."""
        names = trunk_names(index)
        prefix = f'blocks.{index}.'
        if self.mode == 'mx_load':
            tensors = {}
            for filename in sorted({self._key_to_file[n] for n in names}):
                loaded = mx.load(os.path.join(self._dir, filename))
                tensors.update((n, loaded[n]) for n in names
                               if self._key_to_file[n] == filename)
                del loaded
        else:
            tensors = {n: self._read_mmap(n) for n in names}
        mx.eval(*tensors.values())
        return mlx_block.BlockWeights.from_tensors(
            {n[len(prefix):]: v for n, v in tensors.items()})

    def close(self) -> None:
        for mapped, _, _ in self._maps.values():
            mapped.close()
        self._maps.clear()
