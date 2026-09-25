"""Released H3 checkpoint -> MLX tensors and per-block slab files.

Reads the sharded safetensors of `<variant>/transformer` directly with mmap
(no torch, no safetensors package) and hands out MLX arrays in the layout
the MLX block expects:

- the four trunk linears, with q, k and v fused into the per-head
  interleaved row order when the release stores them separately (see
  miowtion.h3.release);
- the AdaLN projection of a block, read separately because inference
  consumes precomputed tables and never streams these 520 MB per block;
- the non-trunk tensors (embedders, token refiner, output heads), small
  enough to keep resident.

Only one tensor is materialized at a time: the caller decides what stays
alive. Writing slabs (write_trunk_slabs) therefore peaks at one block.
"""

from __future__ import annotations

import glob
import json
import mmap
import os
import struct
from collections.abc import Mapping, Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.h3 import release as h3_release
from miowtion.mlx import block as mlx_block
from miowtion.mlx import slab as mlx_slab
from miowtion.utils import progress

# safetensors dtype -> (numpy dtype to read, MLX dtype to view as). bf16 has
# no numpy dtype, so it travels as uint16 and is reinterpreted.
_ST_DTYPES = {
    'BF16': (np.uint16, mx.bfloat16),
    'F32': (np.float32, None),
    'F16': (np.float16, None),
}


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, 'rb') as f:
        (length,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
    header.pop('__metadata__', None)
    return header, 8 + length


class ReleaseReader:
    """Random access to a sharded safetensors checkpoint as MLX arrays.

    Attributes:
        schema: The detected release layout (miowtion.h3.release.SCHEMAS).
        config: The architecture read from config.json.
    """

    def __init__(self, transformer_dir: str):
        """Opens `<variant>/transformer`.

        The index file is optional: the shard headers are the ground truth
        and are cheap to read (one seek each).

        Raises:
            FileNotFoundError: When the directory holds no safetensors.
        """
        shards = sorted(glob.glob(os.path.join(transformer_dir,
                                               '*.safetensors')))
        if not shards:
            raise FileNotFoundError(f'no safetensors in {transformer_dir}')
        self._dir = transformer_dir
        self._headers: dict[str, dict] = {}
        self._bases: dict[str, int] = {}
        self._key_to_file: dict[str, str] = {}
        for path in shards:
            header, base = _read_header(path)
            self._headers[path] = header
            self._bases[path] = base
            self._key_to_file.update({k: path for k in header})
        self._maps: dict[str, mmap.mmap] = {}
        self.config = h3_config.H3Config.from_pretrained(transformer_dir)
        self.schema = h3_release.detect_schema(self._key_to_file)

    def keys(self) -> set[str]:
        return set(self._key_to_file)

    def check_complete(self) -> None:
        """Raises KeyError when a tensor the DiT needs is absent."""
        h3_release.check_complete(self.schema, self._key_to_file,
                                  self.config.num_layers,
                                  self.config.num_refiner_layers)

    def _map(self, path: str) -> mmap.mmap:
        if path not in self._maps:
            with open(path, 'rb') as f:
                self._maps[path] = mmap.mmap(f.fileno(), 0,
                                             prot=mmap.PROT_READ)
        return self._maps[path]

    def read(self, key: str) -> mx.array:
        """One tensor, copied out of the mapping into an MLX buffer.

        Raises:
            KeyError: When the key is absent.
            ValueError: On an unsupported dtype.
        """
        if key not in self._key_to_file:
            raise KeyError(f'{key} is not in {self._dir}')
        path = self._key_to_file[key]
        meta = self._headers[path][key]
        if meta['dtype'] not in _ST_DTYPES:
            raise ValueError(f'{key}: unsupported dtype {meta["dtype"]}')
        np_dtype, view = _ST_DTYPES[meta['dtype']]
        start, stop = meta['data_offsets']
        itemsize = np.dtype(np_dtype).itemsize
        host = np.frombuffer(self._map(path), dtype=np_dtype,
                             count=(stop - start) // itemsize,
                             offset=self._bases[path] + start)
        value = mx.array(host.reshape(meta['shape']))
        return value.view(view) if view is not None else value

    def close(self) -> None:
        for mapped in self._maps.values():
            mapped.close()
        self._maps.clear()

    def __enter__(self) -> ReleaseReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def fuse_qkv(q: mx.array, k: mx.array, v: mx.array, num_heads: int,
             head_dim: int) -> mx.array:
    """Separate projections -> one per-head interleaved weight.

    Row h * 3 * head_dim + part * head_dim + d of the result is row
    h * head_dim + d of the part's projection, which is the layout the MLX
    block slices by head group.

    Args:
        q: [num_heads * head_dim, in] q projection.
        k: Same shape.
        v: Same shape.
        num_heads: Heads.
        head_dim: Channels per head.

    Returns:
        [3 * num_heads * head_dim, in] with q's dtype.

    Raises:
        ValueError: On a shape mismatch.
    """
    rows = num_heads * head_dim
    for name, part in (('q', q), ('k', k), ('v', v)):
        if part.shape != (rows, q.shape[1]):
            raise ValueError(f'{name} has shape {part.shape}, expected '
                             f'{(rows, q.shape[1])}')
    stacked = mx.stack([p.reshape(num_heads, head_dim, -1)
                        for p in (q, k, v)], axis=1)
    return stacked.reshape(3 * rows, -1)


def _gather(reader: ReleaseReader, keys: Mapping[str, tuple[str, ...]],
            config: h3_config.H3Config) -> dict[str, mx.array]:
    """Reads a name -> release keys mapping, fusing q/k/v triples."""
    out = {}
    for name, sources in keys.items():
        if len(sources) == 1:
            out[name] = reader.read(sources[0])
        elif len(sources) == 3:
            parts = [reader.read(s) for s in sources]
            out[name] = fuse_qkv(*parts, config.num_heads, config.head_dim)
            del parts
        else:
            raise ValueError(f'{name}: expected 1 or 3 sources, got '
                             f'{len(sources)}')
    return out


def trunk_tensors(reader: ReleaseReader, index: int) -> dict[str, mx.array]:
    """Trunk tensors of block `index`, keyed for BlockWeights.from_tensors."""
    keys = h3_release.block_keys(reader.schema, index)
    return _gather(reader, keys, reader.config)


def adaln_tensors(reader: ReleaseReader, index: int) -> dict[str, mx.array]:
    """The AdaLN projection of block `index` (`weight`, `bias`)."""
    keys = h3_release.block_keys(reader.schema, index, adaln=True)
    keys = {n.split('.')[-1]: k for n, k in keys.items()
            if n in h3_release.ADALN_NAMES}
    return _gather(reader, keys, reader.config)


def refiner_tensors(reader: ReleaseReader, index: int) -> dict[str, mx.array]:
    """Trunk tensors of token refiner block `index`."""
    return _gather(reader, h3_release.refiner_keys(reader.schema, index),
                   reader.config)


def non_trunk_tensors(reader: ReleaseReader) -> dict[str, mx.array]:
    """Embedders, time embedder, refiner final norm and output heads."""
    return _gather(reader, h3_release.non_trunk_keys(reader.schema),
                   reader.config)


def write_trunk_slabs(reader: ReleaseReader, out_dir: str,
                      blocks: Sequence[int], bits: int = 0,
                      group_size: int = 64) -> None:
    """Writes one slab per trunk block (see miowtion.mlx.slab).

    Args:
        reader: An open ReleaseReader.
        out_dir: Destination directory (created if missing).
        blocks: Block indices.
        bits: 0 (bf16), 8 or 4 (MLX affine quantization of the linears).
        group_size: Quantization group.
    """
    os.makedirs(out_dir, exist_ok=True)
    counter = progress.Progress(f'write slabs (bits={bits})', len(blocks))
    for index in blocks:
        weights = mlx_block.BlockWeights.from_tensors(
            trunk_tensors(reader, index))
        if bits:
            weights = weights.quantize(bits, group_size)
        tensors = weights.to_tensors()
        mx.eval(*tensors.values())
        mlx_slab.write_slab(mlx_slab.slab_path(out_dir, index), index,
                            tensors, bits, group_size if bits else 0)
        del weights, tensors
        counter.update(f'block {index}')
