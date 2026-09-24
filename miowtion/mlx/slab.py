"""Per-block slab files: NVMe offloading of the DiT trunk for MLX.

A slab holds the trunk tensors of one block (release layout, see
miowtion.mlx.block), optionally MLX-affine-quantized, as one contiguous file:

    magic 'MIOSLAB1' | uint64 LE header length | JSON header | pad |
    tensor data, every tensor at a SLAB_ALIGNMENT-aligned offset

Slabs are written once from the checkpoint. At inference a SlabReader reads
a block with plain positional reads straight into preallocated MLX buffers
(no intermediate copy, no MLX call off the main thread), bypassing the page
cache by default: 50 blocks do not fit in memory, so caching them only
evicts other data. A BlockPrefetcher reads block i+1 on a background thread
while block i computes; host memory for weights is bounded by the number of
slots.

Invariant: a slot is overwritten only after the GPU finished every
computation that read it (the prefetcher synchronizes before reusing it).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import fcntl
import json
import os
import struct
import time
from collections.abc import Iterator, Mapping, Sequence

import mlx.core as mx
import numpy as np

from miowtion.mlx import block as mlx_block
from miowtion.utils import progress

MAGIC = b'MIOSLAB1'
FORMAT_VERSION = 1
# Tensor data alignment: the 16 KiB VM page of Apple silicon, so that
# uncached reads cover whole pages.
SLAB_ALIGNMENT = 16384
# Bytes per positional read. Several reads in flight keep the SSD queue
# busy; 16 MiB pieces over 4 threads reach the sequential read bandwidth
# (see docs/features/mlx_inference.md).
DEFAULT_PIECE_BYTES = 16 << 20
DEFAULT_READ_THREADS = 4

_DTYPES = {
    'bfloat16': (mx.bfloat16, mx.uint16),
    'float16': (mx.float16, mx.uint16),
    'float32': (mx.float32, mx.float32),
    'uint32': (mx.uint32, mx.uint32),
}
_DTYPE_NAMES = {mx.bfloat16: 'bfloat16', mx.float16: 'float16',
                mx.float32: 'float32', mx.uint32: 'uint32'}


def _align(n: int) -> int:
    return -(-n // SLAB_ALIGNMENT) * SLAB_ALIGNMENT


@dataclasses.dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclasses.dataclass(frozen=True)
class SlabLayout:
    """Parsed slab header.

    Attributes:
        block: Block index.
        bits: Quantization bits of the linear weights (0: bf16).
        group_size: Quantization group (0: bf16).
        tensors: Entries in file order.
    """

    block: int
    bits: int
    group_size: int
    tensors: tuple[TensorEntry, ...]

    @property
    def file_bytes(self) -> int:
        last = self.tensors[-1]
        return last.offset + last.nbytes

    @property
    def data_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def same_shapes(self, other: SlabLayout) -> bool:
        """True when both slabs fit the same buffers (block may differ)."""
        return (self.bits, self.group_size, self.tensors) == (
            other.bits, other.group_size, other.tensors)


def write_slab(path: str, block: int, tensors: Mapping[str, mx.array],
               bits: int = 0, group_size: int = 0) -> SlabLayout:
    """Writes one block's tensors (in the given order) as a slab.

    Args:
        path: Output file.
        block: Block index recorded in the header.
        tensors: Name -> evaluated array (see BlockWeights.to_tensors).
        bits: Quantization bits of the linear weights (0: bf16).
        group_size: Quantization group.

    Raises:
        ValueError: On an unsupported dtype.
    """
    entries = []
    header_guess = 4096 + 256 * len(tensors)
    offset = _align(len(MAGIC) + 8 + header_guess)
    for name, value in tensors.items():
        if value.dtype not in _DTYPE_NAMES:
            raise ValueError(f'{name}: unsupported dtype {value.dtype}')
        entries.append(TensorEntry(name, _DTYPE_NAMES[value.dtype],
                                   tuple(value.shape), offset, value.nbytes))
        offset = _align(offset + value.nbytes)
    header = json.dumps({
        'version': FORMAT_VERSION, 'block': block, 'bits': bits,
        'group_size': group_size,
        'tensors': [dataclasses.asdict(e) for e in entries]}).encode()
    if len(MAGIC) + 8 + len(header) > entries[0].offset:
        raise ValueError('slab header too large')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        # Converting a checkpoint writes tens of GB that will not be read
        # back soon; caching them would only evict other data.
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        f.write(MAGIC + struct.pack('<Q', len(header)) + header)
        for entry, value in zip(entries, tensors.values()):
            f.seek(entry.offset)
            host = np.asarray(value.view(_DTYPES[entry.dtype][1]))
            f.write(memoryview(np.ascontiguousarray(host)).cast('B'))
    os.replace(tmp, path)
    return SlabLayout(block, bits, group_size, tuple(entries))


def read_layout(path: str) -> SlabLayout:
    """Parses a slab header.

    Raises:
        ValueError: On a bad magic or version.
    """
    with open(path, 'rb') as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f'{path}: not a slab (magic {magic!r})')
        (length,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
    if header['version'] != FORMAT_VERSION:
        raise ValueError(f'{path}: slab version {header["version"]}')
    tensors = tuple(TensorEntry(t['name'], t['dtype'], tuple(t['shape']),
                                t['offset'], t['nbytes'])
                    for t in header['tensors'])
    return SlabLayout(header['block'], header['bits'], header['group_size'],
                      tensors)


def slab_path(directory: str, block: int) -> str:
    return os.path.join(directory, f'block_{block:03d}.slab')


def _writable_view(a: mx.array, entry: TensorEntry) -> memoryview:
    """Byte view of an evaluated MLX buffer (unified memory, no copy)."""
    view = np.asarray(a.view(_DTYPES[entry.dtype][1]))
    if not (view.flags.writeable and view.flags.c_contiguous
            and view.nbytes == entry.nbytes):
        raise RuntimeError(f'{entry.name}: MLX buffer is not a writable '
                           'contiguous allocation')
    return memoryview(view).cast('B')


class SlabReader:
    """Reads slabs into `slots` preallocated sets of MLX buffers.

    All slabs must share one layout (same shapes and quantization).
    """

    def __init__(self, paths: Sequence[str], slots: int = 2,
                 nocache: bool = True, threads: int = DEFAULT_READ_THREADS,
                 piece_bytes: int = DEFAULT_PIECE_BYTES):
        """Opens the slabs and allocates the buffers.

        Args:
            paths: Slab file per block index (paths[i] holds block i).
            slots: Buffer sets (blocks held in memory at once).
            nocache: Bypass the page cache (F_NOCACHE).
            threads: Concurrent positional reads per block.
            piece_bytes: Bytes per read.

        Raises:
            ValueError: On inconsistent layouts or bad arguments.
        """
        if slots < 1 or threads < 1 or piece_bytes < SLAB_ALIGNMENT:
            raise ValueError('slots, threads >= 1 and piece_bytes >= '
                             f'{SLAB_ALIGNMENT} required')
        self.layouts = [read_layout(p) for p in paths]
        self.layout = self.layouts[0]
        for p, layout in zip(paths, self.layouts):
            if not layout.same_shapes(self.layout):
                raise ValueError(f'{p}: layout differs from {paths[0]}')
        self.paths = list(paths)
        self.nocache = nocache
        self.piece_bytes = piece_bytes
        self._pool = concurrent.futures.ThreadPoolExecutor(threads)
        self._fds: dict[int, int] = {}
        self.slots = []
        for _ in range(slots):
            arrays = {e.name: mx.zeros(e.shape, _DTYPES[e.dtype][0])
                      for e in self.layout.tensors}
            mx.eval(*arrays.values())
            views = {e.name: _writable_view(arrays[e.name], e)
                     for e in self.layout.tensors}
            self.slots.append((arrays, views))

    @property
    def slot_bytes(self) -> int:
        return self.layout.data_bytes

    def _fd(self, index: int) -> int:
        if index not in self._fds:
            fd = os.open(self.paths[index], os.O_RDONLY)
            if self.nocache:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self._fds[index] = fd
        return self._fds[index]

    def read(self, index: int, slot: int) -> None:
        """Reads block `index` into buffer set `slot` (blocking).

        The caller must guarantee that no pending MLX computation reads the
        slot.
        """
        fd = self._fd(index)
        _, views = self.slots[slot]
        jobs = []
        for entry in self.layouts[index].tensors:
            view = views[entry.name]
            for start in range(0, entry.nbytes, self.piece_bytes):
                piece = view[start:start + self.piece_bytes]
                jobs.append(self._pool.submit(_pread_exact, fd, piece,
                                              entry.offset + start))
        for job in jobs:
            job.result()

    def weights(self, slot: int) -> mlx_block.BlockWeights:
        """The block currently held by `slot` (views of its buffers)."""
        arrays, _ = self.slots[slot]
        return mlx_block.BlockWeights.from_tensors(
            arrays, group_size=self.layout.group_size, bits=self.layout.bits)

    def close(self) -> None:
        self._pool.shutdown()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def _pread_exact(fd: int, buffer: memoryview, offset: int) -> None:
    done = 0
    while done < len(buffer):
        n = os.preadv(fd, [buffer[done:]], offset + done)
        if n <= 0:
            raise OSError(f'short read at offset {offset + done}')
        done += n


@dataclasses.dataclass
class StreamStats:
    """Per-block timings of a BlockPrefetcher pass (seconds).

    Attributes:
        read: Background read time of every block.
        wait: Time the consumer waited for every block (0 when the read was
            hidden behind the previous block's compute).
    """

    read: list[float] = dataclasses.field(default_factory=list)
    wait: list[float] = dataclasses.field(default_factory=list)


class BlockPrefetcher:
    """Iterates blocks in order, reading `depth` blocks ahead.

    Uses depth + 1 reader slots. The consumer must finish (evaluate) all
    work on a block before asking for the next one; the prefetcher then
    synchronizes the GPU and refills that block's slot.
    """

    def __init__(self, reader: SlabReader, order: Sequence[int],
                 depth: int = 1):
        if depth < 1 or len(reader.slots) < depth + 1:
            raise ValueError(f'depth {depth} needs {depth + 1} reader slots, '
                             f'reader has {len(reader.slots)}')
        self.reader = reader
        self.order = list(order)
        self.depth = depth
        self.stats = StreamStats()
        self._io = concurrent.futures.ThreadPoolExecutor(1)

    def _submit(self, k: int) -> concurrent.futures.Future:
        slot = k % (self.depth + 1)

        def job():
            start = time.perf_counter()
            self.reader.read(self.order[k], slot)
            return time.perf_counter() - start

        return self._io.submit(job)

    def __iter__(self) -> Iterator[tuple[int, mlx_block.BlockWeights]]:
        futures = {k: self._submit(k)
                   for k in range(min(self.depth, len(self.order)))}
        for k, index in enumerate(self.order):
            start = time.perf_counter()
            self.stats.read.append(futures.pop(k).result())
            self.stats.wait.append(time.perf_counter() - start)
            ahead = k + self.depth
            if ahead < len(self.order):
                # Slot of block k - 1: its consumer is done, but GPU work
                # may still be queued.
                mx.synchronize()
                futures[ahead] = self._submit(ahead)
            yield index, self.reader.weights(k % (self.depth + 1))
        self._io.shutdown()


def convert_checkpoint(checkpoint, out_dir: str, blocks: Sequence[int],
                       bits: int = 0, group_size: int = 64) -> None:
    """Writes one slab per block from a released checkpoint.

    The fused QKV keeps the release's per-head interleaved row order (the
    MLX block reads it directly).

    Args:
        checkpoint: miowtion.h3.weights.Checkpoint of `<variant>/transformer`.
        out_dir: Output directory (block_XXX.slab files).
        blocks: Block indices to convert.
        bits: 0 (bf16), 8 or 4 (MLX affine quantization of the linears).
        group_size: Quantization group.
    """
    from miowtion.mlx import interop  # torch is only needed to convert.

    os.makedirs(out_dir, exist_ok=True)
    names = [f'{n}.weight' for n in mlx_block.NORM_NAMES
             + mlx_block.LINEAR_NAMES]
    counter = progress.Progress(f'write slabs (bits={bits})', len(blocks))
    for i in blocks:
        tensors = {n: interop.from_torch(
            checkpoint.read_rows(f'blocks.{i}.{n}', None)) for n in names}
        weights = mlx_block.BlockWeights.from_tensors(tensors)
        if bits:
            weights = weights.quantize(bits, group_size)
        arrays = weights.to_tensors()
        mx.eval(*arrays.values())
        write_slab(slab_path(out_dir, i), i, arrays, bits,
                   group_size if bits else 0)
        counter.update(f'block {i}')
