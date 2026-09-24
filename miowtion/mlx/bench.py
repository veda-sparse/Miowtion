"""Measurements for MLX inference with NVMe offloading.

Synthetic weights with the real shapes (the real checkpoint is not needed to
time I/O or compute), memory accounting of the process and the system, and
the benchmarks behind docs/features/mlx_inference.md. Every benchmark checks
the available memory first and refuses to start when it is too low.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from miowtion.h3 import config as h3_config
from miowtion.mlx import block as mlx_block
from miowtion.mlx import offload
from miowtion.mlx import slab
from miowtion.mlx import sparse_attention
from miowtion.utils import progress

GB = 1e9

# Scale of the synthetic linear weights: keeps activations O(1) through a
# block (the value does not affect timings).
_SYNTHETIC_WEIGHT_STD = 0.02

# ---------------------------------------------------------------------------
# Memory accounting.

# struct rusage_info_v4 (<sys/resource.h>): a uuid, then uint64 fields.
_RUSAGE_V4_FIELDS = (
    'user_time system_time pkg_idle_wkups interrupt_wkups pageins '
    'wired_size resident_size phys_footprint proc_start_abstime '
    'proc_exit_abstime child_user_time child_system_time '
    'child_pkg_idle_wkups child_interrupt_wkups child_pageins '
    'child_elapsed_abstime diskio_bytesread diskio_byteswritten '
    'cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background '
    'cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated '
    'cpu_time_qos_user_interactive billed_system_time serviced_system_time '
    'logical_writes lifetime_max_phys_footprint instructions cycles '
    'billed_energy serviced_energy interval_max_phys_footprint runnable_time'
).split()
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in _RUSAGE_V4_FIELDS]


def process_memory() -> dict[str, float]:
    """This process's memory (GB) and bytes read from disk (macOS).

    phys_footprint is what Activity Monitor calls "Memory"; it includes the
    Metal buffers of MLX (unified memory) but not clean file-backed pages.
    disk_read is physical reads (page-cache hits do not count).
    """
    info = _RusageInfoV4()
    libc = ctypes.CDLL(None)
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4,
                            ctypes.byref(info)) != 0:
        raise OSError('proc_pid_rusage failed')
    return {'footprint': info.phys_footprint / GB,
            'peak_footprint': info.lifetime_max_phys_footprint / GB,
            'resident': info.resident_size / GB,
            'disk_read': info.diskio_bytesread / GB}


def system_memory() -> dict[str, float]:
    """System memory counters from vm_stat (GB) and swap used."""
    text = subprocess.run(['vm_stat'], capture_output=True, text=True,
                          check=True).stdout
    page = int(re.search(r'page size of (\d+) bytes', text).group(1))
    counts = {k.strip(): int(v.strip().rstrip('.'))
              for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.?$', text,
                                     re.MULTILINE)}

    def pages(name):
        return counts.get(name, 0) * page / GB

    swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'],
                          capture_output=True, text=True,
                          check=True).stdout
    used = re.search(r'used = ([\d.]+)M', swap)
    return {
        'free': pages('Pages free'),
        'inactive': pages('Pages inactive'),
        'speculative': pages('Pages speculative'),
        'file_backed': pages('File-backed pages'),
        'compressed': pages('Pages occupied by compressor'),
        'wired': pages('Pages wired down'),
        'available': (pages('Pages free') + pages('Pages inactive')
                      + pages('Pages speculative')
                      + pages('Pages purgeable')),
        'swap_used': float(used.group(1)) / 1e3 if used else 0.0,
        'swapouts': counts.get('Swapouts', 0),
    }


def require_headroom(needed_gb: float) -> dict[str, float]:
    """Raises unless `needed_gb` of memory is available now.

    Raises:
        RuntimeError: When the benchmark would push the system into swap.
    """
    mem = system_memory()
    if mem['available'] < needed_gb:
        raise RuntimeError(f'only {mem["available"]:.1f} GB available, '
                           f'benchmark needs {needed_gb:.1f} GB')
    return mem


def file_resident_fraction(path: str) -> float:
    """Fraction of a file's pages in the page cache (mincore)."""
    size = os.path.getsize(path)
    if size == 0:
        return 0.0
    libc = ctypes.CDLL(None)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_char_p]
    page = os.sysconf('SC_PAGE_SIZE')
    prot_read, map_shared = 1, 1
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError('mmap failed')
        try:
            vec = ctypes.create_string_buffer(-(-size // page))
            if libc.mincore(addr, size, vec) != 0:
                raise OSError('mincore failed')
            resident = np.frombuffer(vec.raw, dtype=np.uint8) & 1
            return float(resident.mean())
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Synthetic weights and inputs.


def synthetic_block(config: h3_config.H3Config,
                    seed: int) -> mlx_block.BlockWeights:
    """Random bf16 trunk weights with the real shapes (release layout)."""
    key = mx.random.key(seed)
    keys = mx.random.split(key, 8)
    hidden, inner = config.hidden_size, config.inner_dim

    def linear(k, out_dim, in_dim):
        return mlx_block.Linear((mx.random.normal((out_dim, in_dim), key=k)
                                 * _SYNTHETIC_WEIGHT_STD).astype(mx.bfloat16))

    def norm(k, dim):
        return (1.0 + 0.1 * mx.random.normal((dim,), key=k)).astype(
            mx.bfloat16)

    return mlx_block.BlockWeights(
        norm1=norm(keys[0], hidden), norm2=norm(keys[1], hidden),
        q_norm=norm(keys[2], config.head_dim),
        k_norm=norm(keys[3], config.head_dim),
        qkv=linear(keys[4], 3 * inner, hidden),
        out=linear(keys[5], hidden, inner),
        fc1=linear(keys[6], 2 * config.ffn_dim, hidden),
        fc2=linear(keys[7], hidden, config.ffn_dim))


def synthetic_inputs(config: h3_config.H3Config, seq_len: int, seed: int = 0
                     ) -> tuple[mx.array, list[mx.array], mx.array,
                                tuple[mx.array, mx.array]]:
    """x [S, hidden], 6 AdaLN tables [6, hidden], index [S], (cos, sin)."""
    key = mx.random.key(seed)
    kx, kt, ki, kp = mx.random.split(key, 4)
    x = mx.random.normal((seq_len, config.hidden_size), key=kx).astype(
        mx.bfloat16)
    tables = [(0.1 * t).astype(mx.bfloat16) for t in mx.split(
        mx.random.normal((36, config.hidden_size), key=kt), 6)]
    index = mx.random.randint(0, 6, (seq_len,), key=ki).astype(mx.int32)
    angles = mx.random.uniform(0, 2 * math.pi,
                               (seq_len, 1, config.rope_dim // 2), key=kp)
    angles = mx.concatenate([angles, angles], axis=-1)
    rope = (mx.cos(angles).astype(mx.bfloat16),
            mx.sin(angles).astype(mx.bfloat16))
    mx.eval(x, index, *tables, *rope)
    return x, tables, index, rope


def write_synthetic_slabs(out_dir: str, config: h3_config.H3Config,
                          blocks: int, bits: int = 0,
                          group_size: int = 64) -> list[str]:
    """Writes `blocks` synthetic slabs; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    counter = progress.Progress(f'synthetic slabs bits={bits}', blocks)
    paths = []
    for i in range(blocks):
        weights = synthetic_block(config, seed=i)
        if bits:
            weights = weights.quantize(bits, group_size)
        tensors = weights.to_tensors()
        mx.eval(*tensors.values())
        paths.append(slab.slab_path(out_dir, i))
        slab.write_slab(paths[-1], i, tensors, bits, group_size if bits else 0)
        del weights, tensors
        mx.clear_cache()
        counter.update(f'block {i}')
    return paths


def _write_safetensors_uncached(path: str,
                                tensors: dict[str, mx.array]) -> None:
    """safetensors writer that bypasses the page cache (F_NOCACHE).

    mx.save_safetensors writes through the page cache, so a file read right
    after being written would measure cached, not SSD, reads.
    """
    import fcntl

    header, offset = {}, 0
    for name, value in tensors.items():
        if value.dtype != mx.bfloat16:
            raise ValueError(f'{name}: only bf16 is written')
        header[name] = {'dtype': 'BF16', 'shape': list(value.shape),
                        'data_offsets': [offset, offset + value.nbytes]}
        offset += value.nbytes
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    with open(path, 'wb') as f:
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        f.write(len(raw).to_bytes(8, 'little') + raw)
        for value in tensors.values():
            f.write(memoryview(np.asarray(value.view(mx.uint16))).cast('B'))


def write_synthetic_safetensors(out_dir: str, config: h3_config.H3Config,
                                blocks: int, blocks_per_shard: int) -> None:
    """Synthetic checkpoint in the release key layout (trunk tensors only)."""
    os.makedirs(out_dir, exist_ok=True)
    weight_map = {}
    shards = -(-blocks // blocks_per_shard)
    counter = progress.Progress('synthetic safetensors shards', shards)
    for s in range(shards):
        tensors = {}
        for i in range(s * blocks_per_shard,
                       min(blocks, (s + 1) * blocks_per_shard)):
            for name, value in synthetic_block(config, i).to_tensors().items():
                tensors[f'blocks.{i}.{name}'] = value
        filename = f'model-{s + 1:05d}-of-{shards:05d}.safetensors'
        mx.eval(*tensors.values())
        _write_safetensors_uncached(os.path.join(out_dir, filename), tensors)
        weight_map.update({k: filename for k in tensors})
        del tensors
        mx.clear_cache()
        counter.update(filename)
    with open(os.path.join(out_dir, 'model.safetensors.index.json'),
              'w') as f:
        json.dump({'weight_map': weight_map}, f)


# ---------------------------------------------------------------------------
# I/O benchmarks.


def ssd_read(paths: Sequence[str], nocache: bool, threads: int,
             piece_bytes: int) -> dict[str, float]:
    """Sequential read throughput over whole files into one reused buffer.

    Args:
        paths: Files read front to back, one after the other.
        nocache: F_NOCACHE (bypass the page cache).
        threads: Reads in flight.
        piece_bytes: Bytes per read.
    """
    import concurrent.futures
    import fcntl

    buffer = memoryview(bytearray(piece_bytes * threads))
    before = process_memory()['disk_read']
    total = 0
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            if nocache:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            size = os.path.getsize(path)
            offsets = list(range(0, size, piece_bytes))
            for k in range(0, len(offsets), threads):
                batch = offsets[k:k + threads]
                jobs = [pool.submit(os.preadv, fd,
                                    [buffer[j * piece_bytes:
                                            (j + 1) * piece_bytes]], off)
                        for j, off in enumerate(batch)]
                total += sum(job.result() for job in jobs)
            os.close(fd)
    seconds = time.perf_counter() - start
    return {'gb': total / GB, 'seconds': seconds, 'gbps': total / GB / seconds,
            'disk_read_gb': process_memory()['disk_read'] - before}


def load_latency(source: str, directory: str, blocks: Sequence[int],
                 nocache: bool = True, threads: int = 4,
                 dequantize: bool = False) -> dict:
    """Per-block load time and memory for one offload option.

    Args:
        source: 'slab', or a SafetensorsSource mode ('mx_load', 'mmap').
        directory: Slab directory or safetensors checkpoint directory.
        blocks: Block indices, loaded in order; each block is dropped
            before the next one is loaded (the streaming pattern).
        nocache: Slab reads bypass the page cache.
        threads: Slab reads in flight.
        dequantize: Also time dequantizing a quantized slab to bf16.

    Returns:
        Per-block seconds and memory counters (see process_memory).
    """
    seconds, footprints, dequant = [], [], []
    disk_before = process_memory()['disk_read']
    if source == 'slab':
        paths = [slab.slab_path(directory, i) for i in blocks]
        reader = slab.SlabReader(paths, slots=1, nocache=nocache,
                                 threads=threads)
        for k in range(len(paths)):
            start = time.perf_counter()
            reader.read(k, 0)
            seconds.append(time.perf_counter() - start)
            if dequantize:
                start = time.perf_counter()
                weights = reader.weights(0).dequantize()
                mx.eval(*weights.to_tensors().values())
                dequant.append(time.perf_counter() - start)
                del weights
            footprints.append(process_memory()['footprint'])
        block_bytes = reader.layout.data_bytes
        reader.close()
    else:
        loader = offload.SafetensorsSource(directory, source)
        block_bytes = 0
        for i in blocks:
            start = time.perf_counter()
            weights = loader.load(i)
            seconds.append(time.perf_counter() - start)
            block_bytes = weights.nbytes
            del weights
            mx.clear_cache()
            footprints.append(process_memory()['footprint'])
        loader.close()
    mem = process_memory()
    result = {
        'source': source, 'nocache': nocache, 'threads': threads,
        'block_gb': block_bytes / GB, 'seconds': seconds,
        'gbps': [block_bytes / GB / s for s in seconds],
        'footprint_after_block': footprints,
        'peak_footprint': mem['peak_footprint'],
        'disk_read_gb': mem['disk_read'] - disk_before,
        'resident_after': mem['resident'],
    }
    if dequant:
        result['dequantize_seconds'] = dequant
    return result


# ---------------------------------------------------------------------------
# Compute benchmarks.


def block_flops(config: h3_config.H3Config, seq_len: int,
                density: float = 1.0) -> dict[str, float]:
    """Matmul FLOPs of one block forward; density 1.0 is dense attention."""
    params = (config.hidden_size * 3 * config.inner_dim
              + config.inner_dim * config.hidden_size
              + config.hidden_size * 2 * config.ffn_dim
              + config.ffn_dim * config.hidden_size)
    return {'gemm': 2.0 * seq_len * params,
            'attention': (4.0 * seq_len * seq_len * config.inner_dim
                          * density)}


def sparse_plan(seq_len: int, density: float, q_block: int,
                k_block: int, seed: int = 0) -> sparse_attention.SparsePlan:
    """A random Veda-shaped plan of the requested density (for timings).

    Which key tiles are selected does not change the cost, only how many, so
    a random selection with the real budget is enough to time the path.

    Raises:
        ValueError: If the tiles do not divide seq_len or the density rounds
            to an empty budget.
    """
    if seq_len % q_block or seq_len % k_block:
        raise ValueError(f'seq_len {seq_len} must be a multiple of q_block '
                         f'{q_block} and k_block {k_block}')
    n_k = seq_len // k_block
    budget = round(density * n_k)
    if not 1 <= budget <= n_k:
        raise ValueError(f'density {density} gives budget {budget} of {n_k} '
                         'key tiles')
    index = sparse_attention.random_index(seq_len // q_block, n_k, budget,
                                          seed=seed)
    mx.eval(index)
    return sparse_attention.SparsePlan(index, q_block, k_block)


def permuted_layer_plan(plan: sparse_attention.SparsePlan, num_heads: int,
                        groups: int, seq_len: int, seed: int = 0
                        ) -> sparse_attention.LayerPlan:
    """The same selection behind one random permutation per head group.

    A real tile plan gives a layer up to two tile shapes, i.e. up to two
    permutations of the sequence; this reproduces the *cost* of that (the
    gather / scatter around attention) without needing a clip geometry.

    Args:
        plan: The selection every group runs.
        num_heads: Heads of the layer; must be divisible by `groups`.
        groups: Head groups, i.e. distinct permutations.
        seq_len: Packed rows (all real, so the permutation is a bijection).
        seed: Permutation seed.

    Raises:
        ValueError: If the heads do not split evenly.
    """
    if groups < 1 or num_heads % groups:
        raise ValueError(f'{groups} groups do not divide {num_heads} heads')
    mx.random.seed(seed)
    per_group = num_heads // groups
    out = []
    for g in range(groups):
        gather = mx.random.permutation(seq_len).astype(mx.int32)
        scatter = mx.argsort(gather).astype(mx.int32)
        mx.eval(gather, scatter)
        out.append(sparse_attention.HeadGroupPlan(
            heads=tuple(range(g * per_group, (g + 1) * per_group)),
            gather=gather, scatter=scatter, plan=plan))
    return sparse_attention.LayerPlan(tuple(out), num_heads)


def attention_only(config: h3_config.H3Config, seq_len: int,
                   plan: sparse_attention.SparsePlan | None,
                   head_chunk: int | None = None, reps: int = 3) -> dict:
    """Times attention alone, dense (plan=None) or block-sparse.

    Isolating attention is what tells us whether the sparse path is worth its
    gather: the block forward mixes in GEMMs that sparsity cannot help.
    """
    heads = config.inner_dim // config.head_dim
    key = mx.random.key(0)
    q, k, v = [mx.random.normal((heads, seq_len, config.head_dim),
                                key=sub).astype(mx.bfloat16)
               for sub in mx.random.split(key, 3)]
    mx.eval(q, k, v)
    mx.reset_peak_memory()

    def run():
        if plan is None:
            out = mx.fast.scaled_dot_product_attention(
                q[None], k[None], v[None],
                scale=config.head_dim ** -0.5)[0]
        else:
            out = sparse_attention.block_sparse_attention(
                q, k, v, plan.index, q_block=plan.q_block,
                k_block=plan.k_block, scale=config.head_dim ** -0.5,
                head_chunk=head_chunk)
        mx.eval(out)

    run()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    seconds = min(times)
    density = 1.0 if plan is None else plan.density(seq_len)
    flops = 4.0 * heads * seq_len * seq_len * config.head_dim * density
    return {
        'seq_len': seq_len, 'density': density,
        'q_block': None if plan is None else plan.q_block,
        'k_block': None if plan is None else plan.k_block,
        'budget': None if plan is None else plan.budget,
        'head_chunk': head_chunk, 'seconds': seconds,
        'tflops': flops / seconds / 1e12,
        'gathered_gb': 0.0 if plan is None else sparse_attention.
        gathered_bytes(seq_len, heads, config.head_dim, plan.q_block,
                       density) / GB,
        'mlx_peak_gb': mx.get_peak_memory() / GB,
    }


def compute_block(config: h3_config.H3Config, seq_len: int, bits: int,
                  qmm: bool, options: mlx_block.BlockOptions,
                  reps: int = 1, profile: bool = True) -> dict:
    """Times one resident block forward on the default device.

    Args:
        config: Shapes.
        seq_len: Packed rows (all real).
        bits: 0 (bf16), 8 or 4.
        qmm: With bits, use quantized matmuls; otherwise dequantize the
            weights first (timed separately) and run bf16 GEMMs.
        options: Chunking.
        reps: Timed repetitions after one warm-up (the minimum is reported).
        profile: Also run one profiled pass (per-stage seconds).
    """
    weights = synthetic_block(config, seed=0)
    dequant_seconds = 0.0
    if bits:
        weights = weights.quantize(bits, 64)
        mx.eval(*weights.to_tensors().values())
        if not qmm:
            start = time.perf_counter()
            weights = weights.dequantize()
            mx.eval(*weights.to_tensors().values())
            dequant_seconds = time.perf_counter() - start
    mx.eval(*weights.to_tensors().values())
    x, tables, index, rope = synthetic_inputs(config, seq_len)
    mx.reset_peak_memory()

    def run():
        out = mlx_block.block_forward(x, weights, tables, index, rope,
                                      seq_len, config, options)
        mx.eval(out)
        return out

    run()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    stages = {}
    if profile:
        out = mlx_block.block_forward(x, weights, tables, index, rope,
                                      seq_len, config, options,
                                      profile=stages)
        mx.eval(out)
    density = (1.0 if options.sparse is None
               else options.sparse.density(seq_len))
    flops = block_flops(config, seq_len, density)
    seconds = min(times)
    return {
        'seq_len': seq_len, 'bits': bits, 'qmm': qmm, 'density': density,
        'head_chunk': options.head_chunk, 'row_chunk': options.row_chunk,
        'eval_chunks': options.eval_chunks, 'seconds': seconds,
        'tflops': (flops['gemm'] + flops['attention']) / seconds / 1e12,
        'attention_flop_share': flops['attention'] / (flops['gemm']
                                                      + flops['attention']),
        'stages': stages, 'dequantize_seconds': dequant_seconds,
        'mlx_peak_gb': mx.get_peak_memory() / GB,
        'peak_footprint': process_memory()['peak_footprint'],
    }


def streamed_run(slab_dir: str, blocks: int, passes: int, seq_len: int,
                 config: h3_config.H3Config,
                 options: mlx_block.BlockOptions, depth: int = 1,
                 nocache: bool = True, dequantize: bool = False) -> dict:
    """Runs blocks from slabs with prefetch, as a denoising step would.

    Args:
        slab_dir: Directory of block_XXX.slab files.
        blocks: Slabs used (block k of the model reads slab k % blocks).
        passes: Model "steps"; blocks * passes block forwards in total.
        seq_len: Packed rows.
        config: Shapes.
        options: Chunking.
        depth: Blocks read ahead.
        nocache: Bypass the page cache.
        dequantize: Quantized slabs are dequantized to bf16 before the
            forward (otherwise quantized matmuls).
    """
    order = [k % blocks for k in range(blocks * passes)]
    reader = slab.SlabReader([slab.slab_path(slab_dir, i)
                              for i in range(blocks)], slots=depth + 1,
                             nocache=nocache)
    x, tables, index, rope = synthetic_inputs(config, seq_len)
    disk_before = process_memory()['disk_read']
    prefetcher = slab.BlockPrefetcher(reader, order, depth)
    counter = progress.Progress(f'streamed blocks (S={seq_len})', len(order),
                                every=max(1, len(order) // 8))
    compute = []
    start = time.perf_counter()
    for _, weights in prefetcher:
        t0 = time.perf_counter()
        if dequantize and weights.quantization[0]:
            weights = weights.dequantize()
        x = mlx_block.block_forward(x, weights, tables, index, rope, seq_len,
                                    config, options)
        mx.eval(x)
        compute.append(time.perf_counter() - t0)
        del weights
        counter.update()
    total = time.perf_counter() - start
    mem = process_memory()
    reader.close()
    return {
        'seq_len': seq_len, 'blocks': len(order), 'depth': depth,
        'bits': reader.layout.bits, 'dequantize': dequantize,
        'block_gb': reader.layout.data_bytes / GB,
        'seconds': total, 'per_block': total / len(order),
        'compute': compute, 'read': prefetcher.stats.read,
        'wait': prefetcher.stats.wait,
        'peak_footprint': mem['peak_footprint'],
        'disk_read_gb': mem['disk_read'] - disk_before,
        'finite': bool(mx.all(mx.isfinite(x)).item()),
    }
