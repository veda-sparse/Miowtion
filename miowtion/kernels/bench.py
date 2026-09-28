"""Block-sparse attention benchmark and correctness probe.

Pattern: for every (head, query block) a random set of key blocks of size
round(density * n_blocks) is kept, the diagonal block always included.
Blocks are 128 x 128 and all full (no padding). Training and inference see a
different pattern on every call, so the benchmark cycles through several
random patterns and reports two times per sparse kernel:
  * kernel: the per-pattern metadata (indices / BlockMask) is prebuilt;
  * +prep:  the metadata is rebuilt from the bool block mask on every call.

Efficiency = dense_time * density / sparse_time (1.0 = perfect skipping).
The dense reference time is the fastest correct dense kernel. Every kernel is
checked against an fp32 reference, which also catches kernels that silently
compute dense attention.

Optional third-party kernel: FastVideo's Triton block-sparse forward (64-row
blocks; each 128 block is expanded to 2x2), loaded from the directory in
$MIOWTION_FASTVIDEO_KERNEL (fastvideo-kernel/python/fastvideo_kernel).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F

from miowtion.kernels import fa4
from miowtion.veda import tiling

_TILE = tiling.TILE_SIZE

# Dense bf16 tensor-core peak, FLOP/s, per torch.cuda.get_device_name(). These
# are the *dense* figures; the 2:1-structured-sparsity numbers the vendors also
# publish are twice these and unreachable here, so quoting them would halve
# every MFU. NB RTX PRO 6000's headline "1 PFLOPS BF16" is the sparse figure.
# TODO(#mfu): miowtion.h3.flops has its own PEAK_BF16_FLOPS for whole-model
# MFU; fold the two tables into one place once that module settles.
PEAK_BF16_DENSE_FLOPS = {
    'NVIDIA GeForce RTX 4090': 165.2e12,
    'NVIDIA RTX PRO 6000 Blackwell Server Edition': 503.8e12,
    'NVIDIA RTX PRO 6000 Blackwell Workstation Edition': 503.8e12,
    'NVIDIA H100 80GB HBM3': 989.4e12,
    'NVIDIA A100-SXM4-80GB': 312.0e12,
}


def attention_flops(seq: int, heads: int, head_dim: int,
                    density: float) -> float:
    """Useful FLOPs of one forward pass: QK^T and PV, 2 FLOP per MAC.

    Skipped blocks are not counted, so a sparse kernel and a dense one are
    credited only for the arithmetic they actually had to do. Softmax
    exponentials are excluded (they are not tensor-core work).
    """
    return 4.0 * seq * seq * heads * head_dim * density


def gemm_ceiling_tflops(device: torch.device, size: int = 8192,
                        iters: int = 20) -> float:
    """Measured cuBLAS bf16 GEMM throughput: the achievable matmul ceiling.

    FA4's SM120 path issues SM80-era mma.sync, not Blackwell's 5th-gen tensor
    core instructions, so this is the honest yardstick for how much of the
    GPU's matmul throughput an attention kernel is leaving unused, separate
    from the vendor's peak.
    """
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3 / iters
    return 2.0 * size**3 / seconds / 1e12


@dataclasses.dataclass
class Result:
    name: str
    kernel_ms: float = float('nan')
    prep_ms: float = float('nan')
    max_err: float | None = None
    efficiency: float | None = None
    efficiency_prep: float | None = None
    tflops: float | None = None
    mfu: float | None = None
    gemm_frac: float | None = None
    note: str = ''


def random_block_mask(heads: int, n_blocks: int, density: float,
                      seed: int, device: torch.device) -> torch.Tensor:
    """[H, n, n] bool with exactly round(density * n) blocks per row."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    keep = max(1, round(density * n_blocks))
    scores = torch.rand(heads, n_blocks, n_blocks, generator=gen)
    diag = torch.arange(n_blocks)
    scores[:, diag, diag] = 2.0
    idx = scores.topk(keep, dim=-1).indices
    mask = torch.zeros(heads, n_blocks, n_blocks, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask.to(device)


def _time_ms(fn: Callable[[int], object], calls: int, warmup: int) -> float:
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(calls):
        fn(i)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / calls


def _reference_rows(q, k, v, mask, q_blocks):
    """fp32 masked attention for a few query blocks. q,k,v: [S, H, D]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    outs = []
    for i in q_blocks:
        rows = slice(i * _TILE, (i + 1) * _TILE)
        s = torch.einsum('qhd,khd->hqk', q[rows].float(), k.float()) * scale
        token_mask = mask[:, i].repeat_interleave(_TILE, -1)[:, None, :]
        p = torch.softmax(s.masked_fill(~token_mask, float('-inf')), -1)
        outs.append(torch.einsum('hqk,khd->qhd', p, v.float()))
    return torch.cat(outs)


def _max_err(out, ref_rows, q_blocks):
    got = torch.cat([out[i * _TILE:(i + 1) * _TILE] for i in q_blocks])
    return (got.float() - ref_rows).abs().max().item()


def _left_packed(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[H, n, n] bool -> ([1,H,n] counts, [1,H,n,n] left-packed indices)."""
    heads, n, _ = mask.shape
    cols = torch.arange(n, device=mask.device).expand(heads, n, n)
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)
    idx = torch.gather(cols, -1, order)[None].to(torch.int32).contiguous()
    return mask.sum(-1)[None].to(torch.int32), idx


class _Backend:
    """prepare(mask) -> meta; run(meta) -> [S, H, D] output."""

    name = ''
    sparse = True

    def prepare(self, mask):
        return mask

    def run(self, meta):
        raise NotImplementedError


class _SdpaDense(_Backend):
    name, sparse = 'sdpa_flash_dense', False

    def __init__(self, q, k, v):
        self.q, self.k, self.v = (t.transpose(0, 1)[None] for t in (q, k, v))

    def run(self, meta):
        del meta
        with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                self.q, self.k, self.v)[0].transpose(0, 1)


class _Fa4Dense(_Backend):
    name, sparse = 'fa4_dense', False

    def __init__(self, q, k, v):
        self.fn = fa4.interface().flash_attn_func
        self.q, self.k, self.v = q[None], k[None], v[None]

    def run(self, meta):
        del meta
        return self.fn(self.q, self.k, self.v)[0][0]


class _Fa4Sparse(_Backend):
    name = 'fa4_block_sparse'

    def __init__(self, q, k, v):
        _, _, block_sparsity, iface, _ = fa4._modules()  # pylint: disable=protected-access
        self.bs, self.fn = block_sparsity, iface.flash_attn_func
        self.q, self.k, self.v = q[None], k[None], v[None]

    def prepare(self, mask):
        cnt, idx = _left_packed(mask)
        # All blocks are full: everything goes into the full list.
        return self.bs.BlockSparseTensorsTorch(
            mask_block_cnt=torch.zeros_like(cnt), mask_block_idx=idx,
            full_block_cnt=cnt, full_block_idx=idx,
            block_size=(_TILE, _TILE))

    def run(self, meta):
        return self.fn(self.q, self.k, self.v, block_sparse_tensors=meta)[0][0]


class _Fa4DenseMask(_Backend):
    """SM8x / SM120 only: the vendored DenseBlockMaskTorch input.

    No index lists are built; the kernels read the block mask directly.
    """

    name = 'fa4_dense_block_mask'

    def __init__(self, q, k, v):
        if (torch.cuda.get_device_capability(q.device)[0]
                not in fa4.PATCHED_MAJOR_ARCHS):
            raise NotImplementedError(
                'DenseBlockMaskTorch is SM8x / SM120 only')
        _, _, block_sparsity, iface, _ = fa4._modules()  # pylint: disable=protected-access
        self.bs, self.fn = block_sparsity, iface.flash_attn_func
        self.q, self.k, self.v = q[None], k[None], v[None]

    def prepare(self, mask):
        return self.bs.DenseBlockMaskTorch(block_mask=mask[None],
                                           block_size=(_TILE, _TILE))

    def run(self, meta):
        out = self.fn(self.q, self.k, self.v, block_sparse_tensors=meta)
        return (out[0] if isinstance(out, tuple) else out)[0]


class _Flex(_Backend):
    name = 'flex_block_sparse'

    def __init__(self, q, k, v):
        from torch.nn.attention import flex_attention  # pylint: disable=import-outside-toplevel
        self.flex_attention = flex_attention
        self.fn = torch.compile(flex_attention.flex_attention, dynamic=False)
        self.seq = q.shape[0]
        self.q, self.k, self.v = (t.transpose(0, 1)[None] for t in (q, k, v))

    def prepare(self, mask):
        cnt, idx = _left_packed(mask)
        # Forward only: skipping the transposed (Q-direction) lists cuts the
        # per-pattern BlockMask cost from ~1.2 ms to ~0.01 ms on a 4090.
        return self.flex_attention.BlockMask.from_kv_blocks(
            torch.zeros_like(cnt), idx, full_kv_num_blocks=cnt,
            full_kv_indices=idx, BLOCK_SIZE=_TILE,
            seq_lengths=(self.seq, self.seq), compute_q_blocks=False)

    def run(self, meta):
        return self.fn(self.q, self.k, self.v,
                       block_mask=meta)[0].transpose(0, 1)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FastVideoTriton(_Backend):
    name = 'fastvideo_triton'

    def __init__(self, q, k, v):
        root = os.environ['MIOWTION_FASTVIDEO_KERNEL']
        kernels = os.path.join(root, 'triton_kernels')
        self.attn = _load_module(
            os.path.join(kernels, 'block_sparse_attn_triton.py'),
            'fastvideo_block_sparse_attn_triton')
        self.index = _load_module(os.path.join(kernels, 'index.py'),
                                  'fastvideo_index')
        self.q, self.k, self.v = (t.transpose(0, 1)[None].contiguous()
                                  for t in (q, k, v))
        self.sizes = torch.full((q.shape[0] // 64,), 64, dtype=torch.int32,
                                device=q.device)

    def prepare(self, mask):
        m64 = mask.repeat_interleave(2, 1).repeat_interleave(2, 2)
        return self.index.map_to_index(m64[None])

    def run(self, meta):
        q2k_idx, q2k_num = meta
        out, _ = self.attn.triton_block_sparse_attn_forward(
            self.q, self.k, self.v, q2k_idx.to(torch.int32).contiguous(),
            q2k_num.to(torch.int32).contiguous(), self.sizes)
        return out[0].transpose(0, 1)


def run(seq: int = 32768, heads: int = 8, head_dim: int = 128,
        density: float = 0.1, seed: int = 0, patterns: int = 8,
        calls: int = 16, warmup: int = 4,
        device: str = 'cuda') -> list[Result]:
    """Runs every available dense and sparse kernel; returns the results."""
    if seq % _TILE:
        raise ValueError('seq must be a multiple of 128')
    dev = torch.device(device)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    q, k, v = (torch.randn(seq, heads, head_dim, generator=gen).to(
        dev, torch.bfloat16) for _ in range(3))
    n_blocks = seq // _TILE
    masks = [random_block_mask(heads, n_blocks, density, seed + i, dev)
             for i in range(patterns)]
    real_density = masks[0].float().mean().item()
    check_blocks = [0, n_blocks // 3, n_blocks - 1]
    ref_sparse = _reference_rows(q, k, v, masks[0], check_blocks)
    ref_dense = _reference_rows(q, k, v, torch.ones_like(masks[0]),
                                check_blocks)
    factories = [_SdpaDense, _Fa4Dense, _Fa4Sparse, _Fa4DenseMask, _Flex]
    if os.environ.get('MIOWTION_FASTVIDEO_KERNEL'):
        factories.append(_FastVideoTriton)
    results = []
    for factory in factories:
        result = Result(getattr(factory, 'name', factory.__name__))
        results.append(result)
        try:
            backend = factory(q, k, v)
            result.name = backend.name
            result.note = 'sparse' if backend.sparse else 'dense'
            metas = [backend.prepare(m) for m in masks]
            out = backend.run(metas[0])
            result.max_err = _max_err(
                out, ref_sparse if backend.sparse else ref_dense,
                check_blocks)
            if backend.sparse:
                dense_err = _max_err(out, ref_dense, check_blocks)
                if dense_err < result.max_err:
                    result.note = 'WRONG: output matches dense attention'
            result.kernel_ms = _time_ms(
                lambda i: backend.run(metas[i % patterns]), calls, warmup)
            if backend.sparse:
                result.prep_ms = _time_ms(
                    lambda i: backend.run(backend.prepare(
                        masks[i % patterns])), calls, warmup)
        except Exception as e:  # pylint: disable=broad-except
            result.note = f'FAILED: {type(e).__name__}: {e}'[:200]
    dense_ms = min((r.kernel_ms for r in results
                    if r.note == 'dense' and r.max_err is not None
                    and r.max_err < 0.1), default=float('nan'))
    for r in results:
        if r.note == 'sparse' and r.max_err is not None and r.max_err < 0.1:
            r.efficiency = dense_ms * real_density / r.kernel_ms
            r.efficiency_prep = dense_ms * real_density / r.prep_ms
    peak = PEAK_BF16_DENSE_FLOPS.get(torch.cuda.get_device_name(dev))
    ceiling = gemm_ceiling_tflops(dev)
    for r in results:
        if r.max_err is None or r.max_err >= 0.1 or r.kernel_ms != r.kernel_ms:
            continue
        flops = attention_flops(seq, heads, head_dim,
                                real_density if r.note == 'sparse' else 1.0)
        r.tflops = flops / (r.kernel_ms / 1e3) / 1e12
        r.gemm_frac = r.tflops / ceiling
        if peak is not None:
            r.mfu = r.tflops * 1e12 / peak
    return results


def format_results(results: list[Result], header: str) -> str:
    """Formats a run's results.

    TFLOP/s counts only the arithmetic a kernel had to do (skipped blocks
    excluded), MFU divides that by the device's dense bf16 tensor-core peak,
    and gemm% divides it by the measured cuBLAS bf16 GEMM ceiling. The last
    column separates "sparsity is not paying off" from "this kernel cannot
    reach the tensor cores' peak in the first place".
    """
    lines = [header, f'{"kernel":20s} {"kernel ms":>10s} {"+prep ms":>9s} '
             f'{"max_err":>8s} {"eff":>5s} {"eff+prep":>8s} {"TFLOP/s":>8s} '
             f'{"MFU":>6s} {"gemm%":>6s}  note']
    fmt = lambda x, spec: '-' if x is None or x != x else format(x, spec)
    for r in results:
        lines.append(f'{r.name:20s} {fmt(r.kernel_ms, ".3f"):>10s} '
                     f'{fmt(r.prep_ms, ".3f"):>9s} {fmt(r.max_err, ".4f"):>8s} '
                     f'{fmt(r.efficiency, ".2f"):>5s} '
                     f'{fmt(r.efficiency_prep, ".2f"):>8s} '
                     f'{fmt(r.tflops, ".1f"):>8s} '
                     f'{fmt(r.mfu, ".1%"):>6s} '
                     f'{fmt(r.gemm_frac, ".1%"):>6s}  {r.note}')
    return '\n'.join(lines)
