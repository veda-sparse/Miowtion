"""Where the fused Sol+Veda call spends its time, stage by stage.

Three costs are tangled in one number: the host side (quantise Q and K,
reduce every tile to a summary), the exact walk over kept blocks, and
the compensation walk, which visits every tile whatever the density is.
The mask separates the last two without touching the kernel: an
all-false mask leaves the exact loop with nothing to do, so it times the
compensation alone, and an all-true mask makes the exact loop dense.

Upstream's sibling INT8 kernel is the ceiling for the exact walk: same
quantisation, same block-sparse loop, no compensation, tuned by its
author. The gap to it is what the compensation costs us.

    python scripts/prof_fused.py --density 0.1 --heads 56
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.kernels import bench                         # noqa: E402
from miowtion.kernels import fa4                           # noqa: E402
from miowtion.kernels import sol_veda                      # noqa: E402
from miowtion.utils import progress                        # noqa: E402

TILE = 128


def time_ms(call, calls: int = 10, warmup: int = 3) -> float:
    """Wall time per call, with the stream drained around the window."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(calls):
        call()
    torch.cuda.synchronize()
    return (time.time() - start) * 1e3 / calls


class _FakeLayout:
    """The fields fa4.block_sparse_attention reads, all tiles full."""

    def __init__(self, counts, n_tiles, seq, device):
        self.valid_count = counts
        self.n_tiles = n_tiles
        self.seq_len = seq
        self.full_tile = torch.ones(n_tiles, dtype=torch.bool, device=device)
        self.kv_ok = torch.ones(n_tiles, dtype=torch.bool, device=device)
        self.slot_valid = torch.ones(seq, dtype=torch.bool, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq', type=int, default=37376)
    parser.add_argument('--heads', type=int, default=56)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--calls', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda')
    gen = torch.Generator(device='cpu').manual_seed(0)
    q, k, v = (torch.randn(args.seq, args.heads, 128, generator=gen).to(
        device, torch.bfloat16) for _ in range(3))
    tiles = args.seq // TILE
    mask = bench.random_block_mask(args.heads, tiles, args.density, 0,
                                   device).contiguous()
    empty = torch.zeros_like(mask)
    full = torch.ones_like(mask)
    counts = torch.full((tiles,), TILE, dtype=torch.int32, device=device)
    _attend, sage = sol_veda._modules()                    # noqa: SLF001
    from miowtion.kernels.sol_veda_patched import attention as fused
    progress.log(f'seq {args.seq}, {tiles} tiles, {args.heads} heads, '
                 f'density {mask.float().mean().item():.4f}, '
                 f'w{fused.LAUNCH["num_warps"]} s{fused.LAUNCH["num_stages"]} '
                 f'C{fused.COLUMNS} M{fused.ROWS} W{fused.KEY_WIDTH}')

    def fused_call(selected, reference=False):
        was, fused.USE_REFERENCE = fused.USE_REFERENCE, reference
        try:
            return fused.launch(q, k, v, counts, selected, sage, TILE)
        finally:
            fused.USE_REFERENCE = was

    stages = [
        ('sdpa dense', time_ms(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                v.transpose(0, 1)[None]), args.calls)),
        ('fa4 block sparse', time_ms(
            lambda: fa4.block_sparse_attention(
                q, k, v, mask,
                _FakeLayout(counts, tiles, args.seq, device)), args.calls)),
        ('fused published (total)', time_ms(
            lambda: fused_call(mask, reference=True), args.calls)),
        ('fused fast (total)', time_ms(lambda: fused_call(mask), args.calls)),
        ('  at density 0 (comp only)', time_ms(
            lambda: fused_call(empty), args.calls)),
        ('  at density 1 (exact dense)', time_ms(
            lambda: fused_call(full), args.calls)),
    ]

    # The host side, each piece on its own.
    stages.append(('  quantize q (tile 128)', time_ms(
        lambda: sage.quantize(q, TILE, pre_scale=128 ** -0.5), args.calls)))
    stages.append(('  quantize k (block 64)', time_ms(
        lambda: sage.quantize(k, 64), args.calls)))
    stages.append(('  summaries (torch)', time_ms(
        lambda: fused._summaries(k, v, counts), args.calls)))  # noqa: SLF001
    stages.append(('  summaries (triton)', time_ms(
        lambda: fused.summarise(k, v, counts), args.calls)))

    index = torch.empty((args.heads, tiles, tiles), device=device,
                        dtype=torch.int32)
    keeps = torch.empty((args.heads, tiles), device=device, dtype=torch.int32)
    import triton

    def build_index():
        fused._index_kernel[(tiles, args.heads)](          # noqa: SLF001
            mask, counts, index, keeps, N=tiles,
            NP=triton.next_power_of_2(tiles), num_warps=4)

    stages.append(('  index compaction', time_ms(build_index, args.calls)))

    # Upstream's sibling: the same INT8 walk without compensation.
    build_index()
    valid = counts.clone()
    sparse_int8 = sys.modules[sage.__name__]
    stages.append(('sage int8 sparse (no comp)', time_ms(
        lambda: sparse_int8.attend(q, k, v, index, keeps, valid),
        args.calls)))

    print(f'\n{"stage":<32}{"ms":>9}')
    for name, ms in stages:
        print(f'{name:<32}{ms:>9.3f}')


if __name__ == '__main__':
    main()
