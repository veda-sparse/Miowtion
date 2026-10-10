"""Cost of zero-order compensation, measured inside each implementation.

A cost ratio only means something against the right baseline. Comparing
the ComfyUI node's fused Sol + Veda kernel (Triton, INT8 for the
selected blocks) against our FA4 path (CuTe, BF16) mixes three
differences at once -- kernel language, precision and compensation -- so
the aligned pair is the node's own two kernels, which differ only in the
compensation.

Reports every arm on one set of tensors, with the GPU to itself: a
contended card gave FA4 6.27 ms and 13.23 ms for the same call in two
runs, and made INT8 look slower than BF16.

Example:
    python scripts/bench_compensation.py --densities 0.05 0.10 0.20
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402
from miowtion.veda import tiling                           # noqa: E402

# 16:9@37 video tokens, the geometry every demo arm was measured on.
_TOKENS = 37 * 24 * 42
_HEADS = 56
_DIM = 128


def _idle(device: int = 0) -> str:
    """Compute processes on the GPU, so a contended run is visible."""
    import subprocess                                      # noqa: PLC0415
    out = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory',
         '--format=csv,noheader', f'--id={device}'],
        capture_output=True, text=True, check=False)
    return out.stdout.strip() or 'none'


def _bench(fn, rounds: int, iters: int) -> dict:
    """Median of per-round means, plus the spread across rounds."""
    for _ in range(5):                      # warm up compile and clocks
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(rounds):
        start = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        means.append((time.time() - start) / iters * 1000.0)
    return {'median_ms': statistics.median(means),
            'min_ms': min(means), 'max_ms': max(means),
            'spread': (max(means) - min(means)) / statistics.median(means)}


def _sol_tau_for(q, k, v, used, target: float, thresh: str) -> tuple:
    """Calibrate Sol's tau to `target` density on these tensors."""
    from miowtion.kernels import sol                        # noqa: PLC0415
    best = None
    for tau in [t / 10 for t in range(2, 36)]:
        got = sol.density(q, k, v, used, tau, thresh)
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (tau, got)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--densities', type=float, nargs='+',
                        default=[0.05, 0.10, 0.20])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tune-int8', action='store_true',
                        help="use the node's best launch config for this "
                        'GPU instead of its shipped default. tools/'
                        'tune_int8.py measured w8 s3 k64 at 1.29x the '
                        'shipped w4 s3 k64 on SM120, and every TMA variant '
                        'slower, so the default is a 1.58x handicap here '
                        'and comparing against it would charge precision '
                        'for an untuned launch.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('this benchmark needs a CUDA device')
    progress.log(f'compute processes on the GPU: {_idle()}')

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'third_party', 'Veda-on-ComfyUI'))
    from veda_comfy.core import selection                   # noqa: PLC0415
    from veda_comfy.kernels.sage import sparse_int8         # noqa: PLC0415

    from sol_attn.triton_ref import sol_attn as sol_triton   # noqa: PLC0415

    from miowtion.kernels import fa4, sol, sol_veda         # noqa: PLC0415

    if args.tune_int8:
        # Both measured on this GPU, each against its own shipped default:
        # the INT8 kernel's w4 s3 k64 -> w8 s3 k64 is 1.58x, the fused
        # kernel's w4 s1 C32 -> w8 s2 C32 is 1.41x. Tuning only one of
        # them would move the ratio by more than the compensation does.
        from miowtion.kernels.sol_veda_patched import attention as fused
        sparse_int8.OVERRIDE = {'tma': False, 'key_block': 64,
                                'num_warps': 8, 'num_stages': 3}
        fused.LAUNCH = {'num_warps': 8, 'num_stages': 2}
        fused.COLUMNS = 32
        progress.log(f'INT8 launch {sparse_int8.OVERRIDE}; '
                     f'fused launch {fused.LAUNCH} C={fused.COLUMNS}')
    from miowtion.veda import attention as veda_attention   # noqa: PLC0415

    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    span = tiling.TiledSpan(start=0, grid=(_TOKENS, 1, 1),
                            shape=tiling.TileShape.parse('128x1x1'))
    layout = tiling.build_tile_layout(
        [span], used=_TOKENS, seq_len=-(-_TOKENS // 128) * 128,
        device=device, tiler=tiling.contiguous_span_tiles)
    n_tiles = layout.n_tiles
    shape = (n_tiles * 128, _HEADS, _DIM)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    counts = layout.valid_count.to(torch.int32)
    progress.log(f'{_TOKENS} tokens, {n_tiles} tiles, {_HEADS} heads, '
                 f'dim {_DIM}')

    for target in args.densities:
        keep = (torch.rand(_HEADS, n_tiles, n_tiles, device=device) < target)
        keep |= torch.eye(n_tiles, device=device, dtype=torch.bool)[None]
        keep &= layout.kv_ok[None, None, :]
        density = keep.float().mean().item()
        index, count = selection.tile_index_list(keep)
        kept, lse = fa4.block_sparse_attention(q, k, v, keep, layout,
                                               return_lse=True)
        tau, sol_density = _sol_tau_for(q, k, v, _TOKENS, density, 'exact')
        arms = {
            'node Veda INT8 (no comp)':
                lambda: sparse_int8.attend(q, k, v, index, count, counts),
            'node Sol+Veda INT8 (comp)':
                lambda: sol_veda.block_sparse_attention(q, k, v, keep,
                                                        layout),
            'ours FA4 bf16 (no comp)':
                lambda: fa4.block_sparse_attention(q, k, v, keep, layout),
            'ours torch comp on FA4':
                lambda: veda_attention.zero_order_compensation(
                    kept, lse, q, k, v, keep, layout),
            f'Sol-Attn CuTe, tau {tau:g}':
                lambda: sol.attention(q[:_TOKENS].contiguous(),
                                      k[:_TOKENS].contiguous(),
                                      v[:_TOKENS].contiguous(),
                                      _TOKENS, tau=tau, thresh_type='exact'),
            f'Sol-Attn Triton, tau {tau:g}':
                lambda: sol_triton(q[:_TOKENS][None].contiguous(),
                                   k[:_TOKENS][None].contiguous(),
                                   v[:_TOKENS][None].contiguous(),
                                   scale=_DIM ** -0.5, tau=tau,
                                   thresh_type='exact'),
        }
        print(f'\n=== mask density {density:.4f} '
              f'(Sol reaches {sol_density:.4f} at tau {tau:g}) ===')
        print(f'{"arm":28s} {"median":>9s} {"min":>9s} {"max":>9s} '
              f'{"spread":>7s}')
        got = {}
        for name, fn in arms.items():
            got[name] = _bench(fn, args.rounds, args.iters)
            row = got[name]
            print(f'{name:28s} {row["median_ms"]:8.2f}ms '
                  f'{row["min_ms"]:8.2f}ms {row["max_ms"]:8.2f}ms '
                  f'{row["spread"]:6.1%}')
        base = got['node Veda INT8 (no comp)']['median_ms']
        comp = got['node Sol+Veda INT8 (comp)']['median_ms']
        fa = got['ours FA4 bf16 (no comp)']['median_ms']
        print(f'  compensation inside the node: {comp / base:.2f}x')
        print(f'  node INT8 against our FA4 bf16: {base / fa:.2f}x')
    progress.log(f'compute processes at the end: {_idle()}')


if __name__ == '__main__':
    main()
