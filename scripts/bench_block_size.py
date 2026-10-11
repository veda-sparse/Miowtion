"""Cost of the routing block size: 64 vs 128 at a fixed density.

Halving the block size quarters the area of one routing decision, so a
router can follow the attention map more closely -- that is Sol-Attn's
geometry and the reason the question comes up. The arithmetic does not
change: at a fixed *density* a 64-block plan touches exactly as many
query-key pairs as a 128-block one, with 4x as many blocks. So whatever
the kernel loses is pure overhead, and that loss is the price of the
finer routing.

Both kernels we have are measured, because they do not pay the same
price: FA4's CuTe block-sparse path has a tile_m constraint, while the
fused Triton kernel takes its block extent as a constexpr. Run under
MIOWTION_TILE_SIZE, which is what every module reads at import:

    MIOWTION_TILE_SIZE=64 python scripts/bench_block_size.py --density 0.1
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.kernels import bench                         # noqa: E402
from miowtion.kernels import compensation_triton            # noqa: E402
from miowtion.kernels import fa4                           # noqa: E402
from miowtion.kernels import sol_veda                      # noqa: E402
from miowtion.utils import progress                        # noqa: E402
from miowtion.veda import tiling                           # noqa: E402

_TILE = tiling.TILE_SIZE


def _time_ms(call, metas: list, calls: int, warmup: int) -> float:
    for i in range(warmup):
        call(metas[i % len(metas)])
    torch.cuda.synchronize()
    start = time.time()
    for i in range(calls):
        call(metas[i % len(metas)])
    torch.cuda.synchronize()
    return (time.time() - start) * 1e3 / calls


def _fused(q, k, v, counts, reference: bool = False):
    """A closure timing the whole fused call at MIOWTION_TILE_SIZE.

    The host side (quantise Q and K, reduce every tile to a summary,
    compact the mask) is inside the timed region on purpose: it touches
    the whole sequence whatever the density is, so a kernel-only number
    would hide where the call actually went. `reference=True` runs the
    published kernel and the torch summaries it shipped with, which is
    the "before" column.
    """
    from miowtion.kernels.sol_veda_patched import attention as fused_mod
    _attend, sage = sol_veda._modules()                    # noqa: SLF001

    def run(selected):
        was, fused_mod.USE_REFERENCE = fused_mod.USE_REFERENCE, reference
        try:
            out, _exact = fused_mod.launch(
                q, k, v, counts, selected.contiguous(), sage, _TILE)
        finally:
            fused_mod.USE_REFERENCE = was
        return out

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq', type=int, default=37376,
                        help='multiple of 128 so 64 and 128 see one length')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--patterns', type=int, default=4)
    parser.add_argument('--calls', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--tune', action='store_true',
                        help='sweep the fused kernel launch constants')
    parser.add_argument('--reference-tune', action='store_true',
                        help='sweep the published kernel too, so that the '
                             'before column is its own best config')
    parser.add_argument('--config', action='append', default=[],
                        help='an explicit launch config, e.g. '
                             'w4s1C32M64W128i1; repeatable')
    parser.add_argument('--check', action='store_true',
                        help='verify both kernels against dense first')
    args = parser.parse_args()
    if args.seq % 128:
        raise ValueError('--seq must be a multiple of 128 so that the two '
                         'block sizes run on exactly the same sequence')

    device = torch.device('cuda')
    gen = torch.Generator(device='cpu').manual_seed(0)
    q, k, v = (torch.randn(args.seq, args.heads, 128, generator=gen).to(
        device, torch.bfloat16) for _ in range(3))
    n_blocks = args.seq // _TILE
    masks = [bench.random_block_mask(args.heads, n_blocks, args.density,
                                     i, device)
             for i in range(args.patterns)]
    density = masks[0].float().mean().item()
    counts = torch.full((n_blocks,), _TILE, dtype=torch.int32, device=device)
    progress.log(f'block {_TILE}, seq {args.seq}, {n_blocks} blocks, '
                 f'{args.heads} heads, density {density:.4f}')

    if args.check:
        # A kernel that is merely fast is not a measurement. At density 1
        # nothing is skipped, so the compensation term is empty and both
        # sparse paths must reproduce dense attention; that is the one
        # check that works identically at either block size.
        full = torch.ones(args.heads, n_blocks, n_blocks, dtype=torch.bool,
                          device=device)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None],
            v.transpose(0, 1)[None])[0].transpose(0, 1)
        got = _fused(q, k, v, counts)(full)
        progress.log(f'fused vs dense at density 1: max abs '
                     f'{(got.float() - reference.float()).abs().max():.4f}')
        if _TILE == 128:
            # The published kernel is the other reference: at the working
            # density its output is what the rewrite has to reproduce,
            # and dense attention says nothing there.
            for name, mask in (('density 1', full), ('the mask', masks[0])):
                was = _fused(q, k, v, counts, reference=True)(mask)
                now = _fused(q, k, v, counts)(mask)
                gap = (now.float() - was.float()).abs().max().item()
                scale = was.float().abs().max().clamp(min=1e-6).item()
                progress.log(f'fast vs published at {name}: max abs {gap:.5f}'
                             f', rel {gap / scale:.5f}, bit-equal '
                             f'{torch.equal(now, was)}')
        got = fa4.block_sparse_attention(
            q, k, v, full, _FakeLayout(counts, n_blocks, args.seq, device))
        progress.log(f'fa4   vs dense at density 1: max abs '
                     f'{(got.float() - reference.float()).abs().max():.4f}')

    rows = []
    dense_ms = _time_ms(
        lambda _m: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None],
            v.transpose(0, 1)[None]),
        [None], args.calls, args.warmup)
    rows.append(('sdpa dense', dense_ms, 1.0))

    try:
        metas = list(masks)
        call = (lambda m: fa4.block_sparse_attention(
            q, k, v, m, _FakeLayout(counts, n_blocks, args.seq, device)))
        ms = _time_ms(call, metas, args.calls, args.warmup)
        rows.append(('fa4 block sparse', ms, dense_ms * density / ms))
    except Exception as error:                             # noqa: BLE001
        rows.append((f'fa4 block sparse FAILED: {type(error).__name__}: '
                     f'{str(error)[:90]}', float('nan'), float('nan')))

    # FA4 for the kept blocks and Triton for the compensation only.
    # The fused kernel's job is both halves at once; this row says what
    # the expensive half costs when it is not in Triton.
    try:
        if True:
            layout = _FakeLayout(counts, n_blocks, args.seq, device)
            ms = _time_ms(
                lambda m: compensation_triton.attention(q, k, v, m, layout),
                list(masks), args.calls, args.warmup)
            rows.append(('fa4 + triton compensation', ms,
                         dense_ms * density / ms))
    except Exception as error:                             # noqa: BLE001
        rows.append((f'fa4 + compensation FAILED: {type(error).__name__}: '
                     f'{str(error)[:60]}', float('nan'), float('nan')))

    # The launch constants are GPU-specific and the two block sizes do
    # not want the same ones, so comparing either at its shipped default
    # charges the block size for a launch config. Both are swept.
    from miowtion.kernels.sol_veda_patched import attention as fused_mod
    metas = [m.contiguous() for m in masks]
    shipped = dict(warps=fused_mod.LAUNCH['num_warps'],
                   stages=fused_mod.LAUNCH['num_stages'],
                   columns=fused_mod.COLUMNS, qrows=fused_mod.ROWS,
                   width=fused_mod.KEY_WIDTH, int8=fused_mod.INT8)

    def measure(config, reference: bool = False) -> float:
        """Time one launch configuration, or inf if it cannot run."""
        fused_mod.LAUNCH = {'num_warps': config['warps'],
                            'num_stages': config['stages']}
        fused_mod.COLUMNS = config['columns']
        fused_mod.ROWS, fused_mod.KEY_WIDTH = config['qrows'], config['width']
        fused_mod.INT8 = config['int8']
        label = (f"w{config['warps']} s{config['stages']} "
                 f"C{config['columns']} M{config['qrows']} "
                 f"W{config['width']} {'int8' if config['int8'] else 'bf16'}"
                 f"{' published' if reference else ''}")
        try:
            ms = _time_ms(_fused(q, k, v, counts, reference), metas,
                          args.calls, args.warmup)
        except Exception as error:                         # noqa: BLE001
            progress.log(f'  {label}: {type(error).__name__}: '
                         f'{str(error)[:60]}')
            return float('inf')
        progress.log(f'  {label}: {ms:.3f} ms')
        return ms

    # Coordinate descent rather than the full product: the five knobs
    # are nearly separable (M and W set the shared-memory footprint,
    # warps and stages how it is covered, C only the summary walk) and
    # the full grid is 72 compiles, which is minutes of nvcc per run.
    if True:
        # The before column has to be the published kernel at *its* best
        # config, not at the constants it shipped: otherwise the rewrite
        # gets credit for a launch config.
        grid = ([dict(shipped, warps=w, stages=st)
                 for w in (4, 8) for st in (1, 2, 3)]
                if args.reference_tune else [dict(shipped)])
        was = min(measure(config, reference=True) for config in grid)
        rows.append(('fused triton (published)', was,
                     dense_ms * density / was))

    if args.config:
        best, best_cfg = float('inf'), dict(shipped)
        for spec in args.config:
            trial = dict(shipped)
            for knob, letter in (('warps', 'w'), ('stages', 's'),
                                 ('columns', 'C'), ('qrows', 'M'),
                                 ('width', 'W'), ('int8', 'i')):
                if letter in spec:
                    tail = spec.split(letter, 1)[1]
                    digits = ''
                    for char in tail:
                        if not char.isdigit():
                            break
                        digits += char
                    trial[knob] = (bool(int(digits)) if knob == 'int8'
                                   else int(digits))
            ms = measure(trial)
            if ms < best:
                best, best_cfg = ms, trial
    else:
        best, best_cfg = measure(shipped), dict(shipped)
    if args.tune:
        for knob, values in (('int8', (True, False)),
                             ('width', (64, 128)),
                             ('qrows', (32, 64, 128)),
                             ('warps', (4, 8)), ('stages', (1, 2, 3, 4)),
                             ('columns', (32, 64))):
            for value in values:
                if value == best_cfg[knob] or (
                        knob in ('qrows', 'width') and value > _TILE):
                    continue
                trial = dict(best_cfg, **{knob: value})
                ms = measure(trial)
                if ms < best:
                    best, best_cfg = ms, trial
    label = (f"w{best_cfg['warps']} s{best_cfg['stages']} "
             f"C{best_cfg['columns']} M{best_cfg['qrows']} "
             f"W{best_cfg['width']} "
             f"{'int8' if best_cfg['int8'] else 'bf16'}")
    if best == float('inf'):
        rows.append(('fused triton FAILED everywhere', float('nan'),
                     float('nan')))
    else:
        rows.append((f'fused triton {label}', best,
                     dense_ms * density / best))

    print(f'\n{"kernel":<38}{"ms":>9}{"efficiency":>12}')
    for name, ms, eff in rows:
        print(f'{name:<38}{ms:>9.3f}{eff:>12.3f}')


class _FakeLayout:
    """The three fields fa4.block_sparse_attention reads, all tiles full."""

    def __init__(self, counts, n_tiles, seq, device):
        self.valid_count = counts
        self.n_tiles = n_tiles
        self.seq_len = seq
        self.full_tile = torch.ones(n_tiles, dtype=torch.bool, device=device)
        self.kv_ok = torch.ones(n_tiles, dtype=torch.bool, device=device)
        self.slot_valid = torch.ones(seq, dtype=torch.bool, device=device)


if __name__ == '__main__':
    main()
