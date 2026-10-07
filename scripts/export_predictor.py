"""Exports a deployable predictor bundle from a training checkpoint.

Takes the live (non-EMA) predictor weights of a checkpoint plus the run's
tile plans and writes one safetensors file that inference can load on its
own (see miowtion/veda/bundle.py for why the plans travel with the
weights). The EMA shadow, the optimizer moments and everything else the
checkpoint carries for resuming are dropped.

Example:
    python scripts/export_predictor.py \\
        --checkpoint runs/stage1_fast_t37_4090/ckpt/step_0000600 \\
        --plan-dir runs/search_turbo8_multigeo_4090/plans \\
        --keep-ratio 0.1 --out weights/veda/stage1_fast_t37_step600.safetensors

    python scripts/export_predictor.py \\
        --checkpoint runs/stage1_r2va/ckpt/step_0000600 \\
        --plan-dir artifacts/init/veda_8nfe_step600_plans \\
        --keep-tiles 32 --ref-keep-tiles 32 --tile-conditions \\
        --out weights/veda/stage1_r2va_step600.safetensors
"""

import argparse
import os

import torch

from miowtion.train import checkpoint
from miowtion.utils import progress
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import plan as veda_plan

_PREFIX = 'predictor.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='training checkpoint directory')
    parser.add_argument('--plan-dir', required=True,
                        help='plans the run trained against')
    parser.add_argument('--out', required=True, help='.safetensors to write')
    parser.add_argument('--keep-ratio', type=float, default=None,
                        help='keep ratio the run trained with; recorded as '
                        'the default budget for inference')
    parser.add_argument('--keep-tiles', type=float, default=None,
                        help='absolute current/target tile budget')
    parser.add_argument('--ref-keep-ratio', type=float, default=None)
    parser.add_argument('--ref-keep-tiles', type=float, default=None)
    parser.add_argument('--tile-conditions', action='store_true',
                        help='record independently tiled visual references')
    parser.add_argument('--dtype', default='bfloat16',
                        choices=sorted(veda_bundle.DTYPES),
                        help='storage dtype; scoring upcasts to fp32 either '
                        'way, so bf16 just halves the file and the copy '
                        'every inference replica holds on its card. '
                        'float8_e4m3fn halves the file again (per-head '
                        'amax scale) but loads back as bf16')
    parser.add_argument('--ema', action='store_true',
                        help='export the EMA shadow instead of the live '
                        'weights (the live weights are the default: the '
                        'predictor is trained, not sampled from)')
    parser.add_argument('--leap', type=float, default=None, metavar='DELTA',
                        help='export DELTA * live + (1 - DELTA) * ema '
                        'instead of either alone, mixed in fp32 before the '
                        'storage dtype is applied. DELTA is the weight on '
                        'the live weights, so 0.8 is mostly live with a '
                        'fifth of the EMA shadow; 1.0 is --ema off and 0.0 '
                        'is --ema. Useful when the run is shorter than the '
                        "EMA's own halflife, where the shadow still "
                        'averages weights the run has moved away from')
    args = parser.parse_args()
    if args.keep_ratio is None and args.keep_tiles is None:
        args.keep_ratio = 0.1
    if args.leap is not None:
        if args.ema:
            raise ValueError('--leap and --ema are exclusive: --leap 0.0 is '
                             '--ema and --leap 1.0 is the live weights')
        if not 0.0 <= args.leap <= 1.0:
            raise ValueError(f'--leap must be in [0, 1], got {args.leap}')

    with progress.Timer(f'load {args.checkpoint}'):
        payload = checkpoint.load(args.checkpoint)
    if args.leap is not None:
        live, shadow = payload['weights'], payload['ema']
        missing = sorted(set(live) - set(shadow))
        if missing:
            raise KeyError(f'{args.checkpoint}: no EMA shadow for '
                           f'{missing[:5]} ({len(missing)})')
        # fp32 throughout: the checkpoint stores fp32 and the storage dtype
        # is applied once, at save time, to the mixed result -- mixing two
        # already-rounded tensors would round twice.
        source = {k: torch.lerp(shadow[k].float(), live[k].float(),
                                args.leap)
                  for k in live}
    else:
        source = payload['ema'] if args.ema else payload['weights']
    weights = {k: v for k, v in source.items() if k.startswith(_PREFIX)}
    if not weights:
        raise KeyError(f'{args.checkpoint} holds no {_PREFIX}* tensors')
    shape = weights[f'{_PREFIX}layers.0.proj_q'].shape  # [heads, 3D, D]
    num_layers = sum(1 for k in weights if k.endswith('.proj_q'))
    plans = veda_plan.PlanTable.load_dir(args.plan_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    veda_bundle.save(args.out, weights, plans, num_layers=num_layers,
                     num_heads=shape[0], head_dim=shape[2],
                     keep_ratio=args.keep_ratio, keep_tiles=args.keep_tiles,
                     ref_keep_ratio=args.ref_keep_ratio,
                     ref_keep_tiles=args.ref_keep_tiles,
                     tile_conditions=args.tile_conditions,
                     source=args.checkpoint,
                     source_weights=(f'leap{args.leap:g}'
                                     if args.leap is not None else
                                     'ema' if args.ema else 'live'),
                     step=int(payload['step']),
                     dtype=veda_bundle.DTYPES[args.dtype])
    size = os.path.getsize(args.out) / 1024 ** 3
    progress.log(f'wrote {args.out}: {size:.2f} GiB, {num_layers} layers x '
                 f'{shape[0]} heads x {shape[2]} {args.dtype}, '
                 f'step {payload["step"]}, '
                 f'plans {sorted(plans.plans)}')


if __name__ == '__main__':
    main()
