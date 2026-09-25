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
"""

import argparse
import os

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
    parser.add_argument('--keep-ratio', type=float, default=0.1,
                        help='keep ratio the run trained with; recorded as '
                        'the default budget for inference')
    parser.add_argument('--ema', action='store_true',
                        help='export the EMA shadow instead of the live '
                        'weights (the live weights are the default: the '
                        'predictor is trained, not sampled from)')
    args = parser.parse_args()

    with progress.Timer(f'load {args.checkpoint}'):
        payload = checkpoint.load(args.checkpoint)
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
                     keep_ratio=args.keep_ratio, source=args.checkpoint,
                     source_weights='ema' if args.ema else 'live',
                     step=int(payload['step']))
    size = os.path.getsize(args.out) / 1024 ** 3
    progress.log(f'wrote {args.out}: {size:.2f} GiB, {num_layers} layers x '
                 f'{shape[0]} heads x {shape[2]}, step {payload["step"]}, '
                 f'plans {sorted(plans.plans)}')


if __name__ == '__main__':
    main()
