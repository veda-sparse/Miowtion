"""Unpacks a released predictor bundle back into training inputs.

The inverse of scripts/export_predictor.py: it takes one deployable
`.safetensors` bundle (predictor weights plus the tile plans, see
miowtion/veda/bundle.py) and writes the two things a training run needs to
continue from it — a weights-only checkpoint for `init_from` and the plan
directory for `plan_dir`.

What a bundle cannot give back, and what that means:

* the optimizer moments and the EMA shadow are not in it, so this is a new
  stage seeded from published weights, not a resumed run: Adam restarts
  cold and the EMA is re-seeded from the weights themselves;
* an fp8 bundle was rounded on export (per-head amax scale), so the weights
  come back as the rounded values, not the trained fp32 ones. That costs
  nothing for inference (the ordering of the top-k is what matters, and it
  is unchanged) but it does mean the continued run starts a hair off the
  original trajectory. `bundle.read_metadata(path)['dtype']` says which
  one you have.

Example:
    python scripts/import_predictor.py \\
        --bundle weights/veda/step600_fp8.safetensors \\
        --out-checkpoint artifacts/init/step600 \\
        --out-plan-dir artifacts/init/step600_plans
"""

import argparse
import json
import os

from miowtion.train import checkpoint
from miowtion.utils import progress
from miowtion.veda import bundle as veda_bundle

_PREFIX = 'predictor.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True,
                        help='.safetensors written by export_predictor.py')
    parser.add_argument('--out-checkpoint', required=True,
                        help='directory for the init_from checkpoint')
    parser.add_argument('--out-plan-dir', required=True,
                        help='directory for the bundle\'s tile plans')
    args = parser.parse_args()

    metadata = veda_bundle.read_metadata(args.bundle)
    with progress.Timer(f'load {args.bundle}'):
        loaded = veda_bundle.load(args.bundle)
    weights = {_PREFIX + name: value
               for name, value in loaded.predictor.state_dict().items()}
    provenance = {
        'imported_from': os.path.basename(args.bundle),
        'bundle_dtype': metadata.get('dtype'),
        'keep_ratio': loaded.target_budget.ratio,
        'keep_tiles': loaded.target_budget.tiles,
        'ref_keep_ratio': (loaded.ref_budget.ratio
                           if loaded.ref_budget else None),
        'ref_keep_tiles': (loaded.ref_budget.tiles
                           if loaded.ref_budget else None),
        'tile_conditions': loaded.tile_conditions,
    }
    for key in ('training_step', 'schedule', 'num_steps', 'variant',
                'teacher_adapter'):
        if key in metadata:
            provenance[key] = metadata[key]
    checkpoint.write_init_payload(args.out_checkpoint, weights, provenance)
    os.makedirs(args.out_plan_dir, exist_ok=True)
    for name, plan in sorted(loaded.plans.plans.items()):
        plan.save(os.path.join(args.out_plan_dir, f'{name}.json'))
    params = sum(v.numel() for v in weights.values())
    progress.log(
        f'{args.out_checkpoint}: {len(weights)} tensors, {params:,} '
        f'parameters (stored fp32 from {metadata.get("dtype")}); '
        f'{args.out_plan_dir}: {len(loaded.plans.plans)} plans '
        f'({", ".join(sorted(loaded.plans.plans))})')
    progress.log(f'target budget {loaded.target_budget}, ref budget '
                 f'{loaded.ref_budget}, provenance {json.dumps(provenance)}')
    # A bundle pairs weights with the plans they were trained against, and
    # a mispairing is silent (the predictor still scores, for a tiling it
    # never saw), so say it where the operator will see it.
    progress.log(f'point the run at BOTH outputs: init_from='
                 f'{args.out_checkpoint}, plan_dir={args.out_plan_dir}')


if __name__ == '__main__':
    main()
