"""Compares predictor bundles of the same step at different precisions.

Rolls one sample along the teacher's few-step trajectory (dense attention
throughout, exactly as stage-1 training does) and, at every measured layer,
scores the blocks with each bundle. Reports each bundle against the teacher
(recall, heat_kept) and against the first bundle's own selection (agree), so
a storage precision can be judged on the only thing that matters: the set of
blocks it picks.

    CUDA_VISIBLE_DEVICES=0 python scripts/predictor_precision.py \\
        --root weights/MiniMax-H3 --adapter weights/turbo_lora/<lora> \\
        --sample-cache artifacts/samples/<cache> --sample-id <id> \\
        --geometry 16:9@102 \\
        --bundle bf16=weights/veda/<step>_bf16.safetensors \\
        --bundle fp8=weights/veda/<step>_fp8.safetensors \\
        --out-dir runs/predictor_precision/<name>
"""

import argparse
import json
import os

import torch

from miowtion.train import data
from miowtion.train import parallel
from miowtion.train import teacher as teacher_lib
from miowtion.train import trajectory as traj_lib
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import mask as veda_mask
from miowtion.veda import quant


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='checkpoint root')
    parser.add_argument('--variant', default='FL2VA')
    parser.add_argument('--schedule', default='turbo')
    parser.add_argument('--num-steps', type=int, default=8)
    parser.add_argument('--adapter', default=None, help='few-step LoRA')
    parser.add_argument('--sample-cache', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--geometry', required=True, help="e.g. '16:9@102'")
    parser.add_argument('--bundle', action='append', required=True,
                        metavar='NAME=PATH',
                        help='a predictor bundle to measure; repeatable. '
                        'The first one is the reference the others are '
                        'compared against')
    parser.add_argument('--keep-ratio', type=float, default=None,
                        help='budget to select at; defaults to the one the '
                        'reference bundle records')
    parser.add_argument('--across-steps', action='store_true',
                        help='allow bundles from different training steps. '
                        'Off by default because this script exists to '
                        'isolate storage precision, and two steps would '
                        'confound rounding with training. On, it becomes a '
                        'checkpoint comparison: one trajectory, one clip, '
                        'every bundle scored on identical inputs, which is '
                        'the only way to read a training trend without the '
                        'per-clip variation that a training log carries.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--q-tile-fraction', type=float, default=0.25,
                        help='share of query tiles measured per layer')
    parser.add_argument('--layer-every', type=int, default=1,
                        help='measure every n-th layer')
    parser.add_argument('--offload-blocks', type=int, default=50)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args(argv)


def _parse_bundles(specs: list[str]) -> dict[str, str]:
    """`['bf16=a.safetensors', ...]` -> ordered {name: path}."""
    out: dict[str, str] = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'--bundle wants NAME=PATH, got {spec!r}')
        name, path = spec.split('=', 1)
        if name in out:
            raise ValueError(f'duplicate bundle name {name!r}')
        out[name] = path
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = _parse_bundles(args.bundle)
    reference = next(iter(paths))
    env = parallel.init_distributed()
    cache = data.SampleCache(args.sample_cache)
    by_id = {s.id: s for s in cache.samples}
    if args.sample_id not in by_id:
        raise KeyError(f'{args.sample_id} not in {args.sample_cache}')
    sample = by_id[args.sample_id]
    geometry = data.parse_geometry(args.geometry)

    bundles = {}
    for name, path in paths.items():
        with progress.Timer(f'load bundle {name} ({path})'):
            bundles[name] = veda_bundle.load(path, env.device)
    steps_recorded = {b.metadata.get('step') for b in bundles.values()}
    if len(steps_recorded) != 1 and not args.across_steps:
        # Comparing precisions of *one* trained predictor is the point; two
        # different steps would confound the rounding with the training.
        raise ValueError(f'bundles come from different training steps: '
                         f'{sorted(steps_recorded)}; pass --across-steps to '
                         'compare checkpoints instead of precisions')
    predictors = {name: b.predictor for name, b in bundles.items()}
    plan_table = bundles[reference].plans
    plan = plan_table.select(geometry)
    keep_ratio = (args.keep_ratio if args.keep_ratio is not None
                  else bundles[reference].keep_ratio)

    tch = teacher_lib.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=sample.task != 't2va',
        audio_references=args.variant == 'Ref2VA',
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows)
    veda_config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=keep_ratio))
    traj = traj_lib.Trajectory(tch.model, cache, sample, geometry,
                               tch.schedule, args.seed, env.device)
    clip = veda_attention.ClipTiling(traj.layout, veda_config, env.device)
    probe = quant.PredictorProbe(
        clip, plan, predictors, reference,
        q_tile_fraction=args.q_tile_fraction, layer_every=args.layer_every,
        generator=torch.Generator().manual_seed(args.seed))
    progress.log(f'{sample.id} @ {geometry.name}: {traj.layout.used} tokens, '
                 f'{tch.schedule.num_steps} steps, keep {keep_ratio}, '
                 f'bundles {", ".join(paths)} (reference {reference})')

    bar = progress.Progress('predictor scoring', tch.schedule.num_steps,
                            every=1)
    with torch.no_grad():
        while not traj.done:
            inputs = traj.inputs()
            probe.step = inputs.step
            video_v, audio_v = tch.model(
                traj.clip, inputs.video_rows, inputs.audio_rows,
                inputs.timestep, probe,
                tch.tables.get(inputs.timestep.timesteps))
            traj.advance(video_v, audio_v)
            per_step = quant.summarize_predictor(
                [r for r in probe.records if r.step == inputs.step])
            bar.update('step {}: {}'.format(inputs.step, '  '.join(
                f'{name} recall {value["recall"]:.4f}'
                for name, value in per_step.items())))

    with open(os.path.join(args.out_dir, 'records.jsonl'), 'w') as f:
        for record in probe.records:
            f.write(json.dumps(record.to_json()) + '\n')
    summary = {
        'sample': sample.id, 'geometry': geometry.name,
        'tokens': traj.layout.used, 'steps': tch.schedule.num_steps,
        'keep_ratio': keep_ratio, 'reference': reference,
        'bundles': {name: {'path': path,
                           'dtype': bundles[name].metadata.get('dtype'),
                           'bytes': os.path.getsize(path)}
                    for name, path in paths.items()},
        'training_step': int(bundles[reference].metadata['step']),
        'q_tile_fraction': args.q_tile_fraction,
        'layer_every': args.layer_every,
        'overall': quant.summarize_predictor(probe.records),
        'per_step': {str(step): quant.summarize_predictor(
            [r for r in probe.records if r.step == step])
                     for step in range(tch.schedule.num_steps)},
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    progress.log(f'wrote {args.out_dir}/summary.json')
    for name, value in summary['overall'].items():
        progress.log(f'  {name:10s} recall {value["recall"]:.4f}  '
                     f'heat_kept {value["heat_kept"]:.4f} / ceiling '
                     f'{value["heat_ceiling"]:.4f}  agree '
                     f'{value["agree"]:.4f}  rel_l2 {value["rel_l2"]:.2e}')


if __name__ == '__main__':
    main()
