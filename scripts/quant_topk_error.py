"""Measures how fp8 / fp4 block scoring moves the Veda top-k selection.

Rolls one sample along the teacher's few-step trajectory (dense attention
throughout, exactly as stage-1 training does) and, at every measured layer,
compares the block top-k chosen from quantized q / k against the one chosen
from the bf16 teacher heat. Nothing is trained and no predictor is needed:
this is about the scoring pass itself.

    CUDA_VISIBLE_DEVICES=3 python scripts/quant_topk_error.py \\
        --root weights/MiniMax-H3 --adapter weights/turbo_lora/<lora> \\
        --sample-cache artifacts/samples/<cache> --sample-id <id> \\
        --geometry 16:9@37 --plan-dir runs/<search>/plans \\
        --out-dir runs/quant_topk/<name>
"""

import argparse
import json
import os

import torch

from miowtion.h3 import model as h3_model
from miowtion.train import data
from miowtion.train import parallel
from miowtion.train import teacher as teacher_lib
from miowtion.train import trajectory as traj_lib
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
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
    parser.add_argument('--geometry', required=True, help="e.g. '16:9@37'")
    parser.add_argument('--plan-dir', required=True)
    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--schemes', nargs='+',
                        default=['bf16', 'fp8_e4m3_head', 'fp8_e4m3_row',
                                 'fp8_e5m2_row', 'nvfp4', 'mxfp8_e4m3',
                                 'mxfp4'],
                        choices=sorted(quant.SCHEMES))
    parser.add_argument('--q-tile-fraction', type=float, default=0.25,
                        help='share of query tiles measured per layer')
    parser.add_argument('--layer-every', type=int, default=1,
                        help='measure every n-th layer')
    parser.add_argument('--offload-blocks', type=int, default=50)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    env = parallel.init_distributed()
    cache = data.SampleCache(args.sample_cache)
    by_id = {s.id: s for s in cache.samples}
    if args.sample_id not in by_id:
        raise KeyError(f'{args.sample_id} not in {args.sample_cache}')
    sample = by_id[args.sample_id]
    geometry = data.parse_geometry(args.geometry)
    tch = teacher_lib.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=sample.task != 't2va',
        audio_references=args.variant == 'Ref2VA',
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows)
    plan = veda_plan.PlanTable.load_dir(args.plan_dir).select(geometry)
    veda_config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=args.keep_ratio))
    traj = traj_lib.Trajectory(tch.model, cache, sample, geometry,
                               tch.schedule, args.seed, env.device)
    clip = veda_attention.ClipTiling(traj.layout, veda_config, env.device)
    probe = quant.QuantHeatProbe(
        clip, plan, args.schemes, q_tile_fraction=args.q_tile_fraction,
        layer_every=args.layer_every,
        generator=torch.Generator().manual_seed(args.seed))
    progress.log(f'{sample.id} @ {geometry.name}: {traj.layout.used} tokens, '
                 f'{tch.schedule.num_steps} steps, schemes '
                 f'{", ".join(args.schemes)}')
    steps = progress.Progress('quantized scoring', tch.schedule.num_steps,
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
            per_step = quant.summarize(
                [r for r in probe.records if r.step == inputs.step])
            steps.update('step {}: {}'.format(inputs.step, '  '.join(
                f'{name} recall {value["recall"]:.4f}'
                for name, value in per_step.items())))
    records = [r.to_json() for r in probe.records]
    with open(os.path.join(args.out_dir, 'records.jsonl'), 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    summary = {
        'sample': sample.id, 'geometry': geometry.name,
        'tokens': traj.layout.used, 'steps': tch.schedule.num_steps,
        'keep_ratio': args.keep_ratio,
        'q_tile_fraction': args.q_tile_fraction,
        'layer_every': args.layer_every,
        'overall': quant.summarize(probe.records),
        'per_step': {str(step): quant.summarize(
            [r for r in probe.records if r.step == step])
                     for step in range(tch.schedule.num_steps)},
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    progress.log(f'wrote {args.out_dir}/summary.json')
    for name, value in summary['overall'].items():
        progress.log(f'  {name:14s} recall {value["recall"]:.4f}  '
                     f'heat_kept {value["heat_kept"]:.4f} / ceiling '
                     f'{value["heat_ceiling"]:.4f}  rel_l2 '
                     f'{value["rel_l2"]:.2e}  max_abs {value["max_abs"]:.2e}')


if __name__ == '__main__':
    main()
