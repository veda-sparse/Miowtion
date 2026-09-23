"""GPU smoke test of the real DiT: sharded load, dense / teacher / scorer.

Uses random text states (no text encoder) and reports time and peak memory
of one forward of each kind. Not a correctness test of the numerics; see the
docs for the visual-check protocol.

Example (3 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node 3 \
        scripts/smoke_dit.py --root weights/MiniMax-H3 --geometry 16:9@37
"""

import argparse
import json
import time

import torch

from miowtion.train import data
from miowtion.train import parallel
from miowtion.train import teacher
from miowtion.train import trajectory
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan
from miowtion.veda import predictor as veda_predictor
from miowtion.veda import search
from miowtion.veda import tiling


def _timed(env, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, {'seconds': round(time.time() - start, 2),
                 'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30,
                                   2), 'rank': env.rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--variant', default='FL2VA')
    parser.add_argument('--geometry', default='16:9@37')
    parser.add_argument('--text-len', type=int, default=300)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--schedule', default='base',
                        choices=('base', 'turbo'))
    parser.add_argument('--num-steps', type=int, default=49)
    parser.add_argument('--adapter', default=None,
                        help='few-step LoRA merged into the teacher')
    parser.add_argument('--offload-blocks', type=int, default=0)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--scorer', action='store_true',
                        help='also run one OracleScorer forward (slow)')
    args = parser.parse_args()

    env = parallel.init_distributed()
    report = {'offload_blocks': args.offload_blocks, 'prefetch': args.prefetch,
              'schedule': args.schedule, 'num_steps': args.num_steps}
    teacher_, report['build_teacher'] = _timed(env, lambda: teacher.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=False, audio_references=False,
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows))
    model, schedule, tables = (teacher_.model, teacher_.schedule,
                               teacher_.tables)
    report['adaln_table_bytes'] = tables.num_bytes()

    geometry = data.parse_geometry(args.geometry)
    gen = torch.Generator().manual_seed(env.rank)
    hidden = torch.randn(args.text_len, 5120, generator=gen).to(
        torch.bfloat16)
    tags = torch.ones(args.text_len, dtype=torch.long)

    class _Cache:
        def text(self, sample):
            del sample
            return hidden, tags

        def conditions(self, sample):
            del sample
            return torch.empty(0, 96), torch.empty(0, 32)

    sample = data.Sample('smoke', 't2va', 'train', (0, args.text_len))
    traj = trajectory.Trajectory(model, _Cache(), sample, geometry, schedule,
                                 seed=env.rank, device=env.device)
    inputs = traj.inputs()
    table = tables.get(inputs.timestep.timesteps)
    report['tokens'] = traj.layout.used

    def dense():
        with torch.no_grad():
            return model(traj.clip, inputs.video_rows, inputs.audio_rows,
                         inputs.timestep, None, table)

    (video_v, audio_v), report['dense'] = _timed(env, dense)
    _, report['dense_warm'] = _timed(env, dense)
    report['dense']['finite'] = bool(torch.isfinite(video_v).all()
                                     and torch.isfinite(audio_v).all())
    report['dense']['video_v_std'] = round(video_v.float().std().item(), 4)

    cfg = model.config
    predictor = veda_predictor.TileScorePredictor(
        cfg.num_layers, cfg.num_heads, cfg.head_dim).to(env.device)
    plan = veda_plan.TilePlan.uniform(geometry, tiling.TileShape(4, 8, 4),
                                      cfg.num_layers, cfg.num_heads)
    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=0.1), teacher_q_tiles=0.1)
    clip = veda_attention.ClipTiling(traj.layout, config, env.device)
    collector = veda_attention.TeacherCollector(
        clip, plan, predictor, torch.Generator().manual_seed(0),
        grad_scale=1.0 / cfg.num_layers)

    def teacher_forward():
        with torch.no_grad():
            return model(traj.clip, inputs.video_rows, inputs.audio_rows,
                         inputs.timestep, collector, table)

    (video_t, _), report['teacher'] = _timed(env, teacher_forward)
    report['teacher']['equals_dense'] = bool(torch.equal(video_t, video_v))
    report['teacher']['kl_mean'] = sum(collector.stats.kl) / cfg.num_layers
    report['teacher']['recall'] = (sum(collector.stats.recall)
                                   / max(1, len(collector.stats.recall)))

    if args.scorer:
        scorer = search.OracleScorer(
            traj.layout, [tiling.TileShape.parse(s) for s in
                          ('4x8x4', '8x8x2')], config, query_tiles=16,
            seed=0, device=env.device)

        def score():
            with torch.no_grad():
                return model(traj.clip, inputs.video_rows, inputs.audio_rows,
                             inputs.timestep, scorer, table)

        _, report['scorer'] = _timed(env, score)
        report['scorer']['layer0'] = scorer.scores[0].mean(1).tolist()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
