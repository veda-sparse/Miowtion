"""Generates audio-video clips with the few-step H3 teacher, dense and/or Veda.

The prompt comes from a sample cache (scripts/encode_samples.py). Each
requested attention mode denoises the same noise (same seed); the latents are
decoded after the DiT is freed. Every mode gets its own <mode>.mp4. With both
modes (the standard comparison) there is also dense_vs_veda.mp4: side by side
with the titles "Dense" / "Veda <S>% Sparsity" and both audio tracks, plus
timing (per step, attention only, speedups; step 0 of each mode includes
kernel compilation and is excluded) and the per-frame PSNR in summary.json.

Example (one GPU):
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/generate.py \\
        --root weights/MiniMax-H3 --schedule turbo --num-steps 8 \\
        --adapter weights/turbo_lora/minimax_h3_turbo_v4_step600_ema.safetensors \\
        --sample-cache artifacts/samples/search_2x3 --sample-id search5s_0000 \\
        --geometry 16:9@37 --attention dense veda \\
        --plan-dir runs/search_turbo8_4090/plans \\
        --checkpoint runs/stage1_turbo8_4090/ckpt/step_0000075 \\
        --out-dir artifacts/generate/search5s_0000
"""

import argparse
import dataclasses
import gc
import json
import os

import torch

from miowtion.infer import decode
from miowtion.infer import pipeline
from miowtion.train import data
from miowtion.train import parallel
from miowtion.train import teacher
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan


def _title(mode: str, keep_ratio: float) -> str:
    if mode == 'dense':
        return 'Dense'
    return f'Veda {100.0 * (1.0 - keep_ratio):g}% Sparsity'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='checkpoint root')
    parser.add_argument('--variant', default='FL2VA')
    parser.add_argument('--schedule', default='turbo')
    parser.add_argument('--num-steps', type=int, default=8)
    parser.add_argument('--adapter', default=None, help='few-step LoRA')
    parser.add_argument('--sample-cache', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--geometry', required=True, help="e.g. '16:9@37'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attention', nargs='+', default=['dense'],
                        choices=pipeline.ATTENTION_MODES)
    parser.add_argument('--plan-dir', default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='training checkpoint with the predictor')
    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--dense-steps', type=int, nargs='*', default=[],
                        help='denoising steps that stay dense in veda runs')
    parser.add_argument('--offload-blocks', type=int, default=40)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--decode-only', action='store_true',
                        help='decode <mode>_latents.pt of an earlier run')
    args = parser.parse_args()

    env = parallel.init_distributed()
    cache = data.SampleCache(args.sample_cache)
    by_id = {s.id: s for s in cache.samples}
    if args.sample_id not in by_id:
        raise KeyError(f'{args.sample_id} not in {args.sample_cache}')
    sample = by_id[args.sample_id]
    geometry = data.parse_geometry(args.geometry)
    if sample.latent_t not in (None, geometry.latent_t):
        raise ValueError(f'{sample.id} was written for latent_t '
                         f'{sample.latent_t}, geometry has {geometry.latent_t}')
    os.makedirs(args.out_dir, exist_ok=True)
    if args.decode_only:
        results = {mode: pipeline.Generated(**torch.load(
            os.path.join(args.out_dir, f'{mode}_latents.pt')))
                   for mode in args.attention}
        if env.is_main:
            _decode_and_compare(args, sample, geometry, results, env)
        return

    tch = teacher.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=sample.task != 't2va',
        audio_references=args.variant == 'Ref2VA',
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows)
    plan = predictor = veda_config = None
    if 'veda' in args.attention:
        if not (args.plan_dir and args.checkpoint):
            raise ValueError('veda needs --plan-dir and --checkpoint')
        plan = veda_plan.PlanTable.load_dir(args.plan_dir).select(geometry)
        cfg = tch.model.config
        predictor = pipeline.load_predictor(
            args.checkpoint, cfg.num_layers, cfg.num_heads, cfg.head_dim,
            env.device)
        veda_config = veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=args.keep_ratio))
        progress.log(f'veda: plan {plan.geometry}, predictor '
                     f'{args.checkpoint}, keep {args.keep_ratio}, dense '
                     f'steps {args.dense_steps}')

    results = {}
    for mode in args.attention:
        results[mode] = pipeline.generate(
            tch.model, tch.schedule, tch.tables, cache, sample, geometry,
            args.seed, env.device, mode, plan, predictor, veda_config,
            args.dense_steps)
        progress.log(f'{mode}: denoised in {results[mode].seconds:.1f} s, '
                     f'attention {sum(results[mode].attention_seconds):.1f} s '
                     f'({results[mode].sparse_calls} sparse / '
                     f'{results[mode].dense_calls} dense attention calls)')
        if env.is_main:
            torch.save(dataclasses.asdict(results[mode]),
                       os.path.join(args.out_dir, f'{mode}_latents.pt'))
    del tch, predictor
    gc.collect()
    torch.cuda.empty_cache()
    if env.is_main:
        _decode_and_compare(args, sample, geometry, results, env)


def _decode_and_compare(args, sample, geometry, results, env):
    """Decodes every mode, writes the videos, timing and comparison."""
    decoder = decode.Decoder(os.path.join(args.root, args.variant),
                             env.device)
    frames = {}
    summary = {'sample': sample.id, 'geometry': geometry.name,
               'seed': args.seed, 'schedule': args.schedule,
               'num_steps': args.num_steps, 'adapter': args.adapter,
               'checkpoint': args.checkpoint, 'keep_ratio': args.keep_ratio,
               'dense_steps': args.dense_steps, 'modes': {}}
    waveforms = {}
    for mode, result in results.items():
        frames[mode] = decoder.video(result.video_rows, geometry)
        waveforms[mode] = decoder.audio(result.audio_rows)
        path = os.path.join(args.out_dir, f'{mode}.mp4')
        decode.write_mp4(path, frames[mode],
                         [(waveforms[mode], _title(mode, args.keep_ratio))],
                         decoder.sample_rate)
        summary['modes'][mode] = {
            'title': _title(mode, args.keep_ratio),
            'total_seconds': result.seconds,
            'seconds_excl_step0': sum(result.step_seconds[1:]),
            'attention_seconds_excl_step0': sum(
                result.attention_seconds[1:]),
            'step_seconds': [round(t, 3) for t in result.step_seconds],
            'attention_seconds': [round(t, 3)
                                  for t in result.attention_seconds],
            'sparse_calls': result.sparse_calls,
            'dense_calls': result.dense_calls}
        progress.log(f'wrote {path}: {frames[mode].shape[0]} frames '
                     f'{frames[mode].shape[2]}x{frames[mode].shape[1]}, '
                     f'{waveforms[mode].shape[1] / decoder.sample_rate:.2f} '
                     's audio')
    if {'dense', 'veda'} <= frames.keys():
        dense, veda = summary['modes']['dense'], summary['modes']['veda']
        summary['speedup'] = {
            'end_to_end': (dense['seconds_excl_step0']
                           / veda['seconds_excl_step0']),
            'attention': (dense['attention_seconds_excl_step0']
                          / veda['attention_seconds_excl_step0'])}
        psnr = decode.psnr_per_frame(frames['veda'], frames['dense'])
        summary['veda_vs_dense_psnr'] = {
            'mean': float(psnr.mean()), 'min': float(psnr.min()),
            'per_frame': [round(float(p), 2) for p in psnr]}
        side = decode.side_by_side([
            decode.add_title(frames[m], _title(m, args.keep_ratio))
            for m in ('dense', 'veda')])
        path = os.path.join(args.out_dir, 'dense_vs_veda.mp4')
        decode.write_mp4(path, side, [
            (waveforms[m], _title(m, args.keep_ratio))
            for m in ('dense', 'veda')], decoder.sample_rate)
        for mode in ('dense', 'veda'):
            m = summary['modes'][mode]
            progress.log(f'{m["title"]:>18s}: {m["seconds_excl_step0"]:.1f} s '
                         f'denoising (steps 1+), attention '
                         f'{m["attention_seconds_excl_step0"]:.1f} s')
        progress.log(f'speedup: end-to-end '
                     f'{summary["speedup"]["end_to_end"]:.2f}x, attention '
                     f'{summary["speedup"]["attention"]:.2f}x; PSNR veda vs '
                     f'dense {psnr.mean():.2f} dB (min {psnr.min():.2f}); '
                     f'side by side: {path}')
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
