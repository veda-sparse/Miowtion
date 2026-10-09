"""Generates audio-video clips with the few-step H3 teacher, dense and/or Veda.

The prompt comes from a sample cache (scripts/encode_samples.py). Each
requested attention mode denoises the same noise (same seed); the latents are
decoded after the DiT is freed. Every mode gets its own <mode>.mp4. With both
modes (the standard comparison) there is also dense_vs_veda.mp4: side by side
with the titles "Dense" / "Veda <S>% Sparsity" and both audio tracks, plus
timing (per step, attention only, speedups; step 0 of each mode includes
kernel compilation and is excluded) in summary.json.

One process drives every visible GPU; set CUDA_VISIBLE_DEVICES to choose.

The launch parameters belong in a versioned config (see configs/infer_*.yaml
and miowtion/infer/config.py); any of them can still be overridden on the
command line:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/generate.py \\
        --config configs/infer_holdout20_step600.yaml

The predictor comes either from a deployable bundle (--predictor, which
carries its own tile plans) or from a training checkpoint plus its plan
directory (--checkpoint --plan-dir).
"""

import argparse
import copy
import dataclasses
import gc
import json
import os

import torch

from miowtion.infer import config as infer_config
from miowtion.infer import decode
from miowtion.infer import pipeline
from miowtion.kernels import sol as sol_kernel
from miowtion.train import data
from miowtion.train import parallel
from miowtion.train import teacher
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import mask as veda_mask
from miowtion.veda import plan as veda_plan

_DECODE_DTYPES = {'bf16': torch.bfloat16, 'fp32': torch.float32}


def _budget(ratio: float | None, tiles: float | None
            ) -> veda_mask.Budget | None:
    if ratio is None and tiles is None:
        return None
    return veda_mask.Budget(ratio=ratio, tiles=tiles)


def _budget_text(budget: veda_mask.Budget) -> str:
    if budget.tiles is not None:
        return f'{budget.tiles:g} tiles'
    return f'{budget.ratio:g} ratio'


def _title(mode: str, target_budget: veda_mask.Budget,
           ref_budget: veda_mask.Budget | None = None) -> str:
    if mode == 'dense':
        return 'Dense'
    target = _budget_text(target_budget)
    if ref_budget is None:
        return f'Veda current={target}'
    return f'Veda ref={_budget_text(ref_budget)}, current={target}'


def build_parser() -> argparse.ArgumentParser:
    """Every option of the script; also what a --config may set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='checkpoint root')
    parser.add_argument('--variant', default='FL2VA')
    parser.add_argument('--schedule', default='turbo')
    parser.add_argument('--num-steps', type=int, default=8)
    parser.add_argument('--adapter', default=None, help='few-step LoRA')
    parser.add_argument('--sample-cache')
    parser.add_argument('--sample-id', nargs='+',
                        help='one or more samples (the model loads once)')
    parser.add_argument('--geometry', nargs='+',
                        help="e.g. '16:9@37'; one per sample, or one for all")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--attention', nargs='+', default=['dense'],
                        choices=pipeline.ATTENTION_MODES)
    parser.add_argument('--plan-dir', default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='training checkpoint with the predictor')
    parser.add_argument('--predictor', default=None,
                        help='predictor bundle (.safetensors) written by '
                        'scripts/export_predictor.py; brings its own plans, '
                        'so it replaces --checkpoint and --plan-dir')
    parser.add_argument(
        '--veda-collect-mib', type=int,
        default=veda_attention.DEFAULT_COLLECT_BYTES // 2**20,
        help='head-chunk bound of the sparse attention (VedaConfig.'
        'collect_bytes); the default fits 24 GB, 2048 runs a whole head '
        'group per launch at latent_t 102')
    parser.add_argument('--keep-ratio', type=float, default=None,
                        help='block budget; defaults to the bundle\'s own '
                        'ratio, else 0.1')
    parser.add_argument('--keep-tiles', type=float, default=None,
                        help='absolute current/target key tiles per query; '
                        'exclusive with --keep-ratio')
    parser.add_argument('--ref-keep-ratio', type=float, default=None,
                        help='reference visual block keep ratio')
    parser.add_argument('--ref-keep-tiles', type=float, default=None,
                        help='absolute reference visual key tiles per query')
    parser.add_argument('--tile-conditions', action=argparse.BooleanOptionalAction,
                        default=None, help='tile visual references and apply '
                        'their independent budget')
    parser.add_argument('--sol-tau', type=float, default=1.0,
                        help="threshold coefficient of --attention sol. "
                        'Larger routes fewer key blocks exactly. It is '
                        'NOT a budget, so matching a fixed-budget router '
                        'at the same sparsity needs calibration: set '
                        'MIOWTION_SOL_DENSITY=1 and the run reports the '
                        'routed fraction it actually reached')
    parser.add_argument('--dense-steps', type=int, nargs='*', default=[],
                        help='denoising steps that stay dense in veda runs')
    parser.add_argument('--offload-blocks', type=int, default=40)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--out-dir',
                        help='output directory; with several samples, one '
                        'subdirectory per sample')
    parser.add_argument('--decode-dtype', default='bf16',
                        choices=sorted(_DECODE_DTYPES),
                        help='video VAE weight dtype (bf16 is 3.2x faster '
                        'than fp32 at 51.3 dB against it)')
    parser.add_argument('--decode-only', action='store_true',
                        help='decode <mode>_latents.pt of an earlier run '
                        '(without it, existing latents are reused and only '
                        'missing (sample, mode) pairs are denoised)')
    parser.add_argument('--config', default=None,
                        help='YAML of these same options (see configs/'
                        'infer_*.yaml); the command line overrides it')
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Merges --config into the defaults, then validates what is required.

    Raises:
        ValueError: If the config holds an unknown key, or an option that
            has no default is set neither in it nor on the command line.
    """
    parser = build_parser()
    config_only, _ = parser.parse_known_args(argv)
    if config_only.config:
        infer_config.merge_into(parser, config_only.config)
    args = parser.parse_args(argv)
    infer_config.require(args, ['root', 'sample_cache', 'sample_id',
                                'geometry', 'out_dir'])
    if args.keep_ratio is not None and args.keep_tiles is not None:
        raise ValueError('--keep-ratio and --keep-tiles are exclusive')
    if args.ref_keep_ratio is not None and args.ref_keep_tiles is not None:
        raise ValueError('--ref-keep-ratio and --ref-keep-tiles are exclusive')
    if (args.keep_ratio is None and args.keep_tiles is None
            and not args.predictor):
        args.keep_ratio = 0.1
    ref_budget = _budget(args.ref_keep_ratio, args.ref_keep_tiles)
    if args.tile_conditions is True and ref_budget is None \
            and not args.predictor:
        raise ValueError('--tile-conditions requires --ref-keep-ratio or '
                         '--ref-keep-tiles')
    if (ref_budget is not None and args.tile_conditions is not True
            and not args.predictor):
        raise ValueError('a reference budget requires --tile-conditions')
    return args


def _resolve_policy(args: argparse.Namespace) -> None:
    """Fills policy fields inherited from a bundle, then validates them.

    Kept out of parse_args so versioned configs can be linted on machines
    that do not hold the large predictor artifact.
    """
    inherited_target = inherited_ref = None
    inherited_tiling = False
    if args.predictor:
        metadata = veda_bundle.read_metadata(args.predictor)
        inherited_target = veda_bundle._read_budget(  # pylint: disable=protected-access
            metadata, 'target', legacy_ratio='keep_ratio')
        inherited_ref = veda_bundle._read_budget(  # pylint: disable=protected-access
            metadata, 'ref')
        inherited_tiling = metadata.get('tile_conditions', 'false') == 'true'
    if args.keep_ratio is None and args.keep_tiles is None:
        inherited_target = inherited_target or veda_mask.Budget(ratio=0.1)
        args.keep_ratio = inherited_target.ratio
        args.keep_tiles = inherited_target.tiles
    if args.ref_keep_ratio is None and args.ref_keep_tiles is None \
            and inherited_ref is not None:
        args.ref_keep_ratio = inherited_ref.ratio
        args.ref_keep_tiles = inherited_ref.tiles
    if args.tile_conditions is None:
        args.tile_conditions = inherited_tiling
    ref_budget = _budget(args.ref_keep_ratio, args.ref_keep_tiles)
    if args.tile_conditions and ref_budget is None:
        raise ValueError('--tile-conditions requires --ref-keep-ratio or '
                         '--ref-keep-tiles')
    if ref_budget is not None and not args.tile_conditions:
        raise ValueError('a reference budget requires --tile-conditions')


def main():
    args = parse_args()
    _resolve_policy(args)
    env = parallel.init_distributed()
    cache = data.SampleCache(args.sample_cache)
    by_id = {s.id: s for s in cache.samples}
    if len(args.geometry) not in (1, len(args.sample_id)):
        raise ValueError('give one geometry, or one per sample')
    jobs = []
    for i, sample_id in enumerate(args.sample_id):
        if sample_id not in by_id:
            raise KeyError(f'{sample_id} not in {args.sample_cache}')
        sample = by_id[sample_id]
        geometry = data.parse_geometry(args.geometry[
            i if len(args.geometry) > 1 else 0])
        if sample.latent_t not in (None, geometry.latent_t):
            # repr, not str: the mismatch is sometimes only in the type
            # (a manifest that quoted latent_t), and '102' vs 102 prints
            # as the same number otherwise.
            raise ValueError(f'{sample.id} was written for latent_t '
                             f'{sample.latent_t!r}, geometry has '
                             f'{geometry.latent_t!r}')
        out_dir = (args.out_dir if len(args.sample_id) == 1 else
                   os.path.join(args.out_dir, f'{sample.id}_{geometry.name}'))
        os.makedirs(out_dir, exist_ok=True)
        jobs.append((sample, geometry, out_dir))

    devices = _devices(env)
    assigned = pipeline.assign_jobs(
        [pipeline.geometry_cost(g) for _, g, _ in jobs], len(devices))
    progress.log(f'{len(jobs)} samples on {len(devices)} GPU(s): '
                 f'{[len(a) for a in assigned]} per GPU')
    if args.decode_only:
        results = [{mode: pipeline.Generated(**torch.load(
            os.path.join(out_dir, f'{mode}_latents.pt')))
                    for mode in args.attention} for _, _, out_dir in jobs]
    else:
        results = _denoise_all(args, env, cache, jobs, devices, assigned)
    summaries: list[dict | None] = [None] * len(jobs)

    def decode_work(rank: int, device: torch.device) -> None:
        decoder = decode.Decoder(os.path.join(args.root, args.variant),
                                 device, _DECODE_DTYPES[args.decode_dtype])
        for index in assigned[rank]:
            sample, geometry, out_dir = jobs[index]
            summaries[index] = _decode_and_compare(
                args, sample, geometry, results[index], decoder, out_dir)

    pipeline.run_on_devices(devices, decode_work)
    if len(jobs) > 1:
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(summaries, f, indent=1)


def _devices(env) -> list[torch.device]:
    """Every visible GPU: one process drives them all (see run_on_devices)."""
    if env.world_size > 1:
        raise ValueError('run generate.py as one process; it uses every '
                         'visible GPU (set CUDA_VISIBLE_DEVICES)')
    return [torch.device('cuda', i) for i in range(torch.cuda.device_count())]


def _denoise_all(args, env, cache, jobs, devices, assigned) -> list[dict]:
    """Denoises every (sample, geometry) in every mode; saves latents.

    The teacher is built once on the first GPU; other GPUs get replicas that
    share its pinned host copy of the offloaded blocks.
    """
    tch = teacher.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=any(s.task != 't2va' for s, _, _ in jobs),
        audio_references=args.variant == 'Ref2VA',
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows)
    plans = predictor = veda_config = None
    if 'veda' in args.attention:
        if args.predictor:
            loaded = veda_bundle.load(args.predictor, devices[0])
            plans, predictor = loaded.plans, loaded.predictor
            source = f'bundle {args.predictor} ({loaded.metadata["source"]})'
        elif args.plan_dir and args.checkpoint:
            plans = veda_plan.PlanTable.load_dir(args.plan_dir)
            cfg = tch.model.config
            predictor = pipeline.load_predictor(
                args.checkpoint, cfg.num_layers, cfg.num_heads, cfg.head_dim,
                devices[0])
            source = f'plans {args.plan_dir}, checkpoint {args.checkpoint}'
        else:
            raise ValueError('veda needs --predictor, or --plan-dir with '
                             '--checkpoint')
        target_budget = _budget(args.keep_ratio, args.keep_tiles)
        ref_budget = _budget(args.ref_keep_ratio, args.ref_keep_tiles)
        veda_config = veda_attention.VedaConfig(
            target_budget=target_budget, ref_budget=ref_budget,
            tile_conditions=args.tile_conditions,
            collect_bytes=args.veda_collect_mib * 2**20)
        progress.log(f'veda: {source}, target {_budget_text(target_budget)}, '
                     f'ref {(_budget_text(ref_budget) if ref_budget else "global")}, '
                     f'tile_conditions={args.tile_conditions}, dense steps '
                     f'{args.dense_steps}')
    replicas = [(tch.model, tch.tables, predictor)]
    for device in devices[1:]:
        with progress.Timer(f'replicate the teacher on {device}'):
            replicas.append((
                parallel.replicate(tch.model, device, args.prefetch),
                tch.tables.to(device),
                copy.deepcopy(predictor).to(device)
                if predictor is not None else None))
    results: list[dict | None] = [None] * len(jobs)

    def work(rank: int, device: torch.device) -> None:
        model, tables, rank_predictor = replicas[rank]
        for index in assigned[rank]:
            sample, geometry, out_dir = jobs[index]
            plan = plans.select(geometry) if plans is not None else None
            progress.log(f'[{device}] {sample.id} on {geometry.name}'
                         + (f' (plan {plan.geometry})' if plan else ''))
            modes = {}
            for mode in args.attention:
                saved = os.path.join(out_dir, f'{mode}_latents.pt')
                if os.path.exists(saved):
                    # Resume: a finished (sample, mode) of an earlier run.
                    modes[mode] = pipeline.Generated(**torch.load(saved))
                    progress.log(f'[{device}] {mode}: reusing {saved}')
                    continue
                modes[mode] = pipeline.generate(
                    model, tch.schedule, tables, cache, sample, geometry,
                    args.seed, device, mode, plan, rank_predictor,
                    veda_config, args.dense_steps, args.sol_tau)
                extra = ''
                if mode == 'sol' and len(sol_kernel.DENSITY_CURVE) > 1:
                    parts = []
                    for cand in sorted(sol_kernel.DENSITY_CURVE):
                        d = sol_kernel.DENSITY_CURVE[cand]
                        parts.append(f'tau {cand:g}: {sum(d) / len(d):.4f}')
                    progress.log(f'[{device}] {sample.id} sol density '
                                 f'curve over {len(d)} calls -- '
                                 + ', '.join(parts))
                    sol_kernel.DENSITY_CURVE.clear()
                if mode == 'sol' and sol_kernel.DENSITY_LOG:
                    log = sol_kernel.DENSITY_LOG
                    extra = (f', sol density {sum(log) / len(log):.4f} '
                             f'[{min(log):.4f}, {max(log):.4f}] over '
                             f'{len(log)} calls at tau {args.sol_tau}')
                    sol_kernel.DENSITY_LOG.clear()
                progress.log(f'[{device}] {sample.id} {mode}: denoised in '
                             f'{modes[mode].seconds:.1f} s, attention '
                             f'{sum(modes[mode].attention_seconds):.1f} s'
                             f'{extra}')
                torch.save(dataclasses.asdict(modes[mode]), saved)
            results[index] = modes

    pipeline.run_on_devices(devices, work)
    del tch, predictor, replicas
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return results


def _decode_and_compare(args, sample, geometry, results, decoder,
                        out_dir) -> dict:
    """Decodes every mode, writes the videos, timing and comparison."""
    frames = {}
    target_budget = _budget(args.keep_ratio, args.keep_tiles)
    ref_budget = _budget(args.ref_keep_ratio, args.ref_keep_tiles)
    summary = {'sample': sample.id, 'geometry': geometry.name,
               'seed': args.seed, 'schedule': args.schedule,
               'num_steps': args.num_steps, 'adapter': args.adapter,
               'checkpoint': args.checkpoint, 'keep_ratio': args.keep_ratio,
               'keep_tiles': args.keep_tiles,
               'ref_keep_ratio': args.ref_keep_ratio,
               'ref_keep_tiles': args.ref_keep_tiles,
               'tile_conditions': args.tile_conditions,
               'dense_steps': args.dense_steps, 'modes': {}}
    waveforms = {}
    for mode, result in results.items():
        frames[mode] = decoder.video(result.video_rows, geometry)
        waveforms[mode] = decoder.audio(result.audio_rows)
        path = os.path.join(out_dir, f'{mode}.mp4')
        decode.write_mp4(path, frames[mode],
                         [(waveforms[mode], _title(
                             mode, target_budget, ref_budget))],
                         decoder.sample_rate)
        summary['modes'][mode] = {
            'title': _title(mode, target_budget, ref_budget),
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
        # No PSNR here. Sparse and dense differ in composition, not just in
        # detail, so a per-frame number says nothing about quality -- and
        # computing it needs two fp32 copies of the clip plus their
        # difference (14 GB at 14.4 s, 1344x768), which on a loaded host
        # spends longer in reclaim than the whole denoise took.
        side = decode.side_by_side([
            decode.add_title(frames[m], _title(
                m, target_budget, ref_budget))
            for m in ('dense', 'veda')])
        path = os.path.join(out_dir, 'dense_vs_veda.mp4')
        decode.write_mp4(path, side, [
            (waveforms[m], _title(m, target_budget, ref_budget))
            for m in ('dense', 'veda')], decoder.sample_rate)
        for mode in ('dense', 'veda'):
            m = summary['modes'][mode]
            progress.log(f'{m["title"]:>18s}: {m["seconds_excl_step0"]:.1f} s '
                         f'denoising (steps 1+), attention '
                         f'{m["attention_seconds_excl_step0"]:.1f} s')
        progress.log(f'speedup: end-to-end '
                     f'{summary["speedup"]["end_to_end"]:.2f}x, attention '
                     f'{summary["speedup"]["attention"]:.2f}x; '
                     f'side by side: {path}')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == '__main__':
    main()
