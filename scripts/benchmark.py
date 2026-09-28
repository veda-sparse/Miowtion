"""Inference performance of the H3 teacher: stage times, step time, MFU.

One process, one GPU, one teacher load, then every (geometry, attention
mode) pair requested. What it reports per run, and why:

* stage times - startup (weight load, LoRA merge, AdaLN tables, predictor)
  is paid once per process and must not be mixed into the step time, while
  the VAE decode is part of the end-to-end latency a user sees;
* steady-state step time - the mean of steps 1..N-1. Step 0 of a run pays
  the FA4 CuTe JIT and the first host-to-device stream of the offloaded
  blocks, so including it overstates a short trajectory by seconds;
* end-to-end - trajectory setup + denoise + decode of one clip, i.e. what
  the startup-amortized latency of the (N+1)-th clip would be;
* MFU - the analytic FLOP count of the run (miowtion.h3.flops) over the
  measured time and the device's dense bf16 peak.

Data parallelism is deliberately out of scope: this measures one GPU's
throughput on one request. The repo has no tensor or sequence parallelism
(miowtion.train.parallel shards with FSDP2/HSDP only), so TP = SP = 1 in
every record here.

Only t2va is benchmarked. ref2va has no encoder yet
(scripts/encode_samples.py rejects it), so there is no supported request
to time; fabricating one would report a number for a path nobody can run.

Two stages, two processes, because a clip's cost is not one number:
'--stage text' times the Qwen3-VL text tower, which runs once per prompt
and does not depend on the geometry or the attention mode; '--stage dit'
times the denoiser and the VAE decode, which do. Loading both towers in
one process would only make them fight over the same 24 GB card.

    python scripts/benchmark.py --root weights/h3 --adapter weights/turbo8 \\
        --sample-cache artifacts/samples/x --sample-id s0 \\
        --geometry 16:9@37 16:9@102 --attention dense veda \\
        --predictor artifacts/bundles/p.safetensors --out runs/bench/a.json

'--random-weights' times the same DiT on a machine without the weights:
the release's config files (the third_party/MiniMax-H3 submodule) give
every shape, the tensors are random (miowtion.h3.synthetic), the prompt is
--text-len random rows and the Veda predictor and plans are random too
(miowtion.veda.bundle.random_bundle). A step's cost depends only on shapes
and on the Veda budget, so step time, attention time and MFU are the real
ones; the startup times are not (no file is read, no LoRA is merged), and
the samples are noise:

    python scripts/benchmark.py --root third_party/MiniMax-H3 \\
        --random-weights --geometry 16:9@37 --attention dense veda \\
        --offload-blocks 0 --out runs/bench/random.json
"""

import argparse
import json
import os
import statistics
import time

import torch

from miowtion.h3 import config as h3_config
from miowtion.h3 import flops as h3_flops
from miowtion.h3 import layout as h3_layout
from miowtion.h3 import schedule as h3_schedule
from miowtion.infer import decode
from miowtion.infer import pipeline
from miowtion.train import data
from miowtion.train import encode
from miowtion.train import parallel
from miowtion.train import teacher
from miowtion.utils import progress
from miowtion.veda import attention as veda_attention
from miowtion.veda import bundle as veda_bundle
from miowtion.veda import mask as veda_mask


# Text rows of holdout14s_0000, the sample of the real-weight records in
# docs/benchmark/performance.md, so random-weight runs have the same layout.
RANDOM_TEXT_LEN = 589
# Seed of the random weights, prompt rows and predictor.
_RANDOM_SEED = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='checkpoint root')
    parser.add_argument('--variant', default='FL2VA',
                        choices=('FL2VA', 'Ref2VA'))
    # ref2va has no encoder yet, so t2va is the only timeable task.
    parser.add_argument('--task', default='t2va', choices=('t2va',))
    parser.add_argument('--schedule', default='turbo')
    parser.add_argument('--num-steps', type=int, default=8,
                        help='NFE of one clip')
    parser.add_argument('--adapter', default=None, help='few-step LoRA')
    parser.add_argument('--stage', default='dit', choices=('dit', 'text'),
                        help="'dit': denoise + VAE decode; 'text': the "
                        'text encoder, which runs once per prompt')
    parser.add_argument('--sample-cache', default=None,
                        help='required by --stage dit')
    parser.add_argument('--prompts', default=None,
                        help='prompt corpus (--stage text)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='text encodes to time (--stage text)')
    parser.add_argument('--max-memory', default=None,
                        help='text encoder budget, e.g. "0=21GiB,cpu=60GiB"')
    parser.add_argument('--sample-id', default=None,
                        help='sample of --sample-cache; defaults to the '
                        'first one of the task')
    parser.add_argument('--geometry', nargs='+', default=[],
                        help="e.g. '16:9@37' (required by --stage dit)")
    parser.add_argument('--attention', nargs='+', default=['dense'],
                        choices=pipeline.ATTENTION_MODES)
    parser.add_argument('--predictor', default=None,
                        help='predictor bundle (veda)')
    parser.add_argument('--keep-ratio', type=float, default=None)
    parser.add_argument('--dense-steps', type=int, nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offload-blocks', type=int, default=40)
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--mlp-chunk-rows', type=int, default=8192)
    parser.add_argument('--decode', action='store_true',
                        help='also time the VAE decode (frames are dropped)')
    parser.add_argument('--random-weights', action='store_true',
                        help='random DiT, prompt and predictor (only the '
                        'config files under --root are read); step times '
                        'are real, samples are noise')
    parser.add_argument('--text-len', type=int, default=RANDOM_TEXT_LEN,
                        help='prompt rows with --random-weights')
    parser.add_argument('--out', required=True, help='JSON to write')
    return parser


def _steady_state(step_seconds: list[float]) -> float:
    """Mean step time without step 0 (kernel JIT, first weight stream).

    Raises:
        ValueError: With fewer than two steps, where there is no steady
            state to report.
    """
    if len(step_seconds) < 2:
        raise ValueError('a steady-state step time needs at least 2 steps')
    return statistics.fmean(step_seconds[1:])


def _parse_max_memory(spec: str | None) -> dict | None:
    """'0=21GiB,cpu=60GiB' -> {0: '21GiB', 'cpu': '60GiB'}."""
    if not spec:
        return None
    budget = {}
    for item in spec.split(','):
        key, value = item.split('=')
        budget[int(key) if key.isdigit() else key] = value
    return budget


def _text_stage(args, device_name: str) -> dict:
    """Times the text tower: load once, then encode one prompt repeatedly.

    The first encode pays the kernel autotune and, when the budget spills
    layers to the CPU, the first pass over the offloaded weights; the
    steady figure is the mean of the rest, matching how the DiT stage
    reports its step time.

    Raises:
        ValueError: Without --prompts, or with fewer than two repeats.
    """
    if not args.prompts:
        raise ValueError('--stage text needs --prompts')
    if args.repeat < 2:
        raise ValueError('--repeat must be at least 2 for a steady state')
    prompts = data.load_prompts(args.prompts, args.task)
    variant_dir = os.path.join(args.root, args.variant)
    with progress.Timer('load the text encoder') as timer:
        encoder = encode.TextEncoder(
            variant_dir, max_memory=_parse_max_memory(args.max_memory))
    seconds, text_len = [], 0
    counter = progress.Progress('encode the prompt', args.repeat)
    for _ in range(args.repeat):
        start = time.time()
        hidden, _ = encoder.encode_t2va(prompts[0])
        seconds.append(time.time() - start)
        text_len = int(hidden.shape[0])
        counter.update(f'{text_len} rows, {seconds[-1]:.2f} s')
    steady = _steady_state(seconds)
    config = h3_config.H3Config.from_pretrained(
        os.path.join(variant_dir, 'transformer'))
    return {
        'device': {'name': device_name, 'count': 1,
                   'torch': torch.__version__},
        'text_encoder': {
            'layers': encode.TEXT_LAYERS,
            'prompts': args.prompts,
            'text_len': text_len,
            'max_memory': args.max_memory,
            'load_seconds': round(timer.seconds, 3),
            'encode_seconds': [round(t, 3) for t in seconds],
            'steady_encode_seconds': round(steady, 3),
            # The DiT's own text tower (the refiner) reruns these rows
            # once per clip; quoted here so the doc can compare them.
            'refiner_flops': h3_flops.refiner_flops(config,
                                                    text_len).as_dict(),
        },
    }


def _record(args, config, sample, geometry, layout, schedule, result,
            mode: str, device_name: str, stages: dict[str, float],
            end_to_end: float) -> dict:
    """One run's request parameters, stage times, step time and MFU.

    The MFU denominators differ on purpose: the steady-state step is priced
    with one velocity evaluation (attention at the run's average sparsity),
    the denoise stage with every step it ran, and neither includes the text
    tower, which runs once in setup.
    """
    keep = args.keep_ratio if mode == 'veda' else 1.0
    slots = int(h3_schedule.build_timestep_state(
        layout, *schedule.timesteps(0)).timesteps.numel())
    dense_step = h3_flops.step_flops(config, layout, 1.0, num_slots=slots)
    # Attention is priced per call, not per step: a veda run leaves some
    # layers and some steps dense (VedaConfig, --dense-steps), and the
    # counts of what actually ran are the only honest weights.
    per_call = h3_flops.attention_flops(layout.used, config.inner_dim)
    calls = result.dense_calls + keep * result.sparse_calls
    attention = int(per_call * calls)
    per_step = dense_step * args.num_steps
    denoise_flops = h3_flops.Flops(attention=attention, linear=per_step.linear,
                                   adaln=per_step.adaln)
    steady_step = h3_flops.Flops(
        attention=attention // args.num_steps,
        linear=dense_step.linear, adaln=dense_step.adaln)
    peak = h3_flops.device_peak(device_name)
    steady = _steady_state(result.step_seconds)
    denoise = sum(result.step_seconds)
    return {
        'task': args.task,
        'sample': sample.id,
        'variant': args.variant,
        'attention': mode,
        'keep_ratio': keep,
        'dense_steps': list(args.dense_steps),
        'sparse_calls': result.sparse_calls,
        'dense_calls': result.dense_calls,
        'request': {
            'aspect': geometry.aspect,
            'resolution': f'{geometry.width}x{geometry.height}',
            'duration_seconds': round(geometry.duration_seconds, 3),
            'frame_count': geometry.frame_count,
            'latent_t': geometry.latent_t,
            'video_grid': list(geometry.video_grid),
            'video_rows': geometry.num_video_tokens,
            'audio_rows': geometry.num_audio_rows,
            'text_len': layout.text_len,
            'seq_len': layout.seq_len,
            'used': layout.used,
            'nfe': args.num_steps,
        },
        'stages': {k: round(v, 3) for k, v in stages.items()},
        'step_seconds': [round(t, 3) for t in result.step_seconds],
        'attention_seconds': [round(t, 3) for t in result.attention_seconds],
        'steady_step_seconds': round(steady, 3),
        # Reported next to the step so a speedup can be split into the
        # part attention won and the part the whole DiT forward kept.
        'steady_attention_seconds': round(
            _steady_state(result.attention_seconds), 3),
        'denoise_seconds': round(denoise, 3),
        'end_to_end_seconds': round(end_to_end, 3),
        'flops': {'step': steady_step.as_dict(),
                  'denoise': denoise_flops.as_dict(),
                  'text_tower': h3_flops.refiner_flops(
                      config, layout.text_len).as_dict()},
        'mfu': {
            'steady_step': round(
                h3_flops.mfu(steady_step.total, steady, peak), 4),
            'denoise': round(
                h3_flops.mfu(denoise_flops.total, denoise, peak), 4),
        },
    }


def _write(args, device_name: str, records: list[dict]) -> None:
    """Writes the whole payload, replacing what is already at --out."""
    payload = {
        'device': {'name': device_name, 'count': 1,
                   'peak_bf16_flops': h3_flops.device_peak(device_name),
                   'torch': torch.__version__},
        'parallel': {'tp': 1, 'sp': 1, 'dp': 1, 'fsdp_shards': 1},
        'offload': {'blocks': args.offload_blocks,
                    'prefetch': args.prefetch,
                    'mlp_chunk_rows': args.mlp_chunk_rows},
        'schedule': {'kind': args.schedule, 'nfe': args.num_steps,
                     'adapter': bool(args.adapter)},
        # Random weights time the real shapes but skip every file read, so
        # a reader must not compare their startup stages with a release run.
        'weights': 'random' if args.random_weights else 'release',
        'runs': records,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=1)


def main() -> None:
    args = build_parser().parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError('benchmark one GPU at a time (CUDA_VISIBLE_DEVICES)')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.stage == 'text':
        payload = _text_stage(args, torch.cuda.get_device_name(0))
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=1)
        progress.log(f'wrote {args.out}')
        return
    if not args.geometry:
        raise ValueError('--stage dit needs --geometry')
    if args.random_weights:
        if args.sample_cache or args.predictor or args.adapter:
            raise ValueError('--random-weights takes no --sample-cache, '
                             '--predictor or --adapter')
        if args.decode:
            raise ValueError('--random-weights cannot --decode: the VAEs '
                             'are only constructible from their weights')
    elif not args.sample_cache:
        raise ValueError('--stage dit needs --sample-cache')
    elif 'veda' in args.attention and not args.predictor:
        raise ValueError('veda needs --predictor')
    if args.keep_ratio is None:
        args.keep_ratio = (
            float(veda_bundle.read_metadata(args.predictor)['keep_ratio'])
            if args.predictor else 0.1)
    geometries = [data.parse_geometry(g) for g in args.geometry]

    if args.random_weights:
        config = h3_config.H3Config.from_pretrained(
            os.path.join(args.root, args.variant, 'transformer'))
        cache = data.SyntheticSampleCache(args.text_len, config.text_dim,
                                          _RANDOM_SEED)
        sample = cache.samples[0]
    else:
        cache = data.SampleCache(args.sample_cache)
        candidates = [s for s in cache.samples if s.task == args.task
                      and (args.sample_id is None or s.id == args.sample_id)]
        if not candidates:
            raise ValueError(f'no {args.task} sample in {args.sample_cache}')
        sample = candidates[0]

    env = parallel.init_distributed()
    device = env.device
    device_name = torch.cuda.get_device_name(device)
    stages: dict[str, float] = {}
    tch = teacher.build_teacher(
        args.root, args.variant, args.schedule, args.num_steps, args.adapter,
        env, visual_conditions=args.task != 't2va',
        audio_references=args.variant == 'Ref2VA',
        offload_blocks=args.offload_blocks, prefetch=args.prefetch,
        mlp_chunk_rows=args.mlp_chunk_rows, stages=stages,
        random_weights_seed=_RANDOM_SEED if args.random_weights else None)
    plans = predictor = veda_config = None
    if 'veda' in args.attention:
        with progress.Timer('load the predictor bundle') as timer:
            if args.random_weights:
                config = tch.model.config
                loaded = veda_bundle.random_bundle(
                    config.num_layers, config.num_heads, config.head_dim,
                    geometries, args.keep_ratio, _RANDOM_SEED, device)
            else:
                loaded = veda_bundle.load(args.predictor, device)
            plans, predictor = loaded.plans, loaded.predictor
        stages['predictor'] = timer.seconds
        veda_config = veda_attention.VedaConfig(
            target_budget=veda_mask.Budget(ratio=args.keep_ratio))
    decoder = None
    if args.decode:
        with progress.Timer('load the VAE decoder') as timer:
            decoder = decode.Decoder(os.path.join(args.root, args.variant),
                                     device, torch.bfloat16)
            # Parked on the host until a decode asks for it: 5 GiB of
            # resident VAE OOMs the denoise at latent_t 102.
            decoder.to(torch.device('cpu'))
        stages['vae_load'] = timer.seconds

    _, tags = cache.text(sample)
    records = []
    for geometry in geometries:
        layout = h3_layout.pack(tags, geometry, sample.keyframes,
                                sample.references)
        for mode in args.attention:
            plan = plans.select(geometry) if plans is not None else None
            progress.log(f'{args.task} {geometry.name} {mode}: seq '
                         f'{layout.seq_len} ({layout.used} used)')
            start = time.time()
            result = pipeline.generate(
                tch.model, tch.schedule, tch.tables, cache, sample, geometry,
                args.seed, device, mode, plan, predictor, veda_config,
                args.dense_steps)
            run_stages = dict(stages)
            run_stages['denoise'] = sum(result.step_seconds)
            run_stages['setup'] = (time.time() - start
                                   - run_stages['denoise'])
            if decoder is not None:
                # The transfer is this harness keeping one process alive
                # across many runs, not part of a decode, so it sits
                # outside the timers.
                torch.cuda.empty_cache()
                decoder.to(device)
                # Split: the video VAE is the expensive one and scales with
                # the clip, the audio VAE is nearly free and does not.
                with progress.Timer('decode video') as timer:
                    decoder.video(result.video_rows, geometry)
                run_stages['decode_video'] = timer.seconds
                with progress.Timer('decode audio') as timer:
                    decoder.audio(result.audio_rows)
                run_stages['decode_audio'] = timer.seconds
                decoder.to(torch.device('cpu'))
            torch.cuda.empty_cache()
            end_to_end = (run_stages['setup'] + run_stages['denoise']
                          + run_stages.get('decode_video', 0.0)
                          + run_stages.get('decode_audio', 0.0))
            records.append(_record(args, tch.model.config, sample, geometry,
                                   layout, tch.schedule, result, mode,
                                   device_name, run_stages, end_to_end))
            progress.log(
                f'{geometry.name} {mode}: steady step '
                f'{records[-1]["steady_step_seconds"]:.2f} s, MFU '
                f'{100 * records[-1]["mfu"]["steady_step"]:.1f}%, '
                f'end-to-end {end_to_end:.1f} s')
            # Rewritten after every run: a sweep is tens of minutes and a
            # later geometry can still run out of memory, which must not
            # cost the runs that already succeeded.
            _write(args, device_name, records)
    progress.log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
