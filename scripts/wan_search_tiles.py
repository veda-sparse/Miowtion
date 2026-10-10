"""Tile-plan search for Wan, the last step of P0-3.

veda/search.py's own runner is built on H3's teacher and trajectory, but
its scoring is tensor-level: `oracle_rel_mse` takes q, k, v and a
candidate tile layout and returns the relative MSE the best mask that
permutation can offer, and `build_plan` votes a plan from those tables.
So a second model needs its activations captured, not that runner
reimplemented.

The activations here come from Wan's own blocks through the processor
swap (miowtion/wan/attention.py), which is verified value-for-value
against the upstream processor, so the q and k being scored are the ones
the model would really attend with.

Example:
    python scripts/wan_search_tiles.py --frames 81 \\
        --layers $(seq 0 29) --out plans/wan_480p
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miowtion.utils import progress                        # noqa: E402
from miowtion.veda import mask as veda_mask                # noqa: E402
from miowtion.veda import search as veda_search            # noqa: E402
from miowtion.veda import tiling                           # noqa: E402
from miowtion.wan import attention as wan_attention        # noqa: E402
from miowtion.wan import layout as wan_layout              # noqa: E402


def _capture_on_trajectory(model, args, device) -> str:
    """Denoise for real and stop at `--capture-step`.

    Scoring random hidden states only sees what the weights impose. A
    plan is about where attention actually goes, which is a property of
    the trajectory, so this runs the released pipeline on a prompt and
    lets the capture processors fire at one step.

    Args:
        model: The transformer whose processors are already swapped.
        args: Parsed arguments; uses prompt, steps, capture_step, seed.
        device: Where to run.

    Returns:
        A description of the activations, for the plan's metadata.
    """
    from diffusers import WanPipeline                       # noqa: PLC0415

    with progress.Timer('load the pipeline (text encoder included)'):
        pipe = WanPipeline.from_pretrained(
            args.root, transformer=model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    reached = {'step': -1}

    def stop_after(_pipe, step, _timestep, kwargs):
        reached['step'] = step
        if step >= args.capture_step:
            raise _Captured
        return kwargs

    try:
        with torch.no_grad():
            pipe(prompt=args.prompt, height=args.height, width=args.width,
                 num_frames=args.frames, num_inference_steps=args.steps,
                 guidance_scale=1.0,
                 generator=torch.Generator(device).manual_seed(args.seed),
                 callback_on_step_end=stop_after,
                 callback_on_step_end_tensor_inputs=['latents'])
    except _Captured:
        pass
    if reached['step'] < args.capture_step:
        raise RuntimeError(
            f'the pipeline stopped at step {reached["step"]} before '
            f'{args.capture_step}; nothing was captured')
    del pipe
    return (f'denoising step {args.capture_step} of {args.steps}, '
            f'prompt {args.prompt[:60]!r}')


class _Captured(Exception):
    """Unwinds the pipeline once the wanted step has run."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='weights/wan/Wan2.1-T2V-1.3B')
    parser.add_argument('--width', type=int, default=832,
                        help='pixel width; Wan2.1-1.3B ships 832x480')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--frames', type=int, default=9)
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[0, 7, 15, 22, 29])
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--max-padding', type=float, default=0.2)
    parser.add_argument('--query-tiles', type=int, default=8,
                        help='video query tiles sampled per candidate; the '
                        'cost knob, and the same sample for every candidate '
                        'so the shapes are compared paired')
    parser.add_argument('--out', default=None, help='write the plan here')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prompt', default=None,
                        help='run the real pipeline for a few steps and '
                        'score the activations at --capture-step, instead '
                        'of scoring random hidden states. Costs the text '
                        'encoder (21 GB) on top of the transformer.')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--capture-step', type=int, default=4,
                        help='which denoising step to score')
    args = parser.parse_args()

    from diffusers import WanTransformer3DModel                # noqa: PLC0415

    device = torch.device('cuda')
    with progress.Timer(f'load {args.root}'):
        model = WanTransformer3DModel.from_pretrained(
            args.root, subfolder='transformer',
            torch_dtype=torch.bfloat16).to(device).eval()
    cfg = model.config
    layout = wan_layout.packed_layout(args.width, args.height,
                                      args.frames, text_len=0)
    grid = layout.target.grid
    candidates = [s for s in tiling.candidate_shapes(grid)
                  if s.padding_ratio(grid) <= args.max_padding]
    if not candidates:
        raise ValueError(
            f'no candidate shape pads {grid} under {args.max_padding:.0%}')
    progress.log(f'{args.width}x{args.height}x{args.frames} -> grid {grid}, '
                 f'{layout.used} tokens, {len(candidates)} candidates, '
                 f'{len(args.layers)} layers')

    captured: dict[int, tuple] = {}

    def capture(q, k, v, layer_index):
        captured[layer_index] = (q.detach(), k.detach(), v.detach())
        return v

    for index in args.layers:
        model.blocks[index].attn1.set_processor(
            wan_attention.make_processor(capture, index))

    if args.prompt is None:
        # Random hidden states: this measures the structure the weights
        # impose, not the activation distribution. Good enough to check
        # the chain, not to adopt a plan from.
        width = cfg.num_attention_heads * cfg.attention_head_dim
        hidden = torch.randn(1, layout.used, width, device=device,
                             dtype=torch.bfloat16)
        latent = (grid[0], grid[1] * 2, grid[2] * 2)
        rope = model.rope(torch.randn(1, cfg.in_channels, *latent,
                                      device=device, dtype=torch.bfloat16))
        with torch.no_grad():
            for index in args.layers:
                model.blocks[index].attn1(hidden, None, None, rope)
        source = 'random hidden states'
    else:
        source = _capture_on_trajectory(model, args, device)
    del model
    torch.cuda.empty_cache()

    budget = veda_mask.Budget(ratio=args.density)
    entries = []
    steps = progress.Progress('score', len(args.layers) * len(candidates))
    generator = torch.Generator().manual_seed(args.seed)
    for index in sorted(captured):
        q, k, v = captured[index]
        table = []
        for shape in candidates:
            tile_layout = tiling.build_tile_layout(
                [tiling.TiledSpan(0, grid, shape)], used=layout.used,
                seq_len=layout.used, device=device)
            blocks = veda_mask.column_blocks(tile_layout, budget, None)
            # Same tiles for every candidate so the comparison is paired;
            # the ids mean different regions under different shapes, which
            # is exactly the thing being scored.
            count = min(args.query_tiles, tile_layout.n_video_tiles)
            rows = torch.randperm(
                tile_layout.n_video_tiles,
                generator=generator)[:count].to(device).sort().values
            with torch.no_grad():
                table.append(veda_search.oracle_rel_mse(
                    q, k, v, tile_layout, blocks, rows))
            steps.update(f'layer {index} {shape}')
        entries.append(veda_search.ScoreEntry(
            clip='random-hidden', step=0, layer=index,
            table=torch.stack(table).cpu()))

    plan = veda_search.build_plan(
        f'wan_{args.width}x{args.height}_t{grid[0]}', grid, candidates, entries,
        num_layers=cfg.num_layers, max_padding=args.max_padding,
        meta={'source': 'scripts/wan_search_tiles.py',
              'layers_scored': sorted(captured),
              'density': args.density,
              'activations': source})
    best = {}
    for entry in entries:
        order = entry.table.mean(-1).argmin().item()
        best[entry.layer] = str(candidates[order])
    print('\nbest shape per scored layer (mean rel-MSE over heads):')
    for layer in sorted(best):
        print(f'  layer {layer:2d}: {best[layer]}')
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f'{plan.geometry if hasattr(plan, "geometry") else "wan"}.json')
        plan.save(path)
        progress.log(f'wrote {path}')


if __name__ == '__main__':
    main()
