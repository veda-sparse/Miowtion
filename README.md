<div align="center">

<img src="assets/miowtion-logo.svg" width="160" alt="Miowtion logo">

# Miowtion

**Sparse acceleration and LoRA fine-tuning of [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) on resource-constrained machines.**

| <a href="docs/INDEX.md"><b>Documentation</b></a> | <a href="#getting-started"><b>Quick Start</b></a> | <a href="docs/pitfalls.md"><b>Pitfalls</b></a> | <a href="AGENTS.md"><b>Contributing</b></a> |

</div>

Miowtion makes the 33B MiniMax-H3 audio-video DiT fast to run and cheap to adapt
on hardware it was not built for: consumer GPUs with 24 GB (verified on 2x / 1x
RTX 4090), with the weights streamed from host memory, and, as work in progress,
Apple silicon with NVMe offloading.

- **Sparse acceleration (Veda).** Video tokens are permuted into 3D tiles of 128
  tokens, a small per-head predictor scores (query tile, key tile) pairs, and only
  the top-scoring blocks are computed by a FlashAttention-4 (CuTe DSL) block-sparse
  kernel (vendored SM8x patch for RTX 30/40). At 90% sparsity one RTX 4090 runs a
  14.4 s 16:9 clip ~3.1x faster end to end than dense attention.
- **LoRA fine-tuning.** FSDP2 training with partial CPU offload, stream-overlapped
  weight copies, precomputed AdaLN tables and memory-bounded (chunked) kernels:
  stage 1 trains the predictor, stage 2 recovers quality with LoRA.

Both H3 checkpoints are supported: **FL2VA** (t2va / fl2va) and **Ref2VA**
(ref2va), with the base 50-step teacher or a merged few-step (4 / 8-step Turbo)
LoRA teacher.

## NEWS

- `2026/09/23`: First public version: H3 DiT port with bit-exact packing,
  per-head dynamic tile permutations, oracle tile-plan search, stage-1 predictor
  training with FSDP2 (verified on real weights on 2x RTX 4090), few-step Turbo
  LoRA teachers, and an FA4 CuTe block-sparse kernel study on SM89.
- `2026/09/23`: Released [`data/prompts/moviegen_video_bench_h3.jsonl`](data/prompts):
  all 1003 MovieGen Video Bench prompts expanded into structured H3 T2VA prompts.
- `2026/09/23`: FA4 block sparsity on SM8x (verified on RTX 4090) through a
  vendored, hash-checked patch of the FA4 SM80 kernels (`miowtion/kernels/fa4_sm8x`).

## Key Features

- **H3 DiT for training** (`miowtion/h3`)
  - Packed-sequence layout for t2va / fl2va / ref2va (fp64 3D RoPE grid, 64-row
    alignment), per-row timestep slots, dual video/audio flow schedules.
  - Strict loader for the released checkpoints (per-head interleaved fused QKV
    reordered at load), shard-local loading into FSDP2 DTensors.
  - Fixed eager bf16 op chain; pluggable attention shared by teacher, student
    and tile search.
- **Veda sparse attention** (`miowtion/veda`)
  - Dynamic tile sizes: up to two (t, h, w) tile shapes per layer (per head),
    padded grids with valid-prefix tiles, segmented reference/target budgets.
  - Equal-kernel-cost budgets with Bresenham rounding, forced diagonal tiles.
  - Predictor (mean/max/min pooling + per-head residual projection, 275M params),
    teacher block heat from the dense LSE (fused Triton kernel), seer KL, recall.
  - FA4 CuTe block-sparse integration (singleton `mask_mod`, full/partial block
    lists, transposed backward lists, architecture guard against silent dense
    fallback), on SM90 / SM100 and, through a vendored patch, on SM8x.
  - Oracle tile-plan search (relative output MSE under the best mask), majority
    vote, at most two shapes per layer.
- **Training** (`miowtion/train`)
  - FSDP2 per-block sharding with partial CPU offload + prefetch, precomputed
    AdaLN tables (drops 13B parameters from the frozen trunk), MLP row chunking,
    host-offloaded optimizer state.
  - The sampler is the training loop: teacher-driven trajectories from pure noise;
    few-step teachers roll out on their own grid.
  - Stage 1 (predictor) and optional stage 2 (LoRA recovery with a frozen
    teacher); atomic checkpoints with resume / init modes.
  - Offline prompt expansion (H3 prompt-writing skill + LLM) and encoding (the
    release's Qwen3-VL text tower and VAEs).
  - Immediate progress logs (done / total, elapsed, ETA) for every long phase.

## Getting Started

We use [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules git@github.com:veda-sparse/Miowtion.git && cd Miowtion
uv venv --python 3.12 --seed && source .venv/bin/activate

# CPU development: unit tests only
uv pip install -e '.[dev]' && pytest tests/unit -q
git config core.hooksPath .githooks   # pre-commit leak checks (see AGENTS.md)

# GPU machines (CUDA 12): torch + FA4 CuTe + Triton + prompt encoding
UV_TORCH_BACKEND=cu126 uv pip install -e '.[dev,gpu,encode]' torchvision accelerate
```

Download the weights (FL2VA fully; for Ref2VA only its DiT differs, the text
encoder / VAEs are byte-identical and can be symlinked). Configs refer to them
through the git-ignored `weights/` directory; if they live elsewhere, symlink
`weights/MiniMax-H3` and `weights/turbo_lora` there instead.

```bash
ROOT=weights/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" \
    "Ref2VA/model_index.json" "Ref2VA/transformer/*" "Ref2VA/tokenizer/*" \
    "Ref2VA/processor/*" --local-dir $ROOT            # ~210 GB
for c in text_encoder video_vae audio_vae; do ln -s ../FL2VA/$c $ROOT/Ref2VA/$c; done
# Few-step teacher (optional): larryvrh/MiniMax-H3-Turbo-Lora
hf download larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors \
    --local-dir weights/turbo_lora
```

### Install with an AI coding agent

```text
Set up Miowtion (this repository) for training on this machine.
1. Read AGENTS.md and docs/INDEX.md first and follow their rules.
2. Detect the platform (nvidia-smi, nvcc --version); create a uv venv with Python 3.12.
3. Install with the gpu/encode extras; run `pytest tests/unit -q`.
4. Run scripts/smoke_dit.py on the real weights and report time and peak memory.
```

## Workflow

| Step | Command | Output |
|---|---|---|
| 1. Prompt expansion | `DEEPSEEK_API_KEY=... python scripts/expand_prompts.py ...` (or use `data/prompts/`) | structured H3 prompts (`.jsonl`) |
| 2. Offline encoding | `python scripts/encode_samples.py --root $ROOT --manifest prompts.jsonl --out $CACHE` | sample cache |
| 3. Tile-plan search | `torchrun --nproc_per_node N scripts/search_tiles.py --config configs/search_turbo8_16x9_t37_4090.yaml` | per-clip scores |
| 4. Build the plan | `python scripts/build_plan.py --scores $RUN/scores/16x9_t37 --out plans/16x9_t37.json` | tile plan |
| 5. Stage-1 training | `torchrun --nproc_per_node N scripts/train.py --config configs/stage1_turbo8_4090.yaml` | predictor checkpoints |
| 6. Generate (dense vs Veda) | `torchrun --nproc_per_node 1 scripts/generate.py ... --attention dense veda` | videos, titled side by side, timing |

Teacher configurations (`schedule` / `num_steps` / `teacher_adapter` in the configs):

| Teacher | Schedule | Configs |
|---|---|---|
| Base, 50 points (49 steps) | `base` / `49` | `*_base50_*.yaml` |
| Turbo LoRA, 8 steps (current target) | `turbo` / `8` | `*_turbo8_*.yaml` |
| Turbo LoRA, 4 steps | `turbo` / `4` | `*_turbo4_*.yaml` |

Tools: `scripts/smoke_dit.py` (real-weight smoke run: load, dense / teacher /
scorer forwards with time and memory), `scripts/bench_sparse_attention.py`
(block-sparse kernels vs dense, new random pattern per call).

## Results (2x RTX 4090, FL2VA, 16:9 5.17 s = 38k tokens)

| Measurement | Value |
|---|---|
| Teacher forward (30 of 50 blocks offloaded, prefetch 1) | 29.3 s, 19.1 GiB peak |
| Stage-1 teacher forward (predictor KL) | 33.6 s, output bit-identical to dense |
| Stage-1 smoke (base teacher, 3 updates) | KL 1.179 → 1.024, recall 0.477 → 0.508 |
| Oracle rel-MSE at 10% keep (base teacher) | per-head plan 0.0651 vs best single shape 0.0689 |

Block-sparse kernels on RTX 4090 (8 heads, 90% sparsity, new pattern per call):

| Kernel | 16k tokens | 32k tokens |
|---|---|---|
| Dense (SDPA flash) | 7.25 ms | 26.9 ms |
| Upstream FA4 block sparse | not supported on SM89 (silently dense) | — |
| FA4 + `miowtion/kernels/fa4_sm8x` (forward) | 0.70 ms, efficiency 0.97 | 2.70 ms, efficiency 1.00 |
| FlexAttention (forward, +BlockMask) | 0.81 ms, 0.91 | 2.90 ms, 0.94 |
| FastVideo Triton VSA | 0.99 ms, 0.74 | 3.89 ms, 0.70 |

Details, open items and verification records: [docs/INDEX.md](docs/INDEX.md).

## Repository Layout

```
miowtion/h3      H3 DiT: geometry, layout, schedule, noise, model, weights
miowtion/veda    tiling, plan, predictor, mask, heatmap, attention, search
miowtion/kernels fa4 (single FA4 entry), fa4_sm8x (vendored SM8x patch),
                 block_heat_triton, reference, bench
miowtion/train   parallel (FSDP2), teacher, trajectory, trainer, checkpoint, lora, data, encode
miowtion/infer   pipeline (dense / Veda denoising), decode (VAEs, mp4, comparison)
scripts/         thin CLI entry points          configs/   run configs
tests/unit       CPU tests (run before every commit)   tests/gpu   GPU tests
docs/            knowledge base: features, pitfalls, dependencies
data/prompts     released prompt sets (see its README for source and license)
third_party/     pinned submodules (MiniMax-H3)
```

## Contributing

Read [AGENTS.md](AGENTS.md): Google Python style, docs updated with every feature
and pitfall, bit-exact alignment for data transforms (visual human sign-off when
bit-exactness is impossible), no machine-local information or secrets in the
repository (enforced by the `.githooks` leak checks), and rules for multiple
agents working in one tree.

## License

MIT (see [LICENSE](LICENSE)). MiniMax-H3 weights and code are subject to their own
licenses. `miowtion/kernels/fa4_sm8x` contains patched FlashAttention files under
their BSD-3-Clause license. `data/prompts/moviegen_video_bench_h3.jsonl` contains
the MovieGen Video Bench prompts and is licensed CC BY-NC 4.0 (non-commercial).
