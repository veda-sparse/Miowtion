# Ref2VA encoding and Veda predictor deployment

## Goal

Run reference-to-audio-video generation with the MiniMax-H3 Ref2VA backbone
and a Veda predictor. The backbone's official text encoder and visual/audio
VAEs are reused; Miowtion handles reference presentation, cache serialization
and sparse tile selection.

## Input and cache contract

A JSONL manifest has one object per sample:

```json
{
  "id": "demo",
  "task": "ref2va",
  "prompt": "<complete structured H3 Ref2VA prompt>",
  "latent_t": 37,
  "references": [
    {"modality": "video", "label": "<Video 1>", "path": "media/videos/reference.mp4"},
    {"modality": "audio", "label": "<Audio 1>", "path": "media/audio/video_soundtrack.wav"},
    {"modality": "audio", "label": "<Audio 2>", "path": "media/audio/voice_reference.mp3"}
  ]
}
```

- `task` is `ref2va`, not the shorthand `r2va` used in checkpoint names.
- `prompt` contains the six H3 reference sections, in order:
  `subject_definitions`, `summary`, `retention_analysis`,
  `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
- References keep input order. Numbering is independent per modality:
  `<Picture N>`, `<Video N>`, `<Audio N>`, starting at one.
- Relative media paths resolve against `--dataset-root`; by default this is
  the parent of the manifest directory, so `manifests/prompts.jsonl` and
  `media/` share one dataset root.
- `latent_t` is an integer. It must match the requested output geometry.
- A video soundtrack is not automatically an audio reference. List it
  separately when the prompt refers to it.

Images and videos enter both the H3 text/vision encoder and visual VAE.
Audio labels enter the text presentation; audio waveforms enter AudioVAE.
The audio VAE is loaded lazily so image-only FL2VA encoding keeps its prior
memory requirements.

`encode_samples.py` does not perform PE. For new user inputs, first use the
[official H3 skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md).
The official example below already provides an expanded prompt.

The resulting cache contains `index.json`, `text.safetensors`, and
`cond.safetensors`. Reference block metadata and visual/audio row spans
travel with each sample; ground-truth target videos are not required.

## Official example

`scripts/prepare_official_r2va_demo.py` calls
`miowtion.train.reference_example.prepare_example`. It downloads the
[official request](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/scripts/readme/reproducible-768p-ref2va-request.sh)
and two reference files, preserves the PE verbatim, and exposes the video's
soundtrack as Audio 1 before the external Audio 2 voice reference.
It requires ffmpeg and ffprobe, but no API key or GPU.

```bash
python scripts/prepare_official_r2va_demo.py \
  --out artifacts/examples/minimax_h3_ref2va
python scripts/encode_samples.py --root weights/MiniMax-H3 \
  --manifest artifacts/examples/minimax_h3_ref2va/manifests/prompts.jsonl \
  --out artifacts/samples/demo
python scripts/generate.py --config configs/infer_r2va_preview.yaml
```

The inference preset uses Ref2VA, Turbo 8 steps, a 16:9 output at latent_t 37,
and the published R2VA predictor path. Download the backbone, Turbo LoRA and
predictor first, following the root README. Outputs include `veda.mp4`, its
latents and `summary.json`. Adding `--attention dense veda` also renders a
Dense output and a comparison with separate audio tracks.

## Budgets and checkpoint compatibility

`--keep-tiles` and `--keep-ratio` are mutually exclusive current-video budgets.
`--ref-keep-tiles` and `--ref-keep-ratio` are mutually exclusive visual reference
budgets. `--tile-conditions` enables tiled visual references.
The preview uses 32 reference tiles and 32 current-video tiles independently
per query tile and head, capped by the available valid tiles. Text, VLM
conditioning and reference/target audio remain global and fully attended.

The predictor bundle stores both budgets and `tile_conditions` alongside
weights and tile plans. CLI overrides take precedence; missing options
inherit the bundle's settings. Old T2VA ratio-only bundles continue to load.
FP8 weights have per-head FP32 scales and load as BF16 projections.

```bash
python scripts/export_predictor.py \
  --checkpoint runs/stage1_r2va_fixed32_8gpu/ckpt/step_0000600 \
  --plan-dir artifacts/init/veda_8nfe_step600_plans \
  --keep-tiles 32 --ref-keep-tiles 32 --tile-conditions \
  --dtype float8_e4m3fn --out weights/veda/r2va_step600_fp8.safetensors
```

The default export uses live weights; `--ema` or `--leap` selects a different
source. A packed tile plan describes a geometry, not proof of fine-tuning
at that duration.

## Training and parallel encoding

`configs/stage1_r2va_fixed32_8gpu.yaml` captures the first R2VA fine-tuning
settings: Ref2VA, four latent_t 37 aspect ratios, Turbo 8 steps, 600 updates,
accum 1 and independent 32/32 budgets. Provide its own training manifest,
encoded cache, T2VA initialization checkpoint and matching plans.
The training sampler and checkpoint/resume mechanisms remain unchanged.
With `topk_weight=0`, dense-teacher KL supervision covers the full tile
probability distribution; the tile budgets govern selection and diagnostics.

For data-parallel encoding, shard with `scripts/shard_sample_manifest.py`,
encode each shard into a separate directory, then merge with
`scripts/merge_sample_caches.py` in the original manifest order.
`scripts/validate_sample_cache.py` checks membership and condition/layout
row counts before the large teacher is loaded. Workers must not write into
the same cache.

## Input notes

Long reference videos can have more latent frames than the generated clip.
Reference noise allocation must cover the complete reference length; the
short-reference allocation and random draw order are retained unchanged.

Audio 1 in the official PE is the video soundtrack, not the external voice
file. Treating the only downloaded MP3 as Audio 1 swaps reference roles.
The example preparation explicitly creates all three labelled inputs.
