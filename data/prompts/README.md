# Prompt sets

## `moviegen_video_bench_h3.jsonl`

All 1003 prompts of the MovieGen Video Bench, each expanded into a structured
MiniMax-H3 T2VA prompt (`integrated_multimodal_description` /
`overall_soundscape` / `non_diegetic_music`) following the H3 prompt-writing
skill (`third_party/MiniMax-H3/skills/h3-prompt-writing`). The file is a manifest
for `scripts/encode_samples.py`.

- Source prompts: `benchmark/MovieGenVideoBench.txt` of
  [facebookresearch/MovieGenBench](https://github.com/facebookresearch/MovieGenBench)
  @ `bab7753` (the copy in `guandeh17/Self-Forcing` `prompts/MovieGenVideoBench.txt`
  is identical). `id` `moviegen_NNNN` is the 0-based index of the non-empty line.
- Expansion: DeepSeek `deepseek-flash`, `reasoning_effort=medium` (served as high),
  16:9, durations alternating between 5.167 s (latent_t 37) and 14.375 s
  (latent_t 102). Every record passed `data.validate_prompt` and
  `prompt_expansion.check_expansion`; 156 needed one repair round and 2 needed two.
  The set as a whole has not been reviewed by a human.
- Fields: `id`, `task` (`t2va`), `source_prompt`, `prompt`, `aspect`,
  `duration_seconds`, `latent_t`, `model`, `served_model`, `reasoning_effort`.

| Duration | Prompts | Shots (count: prompts) | Words (min / median / max) |
|---|---|---|---|
| 5.167 s | 502 | 1: 44, 2: 404, 3: 54 | 149 / 269 / 414 |
| 14.375 s | 501 | 1: 11, 2: 11, 3: 261, 4: 216, 5: 2 | 160 / 410 / 709 |

314 prompts contain dialogue (`<d>`) added by the expansion; the source prompts
have none.

Regenerate (the API key is read from the environment only):

```bash
DEEPSEEK_API_KEY=... python scripts/expand_prompts.py \
    --source https://raw.githubusercontent.com/facebookresearch/MovieGenBench/bab7753fce1108def167a1efb4c1ae1d69b8f03a/benchmark/MovieGenVideoBench.txt \
    --id-prefix moviegen --concurrency 100 --output moviegen_bench.jsonl
```

The output then only differs in run-specific fields (`attempts`, `usage`) and in
the sampled LLM text.

### License

The source prompts are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Meta
(MovieGen Bench). This file contains them together with expansions derived from
them, and is distributed under the same CC BY-NC 4.0 license (non-commercial
use only), not under the repository's MIT license.

## `minimax_h3_vbench_pe.jsonl`

The 251 unique source prompts used by the VEDA2 six-dimension VBench v1
protocol, expanded once per source prompt for MiniMax-H3 T2VA. Each row is
reused for all five generation seeds and attention backends, covering 1,255
prompt/seed pairs. Use `source_prompt` to join the VBench sampling manifest
and name videos. Use `prompt` as the model input.

- Source: [VBench v1 prompt metadata][vbench-source] at commit
  `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. The subset covers subject
  consistency, background consistency, motion smoothness, aesthetic quality,
  imaging quality, and dynamic degree.
- Generation: DeepSeek `deepseek-flash` with the official [H3 prompt-writing
  skill][h3-skill], followed by format validation and semantic review. All
  captions use 16:9, 14.375 seconds, 345 frames at 24 fps, and `latent_t=102`.
  The three H3 sections are kept in order and the descriptions use one shot.
- Fields: `id`, `task`, `source_prompt`, `prompt`, geometry, model and template
  provenance, prompt SHA-256, and review metadata. The manifest is readable by
  `scripts/encode_samples.py`.

The JSONL includes unmodified source captions and H3 adaptations derived from
them. The VBench data in this file is distributed under
[Apache License 2.0](LICENSE.VBench).

[vbench-source]: https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/VBench_full_info.json
[h3-skill]: https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing
