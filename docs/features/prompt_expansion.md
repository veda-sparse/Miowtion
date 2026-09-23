# Prompt 扩写（PE）：短 prompt → H3 结构化 T2VA prompt

## 目标
把短视频描述（例如 MovieGenVideoBench，每行一条）扩写成 MiniMax-H3 官方 prompt-writing skill
规定的 T2VA 结构化 prompt（`integrated_multimodal_description` / `overall_soundscape` /
`non_diegetic_music`）。每条 prompt 绑定一个训练几何（宽高比 + 时长），输出 jsonl，交给
`scripts/encode_samples.py` 离线编码（见 `training.md` 流程第 1 步）。目前只支持 T2VA。

## 设计与不变量

### 规则只有一个来源：submodule 里的 skill
- system prompt 在运行时拼装：前言 + `SKILL.md`（去掉 YAML front matter）+
  `references/base-en.txt` 原文 + 我们自己的 T2VA 输出契约（`_T2VA_CONTRACT`）。代码里不复制
  skill 内容，升级 submodule 就等于升级规则。
- H3 仓库里有三份 skill：`skills/`、`.claude/skills/`、`.agents/skills/`。以 `skills/` 为准，
  另外两份缺少 "Tips for Better Results" 一节；`find_skill_dir` 按这个顺序查找。
- 同一次运行里，所有请求的 system prompt 完全相同；每条请求不同的部分（时长、宽高比、原文）只放在
  user 消息里。这样 DeepSeek 的前缀缓存能命中，实测每条约 4.35K/4.5K 个 prompt token 命中缓存，
  命中价是未命中价的 1/50。

### 输出契约（机器可检查的部分由 `check_expansion` 强制）
- 只输出最终 prompt，以 `integrated_multimodal_description: [Shot 1]` 开头，三个字段按顺序各出现
  一次，字段之间空一行。
- 每个字段只写一段：skill 的所有例子都把后续 shot 内联在同一段里，而空行在格式中是字段分隔符。
- `[Shot 1]` 没有时间戳；之后每个 shot 以 `At MM:SS.mmm,` 开头，切点严格递增且小于时长；
  文中任何时间戳都不超过时长。
- 不在文本里写宽高比、画面方向或总时长：这些几何信息通过 latent 网格传给模型，写进文本只会引入
  噪声（踩坑 3）。
- `overall_soundscape` 不能是 `N/A`（skill 只允许在用户明确要求静音时用；我们的源 prompt 都不
  要求静音）。`non_diegetic_music` 允许 `N/A`。
- 接受条件 = `data.validate_prompt(prompt, 't2va')`（和编码脚本用的是同一个检查）+
  `check_expansion`。除去掉首尾空白外不做任何修正：不合格就重试，不静默修补。

### 重试与失败
- 每条最多尝试 1 + `max_retries`（默认 3）次。
- 输出被拒：下一次请求带上被拒的原文（assistant 消息）和 checker 的错误信息，让模型在 repair 轮里
  修正。实测一轮就能修好（见验证记录）。
- API 错误（429/5xx/网络错误/响应格式不对）：指数退避（2、4、8 s）后原样重发。400/401/402/422
  属于调用方的错误，不重试，直接判失败。
- 所有尝试的 usage 都累加进记录，这样成本统计包含重试的开销。
- 最终失败的 prompt 写到 `<output>.failures.jsonl`（含每次的错误和最后一次输出），CLI 退出码为 1。
  失败的 prompt 不会被丢掉而不报告。

### 几何
- 时长先用 `geometry.resolve_geometry` 对齐到 17n+5 帧网格；记录和告诉 LLM 的都是对齐后的真实
  时长（例如 5.17 → 124 帧 = 5.1667 s，user 消息里写 `5.167 seconds` 和时间轴终点
  `00:05.167`）。
- 默认的候选时长取 `latent_t_ladder()`（[5, 15] s）的两端：5.1667 s（latent_t 37）和
  14.375 s（latent_t 102），也就是最短和最长的可训练时长。`alternate` 模式按顺序轮换，
  `random` 模式按 `--seed` 抽取。

### API（DeepSeek，OpenAI 兼容）
- `POST https://api.deepseek.com/chat/completions`；body 包含 `model`、
  `thinking: {"type": "enabled"}`、`reasoning_effort`、`stream: false`。在原生 HTTP 请求里
  `thinking` 是顶层字段；只有用 OpenAI SDK 时才需要放进 `extra_body`。
- 不传 temperature/top_p：thinking 模式会忽略 temperature，并把 top_p 钳到 ≥ 0.95。
- 接口没有 seed 参数，结果不可复现。产物本身就是缓存，放在 `artifacts/`（不入库）。
- 只用标准库（`urllib.request` + `json`），不引入 `openai` 依赖。
- 密钥只从环境变量 `DEEPSEEK_API_KEY` 读取，不写进记录、日志和 `repr`；错误响应里的 `sk-...`
  片段一律打码。

### 输出格式
每行一个 JSON：`id, task, source_prompt, prompt, aspect, duration_seconds, latent_t, model,
served_model, reasoning_effort, attempts, usage`。`usage` 是所有尝试的累加（含
`prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、
`completion_tokens_details.reasoning_tokens`）。`encode_samples.py` 只读 `id/task/prompt`。

## 代码位置与接口
- `miowtion/train/prompt_expansion.py`：`build_system_prompt`、`build_user_message`、
  `build_request_body`、`parse_completion`、`DeepSeekClient`（`from_env`、`complete`，
  transport 可以注入）、`check_expansion`、`expand_one`、`expand_all`、`read_source_prompts`、
  `default_duration_choices`、`assign_durations`、`make_requests`、`write_jsonl`。
- `scripts/expand_prompts.py`：CLI，只做参数解析。

```bash
DEEPSEEK_API_KEY=... python scripts/expand_prompts.py \
    --source https://raw.githubusercontent.com/guandeh17/Self-Forcing/main/prompts/MovieGenVideoBench.txt \
    --count 10 --id-prefix moviegen --concurrency 10 \
    --output artifacts/prompts/moviegen_smoke10.jsonl   # 其余参数用默认值
```

## 测试
`tests/unit/test_prompt_expansion.py`（CPU，不联网，全部用假 transport）覆盖：
- 从 skill 文件拼 system prompt（去掉 front matter、顺序正确），并检查 submodule 里的真实 skill
  能加载；
- user 消息中的时长和时间轴格式，以及不含 "landscape"；
- 请求 body 和 header 的字段，`none` 时关闭 thinking，非法 effort 报错；
- 密钥不出现在 `repr` 和错误信息里，没有环境变量时报错；
- 响应解析，格式不对的响应判为可重试的错误；
- `check_expansion` 接受 skill 格式的 prompt，并逐条覆盖各类拒绝原因（参数化）；
- 重试流程：被拒后进入 repair 轮（检查第二次请求带上了原文和错误）、API 错误按退避重发、
  `finish_reason=length` 判为被拒、多次失败后放弃、401 不重试、并发时保持顺序并报告失败；
- 时长分配、帧对齐、jsonl 能被 `data.load_prompts` 读回。

## 踩坑记录
1. **`reasoning_effort: "medium"` 并不是独立档位**：DeepSeek 文档里只有 none/low/high/max 四档，
   `medium` 按兼容规则映射成 `high`，请求会被接受，不会报错。记录里写的是请求值 `medium`，实际
   效果等同 `high`。如果真想要更低的档，只能选 `low`。
2. **shot 之间有空行**：第一版契约下 8/10 条在 shot 之间插了空行，与 skill 例子的单段格式不一致
   （而且空行本来是字段分隔符）。对策：契约里写明每个字段只写一段，checker 拒收段内换行。
3. **几何信息泄漏进文本**：user 消息写成 "16:9 (landscape)" 后，模型写出了 "a close-up
   landscape shot"。去掉 "landscape" 之后，又出现了 "in a 16:9 frame"。对策：user 消息改成
   "16:9 (width:height)"，契约禁止写宽高比、方向和时长，checker 拒收包含宽高比字符串的 prompt。
4. **并发冷启动时缓存不命中**：10 条请求同时发出时，system prompt 前缀还不在缓存里（v1 只命中了
   8.7K/45K 个 token）；缓存热了以后是 48K/51K。批量很大时只影响第一批并发，不值得专门预热。
5. **模型偶尔重复字段**：v3 中 moviegen_0007 第一次输出里 `overall_soundscape:` 出现了 4 次。
   checker 拦了下来，repair 一轮就通过了。

## 验证记录
2026-09-23，macOS，DeepSeek `deepseek-flash`（响应里的 `model` 字段也是 `deepseek-flash`），
`reasoning_effort=medium`（实际等同 high），并发 10，MovieGenVideoBench 前 10 条，16:9，时长在
5.167 s 和 14.375 s 之间交替。一共三轮，契约逐轮收紧：

| 轮次 | 改动 | 通过 | repair 次数 | wall | token（缓存命中/未命中/输出） | 费用（非高峰价） |
|---|---|---|---|---|---|---|
| v1 | 初版契约 | 10/10 | 0 | 13.6 s | 8704 / 36306 / 15271 | $0.0146 |
| v2 | 要求单段、按真实时长安排动作、去掉 "landscape" | 10/10 | 1（段内空行） | 12.8 s | 45440 / 5107 / 11546 | $0.0078 |
| v3 | 增加宽高比泄漏检查（最终产物） | 10/10 | 1（字段重复） | 17.0 s | 48125 / 2955 / 12487 | $0.0081 |

v3（`artifacts/prompts/moviegen_smoke10.jsonl`）：5.167 s 的条目有 2–3 个 shot（切点在
1.9–3.6 s），14.375 s 的条目都是 4 个 shot（每个约 2.9–4.5 s）；每条 231–643 词；10 条都有
soundscape 和 music。有 2 条自行加了对白（0007、0009），2 条加了画面文字（0007、0009）。
PE agent 逐条检查的结论是：结构和时间轴都符合 skill。这个结论还没有经过人工确认。

## 待办
- 跑全量 1003 条（建议并发 16）。现在是全部完成后才一次写盘；跑全量前考虑改成增量写盘，并加
  `--resume`。
- 决定是否允许 LLM 自行加对白：skill 规定 `<d>` 里只放 "user-provided spoken content"，但
  T2VA 又允许补充细节。目前是允许的。这会影响 H3 音频分支的语音覆盖面。
- fl2va / ref2va 扩写（ref2va 用 `ref-en.txt` 的六段式）。
- 扩大时长和宽高比的分布（目前只用了两端时长和 16:9）。
- 短时长条目有时每个 shot 的动作偏多（例如 0002 有 3 个 shot，每个约 1.6–1.9 s）。如果生成的视频
  显示出节奏问题，再在契约里加 shot 最短时长。
