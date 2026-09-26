# 可视化对比（visual check）

## 目标

AGENTS.md 1.5 规定：**无法位级对齐的路径**（稀疏 kernel 的 softmax 累加顺序、不同精度的
打分器、少步蒸馏的不同 checkpoint）只能靠人看画面来确认，而且必须留下可以复核的证据。
本功能提供生成这份证据的唯一脚本，保证每次做出来的东西一样。

## 设计与不变量

- **产物固定**，不随任务变：
  - `<label>.mp4`：每个输入原样复制一份。证据要能独立于原始 run 存在 —— 拼好的视频没法
    再单独截帧或重新裁剪，而 `runs/` 会被清理。
  - `side_by_side.mp4`：1×N 横向拼接，**每个 pane 自己带标题栏**。
  - `report.json`：参考是谁、每个 pane 的路径（以及显式要了指标时的 PSNR / SSIM）。
  - `diff_<label>.mp4`：逐帧 |a − b| 热力图（`blend=difference` → `eq=contrast=<gain>`
    → `pseudocolor=turbo`），**默认不生成**，`--heatmap` 才有。默认 gain 8，因为原始
    差值几乎总是肉眼不可见。
- **默认不算 PSNR / SSIM**（`--metrics` 才算）：AGENTS.md 1.5.1 规定画面质量由人眼判断，
  标量指标在少步扩散里会骗人 —— 同一个打分器换 fp8 存储、选块一致率 0.9944，视频对
  bf16 也只有 20.4 dB，因为轨迹是混沌的，早期翻掉几个 block 会沿去噪过程放大。指标低
  不等于质量差，所以不拿它下结论；跑 20 条对比时还要多花一倍时间。
- **pane 顺序就是命令行顺序，baseline / 原始在前，生成结果在后**，参考默认是第一个
  （`--reference` 可改）。指标和热力图都是"其它 pane 对参考"，参考自己不和自己比。
- **标题画在 pad 出来的黑边里，不盖住画面**：先每个 pane 各自 `pad` 出 48 px 再
  `hstack`，所以不同长宽比的 pane 混排时标题始终贴着自己的画面。
- **音轨只取参考那一路**（`-map 0:a?`）：各 pane 是同一段素材，N 份音轨叠在一起只会互相
  相位抵消。`?` 让没有音轨的输入也能通过。
- **标题要转义**：`drawtext` 自己有一套语法，`:`、`'`、`%`、`,`、`[`、`]`、`;` 都会被它
  吃掉，`escape_text` 统一加反斜杠。
- **要了指标而指标缺失时直接报错**，不返回空 dict：ffmpeg 在两路尺寸或帧数不一致时会静默不跑滤镜，
  这时候"没有指标"意味着对比无效，不是没有差别。
- 输出目录约定 `artifacts/visual_checks/<feature>/<date>/`（`--feature` 就够，
  `--out-dir` 可覆盖）。

## 代码位置与接口

- `scripts/visual_check.py`
  - `parse_videos(specs)`：`LABEL=PATH` 列表 → 有序 `{label: path}`，空标签 / 重名报错。
  - `escape_text(label)`、`stack_command()`、`heatmap_command()`、`metrics_command()`：
    纯函数，返回 ffmpeg argv，可以单测，不碰文件系统。
  - `parse_metrics(stderr)`：从 ffmpeg 的报告里取平均 PSNR（含 `inf`）和 SSIM 的 `All:`。
  - `main()`：落盘并打印；`--metrics` / `--heatmap` 是两个 opt-in 开关。
- 依赖外部的 `ffmpeg`（`--ffmpeg` 可指定路径），不是 Python 依赖。

## 测试

- `tests/unit/test_scripts_visual_check.py`（10 个）：解析与报错、标题转义、拼接 argv 里
  pane 数与 `hstack=inputs=N` 一致、热力图与指标 argv 的滤镜链、`parse_metrics` 的
  `inf` 与两个都缺时报错。生成的 argv 只做断言，不真的调 ffmpeg。

## 踩坑记录

- 暂无。

## 验证记录

- 2026-09-26 macOS CPU：`pytest tests/unit` 全过（含本功能 10 个）。
- 2026-09-26 macOS：20 条 holdout 样本的 `Dense | Veda t37 step600 | Veda t102 step200
  (fp8)` 三联拼接（`artifacts/visual_checks/holdout20_step200/`），每条另有三路单独
  视频。

## 待办

- 人工确认过的结论要写回对应 feature 文档的「验证记录」（确认人、日期、commit、结论），
  本脚本只负责生成材料。
