# Frozen-teacher Sparse LoRA recovery

## 目标与范围

保存 T2VA 闪光修复实验使用的训练实现：冻结已合并 Turbo 的 Dense teacher，
训练 Sparse student 的新增 LoRA，使同一输入状态下的音视频 velocity 接近 teacher。
这是原有在线轨迹方案，不包含 Dense50 视频重加噪或 M=2 区间监督方案。
代码基于实验使用的仓库版本 `6dfda02`，训练数值实现与历史实验源码快照一致。

## 模型与训练目标

| 参数 | 全层实验设置 |
|---|---|
| Backbone | MiniMax-H3 FL2VA，BF16 |
| Turbo | v4 step600 EMA；8 NFE；先合并到冻结 backbone |
| Predictor | T2VA Veda 8NFE step600 FP8 Preview，冻结 |
| Teacher | Dense attention；仅关闭新增 LoRA |
| Student | 全部 50 层 Sparse attention；启用新增 LoRA |
| Attention 预算 | `keep_ratio=0.1`；强制对角块和 global 行另计 |
| 新 LoRA | 层 0–49；`attn.qkv_proj/out_proj`、`mlp.fc1/fc2` |
| Rank / alpha | 32 / 32；scale 1；无 dropout |
| 参数 | 200 个 Linear，400 个 A/B 张量，149,094,400 个 FP32 参数 |
| 初始化 | Turbo 合并期间关闭新增 hook，合并后重置 A 随机、B 为零 |
| 冻结 | 原 backbone、Turbo、predictor/projector、refiner、输出头 |
| Loss | `MSE(video_student, video_teacher) + MSE(audio_student, audio_teacher)` |
| KL / EMA | 均无；不训练 predictor，不产生 predictor KL |
| Optimizer | AdamW；betas=(0.9, 0.95)，eps=1e-8，weight decay=0 |
| LR schedule | 线性 warmup，之后恒定；全局梯度 L2 裁剪 1.0 |
| 并行 | FSDP2；历史实验使用 4×RTX PRO 6000 96GB |
| Batch | 每卡 1 个状态，accum=4；四卡满批为 16 个监督状态/更新 |
| 采样 | `trajectory`：Dense teacher Euler 推进，逐个监督 Turbo8 的 state0–7 |
| 执行 | GPU 常驻；LoRA 行分块 1024，MLP 行分块 8192，checkpoint group=5 |

同一轨迹的 teacher velocity 同时用于监督和推进下一状态；student 不推进前置轨迹。
随机模式 `random` 每次访问重新采样 prompt/noise 和一个离散状态，teacher 无梯度
重放到该状态。它支持单独实验，不能与顺序模式的 checkpoint 隐式混用。
同一微步各 rank 使用相同几何及状态，prompt 和噪声各自独立。

## 实验配置

配置文件均位于 `configs/`，以下列出完整文件名。

| 配置 | LR | Warmup | 目标更新数 | 已观察结果 |
|---|---:|---:|---:|---|
| `sparse_lora_t2va_r32_all_attn_mlp_lr1e-2_ratio10_100step_sequential.yaml` | 1e-2 | 5 | 100 | 初始全层配方，后续引入时长均衡 |
| `sparse_lora_t2va_r32_all_attn_mlp_lr1e-2_ratio10_100step_sequential_balanced.yaml` | 1e-2 | 5 | 100 | step10/50 推理出现明显噪点 |
| `sparse_lora_t2va_r32_all_attn_mlp_lr1e-3_ratio10_warmup20_100step_balanced.yaml` | 1e-3 | 20 | 100 | 已完成 100 更新；部分位置闪光减弱，未稳定修复全部 case |
| `sparse_lora_t2va_r32_all_attn_mlp_lr1e-2_ratio10_warmup20_150step_balanced.yaml` | 1e-2 | 20 | 150 | 实际中断于 64 更新，最近完整 checkpoint 为 60；画面明显退化 |

最后一组的 150 是计划目标，不能当成已完成工作量。warmup20 未消除本轮高 LR
实验的画面退化。loss 有限或下降不构成质量修复证据。
另外保留 `sparse_lora_t2va_r32_l0_1_49_*` 三层配置作为历史实验/调试入口；
它们的目标模块、预算、学习率和采样设置与全层配置不同。

## 数据

历史数据集为 `t2va_lora_train1k_v2_aspect180`，900 train / 100 validation；
Demo Enhanced PE 685 条原文和 315 条画幅改写。保留原始 prompt、画幅、时长和
内容家族隔离。原始文本、编码缓存、权重及实验产物需单独准备，不随此分支发布。

| 时长 | 总量 | Train | Validation |
|---|---:|---:|---:|
| 5s | 323 | 290 | 33 |
| 10s | 295 | 266 | 29 |
| 15s | 382 | 344 | 38 |

| 画幅 | 总量 | Train | Validation |
|---|---:|---:|---:|
| 16:9 | 420 | 378 | 42 |
| 9:16 | 220 | 198 | 22 |
| 1:1 | 180 | 162 | 18 |
| 4:3 | 180 | 162 | 18 |

文本编码使用 Qwen3-VL 前 50 层、BF16、5120 维；不加 chat template 或 special
tokens，不截断 PE。训练只读取离线文本缓存，监督由在线 Dense teacher 生成。
`duration_balance_window=50` 按同步 prompt 批次均衡时长，并保留前 10 批；
四卡 accum4 顺序训练时，前 100 更新中 5/10/15s 分别为 34/34/32 更新。
更换 world size 或训练长度后须重新检查分布。

## 准备与运行

安装仓库依赖和锁定的 GPU/encode extra，初始化只读参考 submodule：

```bash
git submodule update --init --depth 1
python -m pip install -e '.[dev,gpu,encode]'
```

配置中的 `weights/`、`artifacts/`、`runs/` 均为仓库相对路径；按配置准备
backbone、Turbo 和 predictor bundle。GPU 必须使用包含对应架构的 Torch 构建；
SM120 的 FA4 补丁安装要求见 [依赖文档](../dependencies.md)。

源 JSONL 每行至少包含 `id`、`task: t2va`、结构完整的 `prompt`、`aspect`、
`latent_t`；5/10/15s 对应历史数据中的 37/72/102。可携带 `text_family_id`、
`source_job_id`、`parent_id` 等来源键用于隔离。

```bash
PYTHONPATH=. python scripts/train_sparse_lora.py --prepare-data \
  --manifest artifacts/source/train.h3.jsonl \
  --out artifacts/sparse_lora_data
cat artifacts/sparse_lora_data/full.train.jsonl \
  artifacts/sparse_lora_data/full.validation.jsonl \
  > artifacts/sparse_lora_data/full.encode.jsonl
PYTHONPATH=. python scripts/train_sparse_lora.py --encode \
  --root weights/MiniMax-H3 \
  --manifest artifacts/sparse_lora_data/full.encode.jsonl \
  --out artifacts/samples/sparse_lora_full
PYTHONPATH=. python scripts/train_sparse_lora.py --cpu-preflight \
  --config configs/sparse_lora_t2va_r32_all_attn_mlp_lr1e-3_ratio10_warmup20_100step_balanced.yaml \
  --report artifacts/sparse_lora_cpu_preflight.json
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  scripts/train_sparse_lora.py --train \
  --config configs/sparse_lora_t2va_r32_all_attn_mlp_lr1e-3_ratio10_warmup20_100step_balanced.yaml
```

`--prepare-data` 需要足够的 5s/16:9 样本构造 24 train / 12 validation 的 pilot。
已有历史 split 时直接使用原清单，不重新划分。新数据的 validation 比例按内容组
分配，跨多条记录的家族可能使最终条数偏离精确 90/10。

续训在同一训练命令增加 `--resume runs/<run_name>/step_0000050`，路径须指向
含完整标记的检查点。保存包含新增 LoRA safetensors 和恢复用 state/完成标记；
恢复 LoRA、Adam、RNG、采样游标及尚未结束的 teacher 轨迹，不重启 warmup。
配置、资产、预算、world size 和数据/缓存身份严格检查；不自动转换不兼容实验。

原生推理加载接口为 `sparse_lora.load_sparse_adapter(model, path, contract)`：
先建立相同 backbone/Turbo 和新 LoRA hook，再加载新增 A/B。导出为 Miowtion
的 grouped Q/K/V 布局，不能直接作为原发布模型或 Comfy 的通用 merger 输入。
此分支提供训练/原生加载接口，未将实验专用 Comfy 评测控制器纳入公共生成 CLI。

## 测试与验证记录

CPU 测试覆盖模块选择、参数冻结、零初始化、teacher 稳定性、所有状态的参考
推进、尾批归一化、时长均衡和 checkpoint/Adam/RNG 恢复。GPU 测试覆盖真实
FA4 稀疏反向、融合排列、有序 dQ、50 层小模型更新及重放。

```bash
python -m pytest tests/unit -q
python -m pytest tests/gpu -q
# 四卡合成 FSDP fixture 单独执行：
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  tests/gpu/test_sparse_lora_fsdp_gpu.py --output artifacts/fsdp_check
```

历史验证（2026-10-15 至 2026-10-21，RTX PRO 6000 SM120）：单卡原生随机/
顺序训练及新进程重放、四卡合成 FSDP 尾批/恢复、原生 15s 输入 backward 和
保存均有独立检查；全层配置完成上述真实训练与 T2VA/R2VA 视频评测。
这些是工程及实验记录，尚无整套评测集闪光已修复的结论。

发布前回归（2026-10-29；基线 `6dfda02`，训练源文件与历史快照一致）：
完整 CPU suite 为 303 passed / 13 skipped（可选 MLX 和依赖项）；
RTX PRO 6000 SM120 的完整 `tests/gpu` 为 29 passed，65.14 秒。
覆盖 kernel 对照、50 层小模型反向/更新、尾批和恢复；本次未重跑四卡
原生权重训练。GPU suite 的结果仅覆盖本次列出的硬件及测试输入。

数值边界：有序 dQ 对原 FA4 的 BF16 fixture max abs 为 0.001953125，
相对 L2 约 4.0e-5；对 FP32 sparse reference 的相对 L2 约 0.264%。
forward 和 dK/dV 与原 FA4 逐位一致，重复梯度可逐位重放。
LoRA 行分块 fixture 输出逐位一致，但输入/A/B 梯度 relative L2 分别为
3.85e-6 / 3.61e-7 / 2.51e-7；不同 GEMM/归约形状不保证任意输入逐位相等。
跨实现的最终视频等价性仍需完整人工评测，不作为主分支合入结论。

## 踩坑记录

- 推理 Triton gather/scatter 原来没有 autograd，冻结权重仍需输入梯度。
  使用融合 gather/pool 加逆排列 backward；predictor pool/top-k 不反传。
- SM8x/SM120 的 FA4 dQ 原子累加不能保证恢复重放逐位一致。
  `ordered_dq: true` 保留 FA4 前向/dK/dV，单独用固定顺序 Triton 计算 dQ；
  仅支持已声明的 BF16 MHA D128 架构，默认不影响其他调用。
- FSDP 初始化必须先生成全局 A 再按行切分，避免每卡同 seed 重复 A 行。
- 长片需限制 LoRA FP32 临时量和保存激活；行分块不改变 Veda tile 或预算。
  group checkpoint 保留组内 checkpoint，减少长期保存输入；CPU 暂存默认关闭。
- 末批缺样 rank 仍执行 forward/backward，loss 权重为零；整个累积窗口按
  全局有效状态数归一化，不能分别放大小尾批。
- B 零初始化时首次 A 梯度为零是预期；B 及后续 A/B 梯度须有限且非零。
- 导出张量一致、latent 有限不代表去噪质量正常；高 LR 曾造成明显噪点。
  Comfy 融合算子还可能绕过 MLP hook，须核对完整模块调用后才能比较质量。
