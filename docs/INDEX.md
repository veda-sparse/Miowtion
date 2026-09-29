# docs 目录

本目录是项目知识库，组织方式类似 skill：本文件是目录，每个子功能一个文档。改代码时同步改对应
文档；踩过的坑写进 feature 文档的「踩坑记录」，并在 `pitfalls.md` 加一行索引。

| 文档 | 内容 | 状态 |
|---|---|---|
| [features/h3_model.md](features/h3_model.md) | H3 DiT 训练侧实现：请求几何、打包布局、调度、噪声、模型、权重加载 | 真实权重前向已验证；视频可视化检查待做 |
| [features/veda_tiling.md](features/veda_tiling.md) | tile 排列（按头动态 tile 形状 + 按 tile 变长）、方案表 | CPU 测试通过 |
| [features/veda_predictor_mask.md](features/veda_predictor_mask.md) | 打分器、预算与掩码规则、教师热力图、KL、recall | CPU 测试通过 |
| [features/veda_kernel.md](features/veda_kernel.md) | FA4 CuTe 块稀疏整合（含 vendored 的 SM8x 补丁）、架构支持、测速结果 | SM89 已通过 GPU 测试；SM120 前向 + 反向已验证；SM90/100 待验证 |
| [features/tile_search.md](features/tile_search.md) | oracle 评分、投票、每层 ≤2 种形状、搜索驱动 | 真实权重上 50 步 / 8 步搜索已跑通 |
| [features/training.md](features/training.md) | FSDP2 训练框架：阶段 1/2、少步 LoRA 教师、AdaLN 表、数据、checkpoint | 阶段 1 已在 2×4090 真实权重上跑通；阶段 2 待 GPU 验证 |
| [features/prompt_expansion.md](features/prompt_expansion.md) | prompt 扩写：短 prompt → H3 T2VA 结构化 prompt（DeepSeek；system prompt 由 H3 skill 拼成；校验 + repair 重试） | CPU 测试通过；MovieGenVideoBench 全量 1003 条已扩写并发布到 `data/prompts/` |
| [features/evaluation.md](features/evaluation.md) | 评测集与评测流程 v1：固定的 20 条 holdout（3 种时长 × 4 种纵横比）、数值评测（打分器 recall / heat_kept）与人眼对比（每路单独视频 + 拼接视频）的跑法、成本与已有基线 | v1 已在 1×RTX PRO 6000 上用于数值对比；20 条视频对比进行中 |
| [features/inference.md](features/inference.md) | 推理：少步 LoRA 教师 + Veda 稀疏，稠密 / 稀疏并排对比，VAE 解码与 mp4 | 实现中，4090 上首次生成 |
| [features/mlx_inference.md](features/mlx_inference.md) | Apple silicon 推理：MLX block 前向、Veda 块稀疏（gather 版）、NVMe offloading（slab / mx.load / mmap）、视频 VAE 的 MLX 解码、实测 I/O 与算力 | 可行性研究 + 原型；CPU 测试通过，合成权重实测，真实权重待验证 |
| [features/quant_scoring.md](features/quant_scoring.md) | 低精度块打分：fp8 / nvfp4 / mx 的 fake quant（含 smooth-K）与 top-k 选择误差探针；打分器 bundle 的 bf16 / fp8 存储精度对比 | CPU 测试通过；GPU 实测进行中 |
| [features/visual_check.md](features/visual_check.md) | 可视化对比：1×N 带标题拼接、逐帧差值热力图、PSNR / SSIM，供人工确认无法位级对齐的路径 | CPU 测试通过 |
| [benchmark/performance.md](benchmark/performance.md) | 实测性能：T2VA 推理（各长度的注意力 / step 加速拆解、文本编码、VAE 解码）与训练，稳态 step time、各 stage 耗时、端到端、diffusion MFU；`--random-weights` 在没有权重的机器上量 step | 1×4090 已测；1×RTX PRO 6000 随机权重已测；多卡与 R2VA 待测 |
| [dependencies.md](dependencies.md) | 外部依赖、锁定版本、依赖的内部接口 | 当前 |
| [pitfalls.md](pitfalls.md) | 踩坑总表 | 持续更新 |

## feature 文档模板

```markdown
# <功能名>

## 目标
## 设计与不变量        （为什么这样做；哪些性质不能被破坏）
## 代码位置与接口
## 测试               （unit / gpu，各测什么）
## 踩坑记录           （现象 / 原因 / 对策）
## 验证记录           （日期、机器、commit、结果；无法位级对齐时的人工确认）
## 待办
```
