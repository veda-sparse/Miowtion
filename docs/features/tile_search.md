# Tile 方案搜索

## 目标
为每种几何找出每层、每个头最适合的 tile 形状（每层最多 2 种），作为训练和推理共用的方案表。

## 设计与不变量（`miowtion/veda/search.py`）
- 评分方式是 **oracle 掩码下的相对输出 MSE**：按候选排列重排 → 抽样视频 query tile → 计算真实概率 p
  和稠密输出 → 取块内最大 p 作为块分数 → 用与训练**完全相同**的规则（预算、Bresenham、对角、分段；
  全局列全保留）选块 → 在保留的块上重新归一化 → `Σ‖o_sparse−o_dense‖²/Σ‖o_dense‖²`，只统计真实行。
  用 oracle 而不是训练好的打分器，是为了把 tile 形状的局部性好坏与打分器准不准分开。
- 各候选形状在同一 (step, layer) 上使用同一个 seed 抽样（配对比较）。全局 query 行在所有方案下都
  一样，不参与评分。
- 模型本身必须跑稠密：用稀疏模型给方案评分，等于拿方案评价它自己。
- 驱动 `run_search`：在我们自己的 FSDP2 框架内运行（数据并行，每个 rank 处理不同的 clip，按步保持
  同步）。用稠密教师推进整条轨迹，只在选定的步使用 `OracleScorer`，越过最后一个评分步就停止。
  每个 clip 的结果原子写盘，并附完成标记；缺少标记的文件拒绝加载。
- `build_plan`：先按补齐上限过滤候选 → 对每个 (layer, head) 在各 (clip, step) 上取 argmin，做多数
  投票（平票看平均 MSE）→ 每层保留票数最多的 2 种形状 → 被淘汰形状的头改用保留形状中对该头更好的
  那个。provenance 记录评分项数、clip、step、补齐上限、方案 MSE、最佳单一形状 MSE、搜索时的
  保留比例。
- 默认候选菜单是 6 种形状：8x4x4、2x8x8、4x4x8、4x8x4、8x2x8、8x8x2，在 H↔W 互换下封闭。
- **kernel 路径**（CUDA 上、FA4 块稀疏与 Triton 可用、并且传入了稠密输出与 LSE 时）：
  `OracleScorer` 的稠密 attention 本来就要算，顺带取出 `out` 与 `lse`（按 token，与 tile 排列无关，
  所有候选形状共用）。每个头组一次完成：`gather_tiles` → 按 tile 顺序取 LSE（pad 槽置 0）→
  `block_heat_triton.teacher_heat`（块分数 = 块内 `exp(s·scale − lse)` 的最大值，只算 QKᵀ）→
  `select_video_blocks` → `dense_block_mask(global_rows=False)` → FA4 块稀疏只算抽样的 query tile →
  与稠密输出对应行比较。块分数与参考路径的 fp32 概率块最大值等价（exp 单调、lse 是精确的）。
  其他情况走 `oracle_rel_mse_reference`（逐头 fp32，CPU 单测用）。

- 一次运行可以搜多个几何（`geometries`），模型只加载一次；每个几何使用 latent_t（以及 aspect，
  如果样本里有）匹配的缓存样本。多组 GPU 可以用 `--geometries` 各搜一部分，写到同一个 run 目录，
  再统一 `build_plan`。某一轮的结果文件在所有 rank 上都已完成时跳过该轮（集体决定，避免 FSDP 的
  前向错步），中断后重跑即可续上。
- 9:16 等镜像宽高比不单独搜索：`build_plan.py --mirror-out` 由 16:9 的方案生成（见 veda_tiling.md
  的 `TilePlan.mirrored`）。

## 用法
```bash
# 两组进程各搜一部分几何（例如 GPU 0,1 一组 FSDP，GPU 3 单卡）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29614 scripts/search_tiles.py \
    --config configs/search_turbo8_multigeo_4090.yaml --geometries 16:9@37 16:9@72 4:3@102 16:9@102
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 29613 scripts/search_tiles.py \
    --config configs/search_turbo8_multigeo_4090.yaml --geometries 1:1@37 4:3@37 1:1@72 4:3@72 1:1@102
python scripts/build_plan.py --scores runs/<run>/scores/16x9_t37 --out plans/16x9_t37.json \
    --mirror-out plans/9x16_t37.json
```

## 测试
`tests/unit/test_veda_search.py`：全保留时误差为 0、误差随稀疏度单调上升、scorer 输出逐位等于稠密、
投票与每层 ≤2 种形状、补齐过滤、完成标记。`tests/unit/test_veda_predictor_mask.py`：子集行的
`dense_block_mask` 与全量掩码的对应行逐位相等。`tests/gpu/test_kernels_gpu.py`：kernel 路径与参考
路径的 rel-MSE 对拍（rtol 1e-2；4090 上实测相对差 ≤ 2e-4）。

## 踩坑记录
- **partial query tile 中的 pad 行**：gather 时 pad 槽指向第 0 行，拿到的是第 0 行的 q，会污染块内
  最大概率。现在在 softmax 之后先把这些行清零，再做任何计算。

- **子集行的块列表把全局列算错**（当时的 `kernel_indices`，现已由 `dense_block_mask` 取代）：全局列的起点曾用所选行数 R 代替
  `layout.n_video_tiles`，于是抽样行的列表里混进了视频列、漏掉了真正的全局列，kernel 路径的
  rel-MSE 与参考路径相差最多 88%。现在起点取自 layout，`global_rows=True` 时还检查 R 必须等于视频
  tile 数。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。
- 2026-09-23，RTX 4090，kernel 路径 vs 参考路径（56 头，38k token，6 种形状 × 16 个 query tile，
  保留 10%）：115 ms vs 2.89 s（**快 24 倍**），rel-MSE 最大差 0.018（均值 9.9，随机数据）。GPU 时间
  构成：Triton 热力图 69 ms、tile gather 24 ms、FA4 块稀疏 9 ms、其余 7 ms；同一层的稠密 FA4
  （含 LSE）为 267 ms。
- 2026-09-23，2×4090，FL2VA，16:9 5.17 s（37,24,42），保留 10%，每个 (layer, candidate) 抽 16 个
  query tile：
  - 50 步 base 教师冒烟（1 个 clip，第 0/12 步，候选 4x8x4 / 8x8x2）：方案 MSE 0.0651，最佳单一
    形状 0.0689；50 层全部用到 2 种形状。
  - **8 步 Turbo 教师**（v4_step600_ema 合并，2 个 clip，第 0/2/4/6 步，6 种形状菜单，用时 12 分
    23 秒）：各形状平均 rel-MSE 为 4x8x4 0.0567、2x8x8 0.0572、8x8x2 0.0589、8x4x4 0.0605、
    4x4x8 0.0626、8x2x8 0.0683（补齐超过 20%，被过滤）。方案 MSE 0.0532，最佳单一形状 0.0567，
    按头选择提升 6%。方案中的形状占比：2x8x8 39%、4x8x4 23%、8x4x4 22%、8x8x2 16%。rel-MSE 随
    去噪推进下降：第 0/2/4/6 步分别为 0.076 / 0.057 / 0.050 / 0.043。
