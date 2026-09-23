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

## 用法
```bash
torchrun --nproc_per_node 4 scripts/search_tiles.py --config configs/search_16x9_t37.yaml
python scripts/build_plan.py --scores runs/<run>/scores/16x9_t37 --out plans/16x9_t37.json
```

## 测试
`tests/unit/test_veda_search.py`：全保留时误差为 0、误差随稀疏度单调上升、scorer 输出逐位等于稠密、
投票与每层 ≤2 种形状、补齐过滤、完成标记。

## 踩坑记录
- **partial query tile 中的 pad 行**：gather 时 pad 槽指向第 0 行，拿到的是第 0 行的 q，会污染块内
  最大概率。现在在 softmax 之后先把这些行清零，再做任何计算。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。
- 2026-09-23，2×4090，FL2VA，16:9 5.17 s（37,24,42），保留 10%，每个 (layer, candidate) 抽 16 个
  query tile：
  - 50 步 base 教师冒烟（1 个 clip，第 0/12 步，候选 4x8x4 / 8x8x2）：方案 MSE 0.0651，最佳单一
    形状 0.0689；50 层全部用到 2 种形状。
  - **8 步 Turbo 教师**（v4_step600_ema 合并，2 个 clip，第 0/2/4/6 步，6 种形状菜单，用时 12 分
    23 秒）：各形状平均 rel-MSE 为 4x8x4 0.0567、2x8x8 0.0572、8x8x2 0.0589、8x4x4 0.0605、
    4x4x8 0.0626、8x2x8 0.0683（补齐超过 20%，被过滤）。方案 MSE 0.0532，最佳单一形状 0.0567，
    按头选择提升 6%。方案中的形状占比：2x8x8 39%、4x8x4 23%、8x4x4 22%、8x8x2 16%。rel-MSE 随
    去噪推进下降：第 0/2/4/6 步分别为 0.076 / 0.057 / 0.050 / 0.043。
