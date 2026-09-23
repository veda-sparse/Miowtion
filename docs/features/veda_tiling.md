# Veda tile 排列与方案表

## 目标

把视频 token 重排，使每连续 128 个 token 构成 (T,H,W) 网格上的一个 3D 小块，让块稀疏 kernel
以 128×128 为单位整块计算或整块跳过。

## 设计与不变量

### 动态 tile 大小：两个层面，都必须精确
1. **按头动态形状**：同一层中不同头可以用不同的 (tt,th,tw)（乘积 128），每层最多 2 种。
   形状相同的头组成一个头组，每个头组有自己的一套排列。
2. **按 tile 变长**：网格不整除时要补齐，补齐位置为 −1。每个 tile 内把真实行**稳定地**移到前面，
   所以真实行总是长度为 `valid_count[tile]`（0..128）的**前缀**。池化、热力图和 kernel
   的 mask_mod 都依赖这个前缀性质，不需要逐行掩码。

### 排列构造（`span_tiles`）
- 网格补齐到形状的整数倍 → 重排成 `(h块, w块, t块)` 外层 × `(t, h, w)` 内层 → tile 内部做稳定压缩。
- 每个视频类 span（参考/关键帧和目标）各自使用自己的网格和形状，按序列顺序拼接；目标 span 放在最后。
  排在目标之前的 tile 数记为 `n_ref_tiles`。
- 其余真实行（文本、音频、未切 tile 的条件）是全局 token，按原顺序每 128 行一个 tile，接在所有视频
  tile 之后。pad 段 `[used, seq_len)` 不进任何 tile。
- 条件 span 的形状取补齐最少的；补齐相同时取最接近立方体的。

### 缓存为常量的派生量（`TileLayout`）
`gather_index`（pad 指向第 0 行）、`scatter_index`（pad 指向多开的第 S 行）、`pad_slots`、
`partial_tiles`、`kv_ok`、`full_tile`、`slot_valid`。这些量每个 clip、每种形状、每个设备只构建一次，
热路径上不再做 nonzero/tolist（这类操作会引起设备同步）。

### gather / scatter
- gather 输出 seq-major 的 `[N, H', D]`（即 FA4 的原生 `[B,S,H,D]` 布局），只把 pad 槽置零。
- scatter 写入 `[S+1, H, D]` 缓冲区，pad 统一落到最后一行，逆排列只需一次索引写。

### 方案表（`miowtion/veda/plan.py`）
- `TilePlan`：`shapes` + 每层每头的形状编号 + 网格 + 来源信息（provenance）。与 timestep 无关。
  每层超过 2 种形状直接报错。
- `PlanTable.select`：同几何精确匹配 → 同宽高比按 latent_t 取最近（例如 5.167 s 与 14.375 s 两档
  的分界是 9.771 s）→ 只有在没有同宽高比方案时，才对 H↔W 镜像方案做转置，并选补齐更少的一组。
  竖屏方案绝不再做一次转置。**训练和推理必须使用同一个选择规则。**
- 方案文件里记录的保留比例只作来源信息，训练一律使用运行时参数。

## 测试
`tests/unit/test_veda_tiling_plan.py`：36 种形状、补齐量（(1,8,16)→333、(8,4,4)→330 个 tile）、
排列覆盖、前缀性质、tile 是 3D 盒子、全局在后/参考在前、gather/scatter 往返、方案限制与选择规则。

## 踩坑记录
- `TileLayout` 需要带上 `used`，搜索和测试都要用它计算 pad 段。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。
