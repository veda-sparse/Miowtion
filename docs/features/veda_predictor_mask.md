# Veda 打分器、掩码、教师热力图与损失

## 打分器（`miowtion/veda/predictor.py`）
- 每个 tile 对 q、k 分别做 [mean | max | min] 池化，得到 384 维特征：
  - mean：bf16 求和、fp32 累加，除以有效行数（不先把整个张量转成 fp32）；
  - max/min：先对整块求；**只对不满的 tile** 把 pad 行设为 ±inf 后重算（pad 是 0，而 0 是合法值）；
  - 空 tile 用 `where` 置 0（用乘 0 会遇到 −inf×0=NaN）。
- 池化在 no_grad 下对 detach 后的输入计算，梯度绝不会回流到 trunk。
- 每层 `proj_q`/`proj_k` 形状为 `[56, 384, 128]`，残差式投影 `q̂ = feat @ P[h] + mean`，
  初始化为 N(0, 1e-4)，因此未训练时约等于均值池化的 QK。logits 为 `q̂·k̂/√128`，fp32。
  50 层共 275M 参数。
- 打分器完全复制，不做 FSDP 切分：第 0 维是头维，按头索引无法作用在 DTensor 上。

## 掩码规则（`miowtion/veda/mask.py`，搜索的 oracle 用完全相同的规则）
1. 全局 tile 双向稠密。
2. 只对 video→video 象限做 top-k。
3. 等 kernel 代价预算：`budget = ratio × n_ideal² / n_tiles`，其中 `n_ideal = ⌈真实 token/128⌉`。
   补齐多的方案不能分到更多 key。
4. 小数部分用 Bresenham 分摊，按 **tile 号**索引（而不是在 rows 中的位置），平均保留数与预算的差
   严格小于 1/n。
5. 对角 tile 设为 +inf，占用预算。
6. 空 key tile 设为 −inf。
7. 条件 span 切 tile 时，列分为参考块和目标块，**各自独立 top-k、各自预算**；对角只在所属块内强制。
- `ratio ≥ 1` 表示整块保留；也支持绝对 tile 数（`Budget(tiles=k)`）。
- `dense_block_mask` 是 kernel 的输入（见 veda_kernel.md）：视频行放所选的块，全局列全保留，全局行
  看到所有非空 tile。`global_rows=False` 只给出所选 R 个 query 行（oracle 评分只跑抽样行）；全局列
  的起点永远是 `layout.n_video_tiles`，与所选行数无关。

## 教师热力图与损失（`miowtion/veda/heatmap.py`）
- `heat[i,j] = max_{r∈i, c∈j} exp(q_r·k_c/√D − lse_r)`。LSE 直接取自稠密教师 flash attention 的
  输出，是精确值。缩放放在 tile 内取最大值之后（正缩放与 max 可交换）。
- 每层只对抽样的视频 query tile 做监督（KL 是对 query tile 的平均，所以抽样是无偏的）；比例
  `teacher_q_tiles` **按每个 clip 的 tile 数分别换算**。
- CUDA 上用融合 Triton kernel（`miowtion/kernels/block_heat_triton.py`），只做 QKᵀ 的计算量，不把 [H', rows, N]
  的分数写回显存；CPU 上用分块的 torch 参考实现。bf16 舍入只对每行的块内最大值做一次（舍入单调，
  结果与逐元素舍入后取 max 逐位相同）。启动参数在 4090 上调过：4 warps、每个 program 16 个 key
  tile、不做软件流水（num_stages=1）。
- q / k 的 gather 与池化在 CUDA 上用一个融合 Triton kernel（`kernels/tile_gather_triton.py`）：每个
  (tile, head) 一个 program，按排列读一次 128 行，同时写出 tile 顺序的行和 mean / max / min 特征；
  v 的 gather 和输出的 scatter 也有对应 kernel。gather / scatter / max / min 与 torch 逐位相同，
  mean 的 fp32 求和顺序不同（容差）。训练（TeacherCollector）和推理（SparseStudent）用同一个
  kernel。4090 上 14 个头的一组：gather + 池化 1.30 → 0.34 ms。
- TeacherCollector 与 SparseStudent 都按头分块处理一个头组（每块 tile 顺序的 q 或 k 副本不超过 512 MiB）：整组 56 个头
  的副本在 103k token 时各 1.5 GB，放不进 trunk 旁边。各头独立、KL 按块内头数加权，所以每个头的梯度
  不变（只有浮点舍入可能不同）；抽样的 query 行对整组只抽一次。
- seer KL：student logits 在空列上填 −inf 后做 log-softmax；teacher 热力图按行归一化；
  只在 tgt>0 的位置累加；先在有效行上平均，再在头上平均。
- recall：预测集合与 oracle 集合（同样规则下对热力图做 top-k）的交集比例，只统计 video→video
  象限且去掉对角；当预算里只有对角时返回 NaN（日志中用 nanmean 汇总）。

## 测试
`tests/unit/test_veda_predictor_mask.py`、`tests/unit/test_veda_heatmap_attention.py`：
池化与朴素实现一致、空 tile 无 NaN、未训练打分器约等于均值 QK、Bresenham、等代价预算、对角规则、
分段预算、行子集按 tile 号取 pattern、kernel 索引与稠密掩码互相还原、热力图与暴力计算一致、
KL 在最优点为 0、recall 为 1、TeacherCollector 输出与稠密逐位一致且只训练打分器。

## 踩坑记录
- **Bresenham 的浮点误差**：`2.3−2` 得到 `0.29999999999999982`，`floor(400×frac)` 少算 1 个，
  平均值与预算的差达到 1/n。对策：frac 按 12 位小数取整。
- **Triton 默认的 num_stages=3 让热力图 kernel 慢约 20%**：key tile 循环被软件流水后反而变慢
  （150 ms vs 123 ms）。启动时显式传 `num_stages=1`。
- **只有对角的预算**：tile 很少或预算很低时，去掉对角后两个集合都为空，recall 会被算成 0，
  这是错的。现在返回 NaN。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。
- 2026-09-23，RTX 4090：热力图 kernel 与 torch 参考一致（tests/gpu）。16:9 5.17 s（297 个 tile、
  56 头、d=128）：全部 query tile 123 ms（169 TFLOPS），同一数据上 FA4 稠密前向 263 ms
  （158 TFLOPS），即 0.47 倍 FA4 耗时（计算量是 FA4 的一半）；16 个 query tile 6.7 ms。
  调参前为 170 ms / 9.5 ms（8 warps、每个元素先舍入再取 max），调参前后输出逐位相同。
