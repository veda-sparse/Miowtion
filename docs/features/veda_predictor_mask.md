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
- `kernel_indices`：full（valid_count=128）与 partial 两个左对齐列表，全局列放在最前面；各行的
  实际长度由 cnt 给出。

## 教师热力图与损失（`miowtion/veda/heatmap.py`）
- `heat[i,j] = max_{r∈i, c∈j} exp(q_r·k_c/√D − lse_r)`。LSE 直接取自稠密教师 flash attention 的
  输出，是精确值。缩放放在 tile 内取最大值之后（正缩放与 max 可交换）。
- 每层只对抽样的视频 query tile 做监督（KL 是对 query tile 的平均，所以抽样是无偏的）；比例
  `teacher_q_tiles` **按每个 clip 的 tile 数分别换算**。
- CUDA 上用融合 Triton kernel（`miowtion/kernels/block_heat_triton.py`），只做 QKᵀ 的计算量，不把 [H', rows, N]
  的分数写回显存；CPU 上用分块的 torch 参考实现。
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
- **只有对角的预算**：tile 很少或预算很低时，去掉对角后两个集合都为空，recall 会被算成 0，
  这是错的。现在返回 NaN。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。Triton 热力图 kernel 待 GPU 对拍。
