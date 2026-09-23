# 块稀疏 kernel：FA4 CuTe 整合

## 目标

稀疏路径统一使用 FlashAttention-4（CuTe DSL）的 block sparse 接口：`mask_mod` +
`aux_tensors` + full/partial 两个块列表。

## 设计与不变量（`miowtion/veda/kernels/fa4.py`）
- 输入是 tile 顺序的 `[N, H', D]`，加上 batch 维就是 FA4 的 `[B,S,H,D]`，无需转置。
- `mask_mod` 是模块级单例：FA4 会对 callable 做哈希作为编译缓存的 key，每次调用都新建闭包就会
  重新计算哈希（可能还会重新编译）。它通过 `aux_tensors[0]`（int32 `[N]` 的槽位有效标记，
  编码了每个 key tile 的有效前缀）判断 key 是否有效。
- full 块（valid_count=128）不需要逐 token 掩码，与 partial 块分开放入两个列表。
- `BlockSparseTensorsTorch` 的第 5 个字段是 `cu_total_m_blocks`（varlen 用），**不是** block size。
  block size 必须用关键字参数 `block_size=(128,128)` 传入。
- 反向需要按 Q 方向的（转置后的）块列表：`_transpose_indices`。
- 架构约束（读源码 + 实测）：
  - **SM90**：hdim 128 时 tile 为 128×128、q_stage=1，与 128 行的 Q tile 直接匹配。
  - **SM100**：seqlen_q>128 时接口强制 `q_stage=2`，Q 稀疏块必须是 256 的倍数，所以 128 行的 tile
    会被拒绝。封装里用 `_force_q_stage_one` 覆盖为 q_stage=1（会检查 FA4 内部符号是否存在）。
    **这条路径尚未在 B200 上验证**，需要实测吞吐。
  - **SM8x（4090 等）**：**不支持**，并且是静默失败，详见踩坑记录。`available()` 只对
    SM9x/10x/11x 返回 True；在其他架构上调用会直接抛 `NotImplementedError`。
- 训练阶段 2 只接受 FA4 路径；参考实现（`kernels/reference.py`）仅用于 CPU 单元测试。

## 测速（`scripts/bench_sparse_attention.py`，`miowtion/veda/kernels/bench.py`）
随机块 pattern（每行保留 round(密度×n) 个块，对角必选），block 128，全部为 full 块。
效率 = 稠密耗时 × 密度 / 稀疏耗时。每个 kernel 都与 fp32 参考实现对拍，以检出"静默按稠密计算"。

2026-09-23，RTX 4090（sm89），8 头，d=128，密度 0.1（稀疏度 90%），torch 2.14+cu126，
flash-attn-4 4.0.0b32 @ d15f153：

| kernel | seq 16384 | seq 32768 | 说明 |
|---|---|---|---|
| SDPA flash（稠密） | 7.26 ms | 27.53 ms | 基线 |
| FA4 稠密 | 7.47 ms | 27.86 ms | 在 SM89 上可用且正确 |
| FA4 block sparse (128,128) | 报错 | 报错 | `sparse_block_size[1]=64 must match tile_n` |
| FA4 block sparse (128,64) | 7.48 ms，与**稠密**结果一致 | — | 静默忽略稀疏 |
| flex 块稀疏（仅作对照） | 0.81 ms，效率 0.91 | 2.87 ms，效率 0.97 | torch 2.14 |

## 踩坑记录
- **FA4 在 SM8x 上的块稀疏是静默的稠密计算**：SM80 前向 kernel 的 `__call__` 接收
  `blocksparse_tensors` 参数但没有使用；接口在 SM8x 上 tile_n=64，只允许 64 宽的 KV 块。传入合法的
  (128,64) 块稀疏张量后不报错，输出与稠密结果一致（max err 1.8e-4，与稀疏参考相差 0.28），耗时也
  与稠密相同。对策：`fa4.available()` 按架构白名单返回，其他架构直接抛异常，绝不静默回退。
- **SM100 的 q_stage=2**：接口按 `seqlen_q > tile_m` 自动选 q_stage=2，稀疏 Q 块粒度因此变成 256。

## SM89 块稀疏：FA4 CuTe SM80 路径的补丁（subagent，2026-09-23）
独立的 flash-attention fork（flash-attention @ d15f153，分支 `sm89-block-sparse`，
补丁在 `patches/0001..0003`，尚未合入依赖）：
- 前向：`FlashAttentionForwardSm80` 新增块稀疏主循环，先 partial 后 full，均为倒序；下一个块的
  索引提前读取；下一个 K tile 在当前 tile 做 softmax+PV 时用 cp.async 预取。tile 实测选 128×32
  （kv_subtile=4，每个 SM 放 2 个 CTA）；128×128 会寄存器溢出，慢约 30 倍。
- 反向：SM80 反向补上了 mask_mod 和 Q 方向的块稀疏主循环；SM8x 上前向是稀疏但缺少
  `block_sparse_tensors_bwd` 时直接报错。
- 4090，8 头，密度约 0.1：前向 16k 0.70 ms / 32k 2.70 ms（**效率 0.97 / 1.00**，含主机开销时为
  0.83–0.96）；10% partial 块时为 0.96 / 0.99；S 从 8k 到 100k、H 从 4 到 56 时效率为 0.92–1.02。
  反向效率 0.75 / 0.84（以 SDPA 的稠密反向为基准；以 FA4 自己的稠密反向为基准时为 0.9–1.0）。
  同条件下 flex：前向 e2e 0.35 / 0.66，反向 0.52 / 0.55。
- 正确性：前向误差 5e-4 到 2e-3（与稠密结果的差距为 0.14–0.67，确认不是稠密计算）；反向误差
  1.5e-3 到 4.9e-3。仓库原有的 400 个块稀疏 mask_mod 用例中 360 个通过，其余 40 个是 tile
  (128,112)：SM8x 反向无法支持，会给出明确报错。
- 发现的风险：SM90/SM100 上如果不传 `block_sparse_tensors_bwd`，反向会**静默算出稠密梯度**。
  我们的封装在需要梯度时强制要求 `block_mask` 并构造反向列表，已经规避。

## 验证记录
- 2026-09-23 RTX 4090：见上表。结论：上游 FA4 在 SM89 上不支持块稀疏；我们的补丁已经达到效率
  门槛。待办：把补丁作为锁定依赖合入（fork 或运行期 patch，按 AGENTS.md 第 3 节处理），
  并让 `fa4.available()` 在打了补丁的版本上放行 SM89。
- H100 / B200：待测（正确性对拍 + 效率 ≥ 0.75 的门槛）。

## 待办
- **FP8 sparse**（用户要求，方案待定）：块稀疏 + FP8（或 INT8 QK / FP8 PV，参考 SageAttention /
  SpargeAttn 的量化方式）。需要确定：量化粒度与 kernel 路线（FA4 CuTe FP8 目前只在 SM100 上可用）、
  以及对教师输出的误差预算和可视化验收方式。
- SM90/SM100 上的 GPU 测试：前向/反向与参考实现对拍（只比较有效 query 行）、测速。
- 视用户决策：在 FA4 的 SM80 CuTe kernel 中实现块稀疏迭代（前向 + 反向）。
