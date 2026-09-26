# 块稀疏 kernel：FA4 CuTe 整合

## 目标

稀疏路径统一使用 FlashAttention-4（CuTe DSL）的 block sparse 接口：`mask_mod` +
`aux_tensors` + full/partial 两个块列表。

## 设计与不变量（`miowtion/kernels/fa4.py`）
- 输入是 tile 顺序的 `[N, H', D]`，加上 batch 维就是 FA4 的 `[B,S,H,D]`，无需转置。
- `mask_mod` 是模块级单例：FA4 会对 callable 做哈希作为编译缓存的 key，每次调用都新建闭包就会
  重新计算哈希（可能还会重新编译）。它通过 `aux_tensors[0]`（int32 `[N]` 的槽位有效标记，
  编码了每个 key tile 的有效前缀）判断 key 是否有效。
- full 块（valid_count=128）不需要逐 token 掩码，与 partial 块分开放入两个列表。
- `BlockSparseTensorsTorch` 的第 5 个字段是 `cu_total_m_blocks`（varlen 用），**不是** block size。
  block size 必须用关键字参数 `block_size=(128,128)` 传入。
- 封装的输入是稠密块掩码 `[H', R, n_tiles]`（`mask.dense_block_mask`）。SM8x 上直接作为
  `DenseBlockMaskTorch` 交给 kernel（反向按列读同一个掩码，不需要索引列表和转置）；SM90/SM100 上由
  `fa4.index_lists` 打包成 full/partial 列表，需要梯度时再打包 Q 方向（转置后）的反向列表。
- 架构约束（读源码 + 实测）：
  - **SM90**：hdim 128 时 tile 为 128×128、q_stage=1，与 128 行的 Q tile 直接匹配。
  - **SM100**：seqlen_q>128 时接口强制 `q_stage=2`，Q 稀疏块必须是 256 的倍数，所以 128 行的 tile
    会被拒绝。封装里用 `_force_q_stage_one` 覆盖为 q_stage=1（会检查 FA4 内部符号是否存在）。
    **这条路径尚未在 B200 上验证**，需要实测吞吐。
  - **SM8x（4090 等）**：上游 FA4 **不支持**，并且是静默失败（详见踩坑记录）。我们 vendor 了补丁后
    的四个 FA4 模块（`miowtion/kernels/fa4_sm8x`，见下文），`install()` 成功后 `available()` 才对
    SM8x 返回 True；其他情况一律抛 `NotImplementedError`（附带 install 失败的原因），绝不静默回退。
- 所有 FA4 的 import 都经过 `fa4._modules()`：机器上有 SM8x GPU 时先 `fa4_sm8x.install()`。
  `flash_attn.cute` 的 `__init__` 会 import interface，所以任何地方提前 `import flash_attn.cute`
  都会拿到未打补丁的版本；`install()` 检测到这种情况直接报错。
- 训练阶段 2 只接受 FA4 路径；参考实现（`kernels/reference.py`）仅用于 CPU 单元测试。

## 测速（`scripts/bench_sparse_attention.py`，`miowtion/kernels/bench.py`）
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
- **多卡线程会撞上 import 竞态**：三张卡各一个线程，第一次注意力调用同时进 `_modules()`。
  `functools.cache` 在被包的函数执行期间不持锁，于是三个线程同时 import；而 `install()` 是
  手工 `module_from_spec` + `exec_module` 往 `sys.modules` 里塞模块，绕过了 Python 的
  per-module import lock，别的线程正好拿到"已注册但还没执行完"的模块，报
  `module 'flash_attn.cute.interface' has no attribute 'flash_attn_func'`。20 条 holdout
  的批量生成里 cuda:0 一路跑完、另外两张卡在第一步就挂了（异常要等线程 join 才抛出来，所以
  日志看起来是"跑完才失败"）。对策：`_IMPORT_LOCK` 串行化 `_modules()` 的函数体。
- **补丁必须在第一次 import `flash_attn.cute` 之前装上**：包的 `__init__` 会 import interface，
  interface 又 import 其他三个模块。`install()` 先用 `module_from_spec` 建出不执行 `__init__` 的包
  对象，把 vendored 模块按依赖顺序注册进 `sys.modules`，最后才执行 `__init__`。

## SM89 块稀疏：vendored 的 FA4 SM80 补丁（`miowtion/kernels/fa4_sm8x`）
补丁系列 `patches/0001..0005` 基于 flash-attention d15f153（BSD-3-Clause，`LICENSE` 与 `AUTHORS`
随目录提供），改动集中在五个模块：`block_sparsity`、`block_sparse_utils`、`flash_fwd`、`flash_bwd`、
`interface`。仓库里放的是打完补丁后的这五个文件（再生方法见 `fa4_sm8x/__init__.py` 的 docstring），
运行期由 `install()` 按依赖顺序替换已安装的 FA4 中的同名模块：
- 要求已安装的 flash-attn-4 版本恰好是 `4.0.0b32`，且其中五个原始模块的 sha256 与锁定的基线一致；
  否则报错，不打补丁（防止把补丁套在不匹配的 FA4 上）。
- 升级 FA4 时：在新基线上重新 `git am` 补丁、重新生成五个文件并更新哈希，重跑
  `tests/gpu/test_kernels_gpu.py`。

补丁内容：
- 前向：`FlashAttentionForwardSm80` 新增块稀疏主循环。partial 与 full 合并成一次遍历，**按稠密 kernel
  的顺序**（KV 降序）访问，是否 partial 在循环里按块判断；下一个块的索引提前读取，下一个 K tile
  在当前 tile 做 softmax+PV 时用 cp.async 预取。tile 实测选 128×32（kv_subtile=4，每个 SM 放 2 个
  CTA）；128×128 会寄存器溢出，慢约 30 倍。
- 反向：SM80 反向补上了 mask_mod 和 Q 方向的块稀疏主循环（Q 升序，与稠密一致）；8 warps、
  AtomLayout (2,4,4)，postprocess 的线程数跟随主 kernel。
- `DenseBlockMaskTorch`：`[B|1, H|1, M, N]` 的 bool/uint8 块掩码（0 跳过，1 full，2 partial）加上按
  KV 列的 partial 标记。每个 CTA 用 ballot 把自己那一行（反向是那一列）转成 smem 里的 bitmask 再
  逐位遍历，不需要 argsort 生成列表，也不需要为反向转置。每边最多 2048 个块（26 万 token）。
- 同一签名第二次调用起走启动缓存；不需要梯度时跳过 `autograd.Function`。每次调用的主机时间从约
  94 µs 降到约 30 µs。
- **与稠密的一致性**（补丁作者的对拍）：同一 tile 配置下，稀疏遍历与"稠密 kernel + 等价 mask_mod"
  的 O / LSE / dK / dV 逐位相等（dQ 用 fp32 atomic 累加，稠密 kernel 两次运行之间也不逐位相同）。
  我们的封装里稠密调用用的是 FA4 的默认 tile，稀疏调用用 128×32，所以两者差约 1 个 bf16 ulp
  （4090 实测最大 9.8e-4，见 GPU 测试）。
- 补丁作者在 4090 上的 e2e 测速（H=8，密度 0.10，10% partial 块，每次调用换新掩码；括号内为纯 GPU
  时间），单位 ms：

  | 方法 | 前向 16k | 前向 32k | 前向+反向 16k | 前向+反向 32k |
  |---|---|---|---|---|
  | SDPA 稠密 × 密度 | 0.698 | 2.767 | 2.453 | 10.02 |
  | FA4 DenseBlockMask | 0.754（0.701） | 2.771（2.718） | 2.985（2.892） | 11.30（10.79） |
  | FA4 索引列表（含构建列表） | 1.038 | 3.092 | 3.398 | 11.16 |
  | flex（构建列表 + BlockMask） | 2.177 | 4.339 | 5.531 | 17.31 |

- 4090，8 头，密度约 0.1：前向 16k 0.70 ms / 32k 2.70 ms（**效率 0.97 / 1.00**，含主机开销时为
  0.83–0.96）；10% partial 块时为 0.96 / 0.99；S 从 8k 到 100k、H 从 4 到 56 时效率为 0.92–1.02。
  反向效率 0.75 / 0.84（以 SDPA 的稠密反向为基准；以 FA4 自己的稠密反向为基准时为 0.9–1.0）。
  同条件下 flex：前向 e2e 0.35 / 0.66，反向 0.52 / 0.55。
- 正确性：前向误差 5e-4 到 2e-3（与稠密结果的差距为 0.14–0.67，确认不是稠密计算）；反向误差
  1.5e-3 到 4.9e-3。仓库原有的 400 个块稀疏 mask_mod 用例中 360 个通过，其余 40 个是 tile
  (128,112)：SM8x 反向无法支持，会给出明确报错。
- 发现的风险：SM90/SM100 上如果不传 `block_sparse_tensors_bwd`，反向会**静默算出稠密梯度**。
  我们的封装在需要梯度时总是从块掩码构造反向列表，已经规避。
- 补丁作者踩过的坑：partial 与 full 分开遍历时访问顺序与稠密不同，差 1 个 bf16 ulp，只有按稠密
  顺序合并遍历才能逐位一致；SM80 反向 8 warps 配 (4,2,2) 或 (4,4,2) 时 ptxas 分配寄存器失败，
  AtomLayoutNdKV=8 能编译但梯度静默出错（误差 1.4）；postprocess 的线程数必须等于主 kernel 的
  线程数，否则 dQ/dK/dV 的累加器布局对不上。

## 验证记录
- 2026-09-23 RTX 4090：见上表。结论：上游 FA4 在 SM89 上不支持块稀疏；我们的补丁达到效率门槛，
  已作为 vendored 模块合入（`miowtion/kernels/fa4_sm8x`）。
- H100 / B200：待测（正确性对拍 + 效率 ≥ 0.75 的门槛）。
- `tests/gpu/test_kernels_gpu.py`（2026-09-23，RTX 4090，通过 `fa4_sm8x.install()`）：4 passed。
  FA4 稠密的 LSE 与 fp32 参考一致；Triton 热力图与 torch 参考一致；FA4 块稀疏（含 partial tile）
  与 fp32 参考一致；oracle 的 kernel 路径与参考路径一致（见 tile_search.md）。
- 2026-09-24，RTX 4090，同步到补丁 0001..0005 后：`tests/gpu` 5 passed（新增：全部块都选中时，
  SM8x 稀疏路径与"FA4 稠密 + 等价 mask_mod"一致，最大差 9.8e-4，即 1 个 bf16 ulp，来自两边 tile
  配置不同）。`scripts/bench_sparse_attention.py`（8 头，密度 0.1，每次调用换新掩码；GPU 与别人的
  轻负载进程共用，数字略有噪声）：含掩码准备的效率，DenseBlockMask 16k 0.90 / 32k 1.04，索引列表
  0.84 / 0.99，flex 0.80 / 0.85。

## 待办
- **FP8 sparse**（用户要求，方案待定）：块稀疏 + FP8（或 INT8 QK / FP8 PV，参考 SageAttention /
  SpargeAttn 的量化方式）。需要确定：量化粒度与 kernel 路线（FA4 CuTe FP8 目前只在 SM100 上可用）、
  以及对教师输出的误差预算和可视化验收方式。
- SM90/SM100 上的 GPU 测试：前向/反向与参考实现对拍（只比较有效 query 行）、测速。
- 把教师热力图并进稠密 FA4 前向（一遍出 out / lse / heat）：heat 等于
  `max_r exp(m_rj − lse_r)`，其中 m_rj 是第 r 行在 key tile j 上的分数最大值，online softmax
  本来就会算。需要让稠密 pass 按 tile 顺序（每个头组一次）运行。4090 上全部 query tile 的 Triton
  热力图（~212 ms/层）与稠密 attention（267 ms/层）是同一量级，合并后阶段 1 的教师 attention 预计
  少 ~40%。
