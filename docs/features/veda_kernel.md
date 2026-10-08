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
  - **SM120（RTX PRO 6000 Blackwell 等）**：上游 FA4 直接 `assert` 拒绝块稀疏，**不会**静默算稠密。
    不需要为它另写 kernel：上游的 `FlashAttentionForwardSm120` 只是
    `FlashAttentionForwardSm80` 的子类，唯一的 override 是 `can_implement`（SMEM 99 KB 而不是
    163 KB），并且在 `__init__` 末尾把 `self.arch` 改回 `sm_80`；块稀疏主循环在基类里，因此**被继承
    下来**。所以 SM120 走的就是 SM8x 那条 `DenseBlockMaskTorch` 路径，见下文的 SM120 一节。
- 走 `PATCHED_MAJOR_ARCHS = (8, 12)` 的架构由 vendored 补丁提供块稀疏；`available()` 对它们要求
  `fa4_sm8x.installed()`。
- 所有 FA4 的 import 都经过 `fa4._modules()`：机器上有 SM8x 或 SM120 GPU 时先 `fa4_sm8x.install()`。
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
补丁系列 `patches/0001..0007` 基于 flash-attention d15f153（BSD-3-Clause，`LICENSE` 与 `AUTHORS`
随目录提供），改动集中在五个模块：`block_sparsity`、`block_sparse_utils`、`flash_fwd`、`flash_bwd`、
`interface`。仓库里放的是打完补丁后的这五个文件（再生方法见 `fa4_sm8x/__init__.py` 的 docstring），
运行期由 `install()` 按依赖顺序替换已安装的 FA4 中的同名模块：
- 要求已安装的 flash-attn-4 版本恰好是 `4.0.0b32`，且其中五个原始模块的 sha256 与锁定的基线一致；
  否则报错，不打补丁（防止把补丁套在不匹配的 FA4 上）。
- 另外校验**不替换但依赖其内容**的两个模块 `flash_fwd_sm120` / `flash_bwd_sm120` 的 sha256
  （`_INHERITED_SHA256`）：SM120 的块稀疏完全靠"它们是 SM80 类的薄子类"这一点得到。万一上游给
  SM120 写了独立 kernel，打过补丁的 SM80 主循环就会悄悄不再被用到，SM120 会在稀疏掩码下算出稠密
  结果——这是最难发现的一类错误，所以宁可在 `install()` 阶段直接报错。
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

## SM120 块稀疏（`patches/0006`，前向）
**结论：SM120 不需要新 kernel，只需要拆掉 interface 里的 arch-12 门禁。**
上游的 `flash_fwd_sm120.py` / `flash_bwd_sm120.py` 各只有 61 / 55 行：
`FlashAttentionForwardSm120(FlashAttentionForwardSm80)` 唯一的 override 是 `can_implement`
（把 SMEM 上限从 163 KB 换成 `get_smem_capacity_in_bytes("sm_120")` 的 99 KB），并在
`__init__` 末尾把 `self.arch` 从 `sm_120` 改回 `sm_80`。反向那个子类只 override `can_implement`。
于是：

- 块稀疏主循环、`DenseBlockMask` 的 bitmask 遍历、launch cache 全都在被补丁改过的基类里，
  **SM120 原样继承**。
- `FlashAttentionForwardSm80` 自己没有 `__init__`，基类 `__init__` 不做任何 arch 相关的判断
  （`self.arch = ...` 是最后一行）；真正读 `self.arch` 的 `use_tma_O`、`use_dense_block_mask`、
  `_setup_attributes()` 都在 `__call__` 里，也就是在子类把 arch 改回 `sm_80` **之后**才跑。
  所以那个 arch 覆盖对块稀疏路径是完全生效的，这也是补丁能"白拿"的原因。
- SM120 的 SMEM 容量（99 KB）与 sm86/sm89 相同，所以 `_tile_size_fwd_sm8x_block_sparse` 的
  128×32 / `kv_subtile_factor=4`（48 KB → 2 CTA/SM）直接沿用，只是要在 SM120 上重新实测确认。

`patches/0006` 在 interface 里做的事（前向）：
- `_get_fwd_config`：arch 12 且给了块稀疏张量时改用 `_tile_size_fwd_sm8x_block_sparse`
  （否则会拿到稠密用的 128×64）。
- 去掉 arch-12 前向分支里的 `assert not use_block_sparsity`，并把 `q_subtile_factor` /
  `kv_subtile_factor` 传给 `FlashAttentionForwardSm120`（SM8x 分支本来就传，arch-12 分支漏了）。
- `DenseBlockMaskTorch` 与 `allow_kv_subtile` 接受 arch 12。
- 0005 的低开销 launch cache 在 arch 12 上也注册（其 key 第一项是 `q.device`，所以混插 SM8x +
  SM120 的机器上不会串用）。

`patches/0007` 把反向也打开了：
- 把 arch-12 的反向配置**并进 arch-8 分支**。两者原本只差"块稀疏时 Q/dO 单级缓冲"这条覆盖和
  warp 数，而 SM120 的 smem 与 sm86/sm89 一样，所以一个分支就够。副作用是 arch 12 在 hdim>64
  时从 4 warps (4,4,4) 换成 8 warps (2,4,4)——**实测这本身就让 SM120 的稠密反向快了 16%**，见下。
- arch-12 分支原来给反向 kernel 传的是**空 kwargs**，`mask_mod` 和两个 subtile factor 都丢了；
  现在与 SM8x 传同一套。
- 反向也接受 `DenseBlockMaskTorch`。
- dQ/dK/dV 的 postprocess 线程数扩到 arch 12。fp32 累加器是按主 kernel 的 MMA 线程划分摆放的，
  postprocess 用了不同线程数会**静默算错梯度**；而 arch 12 现在在 hdim>64 时是 256 线程，
  不再是 128，所以这条必须跟着改。

封装侧（`miowtion/kernels/fa4.py`）：`PATCHED_MAJOR_ARCHS = (8, 12)` 取代原先散落的
`major == 8` 判断——install 触发条件、`available()`、走 `DenseBlockMaskTorch` 的分支。
`_force_q_stage_one` 收窄成只对 `(10, 11)` 生效：只有 SM100/SM101 会选 q_stage=2，SM120 本来就是
1，没必要去 patch FA4 内部符号。

### 部署前提（SM120）
- **torch 必须是 cu128 或更新的构建**：cu126 的 wheel 里没有 sm_120，装了也跑不起来。
  用 `torch.cuda.get_arch_list()` 确认含 `sm_120`。实测 torch 2.14.0+cu130 的 arch_list 为
  `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`。
- 先装 torch 再装 flash-attn-4；实测 `nvidia-cutlass-dsl` 4.7.1 也能用（不止 4.2.0）。
- `pip install` 那条 git 依赖会跑 `git submodule update --init --recursive`，把 flash-attention
  的 `csrc/composable_kernel`（ROCm，与 `flash_attn/cute` 无关）也拉一遍并且经常失败。对策：自己
  浅克隆（不带 submodule）到本地再 `pip install <clone>/flash_attn/cute`。

### 实测：SM120 前向（2026-09-28，RTX PRO 6000 Blackwell Server Edition，96 GB）
`scripts/bench_sparse_attention.py`，d=128，密度 0.1（稀疏度 90%），torch 2.14.0+cu130，
flash-attn-4 4.0.0b32 @ d15f153，cutlass-dsl 4.7.1，独占整卡。
MFU 的分母是**稠密** bf16 tensor core 峰值 504 TFLOP/s（厂商宣传的 1 PFLOPS 是 2:1 结构化稀疏的
数字，不能用）；`gemm%` 的分母是同一次运行里实测的 cuBLAS bf16 GEMM 上限（419–438 TFLOP/s）。

真实几何（`H3Config` 的 56 头）：

| seq | kernel | ms | eff | TFLOP/s | MFU | gemm% |
|---|---|---|---|---|---|---|
| 16384 | FA4 稠密 | 21.209 | — | 362.9 | 72.0% | 83.9% |
| 16384 | **FA4 块稀疏** | **2.209** | **0.98** | **353.9** | **70.2%** | **81.8%** |
| 16384 | FA4 DenseBlockMask | 2.277 | 0.95 | 343.3 | 68.1% | 79.3% |
| 16384 | flex（对照） | 2.634 | 0.82 | 296.7 | 58.9% | 68.6% |
| 32768 | FA4 稠密 | 84.650 | — | 363.7 | 72.2% | 83.6% |
| 32768 | **FA4 块稀疏** | **8.664** | **0.99** | **360.9** | **71.6%** | **82.9%** |
| 32768 | FA4 DenseBlockMask | 8.841 | 0.97 | 353.7 | 70.2% | 81.3% |
| 32768 | flex（对照） | 10.000 | 0.86 | 312.7 | 62.1% | 71.8% |

8 头（与 4090 那张表同条件，便于横向比）：16384 稀疏 0.368 ms / eff 0.90 / MFU 60.3%；
32768 稀疏 1.252 ms / eff 0.97 / MFU 70.8%。

56 头的扫描（`fa4_block_sparse` 一列；seq 32768 扫密度，密度 0.1 扫 seq）：

| 变量 | 值 | ms | eff | MFU | gemm% |
|---|---|---|---|---|---|
| 密度 | 0.05 | 4.189 | 1.02 | 74.1% | 84.3% |
| 密度 | 0.10 | 8.381 | 1.03 | 74.0% | 84.3% |
| 密度 | 0.20 | 16.556 | 1.02 | 73.5% | 83.9% |
| seq | 8192 | 0.560 | 0.92 | 64.0% | 73.1% |
| seq | 16384 | 2.209 | 0.98 | 70.2% | 81.8% |
| seq | 32768 | 8.381 | 1.03 | 74.0% | 84.3% |
| seq | 65536 | 33.718 | 1.00 | 72.2% | 89.5% |

- **同一配置两次跑的差异约 3%**（32768 / 56 头 / 0.1 两次分别是 8.664 ms 与 8.381 ms，
  MFU 71.6% 与 74.0%）。所以这里的门槛要按区间看：56 头、seq ≥ 16k 时 MFU 落在 **70–74%**。
  seq 8192 是 64%，低于门槛，原因同 8 头 16k：问题规模小到 0.56 ms，启动与 occupancy 占比上来了。
- 密度从 0.05 到 0.2，MFU 稳在 73.5–74.1%，说明跳块的收益是线性兑现的，没有随密度退化。
- **eff 会略大于 1**（1.02–1.03）：分母用的是 FA4 默认 tile 的稠密耗时，而稀疏路径用的是
  `_tile_size_fwd_sm8x_block_sparse` 选出来的 128×32，本身就比默认 tile 更快，所以"稠密×密度"
  这个理想值被略微超过。不是测错，但也说明 eff 不能当成 >1 就是超线性加速。

### 实测：SM120 反向（补丁 0007 之后）
56 头、d=128、seq 32768。反向的 FLOPs 按前向的 **2.5 倍**算（dQ/dK/dV/dS 四个 S²HD 矩阵乘，
加上 FA 在反向里重算一遍 S）。

| 路径 | warps | bwd ms | TFLOP/s | MFU |
|---|---|---|---|---|
| 稠密，未打补丁的上游 FA4 | 4 (4,4,4) | 270.42 | 284.6 | 56.5% |
| 稠密，打了 0007 | 8 (2,4,4) | **232.79** | **330.6** | **65.6%** |
| 块稀疏（密度 0.102），打了 0007 | 8 (2,4,4) | 26.52 | 294.7 | 58.5% |

- **合并分支带来的 8 warps 让 SM120 的稠密反向快了 16%**（270.4 → 232.8 ms）。这不是顺带的，
  是把 4090 上调出来的配置用到了 SM120 上；前向不受影响（83.6 vs 84.2 ms，在噪声内）。
- **稀疏反向相对稠密反向的效率是 0.90**（理想值 232.79 × 0.102 = 23.7 ms，实测 26.5 ms），
  说明跳块在反向里同样兑现了。
- **但反向的 MFU 只有 58.5%，达不到 70% 的门槛**；注意稠密反向自己也只有 65.6%。也就是说这是
  FA 反向在这张卡上的天花板（dQ 的 fp32 atomic、以及比前向多得多的访存），不是稀疏路径的问题。
  要提反向的 MFU 得另做文章（dQ 的 atomic、split-K 之类），与 SM120 适配无关。
- 前向在同一次测量里是 8.305 ms / MFU 74.7%，与上一节一致。

结论：
- **90% 稀疏下 56 头达到 MFU 70.2%（16k）与 71.6%（32k），过了 ≥70% 的门槛。**
  8 头 16k 只有 60.3%，是问题规模太小（2.2 ms → 0.37 ms，occupancy 与启动开销占比上去了），
  不是稀疏路径的问题：同样 8 头，32k 就回到 70.8%。
- **原先担心"SM120 走 SM80 的 `mma.sync`，到不了 Blackwell 第五代 tensor core 峰值"并没有成立。**
  稠密 FA4 就有 363 TFLOP/s（MFU 72%、实测 GEMM 上限的 84%），稀疏路径紧跟其后。也就是说这条
  kernel 的天花板不是指令代次，而是 attention 本身的 84% GEMM 效率；要再往上就得动 UMMA
  kernel，收益上限只有 ~16%，暂时不值得。
- eff 0.98–0.99 说明稀疏跳块几乎无损地兑换成了时间。`max_err` 5e-4 ~ 9e-4（对 fp32 参考），
  并且没有触发"输出与稠密一致"的告警，确认不是偷偷算稠密。
- 索引列表路径与 `DenseBlockMask` 路径在 SM120 上基本打平（含 prep 时 2.18 vs 2.19、8.81 vs
  8.85 ms）。4090 上 DenseBlockMask 的优势主要来自省掉的约 30 µs 主机开销，在 56 头、毫秒级的
  kernel 时间面前已经无关紧要。

## 验证记录

- 2026-10-09，1×RTX PRO 6000 Blackwell Server Edition（sm_120，96 GB）：装上 FA4 `4.0.0b32 @ d15f153` 后 `pytest tests/gpu -m gpu` **20 passed, 9 skipped**，其中块稀疏前向与反向都走 vendored 的 SM120 补丁。安装注意：从单 SHA 浅克隆装时要设 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=4.0.0b32`，否则 setuptools_scm 找不到 `fa4-vX` tag、回落到 fallback 版本号，`fa4_sm8x.install()` 会拒绝。
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
- 2026-09-28，RTX PRO 6000 Blackwell Server Edition（sm120，96 GB，独占），补丁 0006 之后：
  `pytest tests/gpu -m gpu` **12 passed, 0 skipped**（其中 `test_fa4_block_sparse_matches_reference`
  与 `test_dense_block_mask_all_blocks_match_dense_with_mask_mod` 是稀疏的两条；CuTe 的日志里
  编译出来的类名是 `...flash_fwd_sm120FlashAttentionForwardSm120...`，确认走的确实是 SM120 子类
  继承来的块稀疏主循环，不是回退到别的路径）。测速与 MFU 见上一节。`pytest tests/unit` 342 passed。
  **SM8x 未回归验证**（本机没有 4090）：改动只是把 `major == 8` 放宽成 `in (8, 12)`，
  `_force_q_stage_one` 由 `< 10` 改成 `not in (10, 11)`——对 major 8/9/10/11 的行为逐一核对过是
  等价的，但仍需在 4090 上重跑一次 `tests/gpu` 才能销掉这条。
- 2026-09-28，同机，补丁 0007（反向）之后：`tests/gpu` **13 passed, 0 skipped**，新增
  `test_fa4_block_sparse_backward_matches_reference`（dQ/dK/dV 对参考实现的 autograd 梯度，
  只比有效 query 行；另外钉两条：梯度不能全零，且**必须离稀疏梯度比离稠密梯度更近**——否则
  "块 pattern 被忽略、反向偷偷算稠密"这个已知故障模式测不出来）。反向测速见上一节。
  注意补丁 0007 顺带改了 SM8x 的**共享分支**（原来 arch 8 与 arch 12 各一份配置，现在合并），
  对 arch 8 的取值逐行核对过没有变化，但同样**未在 4090 上回归验证**。

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
