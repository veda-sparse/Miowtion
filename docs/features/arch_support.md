# 架构支持矩阵（sm_70 / sm_75 以及更高）

## 目标

每个 kernel 能跑在哪些 GPU 上，这件事以前是**隐含**在 kernel 自己的写法里的：
bf16 的类型注解、一个 `tl.dot`、FA4 的 cp.async 流水，各自定下了一个下限，但没有任何
地方把这些下限写出来。于是在不支持的卡上，报错发生在 JIT 编译的深处
（`Unsupported dtype`、`mma16816 requires sm80`），既没说是我们哪个 kernel 要的，
也没说缺的是哪条指令。

本文档和 `miowtion/kernels/arch.py` 把下限变成**声明**：入口函数先查表，要么选一条
能跑的路径，要么抛出一个点名了「架构 + kernel + 缺失的操作」的错误。禁止静默回退
（AGENTS.md 1.4）；已有的回退（torch 参考实现、eager 链、`allow_reference_kernel`）
都是调用方显式选择的。

本次工作的直接动因是 sm_70（Volta，V100）与 sm_75（Turing，T4 / RTX 20xx）。
**项目里没有这两种卡**，所以下面每一条都标明了它的来源：硬件事实、读代码得到的结论，
还是未经验证的推断。

## 设计与不变量

### 硬件事实（下限的来源）

| 能力 | 最低算力 | 说明 |
|---|---|---|
| fp16 tensor core（HMMA） | sm_70 | Volta 就有 |
| INT8 tensor core（IMMA） | sm_75 | Volta **没有**，所以 Volta 上无法下降一个 INT8 `tl.dot` |
| bf16（BMMA 与 `cvt.f32.bf16`） | sm_80 | sm_70 / sm_75 **连把 bf16 读上来加宽成 fp32 都做不到**，不是"慢一点"，是编译不出来 |
| `cp.async` | sm_80 | FlashAttention 的流水化 tile 装载依赖它 |
| fp8 运算 | sm_89 | |
| TMA 描述符 | sm_90 | |
| Triton 的 `tl.dot` | sm_75 | Triton 的 NVIDIA 后端已经不再生成 Volta 的 `mma.m8n8k4`，所以 sm_70 有 fp16 tensor core 但 Triton 用不上 |

最后一条是整件事里最违反直觉的地方：**sm_70 的障碍不是 dtype，是 Triton 自己**。
把 kernel 改成 fp16 并不能让它回到 Volta。

### 每个 kernel 的下限

表里的 dtype 列是"该 dtype 需要的最低算力"；没列出的 dtype 表示该 kernel 根本没实现。
代码里的同一张表是 `kernels/arch.py` 的 `REQUIREMENTS`，单测
`tests/unit/test_kernels_arch.py` 的 `_MATRIX` 逐行钉住它。

| kernel | 整体下限 | 决定下限的操作 | bf16 | fp16 | fp32 |
|---|---|---|---|---|---|
| `fa4.dense_attention` | sm_80 | cp.async + sm_80 MMA 形状 | sm_80 | sm_80 | 未实现 |
| `fa4.block_sparse_attention` | sm_80 | 同上；vendored 的 `fa4_sm8x` 补丁打在 SM80 kernel 上，继承同一个下限 | sm_80 | sm_80 | 未实现 |
| `block_heat_triton.teacher_heat` | sm_75 | QK^T 是一个 `tl.dot` | sm_80 | **sm_75** | sm_75 |
| `tile_gather_triton.gather` / `.scatter` | sm_70 | 只有搬运与归约，**没有** `tl.dot` | sm_80 | **sm_70** | sm_70 |
| `elementwise_triton`（RoPE / AdaLN / SwiGLU） | sm_80 | 为了与 eager 逐位相等，每个中间量都舍入到 bf16 | sm_80 | 未实现 | 未实现 |
| `compensation_triton.compensate` | sm_80 | 概率块被 cast 成 bf16 去做 `tl.dot`；但**先卡住的不是这个 cast**——`compensation_triton.attention` 的精确块走 `fa4.block_sparse_attention`，那条本身就是 sm_80 | sm_80 | 未实现 | 未实现 |
| `sol_veda.block_sparse_attention`（融合 INT8） | sm_80 | 选中的块量化成 INT8（sm_75 起），但池化后的 summary 与在线 softmax 仍是 bf16（sm_80 起） | sm_80 | 未实现 | 未实现 |
| `h3.attention.dense_attention` `backend='sdpa'` | sm_70 | PyTorch 的融合 SDPA kernel | sm_80 | sm_70 | sm_70 |
| 同上，`return_lse=True` | sm_80 | 唯一返回 LSE 的是 flash 后端，它要求 sm_80 | sm_80 | sm_80 | 未实现（flash 后端不收 fp32） |
| `kernels.reference.block_sparse_attention` | 无 | 纯 torch，CPU 也能跑 | 任意 | 任意 | 任意 |
| `h3.attention.dense_attention` `backend='math'` | 无 | 纯 torch fp32 | 任意 | 任意 | 任意 |

### 结论：sm_70 / sm_75 上现在能做什么

- **稀疏注意力的快路径（FA4）在 sm_70 / sm_75 上不存在**，而且不是 dtype 的问题：
  CuTe kernel 建在 cp.async 和 sm_80 的 MMA 形状上。`fa4.available()` 在那里是 False，
  `fa4.unavailable_reason()` 会说出缺的是哪一条。想在这两种卡上跑稀疏注意力，只有
  `kernels/reference.py` 这条 token 粒度的 torch 路径，由调用方显式用
  `allow_reference_kernel=True` 选中——它对真实长度的 clip 慢到不可用，是测试用的。
- **打分器的教师热力图（`block_heat_triton`）在 sm_75 上用 fp16 可以跑**，sm_70 不行
  （Triton 的 `tl.dot`）。`heatmap.teacher_heat` 用 `supports()` 路由，所以够不到下限的
  卡会自动走 torch 参考实现，而不是死在 Triton 编译里。
- **tile 排列与池化（`tile_gather_triton`）在 sm_70 上用 fp16 就能跑**，因为里面没有
  `tl.dot`。bf16 仍然要 sm_80。
- **融合补偿（`compensation_triton`）在 sm_7x 上不只差一个 cast。** 这条路径是
  「FA4 跑精确块 + Triton 补偿被丢掉的块」，所以它的第一道门槛是 FA4 的 sm_80，
  bf16 的 `tl.dot` 只是接下来会撞到的第二件事。换句话说**把 cast 改成 fp16 不会让
  这条路径在 sm_75 上跑起来**：得先给它配一个能在 sm_75 上跑的精确块 kernel。
- **融合的 Sol+Veda（INT8）自己就会报对的错**：`kernels/sol_veda.py` 的 `available()`
  用 `get_device_capability()[0] >= 8` 设限，`unavailable_reason()` 返回
  「needs SM80 or newer」，所以它在 Volta / Turing 上本来就是自描述的，不依赖
  `arch.py`。`arch.REQUIREMENTS` 里仍然给它留了一条，只为了让矩阵是完整的一张表
  （并说明为什么 sm_75 的 IMMA 不够：量化只用在选中的块上，summary 与在线 softmax
  还是 bf16）。
- **整模型仍然是 bf16 的**：`h3/model.py` 的 `_BF16` 与 `h3/weights.py` 把权重 dtype
  钉成 bf16。因此即使上面这些 kernel 的 fp16 路径都就位，**端到端跑在 sm_70 / sm_75 上
  还缺一步：把模型与权重加载的 dtype 参数化**。本次没有做这件事（见「待办」），因为
  没有硬件能验证它，而它会动到训练与 checkpoint 的路径。

### fp16 与 bf16 的数值差别（哪里必须留在 fp32）

fp16 的指数范围比 bf16 小得多（最大 65504，bf16 约 3.4e38），所以任何做指数的地方都要
确认它的累加在 fp32：

- `block_heat_triton`：`tl.dot` 的累加器是 fp32，`exp` 也作用在 fp32 上。**唯一的 fp16
  narrowing 是把每行的最大分数舍入到输入 dtype**，这一步是为了对齐参考实现
  （参考实现的 `torch.bmm` 返回输入 dtype）。它在未缩放分数超过 65504 时饱和成 inf，
  换算到 head_dim 128 的缩放后 logit 是 5788——真实注意力的 logit 到不了那里；而且参考
  实现在同一个位置饱和，所以对齐关系不受影响。
- 这一步以前是硬编码的 `.to(tl.bfloat16)`。在 fp16 输入上那会**悄悄**变成一个与参考
  实现不同的舍入（bf16 的尾数更短），所以现在改成跟随 `q_ptr.dtype.element_ty`。
  「先对每个分数舍入再取 max」与「先取 max 再舍入」逐位相等，因为 round-to-nearest-even
  是单调的；单测 `test_rounding_the_row_max_equals_rounding_every_score` 用
  `torch.equal` 在 fp16 和 bf16 上都钉住了这一点。
- `kernels/reference.py` 的 softmax 本来就在 fp32 做（`q.float()`），fp16 输入只有输入
  舍入这一处误差；单测对 fp32 对拍到 2e-3。

## 代码位置与接口

- `miowtion/kernels/arch.py`
  - `capability(device)`：`(major, minor)`；非 CUDA 设备 `raise ValueError`（问一个 CPU
    张量的算力是调用方的路由 bug，不是一个"没有能力的设备"）。
  - `name(cap)`：`'sm_75 (Turing)'`，给报错用。
  - `REQUIREMENTS`：上面那张表，每条带 `floor` / `floor_reason` / `dtype_floors`。
  - `supported(kernel, device, dtype=None)`：路由用，非 CUDA 设备返回 False。
  - `unsupported_reason(...)`：一行理由，点名架构、kernel、缺失的操作。
  - `require(...)`：不支持就 `raise NotImplementedError`。
- `miowtion/kernels/fa4.py`：新增 `unavailable_reason(device, dtype=None)`；
  `block_sparse_attention` 与 `dense_attention` 的报错现在都点名缺失的操作。
  `veda/attention.py` 的两个 student 把这条理由带进了自己的 `RuntimeError`。
- `miowtion/kernels/block_heat_triton.py`：`supports()` 加上架构 / dtype 判断；
  `teacher_heat()` 先 `arch.require`，并拒绝 q / k dtype 不一致（一个 `tl.dot` 吃两个）；
  行最大值的舍入跟随输入 dtype。
- `miowtion/kernels/tile_gather_triton.py`：`supports()` 与 `_check()` 加上同样的判断。
- `miowtion/h3/attention.py`：`backend='sdpa'` 在 CUDA 上先 `arch.require`，避免在
  sm_80 以下被 PyTorch 悄悄送进 math 后端（展开 `[H, S, S]`）。

## 测试

`tests/unit/test_kernels_arch.py`（CPU，51 passed / 12 skipped；skip 的是矩阵里"受支持"的那些组合在报错测试里被跳过）：

- `_MATRIX` 是上面那张表的可执行版本：`(kernel, 算力, dtype) -> 能否跑`，用
  monkeypatch 的 `torch.cuda.get_device_capability` 伪造架构，两个方向都测
  （`supported()` 的布尔值；不支持时 `require()` 抛错，且错误里出现架构号、kernel 名和
  缺失操作的名字）。
- "dtype 没实现" 与 "架构太老" 是两种不同的报错，分别钉住。
- 表自身的一致性：任何 dtype 的下限不得低于 kernel 的整体下限（否则那一行是死条目）。
- CPU 张量：`supports()` 返回 False，`capability()` 抛 ValueError。
- fp16 数值：参考实现在 fp16 下与 fp32 对拍（块稀疏注意力、教师热力图），以及上面那条
  舍入单调性的逐位测试。

**没有 GPU 测试**：项目里没有 sm_70 / sm_75 的卡（见「验证记录」）。

## 踩坑记录

- **以为 sm_70 / sm_75 的唯一障碍是 bf16。** 实际上有三个互相独立的障碍，dtype 只是
  其中一个：(1) FA4 要 cp.async，和 dtype 无关，所以 sm_7x 上**没有任何** FA4 路径；
  (2) Triton 的 NVIDIA 后端不再为 Volta 生成 `tl.dot`，所以 sm_70 上换 fp16 也救不了
  带 `tl.dot` 的 kernel；(3) sm_80 以下没有 `cvt.f32.bf16`，所以一个 bf16 的 kernel
  在那里**不是慢路径而是编译不出来**。对策：能力表按**操作**声明，不按 dtype 声明，
  报错里也说操作名。
- **教师热力图里硬编码的 bf16 舍入。** `.to(tl.bfloat16)` 原本只是为了对齐参考实现的
  bmm 输出 dtype；一旦输入换成 fp16，它就变成一个参考实现没有的额外舍入，而数值对拍
  只会"稍微差一点"，不会报错。对策：舍入跟随 `q_ptr.dtype.element_ty`，并用一个逐位
  单测钉住 max 与舍入可交换。
- **sm_80 以下的 SDPA 会悄悄退回 math 后端。** 这与仓库里已有的"3 维 q/k/v 退回 math"
  是同一类问题的另一面：在 sm_75 上即使张量是 4 维的，bf16 也没有融合 kernel，于是
  38k token 的一次调用要 300+ GiB。对策：`backend='sdpa'` 在 CUDA 上先查表并报错，
  错误信息里指向 `backend='math'`。

## 验证记录

| 日期 | 机器 | commit | 结果 |
|---|---|---|---|
| 2026-10-12 | 仅 CPU（无 CUDA、无 triton） | 本次提交 | `pytest tests/unit` 724 passed / 12 skipped；新增 `test_kernels_arch.py` 51 passed / 12 skipped |

**sm_70 / sm_75 的 GPU 路径全部未验证（GPU 未验证）。** 项目没有这两种卡，本次改动
也没有在任何 GPU 上跑过。具体地，下面这些结论目前只有"读代码 + 硬件文档"的依据：

1. `block_heat_triton` 在 **sm_75 + fp16** 上能编译并跑出与参考实现一致的结果
   （容差同 bf16）。需要一张 T4 / RTX 20xx 跑 `pytest tests/gpu -m gpu`。
2. `tile_gather_triton` 在 **sm_70 / sm_75 + fp16** 上能跑。
3. Triton 确实在 sm_70 上拒绝 `tl.dot`。这是本表里唯一一条依据"上游已移除 Volta MMA
   下降"的推断，没有在 Volta 上跑过；如果实测发现 Triton 仍能生成，`arch.TRITON_DOT`
   应从 `(7, 5)` 放回 `(7, 0)`，`_MATRIX` 里对应的两行会立刻变红。
4. sm_75 上 PyTorch 的 memory-efficient SDPA 能否作为 `return_lse` 的替代路径
   （它返回 logsumexp，但行数会补齐到 32 的倍数、语义未核对）。**没有实现**，因为无法
   验证；现在是显式报错。

## 待办

- 把模型 dtype 参数化（`h3/model.py` 的 `_BF16`、`h3/weights.py`、训练与 checkpoint
  路径），端到端的 fp16 才可能在 sm_7x 上跑起来。现在缺这一步，所以上面的 fp16 kernel
  路径只对单独调用这些 kernel 的场景（打分器训练的教师热力图、tile 搜索）有意义。
- 若将来要在 sm_75 上跑融合补偿，**先要有一个能在 sm_75 上跑精确块的 kernel**（FA4
  不行，见上），之后才轮到把 bf16 跟随输入 dtype。那一步有两处：
  `kernels/compensation_triton.py` 的 `probability.to(tl.bfloat16)`（`_compensate_kernel`
  里那个 `tl.dot`）与 `summaries()` 里 `kc` / `vc` 的 `dtype=torch.bfloat16`
  （两个 kernel 的 `tl.store` 本来就跟随缓冲区的 `element_ty`，所以改分配处即可）。
  同时 `arch.REQUIREMENTS` 里那一条要补上 fp16 的下限。
- `elementwise_triton` 的 fp16 版本：改动是机械的（把 `_bf16` 的位运算舍入参数化），
  但它的价值全在"与 eager 逐位相等"，而那只能在 GPU 上验证，所以没做；sm_80 以下
  现在走 eager 链（`usable()` 返回 False）。
