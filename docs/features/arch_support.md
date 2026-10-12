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
**项目里没有这两种卡**，所以下面每一条都标明了它的来源：硬件事实、AOT 编译实测、
规格书算术，还是未经验证的推断。

**结论先写在这里：这两代卡上用稠密注意力，稀疏没有能回本的 kernel**（为什么见
「结论」一节）。本文档存在的价值因此不是"怎么在 sm_7x 上跑 Veda"，而是让任何人在
五分钟内看出这件事、并且看出它卡在哪条指令上。

## 设计与不变量

### 一个下限其实是两个问题

初版这张表把两件事混成了一件，于是有两条写错了（见「踩坑记录」）。必须分开问：

1. **工具链能不能编出来。** 这件事现在是**实测**的，不是回忆：
   `scripts/probe_triton_floor.py` 用 `triton.compile` + `GPUTarget('cuda', cap, 32)`
   为目标算力做 AOT 编译（只要 ptxas，不要卡），再从 `compiled.asm['ptx']` 把下降结果
   读出来。
2. **能不能落到 tensor core 上。** 同一次编译的 PTX 里看 `mma` 出现在哪一代。

实测（Triton 3.4.0，`tl.dot` 128x64x128，64x64x64 结果相同）：

| 算力 | fp16 dot | bf16 dot | fp32 dot | int8 dot | bf16 搬运/加宽 |
|---|---|---|---|---|---|
| 7.0 | `fma` | `fma` | `fma` | **编译失败** | 能编 |
| 7.5 | `fma` | `fma` | `fma` | **编译失败** | 能编 |
| 8.0 / 8.6 / 8.9 / 12.0 | `m16n8k16` | `m16n8k16` | `m16n8k8` | `m16n8k32` | 能编 |
| 9.0 | `wgmma` | `wgmma` | `wgmma` | `wgmma` | 能编 |

三条结论，每一条都和"想当然"的版本不同：

- **浮点 `tl.dot` 从 7.0 就能编**。所谓"Triton 删掉了 Volta 的 `mma.m8n8k4`"是错的。
- **但 7.0 和 7.5 都到不了 tensor core**，两者一律下降成 CUDA core 的 `fma`；
  `mma.sync` 要到 8.0 才出现。所以在 sm_7x 上，一个带 `tl.dot` 的 Triton kernel
  不是"慢一点"，而是整个离开了 tensor core 这条路。
- **int8 `tl.dot` 在 7.0 / 7.5 上根本编不出来**（四种形状都失败在
  `TritonGPUAccelerateMatmul`）。Turing 硬件有 IMMA，Triton 3.4 就是不发它。这比
  "IMMA 不够用"强得多：量化的精确块那条臂在 sm_7x 上**不存在编译结果**。

**AOT 能编 ≠ 在那代硅上跑对。** 上面每一格都只是工具链的回答；bf16 在 sm_7x 上靠某种
软件转换编了出来，但没有任何一张 7.x 的卡真的跑过它（见「验证记录」）。

### 硬件事实（另一半依据）

| 能力 | 最低算力 | 说明 |
|---|---|---|
| fp16 tensor core（HMMA） | sm_70 | Volta 就有，但 Triton 3.4 在 8.0 以下不发 |
| INT8 tensor core（IMMA） | sm_75 | Turing 有，但 Triton 3.4 在 8.0 以下不发 |
| bf16 tensor core 与 `cvt.f32.bf16` | sm_80 | Triton 仍然能为 sm_70 编出 bf16 的加宽，所以本表里 bf16 的下限是**我们主动拒绝**，不是编译限制 |
| `cp.async` | sm_80 | FlashAttention 的流水化 tile 装载依赖它；与 dtype 无关，是 FA4 在 sm_7x 上没有任何路径的原因 |
| fp8 运算 | sm_89 | |
| TMA 描述符 | sm_90 | |

### 一条 gate 取两者的较高者

`arch.py` 的 `Requirement.floor` 是**实际的 gate**，`compile_floor` 在两者不同时记下
"工具链其实能编到哪"。这样后来的人一眼能看出某条 gate 是**我们的取舍**（kernel 在那里
会输给自己的 fallback），而不是一个他可以去修的编译错误。

热力图 kernel 就是这种情况：它从 sm_70 就能编，但在 sm_7x 上只有 CUDA core 的 `fma`，
而它要替代的 torch 参考实现**仍然能通过 cuBLAS 吃到 tensor core**。按规格书算一笔
（16:9@37 的工作点：38k slots、56 头、d=128，每层 20.7 TFLOP，中间分数 162 GB）：

| 卡 | 融合 kernel（`fma`） | 参考实现（cuBLAS TC + HBM 往返） | 比值 |
|---|---|---|---|
| V100 | 1.32 s | 0.62 s | 2.1x **慢** |
| T4 | 2.56 s | 1.42 s | 1.8x **慢** |
| RTX 2080 Ti | 1.54 s | 0.78 s | 2.0x **慢** |

这是规格书算术，不是实测（没有这些卡）。但方向足够清楚：**在 sm_7x 上把热力图路由进
融合 kernel 是一次反向优化**，所以 gate 留在 sm_80，`supports()` 在那以下返回 False，
`heatmap.teacher_heat` 自动走参考实现。

### 每个 kernel 的下限

表里的 dtype 列是"该 dtype 需要的最低算力"；没列出的 dtype 表示该 kernel 根本没实现。
代码里的同一张表是 `kernels/arch.py` 的 `REQUIREMENTS`，单测
`tests/unit/test_kernels_arch.py` 的 `_MATRIX` 逐行钉住它。

| kernel | 整体下限 | 决定下限的操作 | bf16 | fp16 | fp32 |
|---|---|---|---|---|---|
| `fa4.dense_attention` | sm_80 | cp.async + sm_80 MMA 形状 | sm_80 | sm_80 | 未实现 |
| `fa4.block_sparse_attention` | sm_80 | 同上；vendored 的 `fa4_sm8x` 补丁打在 SM80 kernel 上，继承同一个下限 | sm_80 | sm_80 | 未实现 |
| `block_heat_triton.teacher_heat` | sm_80（**能编到 7.0**，见上：gate 是取舍不是编译限制） | QK^T 是一个 `tl.dot`，8.0 以下下降成 CUDA core 的 `fma` | sm_80 | sm_80 | sm_80 |
| `tile_gather_triton.gather` / `.scatter` | sm_70 | 只有搬运与归约，**没有** `tl.dot`，所以不涉及 tensor core | sm_80（主动拒绝：Triton 能编，但没有 7.x 的卡跑过那套软件转换） | **sm_70** | sm_70 |
| `elementwise_triton`（RoPE / AdaLN / SwiGLU） | sm_80 | 为了与 eager 逐位相等，每个中间量都舍入到 bf16 | sm_80 | 未实现 | 未实现 |
| `compensation_triton.compensate` | sm_80 | 概率块被 cast 成 bf16 去做 `tl.dot`；但**先卡住的不是这个 cast**——`compensation_triton.attention` 的精确块走 `fa4.block_sparse_attention`，那条本身就是 sm_80 | sm_80 | 未实现 | 未实现 |
| `sol_veda.block_sparse_attention`（融合 INT8） | sm_80 | **Triton 3.4 在 8.0 以下拒绝 int8 `tl.dot`**（失败在 `TritonGPUAccelerateMatmul`），Turing 有 IMMA 也没用；bf16 的 summary 与在线 softmax 只是下一道障碍 | sm_80 | 未实现 | 未实现 |
| `h3.attention.dense_attention` `backend='sdpa'` | sm_70 | PyTorch 的融合 SDPA kernel | sm_80 | sm_70 | sm_70 |
| 同上，`return_lse=True` | sm_80 | 唯一返回 LSE 的是 flash 后端，它要求 sm_80 | sm_80 | sm_80 | 未实现（flash 后端不收 fp32） |
| `kernels.reference.block_sparse_attention` | 无 | 纯 torch，CPU 也能跑 | 任意 | 任意 | 任意 |
| `h3.attention.dense_attention` `backend='math'` | 无 | 纯 torch fp32 | 任意 | 任意 | 任意 |

### 结论：sm_70 / sm_75 上现在能做什么

**一句话的指导：在 sm_70 / sm_75 上跑稠密注意力。这两代卡上没有任何能回本的稀疏
kernel，我们也不打算做一个。** 理由不是 FA4 缺 cp.async——那只是**第一个**撞到的东西，
而且听起来像"绕过去就有快路径"。真正的原因在下面那张实测表里：**Triton 在 8.0 以下拿
不到 tensor core**，所有 `tl.dot` 都是 CUDA core 的 `fma`（int8 的更是连编都编不出来）。
于是任何在 sm_7x 上重写的块稀疏 kernel 都要用 1/8 的算力去换密度带来的 FLOP 节省，
按 `密度 × 8` 算，ρ=0.10 只比稠密好 1.25 倍、ρ≈0.12 就打平——**省下来的 FLOP 还不够
补上离开 tensor core 的损失**。稀疏在这里不是"难做"，是"不划算"。

下面是逐项的细节。

- **稀疏注意力的快路径（FA4）在 sm_70 / sm_75 上不存在**，而且不是 dtype 的问题：
  CuTe kernel 建在 cp.async 和 sm_80 的 MMA 形状上。`fa4.available()` 在那里是 False，
  `fa4.unavailable_reason()` 会说出缺的是哪一条。但**不要把 cp.async 当成那道墙本身**
  （见上）：即使有人把精确块换成一个不用 cp.async 的 kernel，只要它是 Triton 写的，
  它就在 CUDA core 上。想在这两种卡上跑稀疏注意力，只有 `kernels/reference.py` 这条
  token 粒度的 torch 路径，由调用方显式用 `allow_reference_kernel=True` 选中——它对真实
  长度的 clip 慢到不可用，是测试用的。
- **打分器的教师热力图（`block_heat_triton`）在 sm_7x 上我们不路由进去**，尽管它能编：
  8.0 以下 Triton 只给 CUDA core 的 `fma`，按上面那笔算术它比 torch 参考实现慢约 2 倍，
  而参考实现靠 cuBLAS 仍然吃到 tensor core。`heatmap.teacher_heat` 用 `supports()` 路由，
  所以这两代卡上自动走参考实现——**功能上不缺这一块，只是没有加速**。
  （这条取代了本文件初版的说法「sm_75 + fp16 可以跑」：能跑，但跑了更慢。）
- **tile 排列与池化（`tile_gather_triton`）在 sm_70 上用 fp16 就能跑**，而且这里的
  "能跑"是实打实的：kernel 里没有 `tl.dot`，所以 tensor core 这件事根本不参与，sm_70
  在搬运与归约上不比 sm_80 吃亏。bf16 仍然挡在 sm_80，但那是**主动拒绝**：Triton 能为
  sm_70 编出 bf16 的加宽，只是没有卡跑过它，而 fallback（`veda.tiling` 的 torch gather）
  是已知好的。放开它只需要把 `_TRITON_MOVE_DTYPES` 里 bf16 的下限改成 `(7, 0)`。
- **融合补偿（`compensation_triton`）在 sm_7x 上不只差一个 cast。** 这条路径是
  「FA4 跑精确块 + Triton 补偿被丢掉的块」，所以它的第一道门槛是 FA4 的 sm_80，
  bf16 的 `tl.dot` 只是接下来会撞到的第二件事。换句话说**把 cast 改成 fp16 不会让
  这条路径在 sm_75 上跑起来**：得先给它配一个能在 sm_75 上跑的精确块 kernel。
  而且那个替代 kernel **不能是 Triton 写的**，否则它只有 CUDA core 的 `fma`。
  一笔很短的算术：稠密注意力走 cuBLAS / SDPA 吃 tensor core，而 Triton 的 fma 路径
  是 fp32 CUDA core，两者在这几代上差约 8 倍（V100 125 : 15.7 TFLOPS，T4 65 : 8.1），
  所以稀疏 kernel 相对稠密的耗时比约是 `密度 × 8`：**ρ=0.10 时只比稠密好 1.25 倍，
  交叉点在 ρ≈0.12，再稠一点就不如直接跑稠密**。（算式里的两个吞吐是规格书，没有这些卡
  可测；但**取 fp32 那一列这件事是实测的**：`probe_triton_floor.py --ops` 数了下降后的
  PTX，sm_70 / sm_75 上一个 128x64x128 的 fp16 / bf16 dot 是 8192 条 `fma.rn.f32`、
  一条 `fma.rn.f16x2` 都没有——Triton 为 fp32 累加器把操作数加宽了，所以**打包 fp16 的
  那一列（T4 16.2 而不是 8.1）不适用**，比值不会减半。）
- **融合的 Sol+Veda（INT8）自己就会报对的错**：`kernels/sol_veda.py` 的 `available()`
  用 `get_device_capability()[0] >= 8` 设限，`unavailable_reason()` 返回
  「needs SM80 or newer」，所以它在 Volta / Turing 上本来就是自描述的，不依赖
  `arch.py`。`arch.REQUIREMENTS` 里仍然给它留了一条，只为了让矩阵是完整的一张表，
  并且现在那一条给出的是更强的理由：**Triton 3.4 在 8.0 以下连 int8 `tl.dot` 都编不
  出来**（实测），所以这条臂在 sm_7x 上不是"IMMA 不够"而是没有编译结果；bf16 的
  summary 与在线 softmax 只是下一道障碍。
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

- **以为 sm_70 / sm_75 的唯一障碍是 bf16。** 障碍不止一个、也不都在 dtype 上：
  FA4 要的是 cp.async（与 dtype 无关，所以 sm_7x 上**没有任何** FA4 路径），Triton 的
  `tl.dot` 在 8.0 以下拿不到 tensor core，int8 `tl.dot` 在 8.0 以下干脆编不出来。
  对策：能力表按**操作**声明，不按 dtype 声明，报错里也说操作名。
- **初版这张表里关于 Triton 的两条是凭回忆写的，而且两条都错了。** 一条说"Triton 删掉
  了 Volta 的 `mma.m8n8k4`，所以 `tl.dot` 的下限是 7.5"——其实 7.0 就能编，而 7.5 也
  **拿不到** tensor core，真正的分界是 8.0；另一条说"8.0 以下没有 `cvt.f32.bf16`，所以
  bf16 kernel 编不出来"——Triton 照样为 7.0 编出了 bf16 的加宽。两条错误的方向还相反：
  一条把门槛抬高了（7.0 其实能编），一条把理由说死了（bf16 其实能编，是我们选择不用）。
  **代价**：整张 sm_7x 的表都建在这两条上，而它们本来一分钟就能测出来。对策：
  `scripts/probe_triton_floor.py` 做 AOT 编译并把下降结果（`mma` 形状还是 `fma`）打出来
  ——**不要只看编译成功与否**，"能编"和"能用"在这里差 8 倍算力；`Requirement` 里
  `floor` 与 `compile_floor` 分开记，让"我们的取舍"与"工具链的限制"不再混为一谈。
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
| 2026-10-12 | 仅 CPU（无 CUDA、无 triton） | e09076f | `pytest tests/unit` 724 passed / 12 skipped；新增 `test_kernels_arch.py` 51 passed / 12 skipped |
| 2026-10-12 | 有 Triton、**不用 GPU**（AOT 编译只要 ptxas） | fa897fe | `scripts/probe_triton_floor.py`：7 个算力 × 4 种 dtype × 2 种 kernel 形状的编译结果与下降指令，见上面那张实测表 |
| 2026-10-12 | 仅 CPU | 本次提交 | 按实测结果改掉两条错的下限、分出 `compile_floor` 并加测；`pytest tests/unit` 763 passed / 12 skipped |

**sm_70 / sm_75 上没有跑过任何东西（GPU 未验证）。** 项目没有这两种卡。工具链层面的
问题（能不能编、下降成什么）现在是实测的；下面这些仍然只有"规格书 + 读代码"的依据：

1. `block_heat_triton` 在 sm_7x 上**比参考实现慢约 2 倍**这笔规格书算术
   （`fma` 对 cuBLAS tensor core）。它是现在把 gate 留在 sm_80 的唯一理由，所以如果
   有人拿到 T4 / RTX 20xx，值得实测一次：万一实际是"差不多快"，gate 就应该下移到
   `TRITON_COMPILES`，而 `compile_floor` 正是为这个可能性记下来的。
2. `tile_gather_triton` 在 **sm_70 / sm_75 + fp16** 上能跑。
3. ~~Triton 确实在 sm_70 上拒绝 `tl.dot`~~ —— **已测，而且结论是反的**：浮点
   `tl.dot` 从 7.0 就能编，7.0 / 7.5 都只到 `fma`，int8 `tl.dot` 在两者上都编不出来
   （`scripts/probe_triton_floor.py`，fa897fe）。AOT 编译只需要 ptxas，所以这一条不再
   欠硬件。仍然欠硬件的是：那套 bf16 的软件转换在 7.x 硅上**跑得对不对**（本表因此
   把 bf16 的下限保留在 sm_80，并明确写成"主动拒绝"）。
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
