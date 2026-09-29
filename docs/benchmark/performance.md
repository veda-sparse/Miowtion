# 实测性能

本文件记录 Miowtion 在真实权重上跑出来的性能数字：推理和训练各一节，每条记录都带上
生成特性（seq 长度、纵横比、时长、NFE）和硬件特性（几张什么卡、TP/SP、是否 offloading），
这样一条数字在别的机器上才有意义。数字全部由 `scripts/benchmark.py`（推理）和训练器自己的
JSONL（训练）写出，不是手抄的日志。

## 1. 口径（先读这节，否则数字会被误读）

- **稳态 step time**：第 1..N-1 步的平均，**不含第 0 步**。第 0 步要付 FA4 CuTe 的 JIT 编译
  和 offload block 的首次 H2D，把它算进去会让少步轨迹凭空慢几秒。
- **各 stage 的时间**：启动开销（权重加载、LoRA 合并、AdaLN 表、打分器 bundle、VAE 加载）
  一个进程只付一次，**不能摊进 step time**；文本编码一条 prompt 付一次；VAE 解码每条片子
  付一次，它属于用户看到的端到端延迟。
- **端到端**：轨迹 setup + 去噪 + VAE 解码，也就是同一个进程里第 N+1 条片子的延迟
  （启动开销已摊掉）。**不含**文本编码——它是离线一次性的，单独列在 §4。
- **diffusion MFU**：`miowtion/h3/flops.py` 的解析 FLOP 数除以实测时间，再除以设备的
  **稠密 bf16 峰值**（RTX 4090 = 165.2 TFLOP/s，不是 2:4 稀疏的 330）。FLOP 数按每个 MAC
  2 FLOP 数所有 matmul；线性层按 padding 后的 `seq_len` 计，注意力只按真实行数 `used` 计
  （`h3/attention.py` 把 q/k/v 切到 `[:used]`）。RMSNorm / SiLU / RoPE / AdaLN 调制 /
  softmax 的 exp 不计，它们都是 O(seq_len·hidden)，合计 <1%。
  该解析式在 tiny config 上与 `torch.utils.flop_counter.FlopCounterMode` **逐位相等**
  （`tests/unit/test_h3_flops.py`），不是估算。
  Veda 的注意力按**实际调用数**计价（`dense_calls` / `sparse_calls`），所以部分层、
  部分步保持稠密时 MFU 依然诚实。
- **不考虑 DP**：这里量的是一张卡在一条请求上的吞吐。多卡推理是一卡一条样本（数据并行），
  规模化的吞吐不在本文件范围内。
- **没有 TP / SP**：仓库只有 FSDP2/HSDP（`miowtion/train/parallel.py`），所以下面所有记录
  都是 TP = SP = DP = 1。

## 2. 硬件与并行配置

| | |
|---|---|
| 卡 | 1×RTX 4090（24 GB），稠密 bf16 峰值 165.2 TFLOP/s |
| torch | 2.14.0+cu126 |
| TP / SP / DP / FSDP shards | 1 / 1 / 1 / 1 |
| offloading | 开：50 个 block 的权重常驻主机内存，prefetch 1，MLP 按 4096 行分块 |
| 主机内存占用 | 一个推理进程为 offload 的 block pin 约 33 GB |

24 GB 卡上 latent_t 102 必须 `offload_blocks: 50` + `mlp_chunk_rows: 4096`，这不是调优选项
而是能否跑起来的前提。

## 3. T2VA 推理

请求：FL2VA + Turbo 8 步 LoRA（`minimax_h3_turbo_v4_step600_ema`），t2va，1344×768（16:9），
文本 589 行，seed 0，Veda 保留 10%（打分器 bundle `t102_step200_fp8`）。
sample `holdout14s_0000`，三个长度共用同一条样本，所以 `text_len` 恒定。

### 3.1 各长度的稳态 step 与 MFU

| latent_t | 时长 | 帧数 | seq_len (used) | 注意力 | 稳态 step | 每步 FLOP | MFU |
|---|---|---|---|---|---|---|---|
| 37 | 5.17 s | 124 | 38336 (38299) | 稠密 | 24.48 s | 3.58 PFLOP | **88.5%** |
| 37 | 5.17 s | 124 | 38336 (38299) | Veda 10% | 13.81 s | 1.69 PFLOP | 74.0% |
| 72 | 10.13 s | 243 | 73984 (73975) | 稠密 | 71.02 s | 10.70 PFLOP | **91.2%** |
| 72 | 10.13 s | 243 | 73984 (73975) | Veda 10% | 29.29 s | 3.64 PFLOP | 75.1% |
| 102 | 14.38 s | 345 | 104576 (104555) | 稠密 | 129.58 s | 19.70 PFLOP | **92.0%** |
| 102 | 14.38 s | 345 | 104576 (104555) | Veda 10% | 45.95 s | 5.60 PFLOP | 73.7% |

稠密 MFU 随长度上升（88.5% → 91.2% → 92.0%）：注意力是纯 GEMM、算术强度最高，
序列越长它在一步里占比越大。Veda 的 MFU 反而更低（74~75%），这是对的——它把最高效的那部分算掉了 90%，
剩下的是被主机内存带宽卡住的线性层，所以"MFU 低"在这里是加速的结果而不是问题。

### 3.2 加速比拆成两层：注意力 vs 整个 DiT forward

| latent_t | 稠密注意力 | Veda 注意力 | **注意力加速** | 稠密 step | Veda step | **step 加速** | 非注意力部分 |
|---|---|---|---|---|---|---|---|
| 37 | 13.49 s | 2.84 s | **4.75×** | 24.48 s | 13.81 s | **1.77×** | 10.99 → 10.97 s |
| 72 | 49.77 s | 8.00 s | **6.22×** | 71.02 s | 29.29 s | **2.42×** | 21.25 → 21.29 s |
| 102 | 99.41 s | 15.77 s | **6.31×** | 129.58 s | 45.95 s | **2.82×** | 30.17 → 30.19 s |

这是本文件最重要的一张表：

- **注意力加速随长度增长**（4.75× → 6.22× → 6.31×，t72 之后开始饱和），因为稠密
  注意力是 O(used²) 而块稀疏是 O(used² × 保留比例) 再加上打分和 gather 的固定开销；序列越长，固定开销被摊得越薄，
  越接近 1/0.1 的理论上限。
- **整个 DiT forward 的加速远低于注意力加速**（1.77× / 2.42× / 2.82×），而且差距由
  **非注意力部分**完全解释：它在稠密和 Veda 两路上几乎逐秒相同（10.99 vs 10.97；
  21.25 vs 21.29），因为那是同一批线性层、同一批权重 H2D 拷贝，稀疏化一点也没碰它。
  它就是 Amdahl 的分母：t37 上非注意力占稠密一步的 45%，所以 step 加速上限是 1.82×，
  实测 1.77×；t72 上占 30%，上限 3.34×，实测 2.42×；t102 上占 23%，上限 4.30×，
  实测 2.82×。上限与实测之间的差是 Veda 自己的成本（打分器前向、tile gather / scatter、
  掩码构造）。
- **所以要继续提速，下一步不在注意力上**，而在这 11 / 21 / 30 s 的权重流式拷贝上（更大的
  `mlp_chunk_rows`、更深的 prefetch、更少的 offload block，都要拿显存换）。

### 3.3 启动开销（每个进程一次）

| stage | 秒 |
|---|---|
| 加载 teacher 权重（50 block offload + pin） | 66.5 |
| 合并少步 LoRA（209 个权重） | 105.7 |
| 预计算 AdaLN 表（8 个 timestep 集 × 50 层） | 10.0 |
| 加载打分器 bundle（fp8） | 3.9 |
| 加载 VAE 解码器 | 64.8 |
| 合计 | **约 4.2 分钟** |

这 4.2 分钟与长度、注意力模式都无关，只付一次。单条片子的绝对延迟要把它加上；跑一批时
它可以忽略。

## 4. 文本编码器（每条 prompt 一次）

Qwen3-VL 文本塔（只实例化 `encode.TEXT_LAYERS = 50` 层），预算 `0=20GiB,cpu=90GiB`，
超出的层落在 CPU：

| | 秒 |
|---|---|
| 加载 | 42.7 |
| 首次编码（含 autotune + 首次走过 CPU 上的层） | 63.9 |
| **稳态编码**（361 行的 prompt） | **3.01** |

文本编码与几何、注意力模式都无关，一条 prompt 付 3 秒；对上百秒的片子是可以忽略的量级。
训练与推理都用预先编码好的 sample cache（`artifacts/samples/*`），所以去噪路径里根本不会
再跑它——这也是它必须单独量的原因。

## 5. VAE 解码（每条片子一次）

| latent_t | 视频解码 | 音频解码 |
|---|---|---|
| 37（124 帧） | 34.7 s / 58.1 s | 1.1 s / 0.1 s |
| 72（243 帧） | 172.0 s / 171.8 s | 0.1 s / 0.1 s |
| 102（345 帧） | 372.8 s / 390.6 s | 0.2 s / 0.1 s |

两个数字分别来自同一几何的稠密轮和 Veda 轮——解码与注意力模式无关，所以它们本应相等。
t72 相差 0.1%、t102 相差 4.8%，**但 t37 相差 67%（34.7 vs 58.1）**：t37 的稠密轮是整个进程
里第一次解码，它是唯一的异常点而不是系统性偏差，所以 t37 的代表值取 58 s，34.7 s 那个数
不要单独引用。见 §9 待办。

视频 VAE 是 2.4B 的 ViT3D，自己带时间维注意力，所以时间随帧数**超线性**增长
（58 s @124 帧 → 172 s @243 帧 → 373 s @345 帧：帧数 ×2.8，时间 ×6.4）。音频 VAE 小到
不影响计时（t37 稠密轮的 1.1 s 是 CUDA 初始化）。

## 6. 端到端（同一进程里的第 N+1 条片子）

| latent_t | 注意力 | setup | 去噪（8 步） | 解码 | **端到端** |
|---|---|---|---|---|---|
| 37 | 稠密 | 5.2 s | 195.9 s | 35.8 s | **236.9 s** |
| 37 | Veda 10% | 0.1 s | 120.8 s | 58.2 s | **179.0 s** |
| 72 | 稠密 | 0.1 s | 568.1 s | 172.1 s | **740.3 s** |
| 72 | Veda 10% | 0.1 s | 234.3 s | 171.9 s | **406.3 s** |
| 102 | 稠密 | 0.2 s | 1036.4 s | 373.0 s | **1409.5 s** |
| 102 | Veda 10% | 0.2 s | 368.0 s | 390.8 s | **759.0 s** |

端到端加速（t72 1.82×、t102 1.86×）低于 step 加速（2.42× / 2.82×），因为 VAE 解码是两路
共有的固定成本：它占 Veda 端到端的 42%（t72）和 51%（t102）。**14 秒的片子稀疏化之后，
一半的墙上时间花在解码而不是去噪上**——片子越长、稀疏化越狠，瓶颈越往解码那边移，
下一个该动的地方在那里。

## 7. 训练（阶段 1，打分器）

阶段 1 的一次 micro-step = **一次稠密 teacher forward（`no_grad`）** + 教师块热力图 +
打分器的前向反向。`accum: 1`，所以一次 update 就是一个 micro-step，日志里的 `seconds`
就是每个梯度步的时间。可训练参数 275,251,200（只有打分器，teacher 全程冻结）。
配置与真实训练一致：8 步 turbo 轨迹、keep 0.1、`teacher_q_tiles 1.0`、offload 50 block、
prefetch 1、`mlp_chunk_rows 4096`、优化器状态 offload 到主机。

| 几何 | updates | 首个 update | **稳态 update** | 峰值显存 | 同几何的推理稠密 step | 热力图 + 打分器的开销 |
|---|---|---|---|---|---|---|
| 16:9@37 | 12 | 44.43 s | **33.53 s** | 10.24 GiB | 24.48 s | +9.1 s（+37%） |
| 16:9@102 | 8 | 201.11 s | **182.71 s** | 18.11 GiB | 129.58 s | +53.1 s（+41%） |

- 首个 update 比稳态慢 11 s / 18 s：FA4 CuTe 的 JIT 和 offload block 的首次 H2D，和推理
  第 0 步是同一回事。
- **教师热力图 + 打分器只占一次 update 的 27~29%**，其余全是那次稠密 teacher forward。
  所以阶段 1 的训练成本基本就是"跑推理但每步都要算热力图"，想加速阶段 1 要先加速稠密前向。
- **diffusion MFU（下界）**：只把稠密 teacher forward 的 FLOP 算进分子（热力图的 Triton
  kernel 和打分器还没进 `miowtion/h3/flops.py`），得到 **64.6%（t37）/ 65.3%（t102）**。
  真实值比这更高，但高不过推理稠密 step 的 88.5% / 92.0%——热力图和打分器的算术强度低。
- 峰值显存 10.2 / 18.1 GiB，都在 24 GB 以内；t102 的 18.1 GiB 已经没有多少余量，所以
  `offload_blocks: 50` 和 `mlp_chunk_rows: 4096` 在训练侧同样是前提而不是调优。
- 这两个 run 只跑十来步、从随机初始化的打分器开始，**不代表收敛质量**，只用来量成本。
  （参考：t37 跑到第 12 步 KL 0.92 / recall 0.53，t102 第 8 步 KL 1.11 / recall 0.57。）

阶段 2（稀疏学生 + LoRA 恢复）还没在 GPU 上验证过，没有数字。

## 8. R2VA

**不测**：`scripts/encode_samples.py` 对 `ref2va` 直接 `raise`（没有参考图编码路径），
现有的 sample cache 全是 t2va。也没有 Ref2VA 的少步 LoRA，Veda 的 tile 方案表也只在 FL2VA
的 t2va 几何上搜过。伪造一条请求能量出数字，但那是给一条谁都跑不起来的路径报性能，
所以这一节留空，等 ref2va 编码落地后再补。

## 9. 待办

- 解释 §5 里 t37 稠密轮视频解码偏快 23 s 的异常（t72 / t102 两轮分别只差 0.1% / 4.8%，
  所以不是系统性的）。怀疑是进程里第一次解码时显存还没被碎片化，但没有证据。
- VAE 解码已经占 Veda 端到端的一半（§6），却从来没有优化过：它是下一个该动的地方，
  而不是注意力。
- 多卡（2×4090）的记录：现在只有单卡。注意仓库没有 TP/SP，多卡只能是 FSDP 分片训练或
  一卡一样本的推理。
- fp8 块稀疏 kernel（§11.4 第 6 条）：SM120 上 fp8 稠密注意力实测 1.8×，但 FA4 的 SM120
  路径只接 fp16 / bf16。这是目前已知**唯一还剩数量级收益**的方向。
- SM120 的 `ncu` 性能计数器在这台机器上没有权限（`ERR_NVGPUCTRPERM`），所以 §11.4 的
  kernel 判断都是以"实测 GEMM 上限"为尺子，没有 pipe 级证据。

## 10. 复现

```bash
# 推理（DiT + VAE 解码）
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --root weights/MiniMax-H3 --variant FL2VA --task t2va \
  --schedule turbo --num-steps 8 \
  --adapter weights/turbo_lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  --sample-cache artifacts/samples/moviegen_holdout20 \
  --sample-id holdout14s_0000 \
  --geometry 16:9@37 16:9@72 16:9@102 --attention dense veda \
  --predictor weights/veda/t102_step200_fp8.safetensors \
  --keep-ratio 0.1 --offload-blocks 50 --prefetch 1 --mlp-chunk-rows 4096 \
  --decode --out runs/bench/t2va_4090_1gpu.json

# 文本编码器（单独一个进程：两个塔挤不进同一张 24 GB 卡）
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --root weights/MiniMax-H3 --variant FL2VA --stage text \
  --prompts data/prompts/moviegen_video_bench_h3.jsonl --repeat 3 \
  --max-memory "0=20GiB,cpu=90GiB" \
  --out runs/bench/t2va_text_encoder_4090_1gpu.json

# 随机权重（§11，只需要 third_party/MiniMax-H3 子模块里的配置文件）
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --root third_party/MiniMax-H3 --random-weights \
  --geometry 16:9@37 16:9@72 16:9@102 --attention dense veda \
  --offload-blocks 0 --mlp-chunk-rows 8192 --out runs/bench/random_pro6000.json

# 训练（阶段 1，accum 1 所以一次 update 就是一个 micro-step）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train.py \
  --config configs/stage1_bench_16x9_t37_1gpu.yaml
```

## 11. 随机权重实测（不需要真实权重）

`scripts/benchmark.py --random-weights` 在只有 release 配置文件（`third_party/MiniMax-H3`
子模块）的机器上量同一个 DiT，用于"拿到一张新卡，先看它能跑多快"。

**为什么数字可信**：一步 DiT 的成本只由形状决定——每个 GEMM、每次注意力、每次 offload
block 的 H2D 拷贝，形状和 dtype 与权重取值无关；Veda 按预算保留固定比例的块，top-k
保留的块数与打分无关。所以稳态 step、注意力时间、MFU 与真实权重同口径。

**替换了什么**（其余路径——meta 构建、offload 放置、pin、严格的名字 / 形状 / dtype 检查、
AdaLN 表——原样执行）：

| 真实输入 | 随机替身 | 代码 |
|---|---|---|
| DiT 权重（safetensors） | `RandomCheckpoint`：release 自己的 index 给出张量名，meta 模型给出形状；矩阵 N(0, 1/fan_in)、norm 为 1、bias 为 0、RoPE 频率用模型自己的 | `miowtion/h3/synthetic.py` |
| 编码后的 prompt | `--text-len` 行随机 hidden（默认 589，与 §3 的样本同长，所以布局相同） | `data.SyntheticSampleCache` |
| 打分器 bundle + 方案表 | 随机初始化的打分器（bf16 常驻，与 fp8 bundle 加载后一致）+ 每个几何一张最小 padding 的 uniform 方案 | `veda.bundle.random_bundle` |
| 少步 LoRA | 不合并（随机权重上的 LoRA 没有意义，且合并只是启动开销） | — |

**不可比的部分**：启动各 stage（没有读文件、没有合并 LoRA）；生成的样本是噪声。
搜索出来的方案表每层最多混用几种 tile 形状，padding 与 uniform 方案略有不同，所以 Veda
的数字与真实 bundle 可能有几个百分点的差别。VAE 解码不支持：release 的 VAE 类只能从
权重文件构造，`--random-weights --decode` 直接报错。JSON 里 `weights: "random"` 标明来源。

### 11.1 1×RTX PRO 6000 Blackwell Server Edition（96 GB）

2026-09-28，torch 2.14.0+cu130，稠密 bf16 峰值 503.8 TFLOP/s；**offload 0**（DiT 常驻显存，
约 45 GB 权重）、`mlp_chunk_rows 8192`、TP = SP = DP = 1；请求与 §3 相同（t2va 16:9，
text_len 589，turbo 8 步，keep 0.1）。每步 FLOP 数与 §3.1 逐位相同（同一布局）。

| latent_t | 注意力 | 稳态 step | 注意力 | 非注意力 | MFU | 4090 同项 step | 相对 4090 |
|---|---|---:|---:|---:|---:|---:|---:|
| 37 | 稠密 | 10.89 s | 5.80 s | 5.08 s | 65.3% | 24.48 s | 2.25× |
| 37 | Veda 10% | 6.35 s | 1.43 s | 4.92 s | 52.8% | 13.81 s | 2.18× |
| 72 | 稠密 | 31.93 s | 22.00 s | 9.93 s | 66.5% | 71.02 s | 2.22× |
| 72 | Veda 10% | 13.74 s | 4.05 s | 9.69 s | 52.5% | 29.29 s | 2.13× |
| 102 | 稠密 | 58.03 s | 44.02 s | 14.00 s | 67.4% | 129.58 s | 2.23× |
| 102 | Veda 10% | 21.85 s | 8.16 s | 13.68 s | 50.9% | 45.95 s | 2.10× |

注意力加速 4.07× / 5.44× / 5.39×，step 加速 1.71× / 2.32× / 2.66×。算力是 4090 的 3 倍，
step 却只快 2.2 倍：MFU 从 4090 的 88~92% 掉到 65~67%，原因见 11.2。

### 11.2 一步的 GPU 时间拆解（torch.profiler）

方法：随机权重、eager（与 benchmark 相同），第 0 步预热（FA4 JIT），第 1 步用
`record_function` 给每个组件打标签、`with_stack` 采集，kernel 通过 launch 的 correlation
归到发起它的最内层标签；**kernel 时间之和、GPU busy 并集、墙上时间分开报**。profiler
本身的开销 < 2%（6.27 → 6.38 s）。这次验证拿不到 GPU 性能计数器（`ncu` 报
`ERR_NVGPUCTRPERM`），所以 kernel 内部的 pipe 利用率没有测，下面对 kernel 的判断以
"实测 GEMM 上限"为尺子：cuBLAS bf16 8192³ 在这张卡上只有 **432.5 TFLOP/s**（峰值的
86%；FA4 与 cuBLAS 在 SM120 上都走 SM80 时代的 `mma.sync`）。

Veda t102 一步（21.6 s kernel 时间，busy 并集 21.6 s，墙上 22.0 s）：

| 组件 | GPU 时间 | 占比 | 说明 |
|---|---:|---:|---|
| FA4 块稀疏 kernel | 7.37 s | 34.1% | 279 TFLOP/s，峰值的 55%，GEMM 上限的 65% |
| MLP（fc1 / fc2 GEMM） | 5.90 s | 27.3% | GEMM 392~416 TFLOP/s，已是上限的 91~96% |
| QKV / out_proj GEMM | 3.89 s | 18.0% | 同上 |
| 访存型逐元素算子 | ≈3.6 s | ≈17% | RoPE 1.09（cat / mul / add / neg 四类 kernel）、AdaLN modulate 0.73、门控残差 0.66、qk RMSNorm + split 拷贝 0.59、SwiGLU silu×mul 0.52 |
| Veda 自身（gather / pool / 打分器 / top-k / 掩码 / scatter） | 0.54 s | 2.5% | |
| GPU 空闲 | 0.41 s | 1.9% | 几乎全在 Veda：`select_video_blocks` 里的 `torch.nonzero` 每次调用同步一次主机（0.21 s），FA4 CuTe 每次调用的主机侧参数适配（0.12 s），`dense_block_mask`（0.08 s） |

稠密一步 GPU 几乎不空闲（t37：idle 0.004 s / 10.8 s），CPU 一直跑在 GPU 前面，launch
开销不是问题。稠密 FA4 kernel 357~368 TFLOP/s（峰值 71~73%，GEMM 上限 83~85%）。

### 11.3 优化后的实测（commit d0fda2d）

三处改动都不改变结果（见 §11.4 的第 1、2、4 条）：头分块改成
`VedaConfig.collect_bytes`（这里用 2048 MiB）、去掉选择路径的两处主机同步、逐元素链融合成
Triton kernel（与 eager 逐位相等）。同一张卡、同一请求：

| latent_t | 注意力 | 改动前 | 分块+去同步 | ＋融合逐元素 | 总计 | MFU |
|---|---|---:|---:|---:|---:|---:|
| 37 | 稠密 | 10.89 s | — | **10.25 s** | −5.9% | 65.3% → 69.3% |
| 37 | Veda 10% | 6.35 s | 6.04 s | **5.49 s** | −13.5% | 52.8% → 61.0% |
| 102 | 稠密 | 58.03 s | — | **55.98 s** | −3.5% | 67.4% → 69.8% |
| 102 | Veda 10% | 21.85 s | 20.18 s | **18.33 s** | −16.1% | 50.9% → 60.6% |

阶段 1 训练（`configs/stage1_bench_random_16x9_1gpu.yaml`，accum 1 所以一次 update 就是一个
micro-step，稳态值）：

| 几何 | 改动前 | 改动后 | 峰值显存 | 4090 同项 |
|---|---:|---:|---:|---:|
| 16:9@37 | 15.29 s | **14.52 s**（−5.0%） | 48.0 GiB | 33.53 s |
| 16:9@102 | 86.38 s | **84.25 s**（−2.5%） | — | 182.71 s |

训练的收益远小于推理：一次 update 里那次稠密 teacher 前向占绝大部分，而教师热力图
（Triton）和打分器都不经过这三处改动。

### 11.4 还能往哪里走（SM120 的实测天花板）

先把尺子定死：这张卡的**实测 bf16 GEMM 上限是 432.5 TFLOP/s**（cuBLAS 8192³，标称峰值
503.8 的 86%），FA4 与 cuBLAS 在 SM120 上都走 SM80 时代的 `mma.sync`。

| kernel | 实测 | 占峰值 | 占 GEMM 上限 |
|---|---:|---:|---:|
| FA4 稠密（t37 / t72） | 363~368 TFLOP/s | 72~73% | 84~85% |
| FA4 块稀疏（t37 / t102，keep 0.1，56 头一次） | 338 / 356 TFLOP/s | 67~71% | 78~82% |
| 线性层 GEMM（qkv / fc1 / fc2 的真实形状） | 392~416 TFLOP/s | 78~83% | 91~96% |

1. ~~头分块~~、2. ~~逐元素融合~~、4. ~~主机同步~~：已实现，见 §11.3。
3. **块稀疏 kernel 的 tile 调参已经没有余量**：把 128×32（在 4090/SM89 上调出来的）换成
   更大的 tile 全都更慢——t102 上 128×64 慢 18%、128×128 慢 44%（hdim 128 下 128×128 要
   96 KB smem，只剩 1 CTA/SM，而稀疏工作量本身不均匀）。256 线程在 t37 上快 1%，噪声级别。
   所以 bf16 下块稀疏 kernel 距上限只剩 ~20%，即整个 Veda step 的 ~5%。
5. **不能把 SM100 的 kernel 搬过来**：SM100 前向建立在 tcgen05（UMMA）+ TMEM 累加器 +
   2-CTA MMA 之上，**SM120 没有这三样**（它是 Blackwell 家族里仍用 warp 级 `mma.sync` 的那一支，
   既没有 Hopper 的 wgmma 也没有 SM100 的 tcgen05）。强行 `_arch=90` 直接被
   `Only SM 9.x is supported` 拒掉。能跨过来的只有**软件结构**：TMA 载入、warp 专精
   （load / MMA / softmax 分工）、mbarrier 多级流水（SageAttention3 的 sm120a kernel 正是
   用 `SM90_TMA_LOAD` + `PipelineTmaAsync` + SM120 的 blockscaled MMA atom 搭起来的）。
   按第 3 条的上限，这套重写在 bf16 下最多值 ~5% 的 step，**性价比低**。
6. **真正的杠杆是精度**：SM120 的 `mma.sync` 在 fp8 下是 bf16 的两倍。实测（稠密、非 causal、
   D=128、56 头，误差对 FA4 bf16）：

   | kernel | seq 38299 | seq 73975 | 相对 FA4 | rel L2 | 最差行 cos |
   |---|---:|---:|---:|---:|---:|
   | FA4 bf16 | 368 TFLOP/s | 368 TFLOP/s | 1.00× | — | — |
   | SageAttention2（INT8 QK + FP8 PV） | 654 | 668 | **1.78~1.82×** | 3.9e-2 | 0.998 |
   | SageAttention3（NVFP4） | 645 | 崩 | 1.75× | 1.9e-1 | 0.915 |

   NVFP4 在这张卡上**不比 fp8 快**（645 vs 654），误差却大一个数量级，而且 seq 73975 时 TMA
   描述符初始化失败、`illegal instruction` 崩掉——不是我们要的方向。fp8 才是。
   但 SageAttention 是**稠密**的：要拿到这 1.8×，得把 fp8 接进 FA4 的 SM120 **块稀疏** kernel，
   而 `flash_fwd_sm120.can_implement` 现在只接受 fp16 / bf16（`q_descale` / `k_descale` /
   `v_descale` 这些接口参数已经在了，只有 SM90/SM100 用得上）。这是下一个补丁（0008）的范围，
   属于"数值会变"的改动：按 AGENTS.md 1.5 需要误差记录 + 可视化对比 + 人工确认。

## 12. Kernel 端到端复测

`scripts/bench_e2e.py` 对每个序列长度执行多轮独立测量，并把原始轮次和聚合结果写入 JSON。
这里的 `kernel_ms` 只测已经准备好的掩码，`prep_ms` 包含每次调用的掩码准备、kernel 启动和
同步，更接近真实 Veda 调用的端到端成本。聚合同时保存 median、p95、min、max，避免一次
JIT 或系统抖动被误认为稳定性能。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_e2e.py \
  --seq 16384 32768 --heads 8 --head-dim 128 --density 0.1 \
  --repeat 3 --calls 16 --out runs/bench/e2e.json
```

JSON 中每个序列长度下的 `rounds.<seq>.<kernel>.aggregate` 是可直接用于报告的汇总；完整的
`rounds` 保留了每轮误差和效率，便于发现某个后端退化或错误地退回稠密计算。

2026-09-28，1×RTX PRO 6000 Blackwell Server Edition，SM120，torch 2.14.0+cu130，8 头、
d=128、密度 0.1、3 轮（median，单位 ms）：

| 序列 | FA4 dense | FA4 block sparse | FA4 DenseBlockMask | block sparse + prep |
|---:|---:|---:|---:|---:|
| 16384 | 3.411 | 0.355 | 0.375 | 0.421 / 0.375 |
| 32768 | 11.981 | 1.216 | 1.236 | 1.272 / 1.237 |

最后一列分别是索引列表和 DenseBlockMask 的 `prep_ms`；两者都包含每次调用的准备与 kernel。
两种稀疏后端的误差均低于 `8e-4`，没有检测到退回稠密的情况。
