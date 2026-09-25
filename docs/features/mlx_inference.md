# Apple silicon 推理（MLX + NVMe offloading）

## 目标

在一台 18 GB 统一内存的 Apple silicon 笔记本上跑 H3 DiT 的推理。33B 模型的主干有
50 个 block，bf16 下每个 block 约 0.77 GB，权重总量（约 38 GB）远超内存上限，因此
**权重常驻 NVMe，按 block 流式读入，读一个 block 的同时算上一个 block**，任何时刻
内存里只有固定的几个 block。

这是 CUDA 路径（`miowtion/train/parallel.py` 的 `HostSlabs` / `BlockStreamer`：权重放
主机内存、按 block H2D 拷贝）在 Apple silicon 上的对应物，只是"慢的那一层"从 PCIe
换成了 SSD。

本文是可行性研究 + 原型的结论，所有数字都是实测（合成权重、真实 shape），不是估算。

## 设计与不变量

### 为什么用 per-block slab 文件

发布的 checkpoint 是 safetensors 分片。直接从分片读有两种做法，都走 page cache；而
50 个 block 根本放不进 page cache，缓存它们只会把别的数据挤出去。所以默认方案是把
checkpoint 转成**每个 block 一个连续文件**（slab）：

```
magic 'MIOSLAB1' | uint64 LE header 长度 | JSON header | pad |
tensor 数据，每个 tensor 对齐到 SLAB_ALIGNMENT
```

- 对齐到 16 KiB（Apple silicon 的 VM page），这样绕过 page cache 的读覆盖整页。
- 读的时候用多个并发 `pread` **直接读进预分配好的 MLX buffer**（统一内存，
  `np.asarray` 拿到可写视图），没有中间拷贝，后台线程上不调用任何 MLX API。
- 默认 `F_NOCACHE`：读进来的权重这一步用完就不再用，缓存它没有收益。

### 不变量

- **slab 是纯数据变换**：写进去再读出来必须逐位相等（bf16 / 8 bit / 4 bit 都是）。
- **一个 slot 只有在 GPU 读完它之后才会被覆盖**：`BlockPrefetcher` 在复用 slot 前
  先 `mx.synchronize()`。这是流式路径唯一的正确性要求，违反了就是静默的数据竞争。
- **权重占用的内存由 slot 数决定，与模型大小无关**：depth=1 的预取用 2 个 slot。
- **分块不改变结果**：block 里除 attention 外都是逐行的，按 head 分块（QKV +
  attention）和按行分块（norm、输出投影、残差、MLP）只是重排，用来压峰值内存。

### 为什么块稀疏要"聚合"而不是"加掩码"

MLX 的融合 attention（`mx.fast.scaled_dot_product_attention`）接受布尔掩码，但掩码
是在 Q@K.T 的 tile 矩阵乘**之后**才施加的：只有内置的 `"causal"` 会缩短 key 循环的
上界，任意块掩码不会。所以给它一个 90 % 稀疏的块掩码只能拿到正确性和 O(S) 的内存，
拿不到 FLOP（实测 0.97×，等于没有）。

Veda 的关键性质是**每个 query tile 的预算固定**（选中的 key tile 数都一样）。于是可
以把选中的 key tile 直接 gather 成一个规整的
`[n_query_tiles, heads, budget * k_block, head_dim]` 张量，再交给**一次批量的稠密**
attention：kernel 看到的问题宽度就真的是 `budget * k_block` 而不是 `S`。

- **不需要写 Metal kernel**：实测 9–10 倍加速（见「实测」），已经接近密度的理论上界。
- 代价是 gather 要重写 `density * S^2 / q_block` 行，所以 query tile 要够大；Veda 的
  tile 本来就是几千行这个量级。
- 不变量：结果必须与"等价块掩码下的稠密 attention"**逐位相等**——两条路径看到的是
  同一批 key、同样的顺序，没有理由不等。单测用 `mx.array_equal` 卡死这一点。
- 稀疏的 query 行数（`used - dense_rows`）必须是 `q_block` 的整数倍、`used` 必须是
  `k_block` 的整数倍，否则 padding 行会被卷进 attention。不满足时直接报错，不静默
  回退到稠密。
- 真实的 Veda 选择并不是"预算严格相同"，见下面「接上真实的 Veda 选择」：短的行用
  `keep` 补齐并掩掉，全局行走 `dense_rows`，partial tile 走 `key_valid`。

### 数值

MLX block 的算子顺序对齐 `miowtion/h3/model.py`，让每一步逐元素运算在同一个位置
round 到 bf16：RMSNorm 的统计量和乘权重在 fp32 里算、只 round 一次；AdaLN 调制和
门控残差在 bf16 里算；SiLU 在 fp32 里算、round 一次。

Euler 步是个例外：torch 的 `add_(v, alpha=)` 发的是 fused multiply-add（只 round
一次），MLX 只能分开乘和加（round 两次），差最多 1 ulp（实测 2.4e-07，数量级为 1 的
值上）。MLX 既没有 fma 算子，GPU 上也没有 fp64；唯一能复现 torch 的办法是把这一步
走一次 GEMM，把轨迹钉死在 matmul 的累加顺序上，不值得。

结果：**调制、RoPE、SwiGLU 与 torch 逐位相等**；整个 block 不是——RMSNorm 的统计
量、GEMM、attention 的归约顺序不同（见「踩坑记录」）。

trunk 之外同样的做法：两张超越函数表（RoPE 的 cos/sin、时间步的正弦特征）在主机侧
用 numpy 算，而不是用 MLX 算子，因为它们很小（`[S, 96]` 和 `[M, 256]`）、纯逐元素，
放在主机上不花钱却能贴着 torch 参考。**RoPE 的表与 torch 逐位相等**；时间步的正弦
特征差 **1 ulp**（`freq_dim=256` 的 128 个频率里有 14 个，numpy 与 torch 的 fp32
`exp` 最后一位不一致，最大绝对误差 2.98e-08；指数的自变量本身是逐位相等的）。
指数表是常量、只在每条轨迹算一次，之后进的是 fp32 GEMM 再 round 到 bf16，
1 ulp 远在 bf16 的分辨率之下，所以不另做可视化确认；单测钉住"≤1 ulp 且
`t = 0` 那一行精确相等"。

### 真实权重上怎么判断"对齐"

两个后端不可能逐位相等（matmul 和 softmax 的归约顺序不同），所以单看
`mlx-vs-torch` 说明不了问题：它既可能是 bf16 舍入，也可能是移植 bug。
`check.compare_block` 用同一份权重多跑一条 **fp32 torch gold**，看的是
`mlx-vs-fp32` 有没有比 `torch-vs-fp32` 更差——没有更差，就说明 MLX 这条路
和参考实现自己的 bf16 路一样好，差的那部分是 bf16 的分辨率，不是移植。

只对一个 block 做：block 里已经包含了移植过的每一个算子，而 torch 参考是 CPU
eager 的，seq=4096 时一个 block 就要 26 s，50 个 block 没有意义。

### 文本编码器也要流式：50 GB 的塔，一次一层

H3 的条件是 Qwen3-VL（32B）前 `TEXT_LAYERS = 50` 层的 hidden states。按
bf16 算每层约 1 GB、50 层约 50 GB，是 18 GB 机器的三倍，`transformers` 的
`device_map='auto'` 只能退化成磁盘 offload。所以文本塔和 DiT trunk 用同一套
办法：**每层一个 slab，一次只驻留一层**。两者的区别在于频率——trunk 每个去噪
步都要重读一遍，文本塔整段只读一次，几百个 token 的算力可以忽略，全部时间都
是 I/O。

只移植文本这一路。T2VA 的 prompt 是纯文本，视觉塔、deepstack 合并、图像
mrope 分段都不会执行；三个 mrope 轴此时携带同一个 position，
`Qwen3VLTextRotaryEmbedding.recomposition_frequencies` 在每个轴上取到的是同一
个频率，于是旋转退化成普通的 rotate-half RoPE，正好是 `mx.fast.rope` 的语义
（单测里对着 transformers 钉住）。

层权重保持发布的名字和形状，不融合也不置换，所以 slab 是发布张量的逐字节拷贝。

## 代码位置与接口

| 文件 | 作用 |
|---|---|
| `miowtion/mlx/block.py` | 一个 trunk block 的前向；`BlockWeights`（发布 layout，含量化）、`BlockOptions`（head / row 分块） |
| `miowtion/mlx/sparse_attention.py` | Veda 块稀疏：`SparsePlan`、`HeadGroupPlan` / `LayerPlan`（每层两个头组、各自的排列）、gather 版 `block_sparse_attention`、稠密参考、代价模型 |
| `miowtion/mlx/veda_plan.py` | 把 `veda.mask.Selection` 转成 `SparsePlan` / `LayerPlan`；`layer_plan_from_scores` 从 `TilePlan` + `ClipTiling` + 打分直接得到一层的 plan（真实 Veda 掩码的唯一入口）；`clip_plans`（静态打分器，整个 trunk 一次建好）、`ActivationPlanner` / `clip_planners`（读激活的打分器，block 在自己的 head chunk 里调用）、`tile_geometry`、`uniform_tile_plan`（没有搜索过的方案表时的 baseline）、`random_scores`（替身打分器，**只能用来测速**） |
| `miowtion/mlx/veda_predictor.py` | Veda 打分器的 MLX 版：`gather_tiles` / `pool_tiles`（mean/max/min）、`tile_logits`、`score_tiles`（一层一个 head chunk 的 q/k → tile 分数）；投影可选，缺省即 mean-pooled QK |
| `miowtion/mlx/interop.py` | torch ↔ MLX 的逐位转换（numpy 没有 bf16，按 16 bit 原始位走） |
| `miowtion/mlx/slab.py` | slab 格式、`SlabReader`（pread 进预分配 buffer）、`BlockPrefetcher`、`convert_checkpoint` |
| `miowtion/mlx/offload.py` | 不转换的替代方案：直接读发布的 safetensors（`mx.load` 惰性加载 / 每个分片一个 mmap） |
| `miowtion/mlx/model.py` | 一次速度评估：`clip_inputs`（一条轨迹算一次）、`precompute_adaln`（一遍扫过 26 GB 的 AdaLN 投影，只留下表）、`velocity`（trunk 流式过一遍 block） |
| `miowtion/mlx/dit.py` | trunk 之外的部分：`NonTrunkWeights`（常驻）、RoPE / 时间步表、AdaLN 表预计算、token refiner、embed、final layer |
| `miowtion/mlx/pipeline.py` | 去噪循环：`Trajectory`（Euler 步）、`generate`（每步重新流式读一遍 trunk）、`schedule_timestep_sets`、`run_clip`（从发布目录到去噪后的行，整段只有一个入口）、`SparseRequest` / `build_plans`（一条轨迹建一次 Veda plan） |
| `miowtion/mlx/convert.py` | 读发布的 checkpoint（`ShardedSafetensors`：mmap safetensors → MLX array；`ReleaseReader` 另加 H3 的 config 与 schema），融合 q/k/v、写 trunk slab |
| `miowtion/mlx/text_encoder.py` | Qwen3-VL 文本塔：`TowerConfig`、`LayerWeights`、`layer_forward`（GQA + 因果注意力 + SwiGLU）、`encode`、`token_ids`、`write_tower_slabs` / `slab_layers`（按层流式） |
| `miowtion/infer/decode.py` | 两套发布布局的 VAE 解码：`Decoder`（发布包，diffusers 0.32.2）与 `DiffusersDecoder`（diffusers 移植版的 `vae/` + `audio_vae/`，需要 diffusers ≥ 0.36），接口一致 |
| `miowtion/mlx/check.py` | 真实权重上的数值对照：同一个发布 block（`compare_block`）或文本塔层（`compare_text_layer`）跑 MLX、torch bf16、torch fp32 三条路，给出 `BlockComparison` |
| `miowtion/mlx/bench.py` | 测量用：合成权重、进程与系统内存统计、各项 benchmark |
| `scripts/mlx_bench.py` | 命令行入口，每项测量单独一个进程，结果按 JSON 行输出 |
| `scripts/mlx_convert.py` | 把发布的 transformer 目录（`--transformer`）或文本编码器（`--text-encoder`）转成 per-block / per-layer slab（可指定区间与量化位宽） |
| `scripts/mlx_check.py` | 真实权重的对照，一行一个序列长度（`--out` 另存 JSON 行）；`--transformer` 对 trunk block，`--text-encoder` 对文本塔的一层 |
| `scripts/mlx_encode_text.py` | 用真实 prompt 跑文本塔，写出 DiT 条件用的 hidden states（`--slabs` 走 slab 快路径） |
| `scripts/mlx_generate.py` | 生成一个片段：prompt（或已编码的 hidden states）→ 去噪后的 video / audio latent 行（`.npy` + `meta.json`）；`--sparse-ratio` 打开 Veda 稀疏，`--sparse-scorer` 选打分器 |
| `scripts/mlx_decode.py` | 把上面那些行交给两个 VAE（torch/MPS），写出 mp4 |

### 两套发布命名

上游重新发布过一次 checkpoint：最早的 `MiniMaxH3DiTModel` 用 H3 原生名字、fused QKV 是
一个张量；现在 HuggingFace 上的是 diffusers 移植版 `MiniMaxH3Transformer3DModel`，block
在 `transformer_blocks.N.` 下，attention 拆成 `attn.to_q/to_k/to_v` 三个投影，embedding 和
输出层也改了名（`proj_in`、`context_embedder`、`norm_out`…）。名字映射集中在
`miowtion/h3/release.py`（只管名字，不碰张量），`ReleaseReader` 从权重名自动判断是哪一套，
`check_complete()` 在转换前把缺失的键一次性报出来——少一个层只会在深处炸 shape，或者更糟，
根本不炸。名字表已经用真实 checkpoint 的 index（638 个键）核对过：0 个缺失、0 个没被认领。

权重的 fused QKV 保持发布的 **per-head 交错行序**（`[h0: q k v, h1: q k v, ...]`），
这样 block 可以直接从 checkpoint 流式读入而不用做行置换，而且连续的一组 head 正好是
一段连续的行（head 分块要用）。torch 侧是 `[q_all; k_all; v_all]`，
`interop.block_weights_from_torch` 会置换回来。

## 实测

机器：Apple M3 Pro，18 GB 统一内存，12 核；mlx 0.32.2。合成权重，真实 shape
（hidden 5376，56 head × 128，attention inner 7168，ffn 14336，50 block，bf16）。

### SSD 原始吞吐（3.08 GB 顺序读）

| 方式 | 1 线程 | 4 线程 |
|---|---|---|
| `F_NOCACHE`，1 MiB 一次 | 3.10 GB/s | 4.65 GB/s |
| `F_NOCACHE`，16 MiB 一次 | 5.93 GB/s | 6.28 GB/s |
| `F_NOCACHE`，64 MiB 一次 | 6.30 GB/s | 6.43 GB/s |
| 冷启动、走 page cache | — | 6.29 GB/s |
| page cache 命中 | — | **21.8 GB/s** |

结论：**冷读上限约 6.4 GB/s**，要用 ≥16 MiB 的大块读、多个请求在途；1 MiB 的小块
读只有一半。缓存命中能到 21.8 GB/s，但对 50 个 block 的主干来说缓存必然全部落空。

### 单个 block 的加载延迟

| 方案 | 每 block 大小 | 平均延迟 | 有效带宽 | 用完后常驻 | 说明 |
|---|---|---|---|---|---|
| slab + pread（4 线程，NOCACHE） | 0.771 GB | **118 ms** | 6.50 GB/s | 0.82 GB | buffer 是复用的，不增长 |
| `mx.load`（冷） | 0.771 GB | 118 ms | 6.53 GB/s | 0.05 GB | 丢掉 dict 后内存确实释放 |
| `mx.load`（page cache 命中） | 0.771 GB | 45 ms | 17.0 GB/s | 0.05 GB | 只有单个 block 能命中 |
| mmap + 拷贝（冷） | 0.771 GB | **916 ms** | 0.84 GB/s | 0.05 GB | 缺页驱动，慢 7.8 倍 |
| mmap + 拷贝（页已在） | 0.771 GB | 171 ms | 4.50 GB/s | 0.05 GB | |
| slab，8 bit 量化 | 0.409 GB | 68 ms | 6.05 GB/s | 0.46 GB | 反量化另加 14 ms |
| slab，4 bit 量化 | 0.217 GB | 35 ms | 6.12 GB/s | 0.27 GB | 反量化另加 11 ms |

两个要点：

- **mmap 是最差的选择**。冷读只有 0.84 GB/s，跑不满 SSD 的八分之一：缺页是同步
  的、一次一页，没法把 SSD 的队列喂满。`mx.load` 和显式 pread 都能跑满。
- **`mx.load` 冷读和 slab 一样快，而且内存确实会释放**（丢掉返回的 dict 之后进程
  常驻只剩 0.05 GB）。它的代价是走 page cache（把别的数据挤出去），以及没法在后台
  线程上预取——所以默认还是 slab。

### 单个 block 的计算时间（M3 Pro GPU，bf16）

分块设置 `head_chunk=8, row_chunk=4096`：

| 序列长度 | 时间 | TFLOPS | attention 占 FLOP | attention 占时间 | MLP 占时间 | MLX 峰值 | 进程峰值 |
|---|---|---|---|---|---|---|---|
| 4 096 | 0.689 s | 5.28 | 13.2 % | 14.2 % | 48.8 % | 1.68 GB | 2.47 GB |
| 16 384 | 3.925 s | 5.18 | 37.9 % | 38.9 % | 35.0 % | 2.57 GB | 3.56 GB |
| 38 080 | 14.01 s | 5.06 | 58.6 % | 59.6 % | 22.7 % | 3.42 GB | 4.51 GB |

**分块很重要**：S=16384 不分块（`row_chunk=16384`，不分 head）时 MLX 峰值 5.65 GB、
进程峰值 6.21 GB，分块后降到 2.57 / 3.56 GB，而时间还略快一点（3.925 vs 4.009 s）。
在 18 GB 的机器上这是能不能跑长序列的区别。

**光分块还不够，必须同时开 `eval_chunks`**：MLX 的图是惰性的，不在每个 chunk 之后
`mx.eval`，所有 chunk 的中间量会一直活到最后一次 eval，分块等于白做。实测（稀疏、
`head_chunk=8, row_chunk=4096`）：

| 序列长度 | 关（进程峰值 / 时间） | 开 |
|---|---|---|
| 4 096 | 2.78 GB / 0.590 s | 2.53 GB / 0.583 s |
| 16 384 | 4.15 GB / 2.505 s | **3.06 GB** / 2.471 s |
| 38 912 | 6.44 GB / 6.617 s | **4.43 GB** / 6.458 s |

每个尺寸都是又省内存又快一点，所以 `BlockOptions.eval_chunks` 的默认值是 **True**，
命令行用 `--no-eval-chunks` 才关掉（只为做对照实验）。

量化**不会让计算变快**（M3 Pro 上 GEMM 已经是算力受限，不是带宽受限）：

| 序列长度 | bf16 | 8 bit（quantized matmul） | 4 bit |
|---|---|---|---|
| 4 096 | 0.689 s | 0.723 s | 0.724 s |
| 16 384 | 3.925 s | 4.096 s | 4.090 s |
| 38 080 | 14.01 s | 14.84 s | — |

量化的收益只在 **I/O 量和内存**，不在速度；量化矩阵乘反而慢 2–6 %。它确实省内存：
S=38080 下 8 bit 的进程峰值是 3.59 GB，bf16 是 4.51 GB。

### Veda 块稀疏（gather 版，k_block=128，密度约 10 %）

先单独测 attention（56 head × 128，bf16）。`q_block` 越大 gather 越少、越快：

| 序列长度 | 稠密 | q_block=512 | 1024 | 2048 | 4096 | 最好的加速比 |
|---|---|---|---|---|---|---|
| 8 192 | 365 ms | 52.4 ms | 43.2 ms | 38.8 ms | **37.0 ms** | 9.88×（上界 10.7×） |
| 16 384 | 1 494 ms | 237.9 ms | 193.2 ms | 171.7 ms | **163.4 ms** | 9.14×（上界 9.85×） |
| 38 912 | 8 580 ms | — | — | **944 ms** | — | 9.09×（上界 10.1×） |

**拿到了密度上界的 90–93 %**，说明 gather 的开销基本可以忽略。gather 缓冲随
`q_block` 反比缩小（S=16384、密度 10 % 时 512→1.53 GB，4096→0.19 GB），所以大
`q_block` 是双赢。`head_chunk` 用来压 gather 的峰值内存，而且不花钱：S=38912 下
`head_chunk=4` 是 922 ms / 峰值 4.38 GB，不分块是 944 ms / 6.42 GB。

放回整个 block（`head_chunk=8, row_chunk=4096`，`q_block=2048`）：

| 序列长度 | 稠密 block | 稀疏 block | block 加速 | 其中 attention（稠密 → 稀疏） |
|---|---|---|---|---|
| 4 096 | 0.673 s | 0.590 s | 1.14× | 96.7 → 12.0 ms（8.1×） |
| 16 384 | 3.858 s | 2.505 s | 1.54× | 1 509 → 173 ms（8.7×） |
| 38 912 | 14.52 s | **6.62 s** | **2.19×** | 8 950 → 947 ms（9.45×） |

稀疏之后 attention 只剩 14 % 的时间，**剩下的 86 % 是 GEMM**（S=38912 下
qkv 1.55 s + out 0.55 s + MLP 3.13 s ≈ 5.67 s）。而量化过的 GEMM 并不更快（见上），
所以 6.62 s 就是这台机器在这个序列长度上的地板；再想快只能降分辨率或换机器。

### GEMM 已经到顶，NPU 也帮不上忙

稀疏之后 86 % 的时间是 GEMM，所以单独测了 MLX 的裸 GEMM：**所有形状都是平的
5.88 TFLOPS**（block 的四个 linear、以及 2048/4096/8192 的方阵），block 前向里的
GEMM 时间和裸 GEMM 相差不到 2 %。换 dtype 也没用：

| 方阵 | fp32 | fp16 | bf16 |
|---|---|---|---|
| 4 096 | 5.18 | 5.91 | 5.93 |
| 8 192 | 3.57 | 5.96 | 5.96 |

18 核的 M3 Pro GPU 理论 FMA 峰值约 6.45 TFLOPS，5.96 已经是 **92 %**；而且
fp16/bf16 与 fp32 同速——这代 Apple GPU 没有 NVIDIA tensor core 那种半精度双倍吞吐
（GPU 内的矩阵乘单元要到 M5 的 Neural Accelerators 才有）。所以 `mx.compile`、换
tile 形状之类的优化没有空间，**6.62 s / block 就是这台机器的地板**。

Neural Engine 同样不是出路。用 CoreML 把同一个 fc1 形状的 fp16 GEMM 分别跑在三个
后端上：

| 后端 | 时间 | 吞吐 | 编译 + 加载 |
|---|---|---|---|
| ANE（`CPU_AND_NE`） | 334.8 ms | 3.30 TFLOPS | 33.8 s |
| GPU（`CPU_AND_GPU`） | 188.3 ms | 5.87 TFLOPS | 3.9 s |
| CPU | 325.4 ms | 3.40 TFLOPS | 4.4 s |

**ANE 比 GPU 慢 1.8 倍**（标称的 18 TOPS 是 int8 峰值，不是 fp16 稠密 GEMM）。即使
它更快也用不了：MLX 没有 ANE 后端，唯一的入口 CoreML 要求权重编译进模型，而本方案
的前提正是按 block 从 NVMe 换权重——上表里一个 4 次 matmul 的模型编译加载就要
33.8 s，400 次换权重的代价远超任何收益。ANE 还只有 fp16，会破坏与 torch 的数值对齐。

### 接上真实的 Veda 选择

前面的数字用的是"每个 query tile 预算相同、tile 是均匀 128×128 网格"的理想 plan。
真实的 Veda 选择有三处不均匀，`miowtion/mlx/veda_plan.py` 把它们各映射到
`SparsePlan` 的一个字段，而不是去近似掉：

| Veda 的事实 | plan 里的对应 | 为什么不能忽略 |
|---|---|---|
| 预算按 Bresenham 逐 query tile ±1，且分摊在 reference / target 两个列块上 | `keep`：把短的行补齐到统一宽度后掩掉 | 补位不能靠重复已选的 tile，重复的 key 会在 softmax 里算两次 |
| 全局（文本 / 音频）的行和列是稠密的 | 列并进每行的预算；行是排列后序列的末尾，走 `dense_rows` 单独一次稠密调用 | 全局 query 行看所有列，无法和视频行共用同一个预算 |
| tile 只填了一部分（partial tile） | `key_valid`（来自 `layout.slot_valid`） | 掩码是逐行的，不是逐 tile 的 |

另外 Veda 的 top-k 是**逐头**做的，所以 `index` 支持 `[heads, n_q, budget]`：gather
时把 (head, tile) 压成一个轴，仍然只是一次 `mx.take`。把 `index` 按 tile 号排序之后，
gather 出来的 key 顺序和稠密路径一致，结果**逐位相等**（单测钉死）。

代价是 `q_block` 必须等于 Veda 的 tile（128），而前面测过 `q_block` 越小 gather 越
贵。S=38912、密度 10 %、整个 block：

| 配置 | 每 block | 其中 attention | 进程峰值 |
|---|---|---|---|
| 稠密 | 14.52 s | 8.95 s | — |
| 理想 plan（`q_block=2048`，`head_chunk=8`） | 6.46 s | 0.95 s | 4.43 GB |
| Veda 原生（`q_block=128`，`head_chunk=4`） | 8.01 s | 2.70 s | 8.78 GB |
| Veda 原生（`q_block=128`，`head_chunk=2`，`row_chunk=4096`） | **7.70 s** | 2.43 s | **5.99 GB** |

也就是说真实 plan 比理想 plan 慢 19 %、仍然比稠密快 1.89 倍。`q_block=128` 时
gather 量是 `q_block=2048` 的 16 倍（S=38912 下一遍 33 GB），所以 `head_chunk`
必须调小到 2：head_chunk=4 会多花 2.8 GB 峰值，还因为内存压力更慢。

### 每层两个头组：排列也是 plan 的一部分

`TilePlan` 允许一层用两种 tile 形状，也就是两个**不同的排列**。MLX 侧因此不是"一个
block 一个 plan"，而是 `LayerPlan`：每个头组带自己的 `gather` / `scatter` 和
`SparsePlan`，block 本身始终拿 packed 顺序的 q/k/v，进出 attention 时由头组自己排列。
这样 block 不需要知道 Veda 的几何，头组也可以各用各的 tile 形状。

- `gather`：slot → packed 行；padding slot 指向第 0 行，靠 `key_valid` 掩掉，取值
  无所谓。
- `scatter`：packed 行 → slot；输出末尾补一行零，没被任何 tile 覆盖的行指向它
  （`build_tile_layout` 下不会发生，发生就报错）。反排列用 gather 而不是原地
  scatter，路径保持函数式，也避开 MLX 的 in-place 写。
- `head_chunk` 落在头组边界内时不需要额外挑头；跨界时 `layer_attention` 会按组挑出
  各自的头，算完再按输入顺序拼回去。

排列的代价（S=38912，密度 10 %，`q_block=128`，`head_chunk=2`，`row_chunk=4096`）：

| 配置 | 每 block | 其中 attention | MLX 峰值 |
|---|---|---|---|
| 单 plan，不排列 | 7.74 s | 2.51 s | 4.84 GB |
| `LayerPlan`，两个头组各一个排列 | 7.97 s | 2.61 s | 4.94 GB |

即 **+3.0 %**（+0.23 s）。微基准对得上：一次 `[2, 38912, 128]` bf16 的行 gather 是
2.0 ms，每个头组每 block 要做 4 次（q / k / v 和反排列），56 个头 28 个 chunk 就是
0.22 s。这是随机排列的上界，Veda 的排列在 tile 内是连续的，只会更快。

### Veda 接进去噪循环：plan 按层传，谁建取决于打分器

plan 不放进 `BlockOptions` 常量、而是按层传给 `model.velocity`：
`BlockOptions.sparse` 只能表达"每层同一个选择"，而真实的 Veda 每层选的 tile
不一样。`velocity` 里按层 `dataclasses.replace(options, sparse=plans[index])`，
`plans[i] is None` 的层走稠密（`VedaConfig.dense_layers`）。

plan 什么时候建，取决于打分器读不读激活：

- **不读**（`random_scores`，或任何静态回调）：选择只取决于几何和打分，50 层的
  `LayerPlan` 在第一步之前一次建好（`pipeline.build_plans`），8 步共用。
- **读**（默认的 `pooled`）：一层的选择要等到这一步算出这一层的 q/k 才存在，
  所以 `build_plans` 返回的是 50 个 `ActivationPlanner`，block 在自己的 head
  chunk 循环里调用它（`BlockOptions.sparse` 接受一个可调用对象）。

`veda_plan.random_scores` 产生的选择在形状、预算、逐头分布上都和真的一样，
kernel 干的活一模一样，**但保留哪些 tile 是随机的**。所以它只能用来测速，用它生成
的片段没有任何质量意义，文档里凡是用它测出来的数字都必须标明。

### 打分器移到 MLX：每层用自己的 q/k 选 tile

`miowtion/mlx/veda_predictor.py` 是 `veda/predictor.py` 的 MLX 版：每个 tile 按
真实行做 [mean | max | min] 池化，逐头的残差投影把 3D 特征映射回 D 维
（`q_hat = pool_q @ P_q[h] + mean_q`），logits 是 `q_hat · k_hat / sqrt(D)`。

**分工按代价切**：池化要读整块 q 和 k（一个 head chunk 约 26 MB），留在 MLX；
top-k、Bresenham 预算切分和 plan 构造只在 tile 网格上动，一层的 logits 对 12k
token 的片子只有约 2 MB，继续用 torch 侧已经验证过的 `select_video_blocks`
（`veda_plan.ActivationPlanner`）。这样既不用把 mask.py 重写一遍，往返的也不是
激活而是分数。

**没有训练好的 predictor checkpoint**，所以 `proj_q` / `proj_k` 传的是 `None`，
`embed` 退化成池化的 mean——这正是参考实现在初始化时算的东西
（`P ~ N(0, 1e-4)`），也就是 Veda 用来起步的 mean-pooled QK 块分数。它是**跟内容
走的**，和随机打分器有本质区别，但它**不是训练过的选择**：真接上 checkpoint 只需
给 `ActivationPlanner(projections=...)` 传一个"层号 + 头号 → (proj_q, proj_k)"的
回调。

**对齐**：max / min 池化与 torch 参考**逐位相等**，gather 也是；mean、投影和
logits 的矩阵乘是规约，两边累加顺序不同（实测 1e-6 量级）。真正要求相等的是
**选出来的 tile**，单测里用真实几何比较 `index` / `keep` 逐位相等
（`tests/unit/test_mlx_veda_predictor.py`）。

**代价**（真实权重，S=12608，50 层 × 7 个 head chunk = 350 次打分 + 建 plan）：
一步 107.2 s，比预先建好 plan 的 103.9 s 多 3.3 s（3.2 %），峰值 4.49 GB。

没有搜索过的方案表（`plans/` 为空）时 `uniform_tile_plan` 退回到该几何 padding
最少的 tile 形状——搜索本身也是从这个形状出发的，作为测速 baseline 是诚实的。

### 18 GB 上的序列长度上限

开了稀疏和 `eval_chunks`（`head_chunk=8, row_chunk=4096, q_block=2048`，
密度 10 %）之后：

| 序列长度 | 每 block | 进程峰值 | 400 次前向外推 |
|---|---|---|---|
| 38 912 | 6.46 s | 4.43 GB | 43 min |
| 49 152 | 8.39 s | 4.50 GB | 56 min |
| 65 536 | 12.81 s | 5.55 GB | 85 min |
| 98 304 | 19.86 s | **8.03 GB** | 132 min |

98 304 token 大致是 16:9 / 14.4 s 的 clip，也就是说**内存不再是这台机器的限制，
时间才是**：峰值 8 GB 离 18 GB 还有余量，但一次生成要两个多小时。

### 端到端流式（预取 depth=1，2 个 slot）

每个 block 的读完全藏在上一个 block 的计算后面：

| 序列长度 | 精度 | slab/block | 每 block 墙钟 | 其中计算 | 其中后台读 | 实际等待 | 进程峰值 |
|---|---|---|---|---|---|---|---|
| 4 096 | bf16 | 0.771 GB | 694 ms | 686 ms | 119 ms | **7.3 ms** | 2.70 GB |
| 4 096 | 8 bit | 0.409 GB | 747 ms | 743 ms | 64 ms | 4.0 ms | 1.98 GB |
| 4 096 | 4 bit | 0.217 GB | 733 ms | 731 ms | 35 ms | 2.2 ms | 1.59 GB |
| 16 384 | bf16 | 0.771 GB | 3 940 ms | 3 925 ms | 118 ms | 14.8 ms | 4.02 GB |

开了稀疏（密度 10 %）之后计算变快，读 / 算的比值变差，但仍然完全藏得住：

| 序列长度 | 每 block 墙钟 | 其中计算 | 其中后台读 | 实际等待（首块之后） |
|---|---|---|---|---|
| 4 096 | 608 ms | 590 ms | 118 ms | 4–18 µs |
| 16 384 | 2 548 ms | 2 505 ms | 118 ms | 7–24 µs |

「实际等待」是消费端真正阻塞在 I/O 上的时间，基本只剩第一个 block 的冷启动。
**即使在最短的序列、最大的 bf16 权重、并且开了稀疏的情况下，读 / 算的比值也只有
118/590 ≈ 0.20**，离 I/O 受限还有 5 倍的余量。

### 外推：一次完整生成（50 block × 8 步 = 400 次 block 前向）

| 序列长度 | bf16 稠密 | bf16 + 10 % 块稀疏（实测外推） | 加速 |
|---|---|---|---|
| 4 096 | 4.6 min | 4.1 min | 1.14× |
| 16 384 | 26 min | 17.0 min | 1.54× |
| 38 912 | 97 min | **44 min** | 2.19× |

一次生成总共要读 400 × 0.771 GB ≈ **308 GB**，按 6.5 GB/s 是 47 s，全部藏在计算
后面。稀疏的收益随序列长度增长（attention 占比从 14 % 涨到 59 %），38 912 上能砍掉
一半以上的时间。

**所有配置都是计算受限，没有一个是 I/O 受限的。** 这是本研究最主要的结论：在这台
机器上，NVMe offloading 基本是免费的，瓶颈是 M3 Pro 那 ~5 TFLOPS 的 bf16 算力。

### 磁盘与内存预算（50 个 block）

| 精度 | 磁盘占用 | 2 个 slot 的权重内存 | S=4096 进程峰值 |
|---|---|---|---|
| bf16 | 38.5 GB | 1.54 GB | 2.70 GB |
| 8 bit | 20.5 GB | 0.82 GB | 1.98 GB |
| 4 bit | 10.8 GB | 0.43 GB | 1.59 GB |

### 真实权重：一个 block 的三方对照

`scripts/mlx_check.py --transformer <发布目录> --block 0 --seq-len 512 4096`
（真实 T2VA transformer，hidden 5376、56 头、head_dim 128，block 0.718 GiB bf16）：

| block | seq | mlx vs torch(bf16) | mlx vs fp32 | torch(bf16) vs fp32 | torch | mlx |
|---|---|---|---|---|---|---|
| 0 | 512 | 3.78e-03 | 8.920e-03 | 8.904e-03 | 1.82 s | 0.07 s |
| 0 | 4096 | 2.68e-03 | 6.074e-03 | 6.076e-03 | 25.71 s | 0.67 s |
| 1 | 512 | 3.36e-03 | 1.237e-02 | 1.237e-02 | 1.83 s | 0.07 s |

即 **MLX 离 fp32 的距离和 torch 自己的 bf16 路一模一样**（差在第三位有效数字
以内，两边互有胜负），剩下的 2.7e-03 ~ 3.8e-03 就是 bf16 本身的分辨率
（bf16 的 eps 是 7.8e-03）。torch 那一列是 CPU eager，不是性能对照。

同一个 block（seq 4096）量化之后（`--bits`，group 64，torch 侧仍是发布权重）：

| 权重 | mlx vs fp32 | 相对 bf16 | 单 block 计算 |
|---|---|---|---|
| bf16 | 6.074e-03 | 1.0× | 0.67 s |
| 8 bit | 1.161e-02 | 1.9× | 0.79 s |
| 4 bit | 1.259e-01 | 20.7× | 1.72 s |

**4 bit 不能用**：单层就 12.6 %，50 层还要叠。8 bit 只在磁盘放不下 bf16 时用，
而且它连计算都更慢（+18 %），省的只是 I/O 和内存。

### VAE 解码：先留在 torch/MPS（已被下一节取代）

视频 VAE 是 2.4B 的 ViT3D，音频 VAE 很小，两个都只在整段结束时跑一次，
所以没有理由在 DiT 定下来之前先移植它们（视频这一侧后来还是移植了，见
「视频 VAE 移植到 MLX」；音频 VAE 仍在 torch）。MPS 上（`--device mps`，视频
bf16、音频 fp32，1 秒 16:9 片段，39 帧）：

| 阶段 | 1344×768 | 224×128（短边 128） |
|---|---|---|
| 两个 VAE 载入 | 13.7 s | 13.5 s |
| 视频解码（39 帧） | 167.2 s | 13.1 s |
| 音频解码（2×52000 样本） | 1.7 s | 3.1 s |

也就是说解码在整条链路里只占 8 步去噪（1 050 s）的 16 %，画布缩到 1/6
时间掉到 1/13——解码是纯算力，没有流式开销。

### 去噪一步的时间分解（真实权重，S=12608）

先确认算力有没有跑满：`scripts/mlx_bench.py compute --seq-len 12608`（真实形状、
纯计算、不读盘）一个 block 2.690 s、**5.31 TFLOP/s**，是裸 GEMM 上限
（5.88 TFLOP/s）的 90 %。一个 block 内部：

| 阶段 | 时间 | 占比 |
|---|---|---|
| MLP | 1.018 s | 37.8 % |
| attention | 0.900 s | 33.4 % |
| QKV 投影 | 0.508 s | 18.9 % |
| out 投影 | 0.172 s | 6.4 % |
| qk-norm + RoPE | 0.058 s | 2.1 % |
| norm / modulate / 残差 | 0.060 s | 2.2 % |

再确认 I/O 有没有拖后腿：`stream --dir <真实 slab> --blocks 8 --seq-len 12608`
每个 block 2.704 s，其中读 0.118 s（0.77 GB / block，6.5 GB/s），**等待 0.000 s**
（只有第一个 block 等 0.12 s）。预取完全盖住了 I/O，去噪是纯算力。

所以去噪一步的下限是 50 × 2.704 = 135.2 s，实测一步端到端（`mlx_generate.py
--steps 1`）**136.1 s**，多出来的 0.7 % 是非 trunk 的头和 Euler 步。

| 一步 | 一步 | 8 步 | 峰值 |
|---|---|---|---|
| 稠密 | 136.1 s | 18 分 09 秒 | 4.39 GB |
| Veda 稀疏，随机打分器，密度 0.136 | 103.9 s | 13 分 51 秒 | 4.71 GB |
| Veda 稀疏，mean-pooled QK 打分器（默认） | 107.2 s | 14 分 18 秒 | 4.49 GB |

前两行的每 block 时间是 2.704 s（稠密）和 2.078 s（稀疏）。随机打分器那行
（`--sparse-scorer random`，真实的 tile 方案与预算、每头 top-k、每层两个头组）
只说明速度，不说明画面；第三行是默认打分器，每层用自己的 q/k 打分、每步重新选，
多出来的 3.3 s 就是打分加建 plan 的全部代价。attention 从 0.900 降到 0.241 s
（3.7×，不是 9–10×，因为 Veda 的 `q_block=128` 让 gather 量涨了 16 倍），整块
因此只快 1.31 倍。

### 视频 VAE 移植到 MLX

发布的解码是**分块**的：`use_tiling=True`，像素空间 256×256 一块、至少 64 px
重叠（按 16 的倍数加宽），线性交叉淡化后拼回；时间上按 `clip_length=17` /
`token_drop=3` 切成 5 个 latent 帧一段、重叠 2 段。这套几何是权重训练时就定下
的，**不是调参旋钮**，移植时逐位复刻（`split_tiles` 完全相同，`blend` /
`stitch_tiles` 逐位相等）。1344×768、39 帧一段的片子因此是 4×7 个 tile × 2 个
时间块 = 56 次 1797 token 的前向，约 486 TFLOP。

`miowtion/mlx/video_vae.py` 是这个 2.42B ViT3D 的 MLX 实现（fused QKV、fp32
RMSNorm、三轴 RoPE、SwiGLU、逐 block `mx.eval`）。同一段 latent、同一台机器：

| 后端 | 视频解码（39 帧 1344×768） | 峰值 |
|---|---|---|
| torch / MPS（发布实现） | 155.4 s | — |
| MLX，`--tile-batch 2`（默认） | 104.9 s | 5.76 GB |

`tile_batch` 把 N 个 tile 拼成一次前向。一个 tile 本身已经是 1797 行，GEMM 早就
跑满了，所以它几乎只是个内存旋钮（真实权重，合成 latent）：

| tile_batch | s/tile | GEMM | 峰值 |
|---|---|---|---|
| 1 | 1.91 | 4.56 TFLOP/s | 4.78 GB |
| 2 | 1.86 | 4.68 TFLOP/s | 5.03 GB |
| 4 | 1.84 | 4.72 TFLOP/s | 5.55 GB |
| 8 | 1.83 | 4.75 TFLOP/s | 6.11 GB |

权重常驻 4.51 GB（bf16），载入 15 s。**这张表只在机器不紧张时成立**：同样的
batch 8，在系统只剩 18 % 空闲内存时是 6.1 s/tile（0.44 GB 的差别换来 3.3 倍的
时间），载入也从 15 s 变成 38 s。见踩坑记录。

### 真实权重：文本塔一层的三方对照

`scripts/mlx_check.py --text-encoder <发布目录> --block N --seq-len S`
（真实 Qwen3-VL 文本塔，hidden 5120、64 q 头 / 8 kv 头、head_dim 128）：

| layer | seq | mlx vs torch(bf16) | mlx vs fp32 | torch(bf16) vs fp32 |
|---|---|---|---|---|
| 0 | 64 | 1.569e-03 | 2.461e-03 | 2.489e-03 |
| 7 | 256 | 5.155e-03 | 6.373e-03 | 7.750e-03 |
| 30 | 256 | 8.611e-03 | 1.023e-02 | 1.386e-02 |

三层都满足 `mlx_vs_fp32 <= torch_vs_fp32`，而且层越深 MLX 反而更靠近 fp32：
`mx.fast.rms_norm` 和 `mx.fast.scaled_dot_product_attention` 内部用 fp32 累加，
transformers 的 eager 路径是纯 bf16。这同时钉住了 mrope → 普通 rope 的化简
（rope base 写错的话，单层就是 5e-3 量级，见单测里的注释）。

### 真实权重：一条完整轨迹（33B，8 步）

第一次用发布的 33B T2VA transformer 跑完整去噪循环，1 秒 16:9 片段
（`latent_t 12`，打包后 seq 12 352、有效 12 290），turbo 8 步，文本 embedding
用随机张量占位（文本编码器还没下下来），trunk 每步重新流式读一遍：

| 阶段 | 时间 | 备注 |
|---|---|---|
| trunk 转 slab（50 block，bf16，36 GB） | 91 s | `scripts/mlx_convert.py`，一次性 |
| 非 trunk 权重 + token refiner | 一次 | 常驻，不随步数增长 |
| AdaLN 表预计算（8 组时间步，扫过 26 GB 投影） | 43.5 s | 结果只有 138.4 MB，整段只做一次 |
| 每步（50 block） | **131.3 s**（最大 134.0 s） | ≈ 2.6 s/block |
| 8 步合计 | **1 050 s ≈ 17.5 min** | MLX 峰值 4.51 GB |

同一步直接从发布的 safetensors 读（不转 slab）要 **213.9 s**：mmap 缺页驱动
的有效带宽只有 0.17 GB/s，转成 slab 之后读不再是瓶颈，131 s 全花在计算上
（和合成权重的外推一致：seq 16 384 时 26 min / 8 步，这里 seq 12 352 更短）。

**这条轨迹只说明链路跑得通、时间和内存是多少，不说明生成质量**：文本 embedding
是随机的，也还没有 VAE 解码，所以 AGENTS.md 1.5 要求的人工可视化确认尚未进行。

## 18 GB 机器上的推荐配置

- **bf16 slab + `depth=1` 预取（2 个 slot）+ `head_chunk=8, row_chunk=4096`。**
  量化不会更快，所以只要磁盘放得下 38.5 GB 就用 bf16，避免量化误差。
- 磁盘紧张、或者序列长到内存不够时再换 8 bit（I/O 减半、内存省 0.7 GB、慢 5 %）；
  4 bit 留给最极端的情况。
- 不要用 mmap 方案。不想做 checkpoint 转换时用 `mx.load`（一样快），代价是走
  page cache 且不能后台预取。
- 读参数保持 16 MiB 一块、4 线程；调小会直接掉带宽。
- 内存主要被激活占，不是被权重占：务必开分块，并且**保持 `eval_chunks=True`**
  （默认值）——不在 chunk 之间 eval 的话分块等于白做。开了之后 98 304 token 的峰值
  只有 8 GB，18 GB 机器的限制已经从内存变成时间。
- **块稀疏用 gather 版（`BlockOptions.sparse`），`q_block` 取 2048 或更大**；小
  `q_block` 既慢又吃内存。gather 缓冲配合 `head_chunk` 一起压。

## 测试

`tests/unit/test_mlx_block.py`（16 个，缺 mlx 时自动 skip）：

- torch ↔ MLX 转换、RoPE、SwiGLU 逐位相等（`torch.equal`）。
- 整个 block 对 torch 参考实现：fp32 相对 L2 < 1e-5；bf16 < 5e-3，并且额外要求
  MLX 的 bf16 结果离 fp32 参考不比 torch 的 bf16 结果更远。
- 分块（head / row）不改变结果；padding 行不影响真实行的输出。
- 8 / 4 bit 量化的误差量级；`BlockWeights.from_tensors` 对缺失和多余的 tensor 报错。
- 满预算的稀疏 plan 与稠密一致；部分预算确实改变结果；tile 不整除 `used` 或 plan 的
  query tile 数不对时报错。
- 把同一个 plan 拆成两个头组（恒等排列）与单 plan 路径**逐位相等**；`LayerPlan` 的
  头数、`scatter` 长度不对时报错。
- per-head 的选择在 head chunk 下不变（chunk=2 逐位相等，chunk=1 差 1 ulp）。
- `BlockOptions.sparse` 传 planner：每个 head chunk 各调用一次、拿到的正是这个
  chunk 的 q/k 与起始头号，结果与同样的静态 `LayerPlan` **逐位相等**；planner 返回
  的头数不对时报错。

`tests/unit/test_mlx_veda_plan.py`（8 个）：

- plan 展开出来的逐行掩码与 `veda.mask.dense_block_mask`（再按 `slot_valid` 掩掉
  padding 行）**逐位相等**（`torch.equal`）。
- 用该 plan 跑 gather 版，与同一掩码下的稠密 attention 逐位相等。
- 满预算时各头选择相同，head 轴会被收起来（一次 gather 服务整组头）。
- 选择没有覆盖全部视频 query tile 时报错。
- 一层两种 tile 形状：两个头组各自排列，每组的结果与"该组掩码下的稠密 attention
  再反排列"**逐位相等**，头也回到输入顺序；头组不构成划分时报错。
- `layer_plan_from_scores` 走真实几何（16:9、`h3.layout.pack`、`ClipTiling`、
  `TilePlan` 两种形状）：头组、排列长度、密度都对，跑出来的 attention 同样逐位相等。

`tests/unit/test_mlx_veda_predictor.py`（5 个）：

- `gather_tiles` 与 `veda.tiling.gather_tiles` **逐位相等**（纯数据搬运）。
- `pool_tiles`：max / min 与 torch 参考**逐位相等**，mean 相对 1e-6；空 tile 恰好
  是 0（没有漏掉 -inf）。
- `score_tiles` 与 `veda.predictor.LayerPredictor`：带投影、不带投影两条路都对得上
  （1e-5）。
- `ActivationPlanner` 在真实几何上选出的 `index` / `keep` 与"torch 侧同样的池化
  分数 → `select_video_blocks`"**逐位相等**。
- 按 head chunk 建 plan：每个头拿到的选择与整层一次建出来的**逐位相同**，返回的
  plan 只覆盖这个 chunk 的头且从 0 编号。

`tests/unit/test_mlx_convert.py`（7 个）+ `tests/unit/test_h3_release.py`（6 个）：

- `fuse_qkv` 的行序被测试钉住（head h 占 q、k、v 各一段），shape 不对报错。
- 合成的 diffusers 版 checkpoint：schema 自动识别、config 解析、`check_complete`
  少张量时报错；trunk / AdaLN / 非 trunk / refiner 张量与写入的值**逐位相等**
  （`torch.equal`），fp32 的 patch 投影保持 fp32。
- 转换成 slab 再读回来与直接读 checkpoint **逐位相等**（`mx.array_equal`）。
- 名字表对着 `third_party/MiniMax-H3` 里锁定的旧版 index 校验：除了重新计算的
  `rope.inv_freq`，每个张量都被恰好一个名字认领。

`tests/unit/test_mlx_check.py`（4 个）：

- 发布 layout → torch `Block` → 发布 layout 的往返**逐位相等**
  （`mx.array_equal`，含 AdaLN 投影）。置换写反了在"只走一个方向"的测试里
  是看不出来的，所以这里把回来的方向也钉住。
- 合成 release 上 `compare_block` 跑通，且 `as_good_as_torch` 成立。

`tests/unit/test_mlx_pipeline.py` 里另有两个 `run_clip` 的测试：合成发布目录
转 slab 之后整条链路（转 slab → 非 trunk 权重 → AdaLN 表 → 两步去噪）跑通、
形状与几何一致、输出有限；slab 目录只转了一半时在循环开始前就报
`FileNotFoundError`（真实规模下"跑了半小时才发现缺一块"是不可接受的）。为了让
这两个测试在 CPU 上是秒级的，`ClipRequest.short_edge` 可以把画布压到 64 px——
这个旋钮只给冒烟测试用。

同一个文件里还有三个 Veda 的测试：`build_plans` 在真实几何上给出的密度落在预算
附近、每层 plan 覆盖全部头；`VedaConfig.dense_layers` 里的层确实是 `None`（走稠密）；
以及**满预算的稀疏 `run_clip` 与稠密 `run_clip` 逐位相等**（`mx.array_equal`）——
64 px 画布下整段只有一个 tile，稀疏路径相对稠密路径就只多了排列、gather 和反排列
这三步纯数据搬运，按 AGENTS.md 1.5 必须逐位一致，这条把整条链路的排列方向钉死。

`tests/unit/test_mlx_text_encoder.py`（10 个）：

- 单层、整塔、以及"只跑前缀层"（H3 在 `TEXT_LAYERS` 处停下）对
  transformers 自己的 `Qwen3VLTextDecoderLayer` / `Qwen3VLTextModel`：单层
  相对 L2 < 1e-3，整塔 < 1e-2。阈值特意卡在 1e-3：rope base 写错时是 5e-3，
  写对是 3e-4，再松就分不出来了。
- 层权重写成 slab 再读回来**逐位相等**（`mx.array_equal`）；合成发布目录里
  `layer_tensors` / `embed_tokens` 读出来的也与写进去的逐位相等。
- 合成发布目录上 `compare_text_layer` 跑通且 `as_good_as_torch` 成立。
- `LayerWeights.from_tensors` 缺 tensor 报错；token 越界报错；`TowerConfig`
  读发布 config（顺带钉住"每层约 1 GB"这个流式的前提），非 default 的
  rope type 直接报错而不是静默当成 default。

`tests/unit/test_mlx_sparse_attention.py`（13 个）：

- gather 版与"等价块掩码下的稠密 attention"**逐位相等**（`mx.array_equal`）；
  `head_chunk` 不改变结果（同样逐位）。
- 满预算等于稠密；没选中的 key tile 改掉 V 也不影响输出。
- `block_mask_from_index` 标记的正是选中的 tile；非法 tile / 预算 / `head_chunk`
  一律报错；FLOP 与 gather 字节数的代价模型按密度和 `q_block` 正确缩放。
- `keep` 掩掉的补位槽完全惰性：把补位槽指到别的 tile，输出一个 bit 都不变；
  `key_valid` 掩掉的 padding 行同理（改掉它们的 V 不影响输出）。
- 逐头 `index` 与逐头稠密参考逐位相等，且 `head_chunk` 不会把头和它的选择错位。
- `dense_rows` 的尾部行等于普通稠密 attention；某个 query tile 一行 key 都不留时
  报错（否则 softmax 是 NaN）。

`tests/unit/test_mlx_slab.py`（6 个）：

- slab 写—读**逐位相等**（`mx.array_equal`），覆盖 bf16 / 8 bit / 4 bit、
  cached / uncached、不同的线程数和读块大小。
- header 的对齐与重新解析；混合 layout、非法参数、坏 magic 都要报错。
- `BlockPrefetcher` 按顺序给出每一个 block，内容与单独读一致。

## 踩坑记录

- **mmap 读 checkpoint 慢 7.8 倍**。现象：mmap + 拷贝冷读只有 0.84 GB/s，而同一个
  文件用 pread 能到 6.5 GB/s。原因：缺页是同步的、一次一页，SSD 队列一直是空的，
  跑不出并发。对策：用显式的并发 `pread`（或 `mx.load`），不要依赖 mmap 做大块
  顺序读。
- **小块读掉带宽**。1 MiB 一次的读只有 3.1 GB/s（单线程），16 MiB 才到 5.9 GB/s。
  对策：`DEFAULT_PIECE_BYTES = 16 MiB`，多线程并发。
- **不分块的长序列会吃掉一半内存**。S=16384 不分块时进程峰值 6.21 GB，分块后
  3.56 GB，而且分块还更快一点。对策：默认就开 `head_chunk` / `row_chunk`。
- **量化不提速**。本以为 4 bit 能靠省带宽提速，实测反而慢 2–4 %：M3 Pro 上这些
  GEMM 是算力受限的。对策：量化只当成省磁盘 / 省内存的手段。
- **MLX 的融合 attention 不会跳过全被掩掉的块**。现象：给 90 % 稀疏的块掩码，时间
  是稠密的 0.97 倍，等于白给。原因：掩码在 Q@K.T 的 tile 矩阵乘之后才施加，只有内置
  的 `"causal"` 会缩短 key 循环上界。对策：按固定预算把选中的 key tile gather 成规整
  批量问题，走一次稠密 kernel，拿到 9–10 倍（见 `sparse_attention.py`）。
- **稀疏的 `q_block` 不能太小**。现象：S=16384、密度 10 % 时 `q_block=512` 比
  `q_block=4096` 慢 46 %、峰值内存多 2.7 GB。原因：gather 的量正比于 `S/q_block`。
  对策：`q_block` 取 2048 以上，并用 `head_chunk` 压峰值。
- **分了块但峰值内存没下来**。现象：S=38912 开了 `head_chunk=8, row_chunk=4096`，
  峰值仍然 6.44 GB；S=49152 到 9.38 GB。排查时先怀疑 QKV 没分块，读了
  `block_forward` 才排除。原因：MLX 的图是惰性的，不在每个 chunk 之后 `mx.eval`，
  所有 chunk 的中间量会一直活到最后一次 eval，分块只是换了个构图顺序。对策：
  `BlockOptions.eval_chunks` 默认 True（S=38912 峰值 6.44 → 4.43 GB，还快 2 %）。
- **不要指望 GEMM 优化或 ANE**。花时间测了裸 GEMM（5.88 TFLOPS，各形状都一样）
  和 CoreML 的三个后端，结论是 GEMM 已经到理论峰值的 92 %、ANE 比 GPU 还慢
  1.8 倍。对策：记在这里，不要再试第二遍。
- **numpy 没有 bfloat16**。torch ↔ MLX 只能按 16 bit 原始位搬（两边都 view 成
  int16），否则会经过 fp32 损失精度。见 `interop.py`。
- **`mx.load` 的返回值要及时丢掉**。它是惰性的，但 evaluate 过的 array 会一直持有
  数据；把 dict 缓存起来的话，读过的每个 block 都会留在内存里。

- **真实权重对不上，问题却在对照脚本里**。现象：合成权重的单测相对 L2 < 5e-3 全过，
  换成真实发布的 block 0 却是 1.2（等于毫无关系）。原因：临时写的对照脚本把 fused QKV
  的行置换用成了 `perm` 的逆；`interop.block_weights_from_torch` 写的是
  `mlx[perm[r]] = torch[r]`，所以回来要 **gather**（`torch = mlx[perm]`），不是用逆索引。
  只走一个方向的测试永远看不出方向反了。对策：把反方向做成库函数
  `interop.torch_block`，并加"发布 layout → torch → 发布 layout **逐位相等**"的单测；
  真实权重的对照也从一次性脚本改成 `scripts/mlx_check.py`。
- **MLX 的 `astype` 是惰性的，会把 fp32 原张量一直拖着**。现象：bf16 权重集
  4.51 GB，载入峰值却是 9.03 GB，正好两倍；`mx.clear_cache()` 一点用都没有
  （cache 本来就是 0）。原因：`reader.read(k).astype(bf16)` 只建了一个图节点，
  fp32 的输入活到这个节点被 eval 为止；每个 block 才 eval 一次的话，整个
  checkpoint 的 fp32 副本都还在。对策：`_cast()` 读完一个张量就 `mx.eval`，
  超额从"整份 checkpoint"降到"一个张量"（128 MB），峰值 9.03 → 4.60 GB。
- **测出来的时间先要问机器当时紧不紧张**。现象：同一份代码、同一个 batch，
  两次测量差 3.3 倍（1.83 vs 6.1 s/tile），一度以为是 `tile_batch` 线性省时间，
  据此把默认值调到 8。原因：4.5 GB 常驻 + 前一个进程留下的内存压力，MLX 的
  buffer 被换出去，每次前向都在重新缺页。对策：每次测之前看 `memory_pressure`，
  可疑的数字换个时间重测一遍；结论写进文档时把当时的空闲内存也写上。
- **RoPE 的角度漏了 2π**。现象：两条 fp32 路径的相对 L2 是 6e-3（不是 1e-7）。
  原因：发布实现的频率表是 `2π * pos * inv_freq`，移植时只写了 `pos * inv_freq`。
  排查方式是逐段对照中间量（proj_in → rope → block 0 → …），第一段对不上的就是
  rope 的 cos。对策：`rope_tables` 的单测直接钉 numpy 写出来的解析式，而不是钉
  另一份实现。

- **每头一份的选择遇上 head chunk 会直接崩**。现象：真实 Veda plan（每层两个头组、
  每个头自己 top-k）跑起来报 `per-head index has 56 heads, expected 8`。原因：trunk 按
  `head_chunk=8` 分块算注意力，`layer_attention` 把 q 切到 8 个头，却把整组 56 个头的
  `index` / `keep` 原样传给 kernel。合成 plan 是所有头共用一份 2 维 index，所以之前的
  测试全过。对策：`SparsePlan.select_heads` / `HeadGroupPlan.select_heads`，切 q 的同时
  切选择；单测钉"分块与不分块逐位相等"（head_chunk=1 时 batch 为 1，kernel 归约顺序
  不同，差 1 ulp）。

## 验证记录

- 2026-09-24，Apple M3 Pro / 18 GB / mlx 0.32.2，commit 见本次提交：
  `pytest tests/unit` 127 passed。以上全部数字由 `scripts/mlx_bench.py` 在合成
  权重（真实 shape）上实测。
- 2026-09-25，同一台机器：加上 `dit.py`（trunk 之外）、`model.py`（一次速度评估）、
  `pipeline.py`（去噪循环）与其 torch 对照测试后 `pytest tests/unit` 186 passed。
  两步 turbo schedule 的端到端轨迹与同一个 torch 循环相对 L2 < 5e-3。同日用真实发布的
  `diffusion_pytorch_model.safetensors.index.json`（638 个键，50 层 + 2 层 refiner）
  核对 `h3/release.py` 的名字表：完全覆盖。
- 块稀疏与稠密掩码路径**逐位相等**（单测），因此不需要 AGENTS.md 1.5 要求的
  可视化人工确认。
- 2026-09-25，同一台机器：真实发布权重的 block 0 / block 1 用
  `scripts/mlx_check.py` 做了三方对照（见「真实权重：一个 block 的三方对照」），
  `mlx-vs-fp32` 不比 `torch-vs-fp32` 差；`pytest tests/unit` 190 passed。
- 2026-09-25，同一台机器：真实 33B trunk 转成 bf16 slab 后跑完 8 步去噪
  （seq 12 352，见「真实权重：一条完整轨迹」），1 050 s、峰值 4.51 GB，输出
  video `(12096, 96)`、audio `(130, 32)`，数值没有 NaN / 爆炸（video std 0.90、
  audio std 0.23）。
- 2026-09-25，同一台机器：真实发布的文本编码器 layer 0 / 7 / 30 用
  `scripts/mlx_check.py --text-encoder` 做了三方对照（见「真实权重：文本塔
  一层的三方对照」），三层的 `mlx-vs-fp32` 都不比 `torch-vs-fp32` 差；真实
  prompt 跑前 3 层（15 token）1.9 s/层（直接读 checkpoint，mmap 缺页路径），
  峰值 3.27 GB；`pytest tests/unit` 201 passed。
- 2026-09-25，同一台机器：diffusers 移植版的两个 VAE 在 MPS 上解码通过，
  1 秒 16:9（39 帧 1344×768）随机 latent 解出 mp4（见「VAE 解码」）。这只
  说明解码链路通，不说明画面内容对——真实 latent 的可视化确认还没做。
- 2026-09-25，Apple M3 Pro / 18 GB / mlx 0.32.2：视频 VAE 的 MLX 移植。
  纯数据变换（`split_tiles` / `blend` / `stitch_tiles`）与 diffusers 参考**逐位
  相等**；整条解码路径（随机小模型、两边 fp32，`scripts/mlx_check_video_vae.py`）
  相对 L2 1.4e-7 ~ 1.6e-7，量级就是 fp32 的累加顺序，属于 AGENTS.md 1.5 里
  "无法位级对齐"的一类。真实权重、真实 latent 解出的
  `runs/mlx_clip_0000/clip.mp4` **还没有做人工可视化确认**，因此这部分还不能
  按 1.5 判定为通过。`pytest tests/unit` 218 passed。

- 2026-09-26，Apple M3 Pro / 18 GB / mlx 0.32.2：Veda 打分器移到 MLX
  （`veda_predictor.py` + `veda_plan.ActivationPlanner`）。gather 与 max / min
  池化和 torch 参考**逐位相等**，mean / logits 相对误差 1e-6 量级，真实几何下
  选出的 `index` / `keep` 与 torch 路径逐位相等；`pytest tests/unit` 229 passed。
  真实权重一步去噪 107.2 s、峰值 4.49 GB（`--sparse-ratio 0.1 --sparse-scorer
  pooled`）。选择现在跟内容走，但**投影是未训练的**（等价于 mean-pooled QK），
  画面层面的人工确认仍未做。

- 2026-09-26，Apple M3 Pro / 18 GB / mlx 0.32.2，commit 见本次提交：修掉 SwiGLU
  两半接反（见 [h3_model](h3_model.md) 的踩坑记录）之后，第一次跑出**内容正确**的
  片段。同一个 prompt / seed 0 / 8 步 / 1 秒 16:9（768×1344、39 帧）：

  | 路径 | 8 步合计 | 每步 | MLX 峰值 | 末态 vs 初始噪声 |
  |---|---|---|---|---|
  | 稠密 | 1 080.6 s | 135.1 s | 4.53 GB | `\|x−x0\|/\|x0\|` 1.43，cos 0.09 |
  | Veda（pooled-QK，密度 0.136） | 855.1 s | 106.9 s | 4.62 GB | 1.40，cos 0.10 |

  **1.26×**。修之前两条路都只走到 0.34 / cos 0.94（即几乎没动），解出来的画面和
  "直接解码纯噪声"没有区别；现在两条路都解出了 prompt 描述的霓虹街景。视频 VAE 解码
  （MLX）104.8 s、峰值 5.76 GB。产物：
  `artifacts/visual_checks/mlx_veda/2026-09-26-fixed/{dense,veda,side_by_side}.mp4`。
- **Veda 与稠密之间的差异是内容级的，不是扰动级的**：两段片段 PSNR 13.7 dB、
  SSIM 0.648（ffmpeg，veda 对 dense）。同一个场景、同样的风格和构图逻辑，但人物
  位置和招牌排布明显不同。原因是打分器的投影**没有训练**（等价于 mean-pooled QK），
  在密度 0.136 下选出的 tile 已经足以改变轨迹。AGENTS.md 1.5 要求的人工一致性确认
  因此**尚未通过**：速度结论成立，"稀疏不掉质量"的结论不成立。

## 待办

- 接上真实的文本编码器与 VAE 解码（两者留在 torch/MPS），把上面那条轨迹的随机
  文本 embedding 换成真实 prompt，解码成视频，再按 AGENTS.md 1.5 做人工可视化
  确认。DiT 这一侧（转 slab → 去噪循环）已经用真实权重跑通。
- CFG：现在的循环只算一次速度评估，还没有无条件分支（开了就是两倍时间，18 GB
  上要先确认值不值）。
- 打分器已经在 MLX 上按每层的 q/k 打分（mean-pooled QK），但**没有训练好的
  predictor 投影**：要接真实 checkpoint，给 `ActivationPlanner(projections=...)`
  传回调即可，代价是每层多两次 [H', 3D, D] 的矩阵乘和一次权重常驻（50 层 × 56 头
  × 384 × 128 × 2 = 275M 参数，bf16 0.55 GB）。
- `q_block=128` 的 gather 量是 `q_block=2048` 的 16 倍。可以把 16 个相邻 query
  tile 的选择取并集，共享一次 gather，再对每个 tile 单独调一次 SDPA（掩码仍然按
  tile）。并集能省多少取决于相邻 query tile 的选择有多重合，随机打分器上并集≈全集，
  所以这件事必须等真实打分器的选择出来再测。
