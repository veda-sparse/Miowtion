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

### 数值

MLX block 的算子顺序对齐 `miowtion/h3/model.py`，让每一步逐元素运算在同一个位置
round 到 bf16：RMSNorm 的统计量和乘权重在 fp32 里算、只 round 一次；AdaLN 调制和
门控残差在 bf16 里算；SiLU 在 fp32 里算、round 一次。

结果：**调制、RoPE、SwiGLU 与 torch 逐位相等**；整个 block 不是——RMSNorm 的统计
量、GEMM、attention 的归约顺序不同（见「踩坑记录」）。

## 代码位置与接口

| 文件 | 作用 |
|---|---|
| `miowtion/mlx/block.py` | 一个 trunk block 的前向；`BlockWeights`（发布 layout，含量化）、`BlockOptions`（head / row 分块） |
| `miowtion/mlx/interop.py` | torch ↔ MLX 的逐位转换（numpy 没有 bf16，按 16 bit 原始位走） |
| `miowtion/mlx/slab.py` | slab 格式、`SlabReader`（pread 进预分配 buffer）、`BlockPrefetcher`、`convert_checkpoint` |
| `miowtion/mlx/offload.py` | 不转换的替代方案：直接读发布的 safetensors（`mx.load` 惰性加载 / 每个分片一个 mmap） |
| `miowtion/mlx/bench.py` | 测量用：合成权重、进程与系统内存统计、各项 benchmark |
| `scripts/mlx_bench.py` | 命令行入口，每项测量单独一个进程，结果按 JSON 行输出 |

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

量化**不会让计算变快**（M3 Pro 上 GEMM 已经是算力受限，不是带宽受限）：

| 序列长度 | bf16 | 8 bit（quantized matmul） | 4 bit |
|---|---|---|---|
| 4 096 | 0.689 s | 0.723 s | 0.724 s |
| 16 384 | 3.925 s | 4.096 s | 4.090 s |
| 38 080 | 14.01 s | 14.84 s | — |

量化的收益只在 **I/O 量和内存**，不在速度；量化矩阵乘反而慢 2–6 %。它确实省内存：
S=38080 下 8 bit 的进程峰值是 3.59 GB，bf16 是 4.51 GB。

### 端到端流式（预取 depth=1，2 个 slot）

每个 block 的读完全藏在上一个 block 的计算后面：

| 序列长度 | 精度 | slab/block | 每 block 墙钟 | 其中计算 | 其中后台读 | 实际等待 | 进程峰值 |
|---|---|---|---|---|---|---|---|
| 4 096 | bf16 | 0.771 GB | 694 ms | 686 ms | 119 ms | **7.3 ms** | 2.70 GB |
| 4 096 | 8 bit | 0.409 GB | 747 ms | 743 ms | 64 ms | 4.0 ms | 1.98 GB |
| 4 096 | 4 bit | 0.217 GB | 733 ms | 731 ms | 35 ms | 2.2 ms | 1.59 GB |
| 16 384 | bf16 | 0.771 GB | 3 940 ms | 3 925 ms | 118 ms | 14.8 ms | 4.02 GB |

「实际等待」是消费端真正阻塞在 I/O 上的时间，基本只剩第一个 block 的冷启动。
**即使在最短的序列、最大的 bf16 权重下，读 / 算的比值也只有 119/686 ≈ 0.17**，
离 I/O 受限还有 6 倍的余量。

### 外推：一次完整生成（50 block × 8 步 = 400 次 block 前向）

| 序列长度 | bf16 每次生成 | 90 % 块稀疏后（估算） |
|---|---|---|
| 4 096 | 4.6 min | 4.0 min |
| 16 384 | 26 min | 17 min |
| 38 080 | 93 min | **43 min** |

一次生成总共要读 400 × 0.771 GB ≈ **308 GB**，按 6.5 GB/s 是 47 s，全部藏在计算
后面。Veda 的 90 % 块稀疏只作用在 attention 上，按实测的 attention 时间占比线性
折算（MLX 这边不需要稀疏 kernel，只是给出上界）：序列越长收益越大，38 080 上能
省掉一半以上的时间。

**所有配置都是计算受限，没有一个是 I/O 受限的。** 这是本研究最主要的结论：在这台
机器上，NVMe offloading 基本是免费的，瓶颈是 M3 Pro 那 ~5 TFLOPS 的 bf16 算力。

### 磁盘与内存预算（50 个 block）

| 精度 | 磁盘占用 | 2 个 slot 的权重内存 | S=4096 进程峰值 |
|---|---|---|---|
| bf16 | 38.5 GB | 1.54 GB | 2.70 GB |
| 8 bit | 20.5 GB | 0.82 GB | 1.98 GB |
| 4 bit | 10.8 GB | 0.43 GB | 1.59 GB |

## 18 GB 机器上的推荐配置

- **bf16 slab + `depth=1` 预取（2 个 slot）+ `head_chunk=8, row_chunk=4096`。**
  量化不会更快，所以只要磁盘放得下 38.5 GB 就用 bf16，避免量化误差。
- 磁盘紧张、或者序列长到内存不够时再换 8 bit（I/O 减半、内存省 0.7 GB、慢 5 %）；
  4 bit 留给最极端的情况。
- 不要用 mmap 方案。不想做 checkpoint 转换时用 `mx.load`（一样快），代价是走
  page cache 且不能后台预取。
- 读参数保持 16 MiB 一块、4 线程；调小会直接掉带宽。
- 内存主要被激活占，不是被权重占：在 18 GB 机器上真正的上限来自序列长度，务必开
  分块。

## 测试

`tests/unit/test_mlx_block.py`（8 个，缺 mlx 时自动 skip）：

- torch ↔ MLX 转换、RoPE、SwiGLU 逐位相等（`torch.equal`）。
- 整个 block 对 torch 参考实现：fp32 相对 L2 < 1e-5；bf16 < 5e-3，并且额外要求
  MLX 的 bf16 结果离 fp32 参考不比 torch 的 bf16 结果更远。
- 分块（head / row）不改变结果；padding 行不影响真实行的输出。
- 8 / 4 bit 量化的误差量级；`BlockWeights.from_tensors` 对缺失和多余的 tensor 报错。

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
- **numpy 没有 bfloat16**。torch ↔ MLX 只能按 16 bit 原始位搬（两边都 view 成
  int16），否则会经过 fp32 损失精度。见 `interop.py`。
- **`mx.load` 的返回值要及时丢掉**。它是惰性的，但 evaluate 过的 array 会一直持有
  数据；把 dict 缓存起来的话，读过的每个 block 都会留在内存里。

## 验证记录

- 2026-09-24，Apple M3 Pro / 18 GB / mlx 0.32.2，commit 见本次提交：
  `pytest tests/unit` 116 passed。以上全部数字由 `scripts/mlx_bench.py` 在合成
  权重（真实 shape）上实测。
- **尚未用真实权重验证**，也没有做视频层面的可视化对比（本研究不下载权重）。
  在真实 checkpoint 上跑通并做人工一致性确认之前，这里的结论只覆盖性能，不覆盖
  生成质量。

## 待办

- 用真实 checkpoint 跑 `slab.convert_checkpoint`，端到端生成一段视频，并按
  AGENTS.md 1.5 做人工可视化确认。
- 接上 `miowtion/infer` 的去噪循环（目前只有 block 级前向，没有时间步循环、
  文本条件和 VAE 解码）。
- 测 S=65536 及以上，确认 18 GB 上真正的序列上限。
- AdaLN 表的预计算目前还在 torch 侧，MLX 侧只消费表；考虑一并搬过来。
- Veda 的块稀疏在 MLX 上没有 kernel，表里的稀疏收益是按 attention 时间占比折算的
  上界，需要真正实现后复核。
