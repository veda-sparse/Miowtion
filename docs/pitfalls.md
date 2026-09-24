# 踩坑总表

每一行：现象 / 原因 / 对策 / 详细记录所在文档。新坑加在对应分组末尾。

## 模型与数据

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| 模型能跑但结果完全是垃圾 | 发布 checkpoint 的融合 QKV 按头交错存放 | 加载时重排；单测钉死置换 | [h3_model](features/h3_model.md) |
| AdaLN 槽位数写死为 2 时出错 | 第 0 步 t_video=t_audio=0 会合并成 1 个槽位 | 槽位由 `torch.unique(sorted)` 得出，数量不固定 | [h3_model](features/h3_model.md) |
| ref2va prompt 被校验拒绝 | ref2va 是六段式格式，不是 t2va 的三字段 | `validate_prompt(prompt, task)` 按 task 区分 | [training](features/training.md) |
| 多卡时复制的打分器权重不一致 | 初始化用了全局 RNG，而各 rank 的 RNG 状态不同 | 创建打分器前 `torch.manual_seed(config.seed)` | [training](features/training.md) |
| VAE 编码结果不可复现 | VAE 的后验是采样得到的 | 每次编码前把全局 RNG 固定为 seed 42 | [h3_model](features/h3_model.md) |

## Veda

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| 平均保留数比预算少 1/n | `2.3−2 = 0.29999…` 使 `floor(n·frac)` 少算一个 | frac 按 12 位小数取整 | [predictor_mask](features/veda_predictor_mask.md) |
| recall 被错算为 0 | 预算里只有对角块，去掉对角后两个集合都为空 | 返回 NaN，日志用 nanmean 汇总 | [predictor_mask](features/veda_predictor_mask.md) |
| 搜索中 partial tile 的块分数被污染 | pad 槽 gather 到了第 0 行的 q | softmax 之后先清零 pad 行 | [tile_search](features/tile_search.md) |
| Triton 热力图 kernel 慢约 20% | 默认 num_stages=3，key tile 循环被软件流水后反而变慢 | 启动时传 `num_stages=1` | [predictor_mask](features/veda_predictor_mask.md) |
| 长 clip（103k token）的 tile 搜索 OOM | oracle kernel 路径一次 gather 所有头的 q/k/v | 按头分块，单块 ≤ 1 GiB | [tile_search](features/tile_search.md) |
| oracle 的 kernel 路径与参考路径的 rel-MSE 相差最多 88% | 子集行的块列表用行数 R 当全局列起点 | 起点取 `layout.n_video_tiles`；单测对拍子集与全量 | [tile_search](features/tile_search.md) |
| 混合形状的用例没真的测到"按头变长" | 两种形状在 (12,8,16) 上都是 12 个 tile、补齐为 0 | 用 latent_t=11 的网格（12 vs 11 个 tile） | [veda_tiling](features/veda_tiling.md) |

## Kernel

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| FA4 在 4090 上的"块稀疏"与稠密耗时相同、结果也相同 | 上游 SM80 前向 kernel 忽略了 blocksparse_tensors | vendored 补丁（`kernels/fa4_sm8x`）装上才放行 SM8x，否则直接报错 | [veda_kernel](features/veda_kernel.md) |
| SM8x 稀疏与稠密差 1 个 bf16 ulp | partial / full 分开遍历，访问顺序与稠密不同；或稠密调用的 tile 配置不同 | 补丁按稠密顺序合并遍历；逐位对拍时两边用同一 tile | [veda_kernel](features/veda_kernel.md) |
| SM80 反向梯度静默出错（误差 1.4） | AtomLayoutNdKV=8，或 postprocess 线程数与主 kernel 不同 | 8 warps 用 (2,4,4)；postprocess 线程数跟随主 kernel | [veda_kernel](features/veda_kernel.md) |
| 打了补丁的 FA4 没有生效 | `flash_attn.cute` 的 `__init__` 会 import interface，先 import 就拿到了上游版本 | 所有 FA4 import 都经过 `kernels/fa4.py`；`install()` 发现已 import 直接报错 | [veda_kernel](features/veda_kernel.md) |
| FA4 在 SM100 上拒绝 128 行的 Q 块 | 接口在 seqlen>128 时强制 q_stage=2（256 行粒度） | 覆盖为 q_stage=1（待 B200 验证） | [veda_kernel](features/veda_kernel.md) |
| FA4 报出看起来像掩码形状不对的错误 | `BlockSparseTensorsTorch` 的第 5 个字段是 `cu_total_m_blocks` | `block_size` 用关键字参数传 | [veda_kernel](features/veda_kernel.md) |
| flex 每次换 pattern 都很慢 | `from_kv_blocks` 默认为反向计算转置索引（约 1.2 ms） | 只做前向时传 `compute_q_blocks=False`（约 0.01 ms） | [veda_kernel](features/veda_kernel.md) |

## 训练与分布式

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| 生成的 mp4 在 ~2 s 后没有声音 | ffmpeg 4.4 的 `-frames:v` 在视频帧写满时结束整个输出，截断音频 | 不用 `-frames:v`，靠管道 EOF 结束；单测检查音轨时长 | [inference](features/inference.md) |
| `SparseStudent` 报张量行数不匹配（330 vs 336） | 把包含全局行的 logits 交给只要视频行的 `select_video_blocks` | 只取视频行；CPU 测试用带全局 tile 的布局 | [training](features/training.md) |
| 长 clip（78k+ token）在 `apply_rope` 处 OOM | eager RoPE 链的每一步都是整张 [S, H, D] 临时张量 | 按行分块（逐位相同）；训练 offload 40 个 block | [h3_model](features/h3_model.md) |
| 单卡推理 104k token 时 Veda 在 FA4 输出分配处 OOM | SparseStudent 一次处理整个头组 | 与 collector 一样按头分块（逐位不变） | [inference](features/inference.md) |
| 解码一个 14.4 s 的片段要 2.7 分钟 | 视频 VAE 的解码端是 2.42B 的 ViT3D，权重 fp32，用不上 tensor core | 视频 VAE 转 bf16（3.2×，51.3 dB） | [inference](features/inference.md) |
| 关掉 VAE tiling 后画质崩坏（19.7 dB）且更慢 | token id 是 tile 局部坐标，ViT 的 RoPE 没在整帧坐标上训练过 | tiling 是设计前提，不要关 | [inference](features/inference.md) |
| 解码 14.4 s 16:9 时 OOM | `revert_tensor` 要 345 帧 1344×768 的 fp32 副本（约 4 GB） | 按帧分块（32 帧，逐位相同） | [inference](features/inference.md) |
| 两个单卡推理进程被 host OOM killer 杀掉 | 每个进程为 offload 的 block 各 pin ~35 GB | 一个进程驱动多张卡，共享 pinned slab | [inference](features/inference.md) |
| 16:9 14.4 s（103k token）训练在 TeacherCollector 里 OOM | 整个头组的 tile 顺序 q / k 副本各 1.5 GB | 按头分块（≤ 512 MiB），梯度不变 | [predictor_mask](features/veda_predictor_mask.md) |
| 单卡 offload 时预取无效，每个 block 停 ~59 ms | FSDP2 在 world size 1 时 `unshard()` 直接返回，H2D 拷贝在计算流上同步做 | 单进程用 `BlockStreamer`（独立 copy stream + non_blocking + event） | [training](features/training.md) |
| 预取的拷贝仍然没有和计算重叠 | 显存太满，allocator 反复 mapping 失败并同步 | 少放常驻 block，给预取留余量 | [training](features/training.md) |
| "FSDP parameters should be materialized on CPU" | 开 CPU offload 但参数物化在 GPU 上 | offload 的 block 用 `to_empty(device='cpu')` | [training](features/training.md) |
| `FSDPCommContext has no all_gather_copy_in_stream` | 没有 FSDP root，跨 block 预取找不到通信流 | 根模块也 `fully_shard`，复制参数放进 `ignored_params` | [training](features/training.md) |
| 手动 all-reduce 挂起 | 某些 rank 缺梯度，拼出的缓冲区长度不一致 | 缺失的梯度补零 | [training](features/training.md) |
| 保存 checkpoint 时挂起 | `full_tensor()` 是集合通信，rank 0 以外的 rank 提前返回了 | 所有 rank 先取出完整张量，再由 rank 0 写盘 | [training](features/training.md) |
| 训练慢，但诊断量的 FLOPs 明明可以忽略 | 逐层 `kl.item()` 和对设备张量的 Python 分支把 CPU 与 GPU 串起来 | 诊断量留在设备上，每个 micro-step 只做一次 `torch.stack(...).cpu()` | [training](features/training.md) |
| 4090 上第 2 次 update 时 OOM | 打分器的 Adam 状态、梯度和 EMA（约 5.5 GB）常驻显存 | `offload_optimizer`：主权重、动量、EMA 放 pinned 主机内存 | [training](features/training.md) |

## 环境

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| 第三方扩展编译失败，报 "C++20 or later compatible compiler is required" | torch 2.14 需要 C++20 | 编译参数改为 `-std=c++20` | [dependencies](dependencies.md) |
| 运行 transformers 的 Qwen3-VL 报缺少 torchvision | 视频处理器依赖 torchvision | 安装与 torch 匹配的 torchvision | [dependencies](dependencies.md) |

## Prompt 扩写

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| DeepSeek `reasoning_effort: "medium"` 被接受，但没有中档效果 | 档位只有 none/low/high/max，medium 会被映射成 high | 记录请求值并注明实际等于 high；要更低只能用 low | [prompt_expansion](features/prompt_expansion.md) |
| PE 输出 shot 之间有空行，或字段重复 | LLM 输出格式漂移 | 要求每个字段单段；`check_expansion` 拒收；repair 轮重试 | [prompt_expansion](features/prompt_expansion.md) |
| PE 输出里出现 "landscape shot" / "16:9 frame" | user 消息里的几何描述泄漏进了正文 | user 消息只写 "16:9 (width:height)"；checker 拒收宽高比字符串 | [prompt_expansion](features/prompt_expansion.md) |

## Apple silicon（MLX）

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| mmap 读 checkpoint 只有 0.84 GB/s，比 pread 慢 7.8 倍 | 缺页同步、一次一页，SSD 队列始终是空的 | 用并发 `pread` 读进预分配 buffer（或 `mx.load`），不要用 mmap 做大块顺序读 | [mlx_inference](features/mlx_inference.md) |
| SSD 吞吐只有标称的一半 | 每次读 1 MiB 太小 | 16 MiB 一块、4 线程并发，才跑满 ~6.4 GB/s | [mlx_inference](features/mlx_inference.md) |
| 长序列进程峰值接近内存上限 | 未分块时 QKV 与 MLP 的中间量按整个序列物化 | 默认开 `head_chunk` / `row_chunk`；S=16384 峰值从 6.21 降到 3.56 GB，还更快 | [mlx_inference](features/mlx_inference.md) |
| 量化到 4 bit 后不但没变快，反而慢了 | M3 Pro 上这些 GEMM 是算力受限而非带宽受限 | 量化只用来省磁盘和内存，不要指望提速 | [mlx_inference](features/mlx_inference.md) |
| 逐 block 读完后内存持续增长 | `mx.load` 惰性，但 evaluate 过的 array 会一直持有数据 | 每个 block 用完即丢掉返回的 dict，不要缓存 | [mlx_inference](features/mlx_inference.md) |
| torch ↔ MLX 转换后 bf16 精度变差 | numpy 没有 bfloat16，默认路径经过 fp32 | 两边 view 成 int16，按原始 16 bit 搬运 | [mlx_inference](features/mlx_inference.md) |
