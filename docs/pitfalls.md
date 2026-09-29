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
| 8 步生成出来的视频和"直接解码纯噪声"没有区别，torch / MLX 两条路还完全一致 | SwiGLU 的两半接反：发布权重把 fc1 融成 `[up; gate]`，我们当成 `[gate; up]`。形状、范数、两条自研路径的互相对拍全都看不出来 | 三处改成 `up, gate = chunk(2)`；加单测钉死这个顺序；用 `scripts/check_vs_diffusers.py` 逐模块对 diffusers 参考实现 | [h3_model](features/h3_model.md) |
| 同上，但发生在"修好之后"：CUDA 路生成又变成纯噪声，单测全绿 | SwiGLU 融合顺序是发布版本的属性：第一版发布是 `[gate; up]`，diffusers 版是 `[up; gate]`。拿 diffusers 版对出的结论被套用到了两版 | `release.MLP_GATE_FIRST` 按 schema 钉死；`_class_name` 判定发布版本，认不出就报错；加载时交叉校验张量名与模型假设 | [h3_model](features/h3_model.md) |

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
| Veda 每步 GPU 空闲 0.4 s（t102） | `select_video_blocks` 的 `torch.nonzero` 每次调用同步主机；`bresenham_extra` 每次从可分页内存 H2D | 对角块改成 gather / where / scatter；bresenham 表按设备缓存 | [predictor_mask](features/veda_predictor_mask.md) |

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
| 融合 kernel 与 eager 差 1 ulp（modulate 28% 的元素） | Triton 把 `.to(bf16).to(fp32)` 来回转换消掉，再把乘加收缩成 fp32 fma，跳过了 eager 的中间舍入 | bf16 舍入写成整数位运算；启动参数 `enable_fp_fusion=False` | [h3_model](features/h3_model.md) |
| 融合的 silu 对次正规数输出 -0.0 | Triton 默认让 libdevice 走 flush-to-zero | 启动参数 `enable_reflect_ftz=False`；GPU 测试遍历全部 bf16 值 | [h3_model](features/h3_model.md) |
| 猴补 FA4 的 num_stages / num_threads 后测速完全不变、diff 恰好为 0 | FA4 的编译缓存键不含这些参数，第一次编译的 kernel 被复用 | 每个变体前换一个新的 `get_jit_cache()` | [performance](benchmark/performance.md) |
| SM120 上 FA4 `Q_in_regs=True` + 2 stage 结果错误（max diff 5e-2） | 未查明；SM80 主循环的该组合在 SM120 上没有验证过 | 不用；只作为调参时的已知坏组合 | [performance](benchmark/performance.md) |
| 以为 SM120 是 SM100 的小号、照搬 SM100 前向 | SM100 的主循环建在 tcgen05 + TMEM 累加器上，SM120 没有；反过来 SM120 有 SM100 没有的 warp 级块缩放 MMA | 用 `scripts/probe_ptx_isa.py` 让 ptxas 判决，再决定搬哪部分（结构可搬，MMA 不可搬） | [performance](benchmark/performance.md) |

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
| 8 卡训练启动要 23 分钟，其中 18 分钟在 AdaLN 建表 | 表是复制的，每块 `adaln_proj` 1.04 GB × 50 块 = 52 GB，8 个 rank 各读一遍 = 416 GB 走网络盘 | 把 transformer 拷进 `/dev/shm` 再用软链接指过去；启动降到 10 秒 | [training](features/training.md) |
| 训练跑完发现 checkpoint 不在共享盘上，随实例停机丢失 | `ln -sfn 目标 runs` 在 `runs/` **已经是目录**时会把链接建进目录里（`runs/runs`），输出仍写本地盘，且完全没有报错 | 建链接前先确认目标不是已存在的目录（`ls -ld`），建完用 `ls -l` 确认是 `runs -> ...` 而不是 `runs/runs -> ...`；长任务开跑后先确认第一个 checkpoint 落在预期位置 | [training](features/training.md) |

## 环境

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| 第三方扩展编译失败，报 "C++20 or later compatible compiler is required" | torch 2.14 需要 C++20 | 编译参数改为 `-std=c++20` | [dependencies](dependencies.md) |
| 运行 transformers 的 Qwen3-VL 报缺少 torchvision | 视频处理器依赖 torchvision | 安装与 torch 匹配的 torchvision | [dependencies](dependencies.md) |
| 20 条片子去噪跑完 35 分钟，写 mp4 时才报 `No module named 'diffusers'` / `torchvision` / `ffmpeg` | 训练机只装了训练依赖，解码链缺件，而解码是流程最后一步 | 按 [dependencies](dependencies.md) 一次补齐；已跑的去噪不用重来，`generate.py --decode-only` 从 `<mode>_latents.pt` 续 | [dependencies](dependencies.md) |

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
| 给 MLX 的融合 attention 上 90 % 稀疏的块掩码，时间只降到 0.97 倍 | 掩码在 Q@K.T 矩阵乘之后才施加，只有内置 `"causal"` 会缩短 key 循环上界 | 按固定预算把选中的 key tile gather 成规整批量问题，走一次稠密 kernel，拿到 9–10 倍 | [mlx_inference](features/mlx_inference.md) |
| 块稀疏开了反而吃内存、加速不明显 | `q_block` 太小，gather 量正比于 `S/q_block` | `q_block` 取 2048 以上，再用 `head_chunk` 压峰值 | [mlx_inference](features/mlx_inference.md) |
| 分了块，峰值内存却没下来（S=38912 仍然 6.44 GB） | MLX 的图是惰性的，不在 chunk 之间 eval，所有 chunk 的中间量活到最后 | `BlockOptions.eval_chunks` 默认 True，每个 chunk 算完就 `mx.eval`；峰值降到 4.43 GB 且更快 | [mlx_inference](features/mlx_inference.md) |
| 想靠 GEMM 优化 / ANE 再提速 | MLX GEMM 已跑到 5.9 TFLOPS（理论峰值的 92%），ANE 只有 3.3 TFLOPS 且权重被编译进模型 | 不要在这两条路上花时间，瓶颈只能靠稀疏度和序列长度解决 | [mlx_inference](features/mlx_inference.md) |
| 真实 Veda plan 下 `q_block` 只能取 128，内存和时间都变差 | Veda 的选择是逐 128 行 query tile 的，gather 量正比于 `S/q_block` | `head_chunk` 降到 2（S=38912：8.78 → 5.99 GB 峰值，8.01 → 7.70 s） | [mlx_inference](features/mlx_inference.md) |
| 预算不齐的行用"重复已选 tile"补齐，结果不对 | 重复的 key 会在 softmax 里被算两次 | 补位槽用 `SparsePlan.keep` 掩掉，不要重复 | [mlx_inference](features/mlx_inference.md) |
| torch ↔ MLX 转换后 bf16 精度变差 | numpy 没有 bfloat16，默认路径经过 fp32 | 两边 view 成 int16，按原始 16 bit 搬运 | [mlx_inference](features/mlx_inference.md) |
| bundle 存 bf16 却占着 fp32 的显存 | `load_state_dict` 往已有参数里 `copy_`，按**参数**的 dtype 转换，fp32 模块会把 bf16 文件静默升回去 | `bundle.load()` 先 `model.to(stored_dtype)` 再加载；单测断言加载后参数是 bf16 | [inference](features/inference.md) |
| 解码阶段 worker 卡死、100% CPU 却不出文件 | `psnr_per_frame` 一次性造两份 4.3 GB fp32，三个 worker 38 GB，内存吃紧后每次触页走 direct reclaim | 生成路径不算 PSNR；取栈要注意 ptrace_scope=1 只允许 attach 自己的后代进程 | [inference](features/inference.md) |
| 换成只训最长几何后，分配器报 OOM 告警（free 0.6 GiB 却要 3.4 GiB） | 混训时 t102 每 12 条轨迹才来一次，只训 t102 时每次几何切换都是大张量换大张量，缺的是连续性不是总量 | `offload_blocks` 44 → 48，并用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`mlp_chunk_rows` 不能动（影响数值） | [training](features/training.md) |
| 104k token 时报 `expandable_segments: memory mapping failed ... (free: 20 MB)` | 和碎片无关，是显存真的用尽：收集器同时持有 q / k 两份 512 MiB 的 tile 序副本 | `_COLLECT_BYTES` 512 → 256 MiB、`offload_blocks` → 50、启动加 `garbage_collection_threshold:0.8`；头之间独立，chunk 大小不改结果 | [training](features/training.md) |
| 训练跑着时另起教师进程，权重刚载完就被静默 kill | 不是显存：再开一份 33B 教师要约 66 GB pinned 主机内存（`HostSlabs`），训练已经占掉了大部分，内核 OOM killer 不留回溯 | 离线探针和训练错开跑；判断依据是日志停在 `load weights ... done` 且进程无回溯地消失 | [quant_scoring](features/quant_scoring.md) |
| 换了一份发布 checkpoint，config 照样能读出来但模型可能是错的 | `from_pretrained` 对认不出的键静默退回默认值，而上游 diffusers 移植版把键全改名了 | `CONFIG_KEYS` 登记两套拼写，缺字段直接报错 | [h3_model](features/h3_model.md) |
| 主机侧 numpy 算的时间步正弦表与 torch 差 1 ulp，位级对齐的测试挂掉 | numpy 与 torch 的 fp32 `exp` 在 128 个频率里有 14 个最后一位不同（指数的自变量本身逐位相等） | 记录误差（max abs 2.98e-08）、测试改钉"≤1 ulp"；RoPE 的表仍然逐位相等 | [mlx_inference](features/mlx_inference.md) |
| MLX 的 Euler 步与 torch 差 1 ulp | torch 的 `add_(v, alpha=)` 是 fused multiply-add（单次 round），MLX 分开乘加 | 记录误差（2.4e-07）、测试钉 ≤1 ulp；不要为此把这一步改成 GEMM | [mlx_inference](features/mlx_inference.md) |
| 真实权重下 MLX block 与 torch 参考相对 L2 = 1.2（完全对不上），合成权重的单测却全过 | 对照脚本把 fused QKV 的行置换用反了（`perm` 的逆）；单测只走"torch → MLX"一个方向，看不出来 | 反方向写成库函数 `interop.torch_block`，并加往返逐位相等的单测；`mlx[perm[r]] == torch[r]`，回来是 gather | [mlx_inference](features/mlx_inference.md) |
| 用很短的片段（`latent_t` 2）试 VAE 解码，报 `torch.cat(): expected a non-empty list of Tensors` | 视频 VAE 按 `clip_length` 分块解码，并先补上 `token_drop` 个 token；latent 帧数不到一个 chunk 时分块数算成 0 | 冒烟测试压画布（短边）而不是压时长，时长至少取一个完整 chunk | [mlx_inference](features/mlx_inference.md) |
| bf16 权重集 4.5 GB，载入峰值却是 9.0 GB，`mx.clear_cache()` 无效 | MLX 的 `astype` 惰性，fp32 原张量活到结果被 eval；一个 block 才 eval 一次 | 读一个张量就 `mx.eval`（`_cast`），超额降到一个张量 128 MB | [mlx_inference](features/mlx_inference.md) |
| 同一份代码、同一个 batch，两次测速差 3.3 倍 | 机器内存紧张时 4.5 GB 常驻权重被换出，每次前向都在重新缺页 | 测速前先看 `memory_pressure`，把当时的空闲内存一起写进文档 | [mlx_inference](features/mlx_inference.md) |
| 移植 VAE 后两条 fp32 路径相对 L2 是 6e-3 而不是 1e-7 | RoPE 的频率表漏了 `2π` | 逐段对照中间量定位；单测钉 numpy 的解析式，不要钉另一份实现 | [mlx_inference](features/mlx_inference.md) |
| 真实 Veda plan 一跑就报 `per-head index has 56 heads, expected 8` | trunk 按 head_chunk 切 q，却把整层的每头选择原样传给 kernel；合成 plan 是共用的 2 维 index，测不出来 | `select_heads` 跟着 q 一起切；单测钉分块 / 不分块逐位相等 | [mlx_inference](features/mlx_inference.md) |
| `bf16+smoothk` 不是恒等 | 先把 `x - mean` 落回 bf16 再量化，把减法自己丢掉的位算在平滑头上；真实硬件里这一步在 fp32 累加器里 | `fake_quantize` 全程 fp32，最后才 `.to(x.dtype)` | [quant_scoring](features/quant_scoring.md) |
| fp8 bundle 在单测里没有"减半" | safetensors 的 header 在玩具尺寸下比权重还大 | 阈值放到 0.75；真实尺寸上才是 550,635,792 → 275,415,648 字节 | [quant_scoring](features/quant_scoring.md) |
| 多卡生成时两张卡在第一步就挂，报 `flash_attn.cute.interface has no attribute flash_attn_func` | 每卡一个线程同时首次 import FA4；`functools.cache` 执行期间不持锁，而 SM8x 补丁手工把模块塞进 `sys.modules`，绕过 import lock，别的线程拿到半初始化的模块 | `fa4._IMPORT_LOCK` 串行化 `_modules()`；单测用假的 `_import_modules` 钉并发度为 1 | [veda_kernel](features/veda_kernel.md) |
| SM120（RTX PRO 6000）上 Veda 稀疏直接抛 `NotImplementedError`，而且 vendored 补丁根本没装上 | `fa4.available()` 的白名单只有 (9,10,11)，`major == 8` 才触发 `fa4_sm8x.install()`；上游 FA4 又在 arch-12 分支 `assert not use_block_sparsity` | SM120 不需要新 kernel：`FlashAttentionForwardSm120` 只是 SM80 类的薄子类（只 override `can_implement`、把 `self.arch` 改回 `sm_80`），块稀疏主循环被继承。补丁 0006 拆掉 arch-12 门禁，`PATCHED_MAJOR_ARCHS = (8, 12)` | [veda_kernel](features/veda_kernel.md) |
| 在 Blackwell 上装好 torch 却跑不了任何 kernel | cu126 的 wheel 里没有 sm_120 | 装 cu128 及以上；用 `torch.cuda.get_arch_list()` 确认含 `sm_120` | [veda_kernel](features/veda_kernel.md) |
| `pip install` flash-attn-4 的 git 依赖卡死/失败在 `composable_kernel` | pip 对 git 依赖无条件跑 `git submodule update --init --recursive`，把 ROCm 的 `csrc/composable_kernel` 也拉一遍，与 `flash_attn/cute` 毫无关系 | 自己浅克隆（不带 submodule）再 `pip install <clone>/flash_attn/cute` | [veda_kernel](features/veda_kernel.md) |
| 一个进程里 VAE 解码器常驻卡上，latent_t 102 的去噪必 OOM | 2.4B 视频 VAE 占约 5 GiB，正好是 104k token 注意力激活缺的那几 GiB；单次生成不会遇到（解码器是去噪后才建的） | `decode.Decoder.to()` 在两次解码之间把 VAE 停到主机内存，搬运不计入解码计时 | [inference](features/inference.md) |
| 性能扫描崩在最后一个几何，前面成功的几何结果一起没了 | 只在全部跑完后写一次 JSON | 每跑完一个 (几何, 模式) 就重写 `--out` | [inference](features/inference.md) |
| SM120 反向：块稀疏被拒，而且就算放行梯度也会是错的 | 上游 arch-12 反向分支 `assert` 拒绝块稀疏，且给反向 kernel 传的是空 kwargs（`mask_mod` 与 subtile factor 全丢）；dQ/dK/dV 的 postprocess 线程数又写死成"非 arch 8 就 128"，与主 kernel 的 256 对不上时会静默算错梯度 | 补丁 0007 把 arch-12 配置并进 arch-8 分支（smem 相同），传同一套 kwargs，postprocess 线程数扩到 arch 12；测试里钉"梯度必须离稀疏比离稠密更近" | [veda_kernel](features/veda_kernel.md) |
