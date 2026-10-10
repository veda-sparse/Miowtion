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
| 拉完一批生成的视频，拼接脚本安静地少拼了一半 | 按 mp4 总数判断拉全了没有，而 `generate.py` 在 dense+veda 同跑时每个目录还多写一个 `dense_vs_veda.mp4`：16 条应该是 48 个文件，不是 32 个 | 逐目录核对 `dense.mp4` 与 `veda.mp4` 都在，别核对总数 | [evaluation](features/evaluation.md) |
| 对比视频里某两条的 prompt 和画面对不上 | 按 key 在文件名里做子串匹配：`hf_test_1` 也匹配 `hf_test_190`，`hf_test_18` 也匹配 `hf_test_182` | 用精确 key 配，生成后再拿 prompt 哈希回对一遍索引 | [evaluation](features/evaluation.md) |
| manifest 里 `latent_t` 写成字符串，几小时后 `generate.py` 才失败，报错还两边都印 `102` | 编码时原样存进 cache，`'102' != 102` 只在类型上，`str()` 印不出来 | `encode_samples.py` 读 manifest 时就拒绝非 int（不静默转换）；`generate.py` 报错改用 `repr` | [evaluation](features/evaluation.md) |
| 16:9@37（38k token）的稠密步报 `OutOfMemoryError: Tried to allocate 303.50 GiB` | `dense_attention` 的 sdpa 分支在 `return_lse=False` 时传 3 维 q/k/v，PyTorch 所有融合 SDPA kernel 都要求 4 维，dispatcher 静默退回 math 后端展开 `[H, S, S]`；`return_lse=True` 分支本来就是 4 维，所以只有不要 LSE 的纯稠密步会炸 | 无 LSE 分支加批轴 `[None]`；`tests/gpu/test_h3_attention_gpu.py` 用 38010 token 断言单次调用 < 8 GiB | [h3_model](features/h3_model.md) |
| 单测断言「Ω top-k 的 ε 小于 Mx top-k」在随机 Q/K/V 上反过来（1.97 vs 1.77） | 两层原因：`c=0` 时 `Σ_j ξ_uj = 0`，块间误差可抵消，按 `Σ‖ξ‖` 取 top-k 最小化的是三角不等式**上界**而非 ε；iid 高斯又让注意力近均匀，根本没有「该选哪些块」的信号 | H1 是实验要回答的问题，不是可单测的不变量。改成在构造信号上测机制：分数极小（`A`/`Mx` 排名近随机）但 V 范数按块差很多 | [sol_ablation](features/sol_ablation.md) |
| `torch.equal(out, dense_attention(...))` 报 "must be Tensor, not tuple" | `dense_attention` 即使 `return_lse=False` 也返回 `(out, None)` | 解包后再比 | [sol_ablation](features/sol_ablation.md) |
| 一行的 `μ`/`σ` 整行变成 NaN | 代理分数只在视频象限有定义，全局列存的是 NaN（故意的，全局列无条件保留、不参与排名），算矩时把它们一起纳入了 | 所有读 `proxy` 的地方只取前 `n_video_tiles` 列并按 `kv_ok` 过滤 | [sol_ablation](features/sol_ablation.md) |
| 块方差特征在某些 tile 上相对误差到 31% | 用一遍恒等式 `E[x²]−E[x]²` 算方差，而 RMSNorm 过的 key 在同一个空间块内彼此很近，相减两个几乎相等的大数 | `_second_moment` 改成两遍（先均值再减），按行分块限制 fp32 临时显存；单测用「各行几乎相同」的紧致 tile 钉死 | [veda2](features/veda2.md) |
| 打分器把只有 32 行真实 key 的补齐块和 128 行的满块排在同等位置 | 分数是均值池化的 QK，看不到 `valid_count`；而块的注意力**质量**与行数成正比。这在 max 目标下是自洽的，一换成质量目标就是系统性偏差 | 把 `log B_j` 加进 logits（layout 直接读，零成本）；实测补回 22~36% 的 oracle 差距 | [veda2](features/veda2.md) |
| 补偿的等算力对比，中位数之比说赢 14.7% vs 8.7%，逐头配对说只赢 1% | 两个臂的 ε 分布形状不同，「中位数之比」把差距放大了；配对才是同一个头自己和自己比 | 这类同成本对比一律用逐头配对（本例补偿只在 50.8~62.7% 的头上更好） | [veda2](features/veda2.md) |
| 预算分配器在目标等于实测最小密度时报「outside the grid」 | 网格用 `exp(log(lo)+...)` 生成，端点漂移到 lo 之上；而且 n 份精确的 0.05 求和再平均会比 0.05 大一个 ulp | 网格端点钉成 lo / hi；可行性判断用相对松量而不是裸 `<=` | [veda2](features/veda2.md) |
| 把 Veda2 的两个闭式项直接加到**训练过的**打分器上，质量反而变差，而且每个缩放系数都变差 | 用 max 目标训练会把基础项的 spread 压小（logit_std 3.78 → 2.65），携带物理系数的新项在训练过的基础上相对响了约 50%，把它盖住了 | 这两项必须在**训练时就在场**、并对着质量目标训，不能事后 bolt。`--base reset` 才是能组合的那个变体 | [veda2](features/veda2.md) |
| 装上 FA4 之后，三个 CPU 单测在 GPU 机器上变红 | `fa4.available()` 忽略 device 参数，`_fused` / `teacher_heat` 只问 kernel 能不能 import。于是有 GPU 的机器把 CPU 张量、或 kernel 不支持的 head_dim 路由进 kernel，死在里面 | 判定函数要对**张量**负责：`available(device)` 对非 CUDA 返回 False，`tile_gather_triton.supports` / `block_heat_triton.supports` 把 head_dim 与 stride 条件补上 | [veda2](features/veda2.md) |
| 换了打分器结构之后推理报 `log_count and count_term must be set together` | `SparseStudent` 与 `TeacherCollector` 不走 `TileScorePredictor.scores()`，而是直接调 `predictor.layers[i](...)`，因为 `_gather_and_pool` 用的是融合 gather kernel，只产出 [mean\|max\|min] | 新增 `_extra_features()`，在这两个调用点从已经 gather 出来的 tile 行补池化；加打分器特征时记得这两处 | [veda2](features/veda2.md) |
| 从单 SHA 浅克隆装 flash-attn-4，`fa4_sm8x.install()` 报 `0.0.1.dev1+g...; the SM8x patch is built on 4.0.0b32` | 版本来自 `fa4-vX` git tag，浅克隆没有 tag，setuptools_scm 回落到 fallback | 装的时候设 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=4.0.0b32`。真正的来源校验是那五个 vendored 模块的 sha256，版本字符串只是标签 | [veda_kernel](features/veda_kernel.md) |
| 二阶头从 rank 128 降到 16，端到端一点没变快（还是 4.2 s/step、注意力 0.7 s） | 按 FLOP 账以为瓶颈是 n²·rank 那一项，实际开销**全在池化**：融合 gather kernel 只写 `[mean\|max\|min]`，两个二阶矩是事后再调 `pool_tiles`，把 tile 序的 q/k 又读了几遍（中心化方差要两遍） | 把二阶矩塞进 `tile_gather_triton` 的 gather kernel，tile 本来就在寄存器里，连中心化方差都不要额外内存流量。融合后回到 3.94 s/step，与 Veda1 齐平 | [veda2](features/veda2.md) |
| 蒸馏的 KL 降了 65%，而真正在意的 recall 掉了 8 个点 | `seer_kl` 算的是 **forward** KL(teacher‖student)，覆盖型：教师哪里有质量学生就必须给概率，所以**摊平是最便宜的降损方式**。logit spread 为 σ 时 forward KL ≈ σ(1−t_max)，**最小化它等于压缩幅度**而不是学排序 | `seer_kl` 加 `direction`；reverse 是寻峰型，和 top-k 读出对齐，而且教师零 floor 之后有界（约 14 而不是 300） | [veda2](features/veda2.md) |
| lr 1e-3 在 12 步内把物理上正确的初始化覆盖掉了 | Adam 步长 ≈ lr，**与参数自身尺度无关**。`proj` 初值 1e-4、训练后 1.15e-2，所以 1e-3 是**每步 10 倍相对变化**。Veda1 能这么用是因为它跑 600 步 + warm start | 按「参数有用尺度 / 期望步数」定 lr；三类参数（1e-4 / 0.0625 / 1.0）差四个数量级，必须分 param group | [veda2](features/veda2.md) |
| 二阶头在 48 步内把 logit_std 推高 6 倍，损失自己在上升 | 项是 `so_q`·`so_k` 的双线性，对参数是**二次**的；`so_*` 初值 0.0625，每步漂 1.6%、48 步漂 77%，乘积动 3 倍 | `freeze_second_order`，或者改成 `g_h·(归一化项)` 把「多少」和「哪个子空间」解耦 | [veda2](features/veda2.md) |
| 早停用「trailing mean 不再创新高」，把一次只是**平**的长跑砍在第 50 步 | 那个判据分不清「还在路上」和「到终点了」：lr 1e-4 下 50 步只让投影走了有用尺度的一半 | 长跑用 `max_drop`（跌破起点）判**发散**，`patience` 只留给短诊断跑；两个判据都不给直接报错 | [veda2](features/veda2.md) |
| 刚 kill 掉的训练，`pgrep -f 'scripts/train.py'` 还报 RUNNING | pgrep 匹配到了**自己的 ssh 命令串** | GPU 作业用 `nvidia-smi --query-gpu=memory.used` 或 `ps -eo args \| grep -v grep` 确认 | [veda2](features/veda2.md) |
| 声称 `log B` 的收益「取决于补齐比例」，另一个几何上复测不出来 | `log B_j` 是每列常数，只在这些常数**彼此有差别**时才改变 top-k。预测量是 `std_j(log B_j)` 比基础分数 spread，不是补齐比例——而且补齐在各轴上是**乘性**的：16:9@37 两个轴都短，角落 tile 只有 8/128 行，std 0.664；双采几何只有一个轴短，最低 72/128，std 0.160，杠杆小 4 倍。补齐**比例**反而是 19.1% 对 9.6%，指向相反 | `solattn.count_term_leverage(grid, shape)`，纯静态、不需要 GPU，上几何之前先查 | [veda2](features/veda2.md) |
| 四个蒸馏目标连着失败，形状各不相同（压平 / 锐化 / 饱和 / loss 自己涨） | 同一个病：top-k 读出对分数的**逐行正仿射**完全不变，所以逐行偏移和逐行尺度是 kernel 结构上看不见的两个**规范方向**。损失只要对它们敏感，梯度就能在不改变任何一个选择的前提下降损失——那是最便宜的下降方向，它就会去那里 | 目标必须对逐行正仿射不变。把 attention 读成一侧的熵正则 OT，支撑集约束下的最优值是 `log sum_{v in S} exp(s)`，于是正确的损失是**期望保留质量**（`heatmap.transport_loss`）：逐行居中再除以标准差，两个规范方向上的梯度**恰好为零**，而且损失的最优点就是 `kept_over_ceiling` 这个指标本身 | [veda2](features/veda2.md) |
| `pkill` 之后以为旧训练死了，新的却报 `EADDRINUSE: port 29500`，旧的又跑了 14 分钟 | `pkill` 写在一条复合 ssh 里，那条命令非零退出把输出吞了；我拿一行旧的 "GPU idle" 当成了确认。rsync 把源文件换掉对**已经 import 完的进程**毫无作用，所以它一直在跑旧目标 | kill 必须**分三步独立确认**：`ps -eo pid,args` 拿到 PID → `kill -TERM` 再 `-KILL` → **单独一条** `nvidia-smi --query-gpu=memory.used` 读到 0 MiB。另外 torchrun 被杀后会占着 rendezvous 端口，启动一律显式给 `--master_port`（`scripts/launch_training.sh` 的 `PORT=`） | [veda2](features/veda2.md) |
| 蒸馏目标换对了（损失=指标），AdamW lr 1e-3 仍然在 14 个 update 内把四个几何全做坏 | 这是**找尺度**的问题：base projection 初值 1e-4、有用尺度 1.15e-2（115 倍），而 AdamW 的步长 ≈ lr，**与权重当前大小无关**，lr 5e-4 打在 1e-4 的权重上是每步 5 倍相对改变 | 用 Muon（`miowtion/train/muon.py`，`optimizer: muon`）：正交化后把更新 RMS 显式标定到 `lr × muon_rms`，步长是配置的性质而不是梯度大小的函数，`update_rms` / `update_align` 入日志。打分器是纯粹的每头矩阵栈，是 Muon 唯一适用且不用拆优化器的形态 | [veda2](features/veda2.md) |
| 把 `heat_kept/heat_ceiling` 当成训练目标的指标，跨几何看趋势 | 它计入**强制对角**（kernel 规则，不是打分器的决定），而对角块占多少质量是**片子的性质**。实测：同一批打分器、对角质量在六条 clip 间变化，池化 Spearman 从 0.9997 掉到 0.9937、逐对反向从 0.3% 升到 3.1%——也就是它比目标本身更不可跨 clip 比较，而轮转几何的训练正是在跨 clip 比较 | 看 `mask_diagnostics` 新报的 `retained`：排除对角、逐行比值再平均，定义上就是 $\tau\to0$ 的 `-transport_loss`，单测钉在 1e-4。ceiling 仍然要单独看，它说的是教师够不够集中 | [veda2](features/veda2.md) |
| 按「`proj` 要从 1e-4 走到 1.15e-2，是 115 倍旅程」来定步长和早停阈值，结果六次运行全部在覆盖一个已经正确的项 | `LayerPredictor.embed` 是**残差**：`feats @ proj + feats[..., :D]`。所以 **`proj = 0` 时 logit 就是闭式零阶项**，`proj` 是修正量而不是要跑完的路；1.15e-2 只是 Veda1 在另一个目标下学到的修正量大小。实测那一臂第 25 步 `proj_q` RMS 已到 1.65e-2（越过训练后尺度），`update_align` 0.162→0.087，逐几何单调下降，连被优化的损失自己都在上升——而沿 $-\nabla$ 的局部检验证明梯度是对的 | 按**修正量相对残差的大小**定步长：Muon 步长 RMS = `lr × muon_rms`，取 1e-4（= 初始化尺度、训练后修正量的 1%），即 lr 5e-4。momentum 从 0.95 降到 0.5（0.95 的缓冲跨 20 个 update，而几何每步都换且梯度近乎正交）。早停阈值从 0.35 收回 0.06：没有谷底要穿 | [veda2](features/veda2.md) |
| 换成 hybrid 优化器之后，`update_rms` / `update_align` 整整一个 run 都记成 null | 记录那两个数的判据写的是 `isinstance(self.optimizer, muon.Muon)`，而 hybrid 把 Muon 包在 `optim.Hybrid` 里，判据静默变假 | 判据改成问**能力**不问类：`hasattr(self.optimizer, 'last_update_rms')`。这两个数是唯一能实测步长是否符合配置的读数，而步长正是六次训练失败的那个量，所以它不能静默消失（单测钉住 plain / hybrid 都有、AdamW 没有） | [veda2](features/veda2.md) |
| `retained` 在四个几何上都显著上升，而被优化的 `transport` 在两个几何上显著变差 | $\text{slack}=\texttt{retained}+\texttt{transport}$（放松的代价）以 t = 7~12 在四个几何上全部增长，而 `logit_std` 没有趋势——不是尺度跑飞，是**切点附近的分布形状被重塑**。根因：**top-k 对逐行任意严格单调映射都不变**，而 `transport_loss` 只固定了正仿射这个子群；剩下的非线性单调自由度仍是 kernel 看不见的规范，梯度又恰好集中在切点。实测五个保序映射：硬值逐位恒为 0.9141，软值在 0.632~0.841 之间摆 | 判断训练**只看 `retained`**，不看记录的损失。这是方法下界而非 bug：唯一对单调映射不变的统计量是秩，而秩的梯度几乎处处为零；把 sigmoid 宽度改成次序统计量定义，四个重塑里改善三个、在锐化尾部上更差 | [veda2](features/veda2.md) |
| 拿 Sol-Attn 当基线连跑两轮 demo，结论对我们有利，其实两轮 Sol 都只拿到 1.4~1.67% 的预算 | Sol 的路由是**阈值**式的（`mean + tau*std`），tau 决定的是阈值、**不是预算**，它实际路由多少块是激活的性质。我把 `sol_tau: 1.2` 当成了一个稀疏度设置，而它在 H3 激活上只给 1.4~1.67%，对面 Veda 是 5%。标定到 tau 0.7（4.82%）重跑，t37 上结论**反向**：Sol 在 PSNR 上赢 11/11、SSIM 赢 10/11 | 阈值式路由器**必须先标定再比较**，而且发表的是**实测密度**不是目标值（`MIOWTION_SOL_DENSITY` 传裸 tau 就会报实测）。更一般地：等密度≠等成本，同密度下 Sol 的注意力是 Veda2 的 2.11 倍（补偿成本随全部块数走），所以等密度表只是中间量，结论要看等时间 | [veda2](features/veda2.md) |
| 质量表一直只覆盖 12 条 demo 里的 10 条，两个几何从没进过任何一张表 | `generate.py` 在一次运行只含**单个**样本时会把 clip 目录折叠掉，直接写 `<arm>/<mode>.mp4`；而 `quality_vs_dense` 只 glob 三层的 `<arm>/<clip>/<mode>.mp4`。少掉的是 16:9@72 和 9:16@102（以及 t37 的 9:16），**没有任何报错**，表上只是 n 小了 | `_arm_clips` 两种布局都读，折叠的那种从该臂自己的 `summary.json` 取回 sample / geometry；`summary.json` 缺失直接 `raise`（静默丢 clip 比停下来更糟）。单测把两种布局和「同一 mode 来自两个臂」都钉住。补回来之后发现 9:16 的 SSIM 只有 0.38 而 16:9 是 0.66，几何间难度差异极大，更不能让它缺席 | [veda2](features/veda2.md) |
| 9:16 的两条 sol 臂都在 VAE 解码时被 SIGKILL（rc=137），日志里**没有 traceback**，而同几何的 veda / dense 臂解码正常 | 没有 traceback 说明不是 CUDA OOM（那会抛异常），是外部 SIGKILL。**实测的根因是余量：**一个带 offload 的 H3 运行期间 cgroup 的 `memory.current` 稳定在 **107.1 GB / 限额 110.0 GB**，只剩 2.9 GB，所以解码那一下的任何额外分配都在悬崖边上。（我最初归因于 `MIOWTION_SOL_DENSITY` 抬高了常驻内存——**那个解释不成立**：后来测到不开探针的 veda 臂同样是 107 GB。探针顶多是压上去的最后一点。）像素数也不是原因：16:9 是 1344x768、9:16 是 768x1344，一样大 | 两条结论。**一、去噪已完成、latents 已落盘的任务用 `generate.py --decode-only` 重跑**：它直接读 `<mode>_latents.pt`，完全不建 DiT，省掉 8 分钟加载，内存占用也低得多——不要重排一次完整运行。**二、同一张卡上的队列必须串行**：2.9 GB 的余量装不下第二个 H3 进程（单进程 RSS 32.7 GB，还要加上权重的 page cache），所以「并行跑两个任务把墙钟减半」这个想法在这台机器上不可行，先量 `memory.current` 再说 | [veda2](features/veda2.md) |
| 五个 Sol 臂的耗时全部虚高 1.22×，而且这个数进了论文和反驳文件 | `kernels/sol.py` 的 `attention()` 在 `MIOWTION_SOL_DENSITY` 下会在 kernel **之前**再跑一遍 `density()`（上游的 `prepare`），而 `_TimedAttention` 包的是整个函数——**探针在计时区间里面** | 探针类代码必须在计时区间**之外**，或者至少让报告的数区分「含探针」和「不含」。重测：Sol 17.09 s 而不是 20.87 s，对 Veda2 是 1.73× 不是 2.11×。质量不受影响（探针不改输出），所以只有耗时类结论要改 | [veda2](features/veda2.md) |
| 微基准说「INT8 比 BF16 慢 1.78×」，而这在物理上说不通 | 三个错叠在一起：①卡上同时在跑 H3 去噪（同一个 FA4 调用两次测出 6.27 和 13.23 ms）；②基线拿对方的 Triton INT8 比我们的 FA4 CuTe BF16，等于让补偿背上 kernel 语言和精度两笔账；③两个 kernel 都跑出厂默认 launch 配置，而它们依 GPU 而异（对方自带 `tools/tune_int8.py` 正因如此） | 微基准前先 `nvidia-smi --query-compute-apps` 确认卡空闲并把它打进日志；比值只在**同一实现内部**取；两边都调优（实测 1.58× 和 1.41× 的差距，比补偿本身还大）。修正后 INT8 在 20% 密度上**反超** BF16 | [veda2](features/veda2.md) |
| 追了两个假设去解释「闪烁」，都不成立，而真正的原因看一眼画面就知道 | 假设一是 tile 的时间跨度（周期性得分全在 1.0 附近，没有 tile 周期），假设二是打分器（`veda1 − veda2` 的差不显著）。**低预算改变的是生成内容本身**：镜头构图和剪辑点都变了，看起来像闪。Sol 同样偏离 dense，但它自洽所以不闪 | **先抽帧看画面再建模型**。`ffmpeg -vf select=...` 几秒钟就能拼出对照表，而我在标量上绕了两轮。另外「闪」和「忠实于 dense」是两回事，用户要的是后者 | [veda2](features/veda2.md) |
| 第一版闪烁指标说 Sol 的峰值 72.2 最高，而 Sol 恰恰是不闪的那一个 | 均值和峰值被**镜头切换错位**主导：切点挪几帧，那一次转场的帧差就吞掉整条曲线。而且单条 clip 的差距（7 倍）在 11 条聚合后只剩 23% | 时间维指标一律用稳健统计（中位数 / p90 / 去掉最大 5% 后的均值），并且**必须跨 clip 配对**再下结论 | [veda2](features/veda2.md) |
| 四次「不同 plan」的实验跑出逐像素完全相同的视频 | `generate.py` 在给了 `--predictor` 时从 bundle 里取 plans，**`--plan-dir` 被无声丢弃**——那个分支根本没读这个参数 | 显式参数要么生效要么报错，不能静默忽略。已修（覆盖时在 source 里写明），并加单测直接检查该分支读了 `--plan-dir` | [veda2](features/veda2.md) |
