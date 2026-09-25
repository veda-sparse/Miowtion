# 训练框架（FSDP2）：打分器训练与 LoRA 恢复

## 目标
在冻结的 H3 DiT 上训练 Veda 打分器（阶段 1）。可选阶段 2：用 LoRA 恢复稀疏学生的质量。两个阶段都
支持 FL2VA（t2va/fl2va）和 Ref2VA（ref2va）两套权重。

## 设计与不变量

### 流程：先编码，再训练
1. **prompt 扩写**（见 `prompt_expansion.md`，另一个 agent 负责）→ 结构化 prompt 的 jsonl。
2. **离线编码**（`scripts/encode_samples.py`，单独的任务）：text encoder（仓库自带的 Qwen3-VL，
   截断到 50 层并去掉最终 norm）+ 条件 latent（仓库自带的 VAE）写入样本缓存。训练进程里不加载
   text encoder。
3. **tile 搜索** → `plans/*.json`。
4. **DiT 训练**（`scripts/train.py`）。

### 教师与调度（`miowtion/train/teacher.py`）
- 当前目标是**合并少步 LoRA 后的 8 步教师**（Turbo LoRA v4_step600_ema）。4/8/50 步三套配置都有：
  `configs/{search,stage1}_{turbo4,turbo8,base50}*.yaml`。
- `schedule: base` 使用发布版的 50 点网格（49 步，fp32 linspace）。`schedule: turbo` 使用少步网格：
  视频 `σ_i = shift(1 − i/n, 12)`，音频 `shift(unshift(σ_v, 12), 3)`，按 double 闭式计算，与 Turbo
  采样器一致。8 步网格：视频 1, .988, .973, .952, .923, .878, .800, .632, 0；音频 1, .955, .900,
  .833, .750, .643, .500, .300, 0。
- 适配器合并：trunk / refiner / final_layer 的 AdaLN 就地合并 `W += B@A`（fp32 计算，GPU 上算
  delta，再转回 bf16；FSDP 分片只合并本地行）。block 的 `adaln_proj` 已经被表替代，它们的 delta
  在建表时合并进权重。每个 adapter 项都必须有去处，否则报错。"AdaLN 走表"与"全部权重直接合并"
  两条路径的前向逐位相等（单元测试）。
- 注意：Turbo 在 ComfyUI 里默认在运行时施加 LoRA（`W x + B(A x)`）；我们按 `W += B·A` 合并，
  数值上有 bf16 舍入差异，但语义相同。

### 采样器即训练循环（`miowtion/train/trajectory.py`）
- 没有视频数据集。每条轨迹从纯噪声出发，用教师自己的 Euler 步推进；轨迹上每个状态都是一个
  micro-step。**轨迹始终用教师速度推进**（阶段 2 也一样）。
- 少步教师必须在它自己的网格上推进（8 步：σ ≥ 0.632）。
- 几何由**所有 rank 共享、不含 rank 项**的生成器抽取；样本和噪声种子按 rank 各自抽取。
  `geometry_sampling`：`uniform`（每条轨迹独立均匀抽取，默认）或 `cycle`（每一轮按打乱的顺序把
  每个几何各走一遍，混合更均衡；轮次顺序只由 (seed, 轮号) 决定，所以状态就是抽取次数，恢复是精确的）。
- `trajectory_steps`（只用于冒烟测试）：每条轨迹只走前 N 步就换下一条，用来在几次 update 内走遍
  很多几何。正式训练必须走完整条轨迹（默认 None）。
- fl2va 样本的关键帧 latent 绑定画布宽高比；prompt 的分镜时间绑定时长（`latent_t`，由 PE 记录）。
  采样器只挑与当前几何兼容的样本。

### 阶段 1（`miowtion/train/trainer.py`）
- trunk 冻结，全程 no_grad；attention 使用 `TeacherCollector`：先算稠密输出（带 LSE），再对每个头组
  训练打分器，每层的 KL **当场反向**（按头数加权，除以层数和 accum），不在层之间保留计算图。
  输出与稠密教师逐位一致（已在真实权重上验证）。
- 优化器 AdamW(0.9, 0.95)，wd 0，线性 warmup；**梯度按打分器的每一层单独裁剪**（lora / head 各
  自整体裁剪，见下面「逐层裁剪」）；
  EMA 0.995。复制参数（打分器、输出头）的梯度拼成一个 fp32 缓冲区后做一次 all-reduce；缺失的梯度
  补零，否则各 rank 的缓冲区长度不一致，集合通信会挂起。

### 阶段 2（可选）
- 可训练参数：打分器 + trunk qkv/out 投影的 LoRA（rank 64，切分前挂到原 Linear 上，用 forward hook
  注入，基础权重名不变）+ 输出头。
- 冻结教师：关闭 LoRA，并把输出头换回开始时保存的快照。学生：开启 LoRA，使用 `SparseStudent` 和
  逐 block 梯度检查点。学生前向**不收集 KL**（检查点重算时会重复执行）。
- 必须有 FA4 块稀疏 kernel（SM90/100，或打了 vendored 补丁的 SM8x）才会启动；拒绝退回参考 kernel。
- 必须从阶段 1 的 checkpoint（EMA 权重）初始化。

### 监控与梯度截断
- 每次 update 在优化器所在的张量上（offload 时是 host master，不占显存）记录：打分器每层梯度范数、
  本次与上次 update 梯度的余弦（每层和整体；接近 0 说明这个 batch 大小下梯度以噪声为主）、
  优化器一步的相对更新量 ‖Δw‖/‖w‖。写在 `log.jsonl` 的 `grad_norm_layers` / `grad_cos` /
  `grad_cos_layers` / `update_ratio`（`monitor_updates: false` 关闭）。
- 第一次 update 时检查梯度截断：可训练集合以外的参数（冻结的 trunk）只要有 `.grad` 就报错。
  打分器只读取 detach 过的激活（池化在 no_grad 下、融合 kernel 的输出本身不带梯度）。

### 显存与性能（大模型、小显存）
- **AdaLN 表**（`miowtion/train/adaln.py`）：冻结 trunk 时，50 个 `adaln_proj`（13B 参数）只把
  timestep 映射成调制向量。启动时从 checkpoint 逐 block 读取，对调度中出现的所有 timestep 集合计算
  好（49 步约 0.94 GB），之后把这些投影从模型中删除，trunk 从 33B 降到约 20B。
  表的结果与实时计算逐位一致（单元测试 + 真实权重下 `equals_dense`）。
- **FSDP2**：每个 block 一个切分单元；根模块也调用 `fully_shard`，但通过 `ignored_params` 让
  embed/输出/打分器保持复制。根模块必须是 FSDP 模块，否则跨 block 预取会报
  `FSDPCommContext has no all_gather_copy_in_stream`。
- **部分 offload**：`offload_blocks=N` 按 Bresenham 均匀间隔选 N 个 block 放到 pinned 主机内存，
  其余常驻显存；`prefetch` 控制前向预取的 block 数。offload 的 block 必须在 CPU 上物化
  （`to_empty(device='cpu')`）。
- **MLP 行分块**：`mlp_chunk_rows` 限制 `[rows, 2×14336]` 中间张量的大小。GEMM 的行数会影响
  数值，所以搜索、训练、评估必须使用同一个值。
- 阶段 2 的重算：逐 block 的非重入检查点。

2×RTX 4090 实测（5 s 16:9，38010 token，每 rank 一个 clip）：

| 配置 | 加载 | AdaLN 表 | 稠密前向 | 阶段 1 教师前向 | 显存峰值 |
|---|---|---|---|---|---|
| 50 个 block 全 offload | 132 s | 65 s | 33.5 s | 39.2 s | 11.3 GiB |
| offload 30、预取 1 | 45 s | 10 s | 29.3 s | 33.6 s | 19.1 GiB |
| offload 24、预取 2 | — | — | OOM | — | >23.5 GiB |

### 数据（`miowtion/train/data.py`）
- 样本缓存：`index.json` + `text.safetensors`（hidden bf16、tags int8）+ `cond.safetensors`
  （干净的条件行 fp32）。训练时在主机上按需切片。
- prompt 校验按 task 区分：t2va/fl2va 为三字段 + `[Shot 1]`；ref2va 为六段式（`subject_definitions`
  → … → `non_diegetic_music`）。格式不对直接报错，不会静默丢弃。
- 固定种子的 train/test 划分（默认 20 条测试），训练、评估、渲染共用同一个函数。

### checkpoint（`miowtion/train/checkpoint.py`）
- 内容：权重和 EMA（完整张量）、Adam 状态（按参数名）、每个 rank 的采样器位置和噪声生成器、
  共享生成器（保存前跨 rank 校验哈希一致）、步数、配置。
- 写盘：本地临时文件 → 原子重命名 → 带文件大小的完成标记 → 后台复制到持久目录；退出前等待复制
  完成。没有完成标记的 checkpoint 不会被加载。
- 分片张量（LoRA / EMA / Adam 状态）的 `full_tensor()` 是集合通信，必须在**所有 rank** 上完成后，
  rank 0 才能单独写盘。
- 恢复（resume）和初始化（init，只加载权重、严格、可显式丢弃前缀）是互斥的两种方式。world size
  变化后，新的第 r 个 rank 继承旧的第 r % old 个 rank 的采样位置；生成器状态只恢复前 old 个 rank 的。

## 代码位置与接口
`miowtion/train/{parallel,adaln,data,trajectory,trainer,checkpoint,lora,encode}.py`；
入口 `scripts/{train,encode_samples,smoke_dit}.py`；配置 `configs/*.yaml`（字段见 `TrainConfig`）。

## 测试
- `tests/unit/test_train_pipeline.py`：prompt 校验（两种格式）、样本缓存读写往返、AdaLN 表与实时
  计算逐位一致、tiny DiT 上阶段 1 训练 3 步 + checkpoint + 恢复（CPU 端到端）、裁剪分组是每层一个
  且某一层梯度放大 1000 倍时其他层的梯度逐位不变（同样的尖峰在全局裁剪下会拖累所有层）、学习率表
  （常数路径是 warmup 后持平、cosine 在 warmup 结束处不跳变且单调落到下限、非法的
  `lr_decay` / `lr_min_ratio` 被拒）。
- 真实权重：`scripts/smoke_dit.py`（结果见上表，2026-09-23）。

## 多几何混训（`configs/stage1_turbo8_multigeo_4090.yaml`）
- 12 种几何：1:1 / 4:3 / 16:9 / 9:16 × 5.17 / 10.1 / 14.4 s（latent_t 37 / 72 / 102），
  `geometry_sampling: cycle`，每种几何用自己的方案表（9:16 由 16:9 镜像），从单几何打分器热启动。
- 样本：MovieGenVideoBench 扩写集中每个 latent_t 各 150 条（10 s 的 150 条单独扩写）；另外随机
  留出 20 条（随机宽高比与时长）只用于最终的稠密 / 稀疏对比，不参与训练。
- lr 1e-3（用户指定），warmup 20（lr 是参考配方的 10 倍，第一步不能把权重整个替换掉），accum 2
  （每次 update 4 个状态）；warmup 之后是常数 lr。衰减见下面的「学习率与分阶段」。
- checkpoint：每 50 次 update 存一次，**一个都不删**。权重 + EMA 是训练历史（打分器 275.25M
  参数 → 1.03 GiB + 1.03 GiB），2000 次 update 共约 82 GiB。Adam 的一阶 / 二阶矩另外占
  2.05 GiB，单独写在 `optim.pt` 里，只有最近 `keep_optimizer: 2` 个 checkpoint 保留（恢复只会从
  最新的那个开始），更老的只删 `optim.pt`，目录和 `state.pt` 永远保留。
- `teacher_q_tiles: 1.0`：监督每一个 query tile。抽样（旧默认 0.25）虽然无偏，但把梯度的方差和
  诊断量的噪声都放大了；教师热力图本身是 O(S²·D)，抽样省下的那部分不值得用训练信号去换。
- `offload_blocks: 44`：104k token 时 30 / 40 都会 OOM（先后在 RoPE、TeacherCollector、AdaLN 调制
  处，均已分块修复），44 时峰值 20.9 GiB。多卡时 FSDP 的 copy-in 流让拷贝与计算重叠。
- 冒烟（`configs/stage1_turbo8_multigeo_smoke_4090.yaml`，每个几何 1 步）：micro-step 1:1 t37
  16 s、16:9 t37 31 s、1:1 t72 36 s、4:3 t72 54 s、1:1 t102 63 s、4:3 t102 81 s、16:9 / 9:16
  t102 150 s；初始 KL 0.94–1.64，recall ≈ 0.52–0.54。

## 逐层裁剪（`trainer._clip_groups`）
50 个层打分器是 50 个互相独立的模型：参数各自独立、各有一个 KL 项、层与层之间没有梯度往来。
所以裁剪必须逐层做，全局裁剪是一种耦合——而且是不对称的耦合：

- 实测第 49 层的梯度范数是中位层的 **53–71 倍**，占整个打分器梯度能量的 **98%**（阶段 A 第 600
  次 update：l49 = 0.324，中位 = 0.0061）。它的 KL 也高得多（起步 6.23、其他层约 0.5；600 步后
  降到 0.92），说明最后一层的教师分布确实更难拟合，不是坏了。
- 阶段 A 全程全局范数 max 0.894，`grad_clip` 是 1.0，**一次都没触发**，所以历史上没有污染过别的
  层。但余量只剩 12%，而全局范数的最大值几乎就等于第 49 层的最大值（0.886）：只要第 49 层抖一下
  越过 1.0，其余 49 层的步长会被同一个系数一起缩小，纯粹是无关层受牵连。
- **不需要拆优化器**：AdamW 是逐元素的、`weight_decay = 0`，一个 AdamW 管 50 层与 50 个各自的
  AdamW 在数学上完全等价（Adam 的二阶矩本来就是逐参数的，不跨层归一）。唯一的耦合就是那个全局
  裁剪，改掉它就够了。
- 日志里的 `grad_norm.predictor` 仍然是一个数（各层范数的平方和开根，`summarize_norms`），逐层的
  值在 `grad_norm_layers` 里，语义不变。

## 学习率与分阶段（`trainer.learning_rate`）
- 学习率永远是「线性 warmup × 衰减系数」的乘积，不是分段拼接：`lr_decay: none`（默认）时
  warmup 之后恒定；`lr_decay: cosine` 时 cosine 从 warmup 结束处开始、到第 `steps` 次 update
  落到 `lr_min_ratio × lr`。两者相乘而不是接续，是为了让衔接处没有跳变（warmup 期间 cosine
  的自变量被截到 0，系数恒为 1），并且恢复训练时只看绝对的 `step`，不需要额外状态。超过
  `steps` 继续训练时钳在下限，不会翻上去。
- **阶段 A（`stage1_fast_t37_4090.yaml`）只训 latent_t 37、常数 lr 1e-3**：一次 t102 update 的
  代价是 t37 的 5–8 倍，只换来同样的一次梯度，循环 12 种几何会把约 70% 的墙钟花在 1/3 的
  update 上。
- **阶段 B（`stage1_refine_t102_4090.yaml`）只训最长的 latent_t 102 的四个宽高比、
  lr 1e-3 cosine 衰减到 1e-4**：从阶段 A 第 600 次 update 接着训 600 次。不混长度的理由是
  预算：step-600 的对比显示阶段 A 的打分器已经能外推到 t72 / t102，而 t102 正是加速比最高的
  一档（端到端 2.76× vs t37 的 1.57×），所以把全部墙钟花在没训过、收益也最大的长度上；再混
  t37 只是在重新拟合已经拟合好的东西。t72 靠两个训过的长度之间的内插。
- 阶段 B 要衰减的理由：这一阶段打分器已经收敛，只是去适配更长的序列。常数 lr 下按几何循环时，
  最后一个几何的梯度会直接决定最终权重，衰减让尾部变成平均而不是覆盖。batch 取 4（accum 2）
  而不是阶段 A 的 8：一次 t102 micro-step 要 63–150 s，batch 8 的 600 次 update 会超过三天。

## 训练动态的诊断量（`miowtion/veda/{attention,heatmap}.py`）
每个 micro-step 记录，日志里每 `log_every` 步汇总一次：

| 字段 | 含义 | 怎么看 |
|---|---|---|
| `kl` | 所有层按头数加权平均的 seer KL | 主损失 |
| `kl_layers` | 逐层 KL（只是 rank 0 自己的 micro-step） | 看深度上哪些层落后；跨 rank 平均只会把几何差异抹平 |
| `logit_std` | 打分器 logits 在 key tile 维上的标准差 | 塌缩检测：打分器给所有 tile 打一样的分时趋于 0 |
| `heat_kept` | 预测掩码保住的教师块热量占比 | 真正关心的量：掩码丢掉了多少注意力质量 |
| `heat_ceiling` | oracle（按教师热量取 top-k）保住的占比 | 当前预算下的上限。`heat_ceiling` 本身低说明教师不够集中，训练救不回来 |
| `recall` | 预测掩码与 oracle 掩码（去掉对角）的交集 / oracle | 与 `heat_kept` 互补：不加权的命中率 |
| `topk_bce` | oracle top-k 的平衡 BCE（`topk_weight > 0` 时才有） | 直接看"块在不在 oracle 集合里"这个目标；ln 2 ≈ 0.693 相当于无信息 |

代价：这些量都在 **÷128 的压缩 tile 网格**上算（103k token 时是 `[28, 810, 810]` ≈ 18M 元素），
相对产生它们的 O(S²·D) 教师热力图约 1e-6，可以忽略，所以 `recall_every: 1`，每层都算。真正的开销
不是 FLOPs 而是**设备同步**：原先每层每头组都 `kl.item()`，并且 `mask_recall` 里对设备张量做了
Python 分支。现在全部累加成 0 维设备张量，每个 micro-step 只有 `LayerStats.resolve()` 里的一次
`torch.stack(...).cpu()`。

`topk_weight`（默认 0）把 `heatmap.oracle_bce()` 按该权重加到 KL 上。KL 拟合教师的整个分布，而
kernel 只读 top-k 的排序，两者会分开；见 `docs/features/veda_predictor_mask.md` 的 A/B 记录。

## 踩坑记录
- **诊断量里的 `.item()` 比诊断量本身贵得多**：逐层 `.item()` 会把 CPU 和 GPU 串起来，让前向失去
  流水；诊断张量一律留在设备上，一个 micro-step 只做一次传输。
- CPU offload 要求被 offload 的 FSDP 参数先在 CPU 上物化，否则报 "FSDP parameters should be
  materialized on CPU"。
- 没有 FSDP root 时，`set_modules_to_forward_prefetch` 报 `FSDPCommContext` 缺少
  `all_gather_copy_in_stream`。
- 复制参数手动 all-reduce 时，缺失的梯度必须补零，否则各 rank 缓冲区长度不一致，通信挂起。
- DTensor 的 `full_tensor()` 必须在所有 rank 上调用，之后才能提前 return 让 rank 0 单独写盘。
- **FSDP2 在 world size 1 时预取完全不起作用**：`unshard()` 在单卡时直接返回，H2D 拷贝在
  `wait_for_unshard()` 里、在计算流上、在 block 开始时才做，offload 的 block 每个都停 ~59 ms
  （4090 上 16 s 的一步里有 2.4 s）。单进程改用 `parallel.BlockStreamer`：offload 的 block 不交给
  FSDP，参数放 pinned host 内存，前向 pre-hook 在独立的 copy stream 上用 non_blocking 拷贝预取下一
  个 block，计算流只等自己 block 的 event；`record_stream` 防止显存被提前复用。输出与 FSDP 路径
  逐位相同。
- **H2D 带宽受限于 GPU 的 PCIe 链路**：x4 链路上 pinned、pageable、整块 slab 的拷贝都是约 6 GB/s，
  拷贝方式无关，链路才是上限。所以 `BlockStreamer` 的改进目标是"藏在计算后面"和"不让分配器抖动"，
  而不是提高单次拷贝的带宽：offload 的 block 打包进 pinned `HostSlabs`（每个 block 每种 dtype 一个
  连续缓冲区，一次 DMA），设备端用 `prefetch + 1` 个预分配的环形缓冲区（不经过 caching allocator），
  环形槽位只在计算流用完上一个 block 后才被覆盖（event）。`parallel.replicate` 在另一张卡上复制
  常驻参数，offload 的 block 与源模型共用同一份 pinned 主机内存，多卡推理不增加主机内存。
- 显存太满时 caching allocator 反复 "memory mapping failed" 并重试（会同步），预取的重叠就被抵消了
  （稠密推理 40 个 block offload 时只快了 0.3 s）。常驻 block 要给预取的拷贝留出余量。
- **`SparseStudent` 把全部 tile 的 logits 交给了只接受视频 query 行的 `select_video_blocks`**：
  有文本等全局 tile 时两者行数不同（330 vs 336），第一次在 GPU 上跑稀疏推理时报错。原来没有任何
  测试覆盖 `SparseStudent`。现在只取视频行，新增的 CPU 测试用带全局 tile 的真实打包布局，并检查全
  保留预算时与稠密一致。

- **只训 t102 时 offload 44 不够，分配器在几何切换处清缓存重试**：12 几何混训时 44 是够的
  （104k token 峰值 20.9 GiB），但那种配比下 t102 每 12 条轨迹才来一次；只训 t102 时每次几何
  切换都是大张量换大张量，日志里出现
  `CUDACachingAllocator ... memory allocation failed with OOM ... (free: 0.6 GiB)`——
  1:1（60k）换到 4:3（78k）就要 3.4 GiB 连续显存。这是 **告警不是崩溃**（分配器清掉缓存后重试
  成功），但它说明已经没有余量，再往上到 16:9 / 9:16 的 104k 必然硬 OOM。对策：`offload_blocks`
  提到 48，并用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启动——失败的是连续性
  （碎片）而不是总量，expandable segments 正好治这个。`mlp_chunk_rows` **不能**动：GEMM 的行数
  影响数值，必须与 tile 搜索时一致。

## 验证记录
- 2026-09-23 macOS CPU：unit 全部通过（阶段 1 端到端 + 恢复；optimizer offload 与不 offload
  逐位一致；Turbo 适配器下表路径与全合并路径逐位一致）。
- 2026-09-23 2×4090，50 步 base 教师，阶段 1 冒烟（10 条 PE prompt，5.17 s 16:9，冒烟方案表，
  accum 2，optimizer offload）：3 次 update，KL 1.179 → 1.072 → 1.024，recall 0.477 → 0.495 →
  0.508；每个 micro-step 约 31 s，每次 update 约 64 s，显存峰值 19.1 GiB。
- 2026-09-23 2×4090：真实 FL2VA 权重，稠密/教师前向正常，教师输出与稠密逐位一致，初始 KL≈1.36，
  recall≈0.44。

## 待办
- 阶段 1 在真实权重上的多步训练曲线（需要 prompt 语料，PE agent 正在准备）。
- 更长的 clip（15 s，约 10 万 token）在 24 GB 卡上的显存方案（进一步分块 qkv / 更多 offload）。
- ref2va 条件编码（参考视频的 VLM presentation、音频 VAE）以及 ref2va 的训练冒烟。
- 阶段 2 需要 SM90/100 的机器（或者 SM89 上的 FA4 块稀疏，subagent 正在开发）。
