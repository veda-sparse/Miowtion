# 推理：少步 LoRA 教师 + Veda 稀疏

## 目标
用合并了少步 Turbo LoRA 的 H3 DiT 生成音视频：稠密注意力（教师）或 Veda 块稀疏注意力（训练好的
打分器 + 方案表 + FA4 块稀疏），同一个 seed 下两者并排对比，作为稀疏效果的可视化验收依据。

## 设计与不变量（`miowtion/infer/`、`scripts/generate.py`）
- 去噪循环直接用训练的轨迹（`train/trajectory.Trajectory`）：同样的打包布局、初始噪声、AdaLN 表和
  双时钟 Euler 步。所以稠密推理就是打分器训练时看到的教师 rollout；Veda 推理只换注意力函数
  （`veda/attention.SparseStudent`），`--dense-steps` 指定的步保持稠密。
- 音频速度：Turbo LoRA 的参考生成器在前向里把音频速度乘上 dσ_audio/dσ_video，采样时再除回去，
  净效果是"音频 σ 增量 × 原始速度"，与我们轨迹的做法一致，不需要额外缩放。
- 打分器有两种来源：训练 checkpoint（`--checkpoint` + `--plan-dir`，默认 EMA 权重），或者**部署用的
  bundle**（`--predictor`，`miowtion/veda/bundle.py`）。方案表按几何用 `PlanTable.select` 选，
  必须与训练时同一套方案。
- **bundle：权重和 tile 方案打包在一起**（`scripts/export_predictor.py` 生成）。训练 checkpoint
  不是部署件：它带着 EMA 影子（推理永远不读，却占一倍体积）、是 `torch.load` 的 pickle，而且
  **完全没有记录自己是在哪套 tile 形状上训练的**——那些在另一个方案目录里，要单独带上并且名字要对。
  配错是静默的：打分器照样出分，只是那是它没见过的 tiling 的分。bundle 是一个 safetensors 文件，
  张量是 live（非 EMA）权重，`__metadata__`（safetensors 定义为 str→str）里放整张方案表的 JSON
  以及 `keep_ratio` / 来源 checkpoint / step，于是几何、tile 形状和权重一起走，配错在构造上就不可能。
  导出时选 live 还是 EMA 只决定一次，记在 `source_weights` 里。
- **bundle 默认存 bf16**。`LayerPredictor.embed` 对取出的投影显式 `.float()`，打分算术无论存什么
  dtype 都是 fp32，所以 fp32 落盘只是把文件和**每个推理 replica 各自常驻在卡上的那一份**都翻倍
  （50×56×384×128×2 = 275M 参数，fp32 1.03 GiB，bf16 0.51 GiB；三卡三份就是 3 GiB 对 1.5 GiB，
  正好压在 24 GB 卡最紧张的地方）。dtype 记在 metadata 里，没有这个键的旧 bundle 按 fp32 读。
  **`load_state_dict` 是往已有参数里 copy_，会按参数的 dtype 转换**，所以 `load()` 必须先把模块
  `.to(stored)` 再加载，否则 bf16 文件被静默升回 fp32，省下的显存又还回去了。
  不提供 fp8：e4m3 只有 3 位尾数，足以打乱 block 的 top-k 排序，要做得先有 per-head scale。
- **启动参数放在版本化的配置里**（`configs/infer_*.yaml` + `miowtion/infer/config.py`）：推理的
  旋钮和训练一样多（offload 深度、chunk 行数、方案、保留比例），而配错不会立刻失败——24 GB 卡上
  短几何跑得好好的，一小时后到第一条 14.4 s 的 clip 才 OOM。配置合并进 argparse 的默认值，
  命令行仍然可以覆盖任何一项；配置里出现未知 key 直接报错。
- 解码是编码（`train/encode.py`）的逆过程：视频行 `unpatchify` → `× latents_std + latents_mean`
  → 发布版 VAE 的 `decode_base(z, frame_num)` → `processor.revert_tensor`（[0,1]）；音频行是
  按声道优先的 `[2·audio_t, 32]`，反归一化后按两个单声道的 batch 送进 mono 音频 VAE（32 kHz，每个
  latent 帧 800 个采样点）。响度保护与 Turbo 参考生成器相同：std×5 > 1 时整体缩小。
- **视频 VAE 的解码端是 2.42 B 参数的 ViT3D**（`use_vit_decoder`，36 层、dim 2048、32 头、gated
  SiLU FFN；编码端才是 3D CNN），权重是 fp32，所以解码本身是分钟级的工作量，不是 bug。
  默认把它转成 bf16（`--decode-dtype`）：fp32 既用不上 tensor core，SDPA 也退出 flash 路径。
  音频 VAE 太小，保持 fp32。
- **tile 解码不能关**：`create_token_ids` 给的是 **tile 局部坐标**，ViT 的 RoPE 只在 256 px tile
  的坐标范围内训练过；整帧解码时坐标跑到 84×48，属于分布外（实测 PSNR 只有 19.7 dB），而且因为
  attention 变成 S=20160 的 O(S²)，总时间反而更长。所以 tiling 是这个 decoder 的设计前提。
- **不值得在 VAE 里上稀疏**：tiled + bf16 时 attention 只占解码时间的 9%（S=1280），上限太低。
  只有整帧那条（错误的）路上 attention 才占 51%。
- 先释放 DiT 再加载 VAE。视频 VAE 开启空间 tile 解码（256，重叠 64）。反归一化
  （`revert_tensor`）按帧分块（`_REVERT_CHUNK_FRAMES = 32`）：它是逐元素的，分块结果逐位相同，
  只为避免 345 帧 1344×768 的 fp32 副本（4 GB）撑爆 24 GB 卡。
- **多卡在同一个进程里**（`--devices 0,1` 或 `CUDA_VISIBLE_DEVICES`）：模型只构建一次，
  `train/parallel.replicate` 把常驻参数深拷到其他卡，**offload 的 block 继续共享同一批 pinned
  slab**（host 内存不随卡数增长，这是关键：每个进程自己 pin 一份的话两进程就吃掉 70 GB）。
  AdaLN 表用 `AdalnTables.to(device)` 复制，打分器 deepcopy。每张卡一个线程，
  `assign_jobs` 用"最长优先 + 最小负载"把样本分配到卡上（代价 `geometry_cost` = token 数的平方，
  即注意力的量级），并列时给编号小的卡，所以分配是确定的。任何一个线程抛异常都会重新抛到主线程。
- FA4 的 host 端 launch 用 `_CALL_LOCK` 串行化：首次调用某个签名会经 CuTe DSL / MLIR 做 JIT，
  没有文档保证线程安全。只锁住 launch，kernel 仍在各自的卡上并发执行。
- **标准对比方式**（以后所有稠密 / 稀疏对比都按这个来）：每种模式单独一个 `<mode>.mp4`；另有
  `dense_vs_veda.mp4`，左右拼接，顶部标题栏分别是 "Dense" 和 "Veda <S>% Sparsity"
  （S = 100 × (1 − 保留比例)），带两条音轨（稠密在前）；`summary.json` 记录每步耗时、注意力 GPU
  时间（CUDA event，Veda 包括打分、选块和 gather）、端到端与注意力加速比。第 0 步包含
  kernel 编译，不计入加速比。还会保存 `<mode>_latents.pt`。
- **生成路径不算 PSNR**。稀疏与稠密的差别是构图级的（不是细节级的），逐帧 PSNR 说明不了质量——
  移植后的第一次目视检查里两边构图完全不同，PSNR 只有 10.9 dB，而画面各自都是好的。质量判断走
  并排视频的人工确认（1.5 节）。`decode.psnr_per_frame` 保留给同一条 latent 的数值对照（例如
  解码 dtype 的 bf16 vs fp32）。

## 用法
一个进程用掉所有可见的卡，用 `CUDA_VISIBLE_DEVICES` 选卡：
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/generate.py \
    --config configs/infer_holdout20_step600.yaml
```
导出一个部署用的 bundle（live 权重 + 方案表）：
```bash
python scripts/export_predictor.py \
    --checkpoint runs/<run>/ckpt/step_0000600 \
    --plan-dir runs/<search>/plans --keep-ratio 0.1 \
    --out weights/veda/<run>_step600.safetensors
```
之后推理只要 `--predictor weights/veda/<run>_step600.safetensors`，不再需要
`--checkpoint` / `--plan-dir`；`--keep-ratio` 不给时用 bundle 自己记的那个。

prompt 先用 `scripts/encode_samples.py` 编码进样本缓存。`--sample-id` / `--geometry` 可以给多个
（模型只加载一次，每个样本一个子目录，另有汇总的 `summary.json`）。同一个输出目录再跑一次时，
已经完成的 `<mode>_latents.pt` 会被复用，只补缺的 (样本, 模式) 组合。

## 测试
`tests/unit/test_veda_bundle.py`：bundle 往返逐位相等、方案随权重一起走并且能按几何选出来、
只读 metadata 不读张量、外来的 safetensors 文件被拒、metadata 与张量形状不符被拒、
fp32/bf16 两种存储都逐位往返、bf16 是默认且加载后不被升回 fp32、缺 dtype 键的旧 bundle 按 fp32
读、不支持的存储 dtype 被拒、bf16 的舍入不改变 block 的 top-k 选择。
`tests/unit/test_configs.py`：`configs/` 下每个 yaml 都能解析成可运行的参数（`infer_*` 进
`scripts/generate.py` 的 parser，`search_*` 进 `SearchConfig`，其余进 `TrainConfig`）。
`tests/unit/test_infer_decode.py`：视频 latent 的反归一化是编码的逆（恒等统计量时逐位相等）、
音频行按声道优先、PSNR。端到端只有 GPU 上的实际生成（见验证记录）。

## 踩坑记录
- 发布版视频 VAE 的代码依赖 diffusers；t2va 编码用不到 VAE，所以第一次解码时才暴露（已加入
  `encode` extra）。
- **mp4 里的音频只有 2.2 s（视频 5.17 s）**：`write_mp4` 用了输出选项 `-frames:v N`；ffmpeg 4.4
  写满 N 帧视频就结束整个输出，而原始视频从管道进来比音频编码快得多，音频被截在当时的位置。稠密和
  稀疏都受影响，本机较新的 ffmpeg 不出现。去掉 `-frames:v`（管道 EOF 自然结束），单测用满尺寸
  噪声帧检查两条音轨的时长（在 4.4 上修复前失败、修复后通过）。
- **单卡推理 16:9 14.4 s（104k token）在 Veda 的 FA4 输出分配处 OOM**：SparseStudent 一次处理整个
  头组（q/k/v/out 的 tile 顺序副本各约 1 GB）。改为与 TeacherCollector 一样按头分块，结果逐位不变。
  多样本生成改为可续跑：已存在的 `<mode>_latents.pt` 直接复用，只补缺的 (样本, 模式)。
- 第 0 步包含 FA4 / Triton kernel 的编译（Veda 第 0 步 22.1 s，之后每步 15.0 s），计时和加速比
  必须按 warmup 之后的步折算。
- **解码 14.4 s 16:9 时 OOM**：`revert_tensor(recon.float())` 要把 [1,3,345,768,1344] 转成 fp32
  （约 4 GB）再加上输出，单卡放不下。按帧分块后峰值有界，单测检查分块与不分块逐位相同。
- **解码阶段三个 worker 各卡死半小时，100% CPU、无 I/O、GPU 全空**：`psnr_per_frame` 里
  `a.astype(np.float32) - b.astype(np.float32)` 对 14.4 s 的片子要两份 4.3 GB fp32 加上差值，
  每个 worker 约 12.8 GB、三个就是 38 GB；主机内存被 pinned slab 和 page cache 占满后，每次首次
  触页都走 direct reclaim，全是内核态 CPU，所以看着像在算其实在回收。ptrace_scope=1 时
  `py-spy` 只能 attach 自己的后代进程，detach 出去的长任务要 sudo 才能取栈。对策：生成路径直接
  不算 PSNR。
- **两个单卡推理进程被 host OOM killer 干掉**：每个进程为 offload 的 44 个 block pin 约 35 GB，
  125 GB 的机器上两个进程就只剩十几 GB。改成一个进程驱动多张卡、共享 pinned slab。
- **一个进程里先解码再去噪，latent_t 102 必 OOM**：`scripts/benchmark.py` 在启动时就把 VAE
  解码器建在卡上，好让每个 (几何, 模式) 都能量到解码时间；2.4B 的视频 VAE 常驻约 5 GiB，
  短片子还挤得下，到 104k token 的注意力激活就只差这几 GiB（`k_norm` 处要 1.40 GiB，卡上
  只剩 1.17 GiB）。单次生成不会遇到：`scripts/generate.py` 是去噪完再建解码器。对策：
  `decode.Decoder.to()` 把两个 VAE 在两次解码之间停到主机内存，解码前再搬回卡上，搬运不计入
  解码计时（它是这个 harness 一个进程跑多轮的产物，不是解码的一部分）。
- **崩在最后一个几何上，前面跑完的几何一并丢失**：扫描只在全部跑完后才写 JSON，@37 和 @72
  的结果只剩日志里的一行摘要。对策：每跑完一个 (几何, 模式) 就重写一次 `--out`；一轮扫描
  几十分钟，后面的几何随时可能 OOM，不能连累已经成功的。

## 验证记录
- 2026-09-25，3×RTX 4090（每卡一个几何，权重 offload 到主机内存），commit 608ffd4，
  `configs/infer_holdout20_step600.yaml`，20 个 holdout prompt（不在打分器训练集中），
  FL2VA + Turbo 8 步，seed 0，打分器为阶段 1 第 600 次 update 的 live 权重（`--checkpoint`
  路径，不是 bundle），保留 10%。计时都不含第 0 步（kernel 编译）：

  | 几何 | 样本数 | 稠密 | Veda | 端到端 | 注意力 |
  |---|---|---|---|---|---|
  | latent_t 37（5.17 s） | 5 | 553 s | 345 s | 1.57× | 4.55× |
  | latent_t 72（10.1 s） | 8 | 3043 s | 1342 s | 2.21× | 6.14× |
  | latent_t 102（14.4 s） | 7 | 5187 s | 1852 s | 2.76× | 6.66× |
  | 合计 | 20 | 146.4 min | 59.0 min | 2.24× | 5.92× |

  单样本的极值：16:9 latent_t 102 端到端 3.08×、注意力 6.87×（稠密 982 s / 注意力 695 s →
  Veda 319 s / 103 s）；1:1 latent_t 37 端到端 1.44×、注意力 3.87×（77 s / 32 s → 54 s /
  8.3 s）。**加速比随序列长度单调上升**：注意力是 O(n²) 而其余部分是 O(n)，短片子里注意力
  只占稠密的 42%（1:1 t37），长片子占 71%（16:9 t102），所以同样 6–7× 的注意力加速在端到端
  上的上限完全不同。打分器只在 5.17 s 上训过，10.1 / 14.4 s 是外推。
  并排视频与 `summary.json` 在 `artifacts/generate/holdout20_step600/`（未入库），**待人工
  确认画质**。
- 2026-09-25，CPU，bundle 存储 dtype 的 fp32 ↔ bf16 对照（阶段 1 第 600 次 update 的 live 权重，
  50 层 × 56 头 × 128，随机池化特征，200 个 tile，保留 10% 即每个 query tile 取 20 个 block）：

  | 层 | max\|Δlogit\| | 相对误差 | top-20 重合率（均值 / 最差） | 完全一致的 query tile |
  |---|---|---|---|---|
  | 0 | 4.6e-3 | 7.5e-4 | 0.99943 / 0.95 | 98.86% |
  | 25 | 3.7e-3 | 7.9e-4 | 0.99927 / 0.95 | 98.54% |
  | 49 | 5.7e-3 | 5.7e-4 | 0.99936 / 0.95 | 98.72% |

  最差的 query tile 在 20 个 block 里换掉 1 个。**注意这是在随机特征上测的**：真实激活的池化特征
  相关性强得多，logit 之间的间距分布不同，所以这组数字是量级参考，不是端到端结论；端到端要看下一
  次带 `--predictor` 的生成与稠密的对比。
- 2026-09-24，RTX 4090 单卡，解码 14.4 s 16:9（345 帧 1344×768）的同一批 latent：

  | 配置 | 时间 | attention 占比 | 峰值显存 | vs fp32 的 PSNR |
  |---|---|---|---|---|
  | fp32 + tiling | 165.3 s | 20%（32.6 s，20160 次调用） | 16.1 GiB | — |
  | **bf16 + tiling（默认）** | **50.9 s** | 9%（4.5 s） | 9.3 GiB | **51.3 dB** |
  | bf16 + 整帧 | 59.2 s | 51%（30.0 s，720 次调用） | 9.9 GiB | 19.7 dB |


- 2026-09-24，RTX 4090 单卡（40 个 block offload），FL2VA + Turbo v4_step600_ema 8 步，16:9
  5.17 s（38k token），prompt `search5s_0000`（不在打分器训练集中），seed 0，打分器为阶段 1 第 50 次
  update 的 EMA，保留 10%：
  - 稳态每步（第 1–7 步）：稠密 25.5 s（注意力 13.3 s），Veda 15.0 s（注意力 2.8 s）；**端到端
    1.70×，注意力 4.76×**。
  - 单进程改用 `BlockStreamer`（独立 copy stream 预取 offload 的 block）后：稠密 25.3 s/步，
    Veda 13.6 s/步，**端到端 1.85×**，两种模式的 latent 都与 FSDP 路径逐位相同。
  - 画面：两者都清晰、符合 prompt（东京夜街、黑皮衣红裙、墨镜、黑包、霓虹、湿地面），稠密结果也是
    移植模型的第一次目视检查。Veda 的构图与稠密不同（店面布局、机位），所以逐帧 PSNR 只有 10.9 dB
    （最低 8.6 dB）：差异来自内容而不是画质退化。待人工确认；下一步试第 0 步保持稠密。

## 待办
- 在脚本里直接编码 prompt（现在要先写样本缓存）。
- 按时间步的方案表接入后，推理按当前步的 timestep 选表。
- 人工看过 dense / veda 并排视频后，在这里记录确认人、日期、commit 和结论（AGENTS.md 1.5）。
