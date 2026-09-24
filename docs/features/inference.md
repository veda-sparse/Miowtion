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
- 打分器从训练 checkpoint 严格加载（默认 EMA 权重），方案表按几何用 `PlanTable.select` 选，必须与
  训练时同一个方案目录。
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
  时间（CUDA event，Veda 包括打分、选块和 gather）、端到端与注意力加速比、逐帧 PSNR。第 0 步包含
  kernel 编译，不计入加速比。还会保存 `<mode>_latents.pt`。

## 用法
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/generate.py \
    --root weights/MiniMax-H3 --schedule turbo --num-steps 8 \
    --adapter weights/turbo_lora/minimax_h3_turbo_v4_step600_ema.safetensors \
    --sample-cache artifacts/samples/search_2x3 --sample-id search5s_0000 \
    --geometry 16:9@37 --attention dense veda \
    --plan-dir runs/search_turbo8_4090/plans \
    --checkpoint runs/stage1_turbo8_4090/ckpt/step_0000050 \
    --out-dir artifacts/generate/search5s_0000
```
prompt 先用 `scripts/encode_samples.py` 编码进样本缓存。`--sample-id` / `--geometry` 可以给多个
（模型只加载一次，每个样本一个子目录，另有汇总的 `summary.json`）。

## 测试
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
- **两个单卡推理进程被 host OOM killer 干掉**：每个进程为 offload 的 44 个 block pin 约 35 GB，
  125 GB 的机器上两个进程就只剩十几 GB。改成一个进程驱动多张卡、共享 pinned slab。

## 验证记录
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
