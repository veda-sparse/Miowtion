# 外部依赖

| 依赖 | 方式 | 锁定版本 | 用途 | 我们依赖的内部接口 |
|---|---|---|---|---|
| MiniMax-H3（GitHub 仓库） | git submodule `third_party/MiniMax-H3`，浅克隆 | `d21241f` | config、tokenizer/processor、VAE 代码、prompt-writing skill | `FL2VA/transformer/config.json` 的键名；`model_index.json` 中的 `_minimax_h3.sigma_shift_scales`；`video_vae.minimax_h3_video_vae.MiniMaxH3VideoVAE`（`encode_images`/`encode_videos`）；AudioVAE posterior-mean 编码 |
| MiniMax-H3 权重（HF `MiniMaxAI/MiniMax-H3`） | `hf download`，不进 git | revision `42ed227` | DiT（FL2VA / Ref2VA）、text encoder、VAE | 原始（非 diffusers）格式的 key 与 dtype；融合 QKV 按头交错 |
| MiniMax-H3 Turbo LoRA（HF `larryvrh/MiniMax-H3-Turbo-Lora`） | `hf download`，不进 git | revision `43a7455`；推荐 `minimax_h3_turbo_v4_step600_ema.safetensors` | 少步（4/8 步）教师：合并进冻结 trunk | key 为 `<module>.lora_A/B.weight`，模块名与发布 checkpoint 一致（含 `blocks.N.adaln_proj.linear`、token refiner、final_layer AdaLN）；qkv 为 [q;k;v]、fc1 为 [gate;up]（与 ComfyUI 版 H3 一致）；`W_eff = W + B@A`，alpha=rank |
| torch | pip | ≥2.8（已验证 2.14+cu126） | 全部 | FSDP2 `fully_shard(ignored_params=…)`、`CPUOffloadPolicy`、`set_modules_to_forward_prefetch`；`torch.distributed.tensor._utils.compute_local_shape_and_global_offset`；`aten._scaled_dot_product_flash_attention`（取 LSE） |
| flash-attn-4（FA4 CuTe） | pip git，`gpu` extra | `d15f1531a460ba456f41b01a774f33ab2db8febf` | 稠密 attention + LSE；块稀疏 kernel | `interface.flash_attn_func`（`mask_mod`、`aux_tensors`、`block_sparse_tensors(_bwd)`、`return_lse`）；`block_sparsity.BlockSparseTensorsTorch`；**运行期 patch** `interface._get_fwd_config`（SM100 上强制 q_stage=1，会检查 `FwdConfig.q_stage` 是否存在） |
| FA4 SM8x / SM120 块稀疏补丁 | **vendored**：`miowtion/kernels/fa4_sm8x`（补丁后的 5 个模块 + `patches/0001..0007` + BSD-3 `LICENSE` / `AUTHORS`） | 基线 flash-attn-4 `4.0.0b32` @ d15f153，五个原始模块按 sha256 校验；另外校验**不替换但依赖其内容**的 `flash_fwd_sm120` / `flash_bwd_sm120`（`_INHERITED_SHA256`）——SM120 的块稀疏靠它们是 SM80 类的薄子类得到，上游若改写就必须报错，否则会在稀疏掩码下静默算出稠密结果 | SM8x（4090 等）与 SM120（RTX PRO 6000 Blackwell 等）上的块稀疏前向 / 反向；`DenseBlockMaskTorch` | **运行期 patch**：`fa4_sm8x.install()` 按依赖顺序替换 `flash_attn.cute.{block_sparsity,block_sparse_utils,flash_fwd,flash_bwd,interface}`；必须在第一次 import `flash_attn.cute` 之前执行（由 `miowtion/kernels/fa4.py` 统一调用）。升级 FA4 时重新 `git am` 补丁、重新生成文件并更新哈希 |
| triton | pip，`gpu` extra | ≥3.3（已验证 3.8） | 教师热力图 kernel | `tl.dot` 等公共 API |
| transformers + accelerate | pip，`encode` extra | 已验证 5.17 / 1.15 | 离线 prompt 编码（Qwen3-VL） | `Qwen3VLForConditionalGeneration`、`model.model.language_model.norm`（替换为 Identity） |
| torchvision | pip | 与 torch 匹配 | transformers 的 Qwen3-VL 处理器 | — |
| diffusers | pip，`encode` extra | `0.32.2`（发布版 `model_index.json` 的版本） | 发布版视频 VAE 的代码依赖它（`ModelMixin` / `ConfigMixin`）：推理解码、fl2va 条件编码 | 只经由发布版 VAE 包间接使用 |
| diffusers（移植版布局） | 未加入 `pyproject.toml` | 需要 ≥ 0.36（`AutoencoderKLMiniMaxH3` / `AutoencoderKLMiniMaxH3Audio`；0.40.0 上实测可用） | `infer.decode.DiffusersDecoder`：解码上游重新发布的 `vae/` + `audio_vae/` | **和上一行冲突**：发布包里的 VAE 代码要 0.32.2，两个版本不能装在同一个环境里。所以这条路只在 Apple silicon 的 MLX 环境里用，`DiffusersDecoder` 惰性 import 并在类缺失时报错，仓库其他地方不依赖它 |
| mlx | pip，`mlx` extra | ≥0.32（已验证 0.32.2） | Apple silicon 推理（`miowtion/mlx`）：block 前向、Veda 块稀疏、NVMe offloading | `mx.quantize` / `mx.dequantize` / `mx.quantized_matmul`（affine，`group_size`、`bits`）；`mx.fast.scaled_dot_product_attention`；`mx.load` 对 safetensors 的惰性加载；`mx.array` 的 buffer 可经 `np.asarray` 拿到可写视图（slab 直接读入）；`mx.synchronize` / `mx.get_peak_memory` / `mx.set_cache_limit`；`mx.take`（块稀疏的 key tile gather）。**注意**：融合 SDPA 的布尔掩码在矩阵乘之后施加，不会跳过整块 |
| ffmpeg（系统二进制，不是 pip 包） | 系统包管理器，或 `pip install imageio-ffmpeg` 后把 `imageio_ffmpeg.get_ffmpeg_exe()` 拷进 PATH | 任意近期版本（已验证 7.0.2 静态版） | `infer.decode.write_mp4` 把帧喂给 ffmpeg 写 mp4；`scripts/visual_check.py` 的拼接与标题栏 | 通过 `subprocess` 调用 `ffmpeg`，走 stdin 的 rawvideo 管道 |
| safetensors / numpy / pyyaml | pip | 见 pyproject | IO | — |

升级规则：每次升级都作为独立提交，并重跑 unit + gpu 测试和对齐检查（见 AGENTS.md）。

## 在"只装了训练依赖"的机器上跑推理

训练只需要 `torch` + `gpu` extra，所以一台专门用来训练的机器上，**解码链是缺的**，而且
每一环都**在几十分钟的去噪之后才报错**（写 mp4 是最后一步）。一次补齐：

```bash
pip install "transformers>=4.57" accelerate "diffusers==0.32.2" pillow
pip install --no-deps torchvision==<与 torch 对齐的版本>   # 见下
pip install imageio-ffmpeg && cp "$(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg
```

- **torchvision 必须 `--no-deps`**：它的 wheel 会把 torch 版本钉死，直接装可能把
  torch 降级/换掉（FA4 是按当前 torch 的 ABI 编的，换掉就全崩）。版本要和 torch 对齐：
  torch 2.14.0+cu130 对应 torchvision 0.29.0+cu130，装完立刻
  `import torchvision; from torchvision.transforms import Normalize` 验证。
  只有发布版 video VAE 用到它（`Normalize`）。
- **不必为纯文本输入装图像栈**：文本编码器的 processor 是惰性的，纯文本编码不需要
  pillow / torchvision。图像、视频输入和解码需要。
- **去噪的结果不会丢**：`scripts/generate.py` 每条片子都把 latent 存成
  `<mode>_latents.pt`，所以解码挂了用 `--decode-only` 续，不用重新去噪。

仅用于测速对照、不参与训练的第三方代码（不进仓库）：FastVideo
fastvideo-kernel（d995516）、SageAttention（d1a57a5）、SpargeAttn（ae5b629）、NVlabs/Sana sol-engine
（Sol-Attn）。

## Ref2VA 编码与官方示例

编码复用 MiniMax-H3 的 Qwen3-VL 与 Visual/Audio VAE；不会调用托管 PE API。
视频输入需要 ffmpeg，官方示例准备还需要 ffprobe。音轨提取为 32 kHz 双声道，
并在 JSONL 中显式列为 audio reference。示例请求固定到 MiniMax-H3 submodule
的 `d21241f0a4b3acbb34c97dae47fa417b7065e438`，解析 JSON 数据而不执行上游 shell。

## Sol-Attn（比较基线，`miowtion/kernels/sol.py`）

| | |
|---|---|
| 用途 | 作为第四条注意力路径（`--attention sol`）做**画质对比**，不作为我们的方案 |
| 来源 | `NVlabs/Sana`，分支 `sol-engine`，子目录 `techniques/sparse_backends` |
| 锁定 SHA | `670482d8a857d578ac8a2ea89b052d0fb47badba`（2026-09-29） |
| 许可证 | Sana 主干 **Apache-2.0**；`sol_attn/THIRD_PARTY_NOTICES.md` 声明内含 FlashAttention（**BSD-3-Clause**）与 cuDNN Frontend block-sparse 参考（**Apache-2.0**）。三者都与 MIT 兼容 |
| 我们依赖的内部接口 | `sol_attn.sol_attn(q, k, v, tau, thresh_type)`（公开 API）；以及 `sol_attn.triton_ref.preprocess.prepare` **仅用于密度标定**（见下） |

**安装必须加 `--no-deps`**：上游要求 torch 2.10 / triton 3.6，而我们是 torch 2.8 / triton 3.4，让 pip 去满足它会把 vendored FA4 所钉的 torch 换掉。实测在我们的版本下它的 **CuTe SM120** 后端照样编译成功（`interface._compiled` 的 key 是 `(12, 0)`），**不是** Triton 回退路径——这一点必须说明，因为上游文档指出 Triton 参考路径「correct but not representative of published speedups」。

**为什么要调用它的 `prepare`**：`tau` 是阈值不是预算。上游的路由规则是逐 (query block, head) 算 `mean + tau * std`，当一个 key block 的 proxy 分数在该 query block 上的均值超过它就精确计算（`triton_ref/fwd.py`：`exact = (sum(scores, 0) / q_len > route_threshold)`）。所以要和固定预算的 router 在同一稀疏度上比，必须**标定 tau 并报告实际达到的密度**。`sol.density()` 调用上游自己的 `prepare` 取 block summary 和阈值，只重写最后那一步比较，这样判定来自上游而不是我们的近似。

**不允许的做法**：假定 tau 对应某个密度。`MIOWTION_SOL_DENSITY=1` 让每次生成都记录实际路由比例，`scripts/generate.py` 会把均值与区间打进日志。

## 对比基线：第三方稀疏注意力实现（只读，仅用于横向对比）

为 `EXPERIMENT_PLAN.md` 的 P0-3 / P0-4 准备。全部以 `third_party/` 下的浅克隆放置、锁定
commit、**不修改内容**，并按 AGENTS.md §3 核对许可证必须与 MIT 兼容。

| 方法 | 仓库 | 锁定 commit | 许可证 | 可用性 |
|---|---|---|---|---|
| Sol-Attn | `NVlabs/Sana`（`sol-engine` 分支，子目录 `techniques/sparse_backends`） | `670482d` | Apache-2.0（内含 FlashAttention BSD-3、cuDNN Frontend Apache-2.0） | **已装并实测**，CuTe SM120 后端 |
| SpargeAttn | `thu-ml/SpargeAttn` | `ae5b629` | **Apache-2.0** | 已克隆，未装 |
| SVG / SVG-EAR | `svg-project/Sparse-VideoGen` | `f89aeda`（含 svg-ear PR #80） | **Apache-2.0** | 已克隆，未装 |
| SLA | `thu-ml/SLA` | `7db4039` | **Apache-2.0** | 已克隆，未装 |
| STA | `hao-ai-lab/FastVideo` | main | **Apache-2.0** | 已克隆，未装 |
| XAttention | `mit-han-lab/x-attention` | `e379887` | **无顶层 LICENSE** | **不可用**：许可证不明，按 §3 不得使用 |

**安装一律 `--no-deps`**，和 Sol-Attn 同一个理由：其中几个要求比本仓库更新的 torch，
让 pip 去满足会替换掉 vendored FA4 所钉的那一套，打断训练与推理栈。

**XAttention 的处理**：仓库没有顶层许可证文件，所以无法确认与 MIT 兼容。
按 AGENTS.md §3，**不使用**，并在论文的横向对比里注明「该方法因许可证不明未纳入」，
而不是悄悄跳过。


## Wan2.1-T2V-1.3B（P0-3 的第二个公开模型）

| | |
|---|---|
| 来源 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（HuggingFace） |
| 本地 | `weights/wan/Wan2.1-T2V-1.3B`，**27 GB，已验证完整** |
| 构成 | `WanPipeline`（diffusers 0.33.0.dev0）：transformer 2 分片 5.29 GiB、text_encoder 5 分片 21.16 GiB、vae 0.47 GiB、scheduler、tokenizer；**incomplete 文件 0 个** |
| 用途 | 横向对比的第二个模型，解决「只有一个非公开模型」这条审稿风险 |

**下载必须走本地 mihomo 代理**（见 agent 的本地 memory），而且**要带重试循环**：前四次单次
尝试全部静默失败,进程还在、`.incomplete` 是 0 字节、大小停在只有 config 的 6.7 MB。
其中有两次是我自己造成的：容器里 `ss -ltn` 查不到 7890 的监听，我据此判断 mihomo 死了，
把 `/etc/profile.d` 已经设对的代理变量 `unset` 了，于是每个请求直奔被阻的 CDN。
**判断代理活没活要读它自己的日志或 `curl -x` 直测，不能看 `ss`。**

### 已经能静态算出来的部分（不需要 GPU，2026-10-10）

Wan 的架构参数从 `transformer/config.json` 读出来：**30 层、12 头、head_dim 128**、
patch `[1,2,2]`、VAE 空间 8× 时间 4×。`head_dim 128` 和 H3 一样，所以 **FA4 块稀疏的
约束和打分器的 `D=128` 都能直接沿用**，变的只有层数和头数（360 个 (层, 头) 对，H3 是
2800，所以逐头统计会噪得多）。

480p 的标准配置 832×480×81 帧 → 潜空间 (21, 60, 104) → patch 后 token 网格
**(21, 30, 52) = 32760 token**。`tiling.candidate_shapes` 给出 21 个候选（H3 的网格是
36 个），按补齐排序：

| 形状 | tile 数 | 补齐 | `std(log B)` | count-term 杠杆 |
|---|---|---|---|---|
| **2x16x4** | 286 | **11.75%** | 0.2105 | **5.4%** |
| 1x16x8 | 294 | 14.87% | 0.2520 | 6.5% |
| 2x8x8 | 308 | 20.34% | 0.3383 | 8.7% |
| 4x8x4 | 312 | 21.90% | 0.5323 | 13.7% |
| 8x16x1 | 312 | 21.90% | 0.2318 | 5.9% |

对照 H3 的 16:9@37 + 4x4x8：`std(log B)` 0.664、杠杆 **17.0%**、最小 tile 只有
**8/128** 行。

**可检验的预测：Veda2 的 `count_term` 在 Wan 上值的钱比在 H3 上少。** 在实际会挑的那个
形状（2x16x4，补齐最小）上杠杆是 **5.4%**，低于 H3 已发布方案表的上限 8.9%；原因是
(21, 30, 52) 分得更齐，最小 tile 有 56/128 行而不是 8/128。想要高杠杆得挑 4x8x4
（13.7%），但补齐要从 11.75% 涨到 21.90%，那又把 tile 数推上去。

这是 `count_term_leverage` 这个函数当初的用途：**上一个新几何之前先查，不要先烧 GPU**。
它是纯静态的，只看网格和形状，不需要权重。

### P0-3 剩下的工作量（权重已不是卡点）

1. **Veda2 到 Wan 的移植**。Veda 的 tile 排列、方案表、FA4 块稀疏整合都绑在 H3 的
   packed layout 上（video / audio / global 三个象限、ragged 补齐、按 (层, 头) 选形状）。
   Wan 没有音频象限，latent 网格也不同，所以 `h3/layout.py` 与 `veda/tiling.py` 的
   接口要为 Wan 重做一遍。
2. **tile 方案搜索**。Veda2 的收益依赖逐 (层, 头) 的 tile 形状，而那是搜出来的
   （`scripts/search_tiles.py`）。Wan 上必须重搜,这是 GPU 工作，按 H3 的经验是小时级。
3. **二阶头的校准**。`ablate_sol.py --second-moments` 要在 Wan 上重跑一次（一条 clip，
   约 1.5 分钟），因为 `C_u`/`C_v` 是模型相关的。
4. **对比方法的接线**。SpargeAttn / SVG+EAR / SLA / STA 都有 Wan 路径（SVG 甚至有
   `svg/models/wan/inference.py`），所以这一层比在 H3 上容易,它们本来就是为 Wan 写的。

