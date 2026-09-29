# 外部依赖

| 依赖 | 方式 | 锁定版本 | 用途 | 我们依赖的内部接口 |
|---|---|---|---|---|
| MiniMax-H3（GitHub 仓库） | git submodule `third_party/MiniMax-H3`，浅克隆 | `d21241f` | config、tokenizer/processor、VAE 代码、prompt-writing skill | `FL2VA/transformer/config.json` 的键名；`model_index.json` 中的 `_minimax_h3.sigma_shift_scales`；`video_vae.minimax_h3_video_vae.MiniMaxH3VideoVAE`（`encode_images`/`encode_videos`） |
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
- **不必为 t2va 装图像栈**：`train.encode.TextEncoder` 的 processor 是惰性的，纯文本
  编码不需要 pillow / torchvision（见 [h3_model](features/h3_model.md)）。解码需要。
- **去噪的结果不会丢**：`scripts/generate.py` 每条片子都把 latent 存成
  `<mode>_latents.pt`，所以解码挂了用 `--decode-only` 续，不用重新去噪。

仅用于测速对照、不参与训练的第三方代码（不进仓库）：FastVideo
fastvideo-kernel（d995516）、SageAttention（d1a57a5）、SpargeAttn（ae5b629）、NVlabs/Sana sol-engine
（Sol-Attn）。
