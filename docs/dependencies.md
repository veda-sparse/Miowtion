# 外部依赖

| 依赖 | 方式 | 锁定版本 | 用途 | 我们依赖的内部接口 |
|---|---|---|---|---|
| MiniMax-H3（GitHub 仓库） | git submodule `third_party/MiniMax-H3`，浅克隆 | `d21241f` | config、tokenizer/processor、VAE 代码、prompt-writing skill | `FL2VA/transformer/config.json` 的键名；`model_index.json` 中的 `_minimax_h3.sigma_shift_scales`；`video_vae.minimax_h3_video_vae.MiniMaxH3VideoVAE`（`encode_images`/`encode_videos`） |
| MiniMax-H3 权重（HF `MiniMaxAI/MiniMax-H3`） | `hf download`，不进 git | revision `42ed227` | DiT（FL2VA / Ref2VA）、text encoder、VAE | 原始（非 diffusers）格式的 key 与 dtype；融合 QKV 按头交错 |
| MiniMax-H3 Turbo LoRA（HF `larryvrh/MiniMax-H3-Turbo-Lora`） | `hf download`，不进 git | revision `43a7455`；推荐 `minimax_h3_turbo_v4_step600_ema.safetensors` | 少步（4/8 步）教师：合并进冻结 trunk | key 为 `<module>.lora_A/B.weight`，模块名与发布 checkpoint 一致（含 `blocks.N.adaln_proj.linear`、token refiner、final_layer AdaLN）；qkv 为 [q;k;v]、fc1 为 [gate;up]（与 ComfyUI 版 H3 一致）；`W_eff = W + B@A`，alpha=rank |
| torch | pip | ≥2.8（已验证 2.14+cu126） | 全部 | FSDP2 `fully_shard(ignored_params=…)`、`CPUOffloadPolicy`、`set_modules_to_forward_prefetch`；`torch.distributed.tensor._utils.compute_local_shape_and_global_offset`；`aten._scaled_dot_product_flash_attention`（取 LSE） |
| flash-attn-4（FA4 CuTe） | pip git，`gpu` extra | `d15f1531a460ba456f41b01a774f33ab2db8febf` | 稠密 attention + LSE；块稀疏 kernel | `interface.flash_attn_func`（`mask_mod`、`aux_tensors`、`block_sparse_tensors(_bwd)`、`return_lse`）；`block_sparsity.BlockSparseTensorsTorch`；**运行期 patch** `interface._get_fwd_config`（SM100 上强制 q_stage=1，会检查 `FwdConfig.q_stage` 是否存在） |
| FA4 SM8x 块稀疏补丁 | **vendored**：`miowtion/kernels/fa4_sm8x`（补丁后的 5 个模块 + `patches/0001..0005` + BSD-3 `LICENSE` / `AUTHORS`） | 基线 flash-attn-4 `4.0.0b32` @ d15f153，五个原始模块按 sha256 校验 | SM8x（4090 等）上的块稀疏前向 / 反向；`DenseBlockMaskTorch` | **运行期 patch**：`fa4_sm8x.install()` 按依赖顺序替换 `flash_attn.cute.{block_sparsity,block_sparse_utils,flash_fwd,flash_bwd,interface}`；必须在第一次 import `flash_attn.cute` 之前执行（由 `miowtion/kernels/fa4.py` 统一调用）。升级 FA4 时重新 `git am` 补丁、重新生成文件并更新哈希 |
| triton | pip，`gpu` extra | ≥3.3（已验证 3.8） | 教师热力图 kernel | `tl.dot` 等公共 API |
| transformers + accelerate | pip，`encode` extra | 已验证 5.17 / 1.15 | 离线 prompt 编码（Qwen3-VL） | `Qwen3VLForConditionalGeneration`、`model.model.language_model.norm`（替换为 Identity） |
| torchvision | pip | 与 torch 匹配 | transformers 的 Qwen3-VL 处理器 | — |
| diffusers | pip，`encode` extra | `0.32.2`（发布版 `model_index.json` 的版本） | 发布版视频 VAE 的代码依赖它（`ModelMixin` / `ConfigMixin`）：推理解码、fl2va 条件编码 | 只经由发布版 VAE 包间接使用 |
| safetensors / numpy / pyyaml | pip | 见 pyproject | IO | — |

升级规则：每次升级都作为独立提交，并重跑 unit + gpu 测试和对齐检查（见 AGENTS.md）。

仅用于测速对照、不参与训练的第三方代码（不进仓库）：FastVideo
fastvideo-kernel（d995516）、SageAttention（d1a57a5）、SpargeAttn（ae5b629）、NVlabs/Sana sol-engine
（Sol-Attn）。
