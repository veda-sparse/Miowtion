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

## Kernel

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| FA4 在 4090 上的"块稀疏"与稠密耗时相同、结果也相同 | 上游 SM80 前向 kernel 忽略了 blocksparse_tensors | vendored 补丁（`kernels/fa4_sm8x`）装上才放行 SM8x，否则直接报错 | [veda_kernel](features/veda_kernel.md) |
| 打了补丁的 FA4 没有生效 | `flash_attn.cute` 的 `__init__` 会 import interface，先 import 就拿到了上游版本 | 所有 FA4 import 都经过 `kernels/fa4.py`；`install()` 发现已 import 直接报错 | [veda_kernel](features/veda_kernel.md) |
| FA4 在 SM100 上拒绝 128 行的 Q 块 | 接口在 seqlen>128 时强制 q_stage=2（256 行粒度） | 覆盖为 q_stage=1（待 B200 验证） | [veda_kernel](features/veda_kernel.md) |
| FA4 报出看起来像掩码形状不对的错误 | `BlockSparseTensorsTorch` 的第 5 个字段是 `cu_total_m_blocks` | `block_size` 用关键字参数传 | [veda_kernel](features/veda_kernel.md) |
| flex 每次换 pattern 都很慢 | `from_kv_blocks` 默认为反向计算转置索引（约 1.2 ms） | 只做前向时传 `compute_q_blocks=False`（约 0.01 ms） | [veda_kernel](features/veda_kernel.md) |

## 训练与分布式

| 现象 | 原因 | 对策 | 链接 |
|---|---|---|---|
| "FSDP parameters should be materialized on CPU" | 开 CPU offload 但参数物化在 GPU 上 | offload 的 block 用 `to_empty(device='cpu')` | [training](features/training.md) |
| `FSDPCommContext has no all_gather_copy_in_stream` | 没有 FSDP root，跨 block 预取找不到通信流 | 根模块也 `fully_shard`，复制参数放进 `ignored_params` | [training](features/training.md) |
| 手动 all-reduce 挂起 | 某些 rank 缺梯度，拼出的缓冲区长度不一致 | 缺失的梯度补零 | [training](features/training.md) |
| 保存 checkpoint 时挂起 | `full_tensor()` 是集合通信，rank 0 以外的 rank 提前返回了 | 所有 rank 先取出完整张量，再由 rank 0 写盘 | [training](features/training.md) |
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
