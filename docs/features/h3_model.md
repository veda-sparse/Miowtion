# H3 DiT 训练侧实现

## 目标

在自己的框架里完整实现 MiniMax-H3 DiT 的前向（FL2VA：t2va/fl2va；Ref2VA：ref2va；两套只有 DiT
权重不同），attention 可插拔，供稠密教师、Veda 学生、tile 搜索评分共用，并能被 FSDP2 逐 block
切分。MiniMax-H3 仓库里已有的部分（config、tokenizer/processor、text encoder 配置、视频/音频
VAE 代码、`model_index.json` 的 sigma 位移）直接复用，不重写。

## 设计与不变量

### 请求几何（`miowtion/h3/geometry.py`）
- 画布：短边 768 → 面积超过 768×1344 时按面积等比缩小 → 宽高各自取整到最近的 32 的倍数。
  16:9 → 1344×768，9:16 → 768×1344，4:3 → 1024×768。
- 帧数：`round(时长×24)` 向上对齐到 17n+5；latent_t = (帧数−5)/17×5+2；音频 `round(实际时长×40)`。
- 5–14.375 s 共 14 档 latent_t：37, 42, …, 102。要求 15.0 s 会对齐到 107（15.083 s），超出上限。

### 打包布局（`miowtion/h3/layout.py`）
- 行序 `[text | 关键帧 | 参考 | 目标音频 | 目标视频 | pad]`，`seq_len` 为 64 对齐或固定 bucket，
  `cu_seqlens=[0, used, seq_len]`，pad 自成一段。
- 位置编码 fp64，**不能取整**（空间轴是小数，取整会合并列）；时间轴每 5 个 latent 帧覆盖
  1,4,4,4,4 个像素帧，每帧 5/3；音频两声道按声道优先排列，w 分别钉在视频宽度轴两端。
- 尾关键帧的 t = 目标起点 + 全部时间跨度 − 5/3（即最后一个像素帧）。
- 参考（ref2va）：image 推进 t 1；audio 推进 T；video 推进 max(音频 T, 视频时间跨度)。
- text 段的 tag 由调用方给出：VLM 视觉 token 为 0（视频），文字为 1；pad 为 −1，DiT 内 clamp 为 0。
- 自检基准：16:9、5 s、353 个文本 token → 视频区间 [767, 38063)，网格 (37,24,42)，seq_len 38080。

### 调度（`miowtion/h3/schedule.py`）
- `linspace(1,0,N)` fp32 → `σ=s·b/(1+(s−1)·b)`（视频 12、音频 3，读 `model_index.json`）→ 末尾补 0。
  N=50 → 49 步；少步 8 步网格为 N=9（1.0, 0.988, …, 0.632, 0）。
- DiT 输入 `t=1−σ`；`v = x0 − noise`；Euler `x += (σ_cur − σ_next)·v`，fp32。
- 每行 timestep：文本/pad/目标视频用 t_video，目标音频用 t_audio，视觉条件 `max(t_video, 0.999)`，
  音频参考 `max(t_audio, 1.0)`。去重后**升序**排成槽位；AdaLN 行 = `clamp(tag,0) + 3×槽位`。
  **槽位数量不固定**（第 0 步 t_video=t_audio=0 合并为 1 个）。

### 噪声（`miowtion/h3/noise.py`）
- 视频噪声在原始 `[1,24,T,H,W]` latent 上 randn 再 patchify（不同于直接生成 `[N,96]`）；音频用同一
  seed 新建生成器。
- 视觉条件：每个条件一个新生成器（同 seed），在 `[1,24,target_T+条件数,H,W]` 上 randn 取前缀，
  `0.999·clean + 0.001·noise`。

### 模型（`miowtion/h3/model.py`）
- 模块名与发布 checkpoint 的 key 一一对应；唯一的非平凡映射是融合 QKV（见下）。
- 数值走固定的 eager 算子链：逐元素运算全部在 bf16 中进行，RMSNorm 用 fp32 累加；fp32 孤岛
  （patch 投影、时间嵌入、输出头）保持 fp32。改变算子顺序就是改变教师。
- 时间嵌入 cos 在前、sin 在后；RoPE 只旋转前 96 维（t/h/w 各 16 个频率），cos/sin 先转 bf16。
  超过 16384 行时 RoPE、AdaLN 调制和带 gate 的残差都按行分块计算（逐元素运算，逐位相同），
  只为限制长 clip 的临时显存。
- `AttentionFn(q, k, v, layer_index)`：稠密教师、TeacherCollector、SparseStudent、OracleScorer
  都实现这个接口。
- 冻结 trunk 时可以用 AdaLN 表（`miowtion/train/adaln.py`）替代 50 个 `adaln_proj`（共 13B 参数），
  这样只剩约 20B 参数需要切分。表的计算与模型内实时计算逐位一致。
- `set_mlp_chunk_rows`：按行分块计算 MLP，降低中间张量峰值（100k 行时为 5.7 GB）。不同行数下
  GEMM 结果可能不同，所以搜索、训练、评估必须使用同一个设置。

### 权重（`miowtion/h3/weights.py`）
- 发布 checkpoint 的融合 QKV 按头交错存放（h0 的 q,k,v，h1 的 q,k,v，…），加载时重排成
  [q_all; k_all; v_all]。**不重排也能跑，不报错，但结果是垃圾**；测试钉死了这个置换。
- 严格加载：缺失或多余的 key、dtype、shape 不对都直接报错。FSDP2 的 DTensor 参数只读本 rank 那一片
  的行（QKV 重排后的行按连续段读取）。

## 代码位置与接口
`miowtion/h3/{config,geometry,layout,schedule,noise,attention,model,weights}.py`

## 测试
- `tests/unit/test_h3_geometry_layout.py`：画布、时长阶梯、自检基准、音频钉位、时间坐标、fl2va、
  ref2va、bucket。
- `tests/unit/test_h3_schedule_noise.py`：两个变体的 sigma 位移、8 步网格、Euler 方向、槽位、
  噪声排列、条件加噪（逐位）。
- `tests/unit/test_h3_model_weights.py`：config、dtype、QKV 置换、交错格式读写往返、严格加载、
  确定性、pad 不泄漏、AdaLN 预计算逐位一致、SwiGLU 的门控半边。
- `scripts/check_vs_diffusers.py`（需要真实权重和 `diffusers>=0.36`，不在 CI 里）：把同一份
  发布权重灌进 `H3DiT` 和 diffusers 的参考实现，逐模块比相对 L2。改动权重语义（融合顺序、
  置换、命名映射）之后必须跑一次。

## 踩坑记录
- **长 clip 在 RoPE 处 OOM**：eager 的 RoPE 链（乘积、rotate_half 的 cat、最后的 cat）每一步都是整张
  [S, 56, 128] 的临时张量，78k token 时约 4 GB，训练在 4:3 14.4 s 上 OOM。按行分块后临时显存有界，
  结果逐位不变。之后 16:9 14.4 s（104k token）又在 `_modulate`（`index_select` 出的 [S, 5376]
  调制向量、乘积、和）处 OOM，调制和 gate 残差同样按行分块。
- **发布的 checkpoint 有两套命名**：最早的 `MiniMaxH3DiTModel`（diffusers 0.32）用
  `blocks.N.attn.qkv_proj.weight` 这类 H3 原生名字，后来上游改成 diffusers 移植版
  `MiniMaxH3Transformer3DModel`（diffusers 0.36），config.json 的键几乎全部改名
  （`ffn_hidden_size` → `ffn_dim`、`latents_dim` → `in_channels`、`rope_inv_freq_len`
  → `rope_freq_dim` 等），权重也改成 `transformer_blocks.N.attn.to_q/to_k/to_v`。
  旧的 `from_pretrained` 只认第一套键名，遇到第二套会**静默退回 dataclass 默认值**——
  这次恰好默认值和真实值一致所以没炸，换个 checkpoint 就会安静地跑错模型。现在
  `CONFIG_KEYS` 同时登记两套拼写，任何字段找不到就直接 `KeyError`。
- 浮点恒等式不能直接在测试里用：`a+b−a` 在浮点下不等于 `b`，逐位测试应当直接复算期望值。
- **SwiGLU 的两半接反，生成出来的就是噪声**：8 步真实权重生成的片段和"直接把纯噪声送进
  VAE 解码"看不出区别；量化的说法是末态 latent 相对初始噪声只走了 `|x−x0|/|x0| ≈ 0.34`、
  `cos ≈ 0.94`，而流匹配走完 Σ Δσ = 1 应当是 ≈1.4 且与噪声基本不相关。原因是发布权重把
  MLP 的两个投影融成 `[up; gate]`（diffusers 的 SwiGLU：`up, gate = proj(x).chunk(2)`，
  返回 `up * silu(gate)`），我们按 `[gate; up]` 解读，对**每一个 block 和 token refiner**
  都算错。这个错误保形状、保范数，而且 torch 侧和 MLX 侧是同一个误解，所以两条自研路径
  互相对拍、以及所有合成权重单测全都是绿的。
  定位办法是引入**第三方参考实现**：`scripts/check_vs_diffusers.py` 把同一份发布权重同时
  灌进我们的 `H3DiT` 和 `diffusers.MiniMaxH3Transformer3DModel`，逐模块比相对 L2。修好后
  token refiner / 打包 embedding / 时间步 embedding / 输出头全部 `rel 0.0`（逐位相等），
  trunk block 在"同样让真实行去 attend pad 行"的前提下也是 `rel 0.0`；默认屏蔽 pad 时
  real 行差 3.6e-3，差异全部来自 diffusers 不带 padding mask。
  教训：只和自己的另一条实现对拍，共享的误解永远测不出来；涉及权重语义（融合顺序、置换、
  半边划分）的地方必须有外部参照。
- **上一条的修复本身又是一个坑：融合顺序是"发布版本"的属性，不是常量**。上面的结论
  （`[up; gate]`）是拿 **diffusers 版发布**（`MiniMaxH3Transformer3DModel`）对出来的，
  但 CUDA 推理和训练加载的是**第一版发布**（`MiniMaxH3DiTModel`，`weights/MiniMax-H3/
  FL2VA/transformer`），它融的顺序正好相反（`[gate; up]`，即 H3DiT 自己的
  `gate, up = fc1(x).chunk(2)`）。把 `Mlp._forward` 改成对所有权重都按 `[up; gate]` 解读
  之后，MLX 路修好了，CUDA 路反而被改坏，生成结果又退回纯噪声，而 `pytest tests/unit`
  和 torch↔MLX 对拍依旧全绿——因为合成权重下两种顺序都自洽。
  排查花了大半天，最省事的几个判据记在这里：
  - 末态 latent 与**同 seed 的初始噪声** `noise.initial_noise(geometry, seed)` 的
    `cos ≈ 0.94`：说明几乎没去噪；
  - 同几何同 seed、**两个不同 prompt** 的末态 latent `cos = 1.0000`（逐位相同）：说明
    条件通路完全没起作用，问题在 DiT 主干而不在 Veda、打分器或数据；
  - `__pycache__/*.pyc` 的时间戳（rsync 保留源文件 mtime）能钉死"新代码第一次被执行"的
    时刻，用来把好坏两批产物和某次合并对上。
  现在的做法：`release.MLP_GATE_FIRST` 按 schema 钉死两种顺序，`H3Config.from_pretrained`
  从 config.json 的 `_class_name` 判定发布版本（认不出来直接 `ValueError`，不猜），
  `Mlp` 用 `config.mlp_gate_first`，`weights.load_dit_weights` 与
  `mlx.convert.ReleaseReader` 在加载时交叉校验"张量名字所属发布"与"模型假设的顺序"，
  不一致就报错。单测同时钉死两种顺序以及这个交叉校验。
  教训：外部参照只能证明**它自己那一版**；结论必须连同"适用于哪个发布"一起落到代码里，
  否则下一次就是把一条路修好、另一条路改坏。

## 验证记录
- 2026-09-26，3×RTX 4090、真实权重、8 步 Turbo LoRA：`[gate; up]`（第一版发布）下
  dense 生成正常，`[up; gate]` 下同 prompt 同 seed 的末态 latent 与初始噪声 cos 0.94、
  两个不同 prompt 的结果逐位相同（纯噪声）。
- 2026-09-23，macOS CPU：unit 全部通过。GPU 前向（真实权重）待验证。
- 2026-09-26，macOS CPU（Apple silicon）、真实权重、`scripts/check_vs_diffusers.py
  --layers 1`：token refiner、打包 embedding、时间步 embedding、AdaLN 索引、视频头、音频头
  对 diffusers 参考实现逐位相等（`rel 0.0`）；trunk block 0 在让真实行也 attend pad 行时
  逐位相等，屏蔽 pad 时 real 行 rel 3.6e-3（diffusers 没有 padding mask）。

## 待办
- 用真实权重在 GPU 上跑通稠密前向并渲染视频做可视化检查（需要 VAE 解码与人工确认）。
- ref2va 参考视频的 VLM presentation（2 fps 采样、时间戳文本）编码尚未实现。
