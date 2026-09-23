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
  确定性、pad 不泄漏、AdaLN 预计算逐位一致。

## 踩坑记录
- **长 clip 在 RoPE 处 OOM**：eager 的 RoPE 链（乘积、rotate_half 的 cat、最后的 cat）每一步都是整张
  [S, 56, 128] 的临时张量，78k token 时约 4 GB，训练在 4:3 14.4 s 上 OOM。按行分块后临时显存有界，
  结果逐位不变。之后 16:9 14.4 s（104k token）又在 `_modulate`（`index_select` 出的 [S, 5376]
  调制向量、乘积、和）处 OOM，调制和 gate 残差同样按行分块。
- 浮点恒等式不能直接在测试里用：`a+b−a` 在浮点下不等于 `b`，逐位测试应当直接复算期望值。

## 验证记录
- 2026-09-23，macOS CPU：unit 全部通过。GPU 前向（真实权重）待验证。

## 待办
- 用真实权重在 GPU 上跑通稠密前向并渲染视频做可视化检查（需要 VAE 解码与人工确认）。
- ref2va 参考视频的 VLM presentation（2 fps 采样、时间戳文本）编码尚未实现。
