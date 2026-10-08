# 社区的双采 / 分块工作流

## 目标

把 ComfyUI 社区里"低分辨率先采、再升分辨率续采"（双采）和"高分辨率分块采样"这两类工作流
的实际参数与调用方式记下来，作为 Veda 在这些工作流下做方案选择和预算适配的事实依据。

这里只记录**读源码确认过的**行为。两个节点包都不是我们的依赖，也不进我们的包；记在这里是因为
它们决定了推理时 Veda 会看到什么样的几何序列，而这一点在我们自己的评测里是看不到的。

## 设计与不变量（这些工作流给 Veda 的约束）

### 一次生成会出现多个几何，而且第二段通常不在训练网格上

两类工作流都把一次生成拆成**两段独立的采样**，中间把 latent 升分辨率：

- `selflift-Avatar`：`low_res_model` 跑前 `transition_step` 次去噪，中间用
  `LatentResizer3D` 升采样，再用 `high_res_model` 跑剩下的步。**时间轴 T 全程不变，只有 H/W 变**
  （`h3_upscaler.py:348-390` 的 `learned_latent_lift` 固定 `target_size=(T, H, W)`）。
  低分辨率边长由 `lowres_scale` 决定并强制取偶：`h = max(2, round(H*scale/2)*2)`
  （`nodes.py:221-229`）。
- YCNodes 的例子工作流：第一段 `SamplerCustomAdvanced`，中间
  `MinimaxH3LatentUpscaler3D` 按 1.4 倍升采样，第二段 `H3TiledSampler`。

**结论**：一次生成里至少两种 (T,H,W)，而第二段的网格是第一段乘一个任意比例，几乎必然落在
训练过的 12 种几何之外。方案表的"最近邻"选择（aspect → 时长 → 补齐）因此是双采下的常规路径，
不是例外路径；两段都要各自选一次方案。

### 布局是每段重建的，不存在跨段复用

两类工作流都走 ComfyUI 正常的 `comfy.samplers` 栈，没有任何一处直接调 `diffusion_model`。
每次 `sample()` 都会新建 `CFGGuider` 并重跑 `process_conds` → `extra_conds`，于是
`MiniMaxH3.extra_conds` 用**该次调用的 latent 形状**重建 `PackedLayout`
（`comfy/model_base.py:2221-2227`），`MiniMaxH3Model._forward` 再校验一次 signature 并写进
`transformer_options["minimax_h3_layout"]`（`comfy/ldm/minimax/model.py:617-624`）。

所以"第二段的 layout 是第一段的残留"这个猜测是**错的**，不要据此设计。

### 采样步下标在第二段从 0 重新开始

`transformer_options["sample_sigmas"]` 由 `CFGGuider.inner_sample`
（`comfy/samplers.py:1228-1229`）按**本段的 sigma 切片**设置：
第一段是 `sigmas[:transition_step+1]`，第二段是 `sigmas[transition_step:]`。
第二段的首个 sigma 已经 < 1。任何"前 N 步走稠密"的按步规则都会在第二段重新触发一次。

### 分块时 position_ids 是全局坐标的切片

`selflift` 的 `highres_tiling` 用 `WrappersMP.DIFFUSION_MODEL` 包住 `_forward`，逐块各跑一次
完整前向（`h3_tiling.py:114-134`）：

- 只沿 H 或 W 切（`axis` 3 或 4），**T 不切，音频整段传给每一块**；
- 每块自建 `PackedLayout`，`seq_len` 和 `segments` 是块的尺寸，但 `video` / `cond` 段的
  `position_ids` 是**整帧坐标切片到该块**（`h3_tiling.py:75-86`），不是从 0 开始；
- 块之间按 patch 单位重叠 `overlap = min(4, max(1, patches//8))`，线性斜坡融合；
- `transformer_options.copy()` 是**浅拷贝**：layout 落在每块自己的 dict 上（没有串块），
  但任何嵌套可变状态是跨块共享的；
- 每步的前向次数变成 `n_tiles` 次，按"每步一次前向"计数的统计会偏大 `n_tiles` 倍。

**对 Veda 的含义**：依赖 position_ids 推网格的代码必须接受非零起点；按头/按块的缓存键里要带上
`seq_len`，否则不同块会互相命中。

## 实际收集到的工作流设置

下面是**下载到的真实工作流**（不是节点默认值）。token 网格 = latent / 2（`PackedLayout` 把
h、w 向上取到 DiT 的 2x2 patch），与打分器 config 里的 `latent_grid` 同一单位。

| 来源 | 阶段 | 像素 | latent | token 网格 | 相对训练网格 |
|---|---|---|---|---|---|
| T2VA 训练几何（参照） | — | 1344x768 | 48x84 | 24x42 | 1.00x |
| selflift（用户日志，精确值） | 低 | 448x256 | 16x28 | 8x14 | **0.33x** |
| selflift | 高 | 864x480 | 30x54 | 15x27 | **0.63x** |
| YCNodes 例子（16:9 @ 0.7 MP） | 低 | 1152x640 | 40x72 | 20x36 | 0.85x |
| YCNodes 例子（升 1.4 倍） | 高 | 1600x896 | 56x100 | 28x50 | 1.18x |

**纵横比对得上，尺度对不上**：四个真实阶段里有三个落在训练网格的 0.33–0.85 倍。方案表按
aspect 选最近邻是选对了，但 tile 形状是**按 token 计**的，所以同一个 8x4x4 在 0.33x 的网格上
覆盖的画面比例是训练时的 3 倍。双采的难点在尺度，不在纵横比——这一点和原先的猜测相反。

### 社区最常见的一套参数（javawock7618 的 INT8 工作流，4 个都一样）

- 第一段：`BasicScheduler simple, 4 步, denoise 1.0` → `SamplerCustomAdvanced`（`euler`）
- 中间：`MinimaxH3LatentUpscaler3D`，`scale by multiplier` **2 倍**（也见 1.5 倍、
  1.4 倍，以及直接给 `target dimensions 1280x704`），align 32
- 第二段：`BasicScheduler simple, 3~4 步, denoise 0.4`，或固定
  `ManualSigmas '0.9035, 0.6316, 0.3158, 0.0000'`
- 全程 `MiniMaxH3SigmaShift [12, 3]`；起始分辨率由 `ResolutionSelector` 给到
  **0.2 / 0.4 / 0.7 MP**，都远低于训练用的约 1.03 MP

即：**第二段是一次低 denoise（0.4）的短精修**，不是从头再采一遍。

### 这不止一个实现

同一套思路至少有四个节点在做，接口各不相同，不能只按 selflift 适配：
`MiniMaxH3TimelineSelfLiftSampler`（上游的 "MiniMax H3 Two-Stage Sampler"）、
`SelfLiftAvatarH3Sampler`（本文读的这个 fork）、
`MinimaxH3LatentUpscaler3DRefineHandoff`（内部自己跑一次采样）、
`VRGDG_MiniMaxH3LearnedLatentUpscale`。升采样器本身也有多个分支
（`xmarre/...-Plus`、`LBH-123-AI/...` 原版）。

上游 README 的经验法则：**高分辨率段占总步数的约 25%**（8 步时低分辨率段 6 步），
与用户实测截图里的 `transition_step=6` 一致。

## 两个节点包的实际参数

### selflift-Avatar（`SelfLiftAvatarH3Sampler`）

| 输入 | 默认 | 含义 |
|---|---|---|
| `transition_step` | 6 | 低分辨率段的去噪次数；约束 `1 ≤ x ≤ len(sigmas)-2` 且 `sigmas[x] < 1` |
| `lowres_scale` | 0.5 | 低分辨率段的空间比例，0.25–1.0 |
| `rho` | 0.0 | 用像素 VAE 锚点纠正的高风险位置比例；0 表示完全跳过 VAE 往返 |
| `w_min` / `w_max` | 0.5 / 1.0 | 纠正强度上下限，约束 `0 ≤ w_min ≤ w_max ≤ 1` |
| `cfg` | 5.0 | 两段共用 |
| `upscaler_model` | 自动选名字里带 h3 的 | `models/latent_upscale_models` 下的 `LatentResizer3D` |
| `high_res_model` | 空 | 为空时第二段复用 `low_res_model` |
| `highres_tiling` | False | 只作用于第二段 |
| `tiling_mode` | auto | auto 按显存估算在 1..8 里搜；manual 用 `tiling_tiles` |
| `tiling_tiles` | 2 | 可选 2/4/6/8 |
| `tiling_axis` | auto | auto 取 patch 网格较长的一边 |

- `sampler` 必须是 `s_churn=0` 的 Euler（`nodes.py:93-99`）。
- NFE：`low_nfe = transition_step`，`high_nfe = len(sigmas)-1-transition_step`
  （`nodes.py:233-236`）。最后一次低分辨率 Euler 更新被丢弃并在升分辨率后解析重建，不额外耗 NFE。
- 日志：`[selflift-Avatar plan] ... hires_model=%s`，其中 `same` 表示两段是**同一个
  ModelPatcher 对象**，`custom` 表示两个不同对象——后者下，只挂在 `low_res_model` 上的补丁在
  第二段完全不生效。
- `SelfLiftAvatarH3TST` 会设置 `optimized_attention_override`（`h3_tst.py:223`），
  与任何同样用这个钩子的节点**直接冲突**。

### YCNodes（`H3TiledSampler` / `H3SigmaRefiner`）

`H3SigmaRefiner`：`extra_steps`(1)、`start_at_sigma`(0.7)、`end_at_sigma`(0.0)、
`spacing`(cosine)。纯 sigma 数组变换，不碰模型。把 `sigmas[idx:]`（首个 ≤ `start_at_sigma` 的
位置起）换成 `(len-idx)+extra_steps` 个插值点。

`H3TiledSampler`：`h_tiles`(2) 实际作用于 **H 轴**、`v_tiles`(2) 作用于 **W 轴**（两个 tooltip
与实现相反）、`tile_overlap`(8)、`max_size_for_no_tile`(24)、`vram_budget_frac`(0.30)。

两个与直觉不符、会影响解读实测结果的点：

1. **它的分块分支实际上跑不起来**。分块时传给 `guider.sample` 的是裸的 5D video 张量，
   `latent_shapes` 长度为 1，而 H3 的入口 `forward` 直接取 `x[1]` 当音频
   （`comfy/ldm/minimax/model.py:561-564`），会 `IndexError`。它自己的 `_single_pass`
   docstring 也写明必须传 NestedTensor。官方例子工作流里
   `max_size_for_no_tile=96` 而 latent 是 56×100，于是 `n_h=n_v=1`，直接短路到
   `_single_pass`——**那条路径等价于一次普通的 `SamplerCustomAdvanced`**。
2. `_clean_minimax_layout`（`h3_tiled_sampler.py:573-604`）会就地删掉
   `guider.original_conds` 里的 `minimax_refs`。第二段因此**没有任何 reference 段**，
   `PackedLayout` 不含 `ref_img` / `ref_audio`，打包序列比第一段短。它同时要删
   `model._cached_extra_conds`，但当前 ComfyUI 没有这个属性，那段是死代码。

官方例子工作流的实测参数：1152×640、243 帧（≈10.1 s）、`BasicScheduler simple` 9 步 →
在第 5 步 `SplitSigmas` → 第一段 5 步、第二段经 `H3SigmaRefiner` 约 5 步，
中间 `MinimaxH3LatentUpscaler3D` 1.4 倍 → 第二段 latent 约 `[1,24,72,56,100]`。
两段都用 `euler_ancestral`。

## 代码位置与接口

本仓库没有代码引用这些节点。相关的是我们这边的"每段各选一次方案"：
`miowtion/veda/plan.py` 的方案表按几何名索引，推理侧（Veda-on-ComfyUI 的
`core/plans.py:select`）按 aspect → 时长 → 补齐选最近邻。

## 测试

无（纯事实记录）。涉及的上游行为由 Veda-on-ComfyUI 的集成测试覆盖。

## 踩坑记录

- **"第二段复用了第一段的 layout"是个想当然的结论**：实际每段都重建。按这个错误前提去
  设计"修复"会白做。先读 `extra_conds` 与 `_forward` 的 signature 校验，再下结论。
- **分块路径的 `position_ids` 不从 0 开始**：按 `position_ids` 反推网格时，要用
  `unique` 后的集合而不是最大值，否则会把块当成整帧。

## 验证记录

- 2026-10-08，读 `slmonker/selflift-Avatar` 与 `yichengup/ComfyUI-YCNodes-MiniMax-H3`
  的源码（各自 depth-1 clone 的当前 main）与 ComfyUI v0.38.2 的 `comfy/samplers.py`、
  `comfy/model_base.py`、`comfy/ldm/minimax/model.py` 得出；未在 GPU 上实跑这两个工作流。
- 2026-10-08，参数表取自实际下载的 5 个工作流：`javawock7618/comfy-MiniMax-H3-workflows`
  的 Bridge / FR / I2V / R2V 四个，以及 YCNodes 仓库自带的 tiled second sampling 例子。
  selflift 那两行是用户实跑的日志（`[selflift-Avatar plan]`），是其中唯一的实测值。

## 待办

- **按尺度补方案表**：真实工作流落在训练网格的 0.33–1.18 倍，而现有 12 种几何只覆盖
  1.0 倍附近。先给 `latent_t_ladder` 之外补一组"低分辨率段"几何（约 0.3x、0.6x、0.85x 的
  16:9 / 9:16 / 4:3 / 1:1），按 `tile_search.md` 的流程搜方案，再并进 bundle。
  这比按节点去适配更通用——四个实现产生的几何是同一批。
- 搜之前先量一下值不值：在 0.33x 的网格上，用 1.0x 搜出来的方案相对 oracle 的
  rel-MSE 比专门搜的差多少。差得不多就只补文档，不补方案。
- 推理侧：`select` 的 cost 现在只看 aspect → 时长 → 补齐，不看尺度。补了多尺度方案之后
  要把尺度距离加进 cost，否则选不到新搜的那些。
