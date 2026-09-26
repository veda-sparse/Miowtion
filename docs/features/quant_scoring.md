# 低精度块打分（fp8 / fp4）对 top-k 的影响

## 目标

Veda 的选择环节（给每个 128×128 块打一个分，每个 query tile 留 top-k）和注意力本身一样是
O(S²D)，所以它是最该降精度的地方。问题只有一个：**降精度之后选出来的块集合还是不是同一个**。

分数本身的绝对误差不是问题 —— top-k 对每行加一个常数、对每行做任何单调缩放都不变。唯一要
测的是预算边界附近的排序会不会翻。本文件记录量化方案、测量方法和结论。

## 设计与不变量

- **fake quant**：量化再反量化，算术仍然在 bf16/fp32 里做。这样任何方案都能在现有硬件上测
  （4090 没有 nvfp4 张量核），而且和参考路径之间唯一的差别就是那一次舍入本身。
- **缩放粒度只取 kernel 真的能用的那几种**。两个按行缩放的向量的点积等于缩放后的点积，
  所以 per-row 的 scale 在 GEMM 这一层是免费的；per-block（NVFP4 的 16、MX 的 32）是
  micro-scaling 格式自带的。方案表：

  | 名称 | 元素 | scale |
  |---|---|---|
  | `bf16` | — | 恒等（对照组） |
  | `fp8_e4m3_head` | e4m3 | 每个头一个 amax scale（最粗） |
  | `fp8_e4m3_row` | e4m3 | 每行每头一个 amax scale |
  | `fp8_e5m2_row` | e5m2 | 同上（动态范围大，尾数只有 2 位） |
  | `nvfp4` | e2m1 | 每 16 个元素一个 e4m3 scale + 每头一个 fp32 全局 scale |
  | `mxfp8_e4m3` | e4m3 | 每 32 个元素一个 e8m0（2 的幂）scale |
  | `mxfp4` | e2m1 | 每 32 个元素一个 e8m0 scale |

  NVFP4 的两级 scale 不是我们加的近似：e4m3 的 block scale 自己也要可表示，所以规范里就有
  一个 per-tensor 的 fp32 外层 scale。

- **smooth-K（`+smoothk` 后缀）**：任何方案名后面加 `+smoothk`，先减去 k 的逐通道均值再
  量化，然后加回来。思路取自 SageAttention（arXiv:2410.02367，只借思路，没有拷贝代码）：
  k 的各个通道带着一个很大的公共偏移，量化范围被这个偏移吃掉，而它对区分不同的 key 毫无
  贡献。
  - **为什么对选择是无损的**：`q·(k − μ) = q·k − q·μ`，而 `q·μ` 只和 query 有关 ——
    每行一个常数，top-k 不变。所以平滑不是近似，它只是把量化误差挪到一个选择看不见的
    地方。`tests/unit/test_veda_quant.py::test_smoothing_does_not_move_the_top_k_of_the_
    exact_scores` 直接对精确分数断言 `argsort` 相等。
  - **只平滑 k，不平滑 q**：减 q 的均值会引入 `μ_q·k`，这一项随 key 变化，会真的改变每行
    内部的排序。所以 `QuantHeatProbe` 对 q 一律用 `base_scheme(scheme)`。
  - **残差保持 fp32**：`x.float() - mean` 之后才交给量化器。减法是量化器在自己的累加器里
    做的预处理，先把它舍入回输入 dtype 等于让平滑替硬件不会犯的错误背锅（见踩坑记录）。
- **`bf16` 是对照组**，它必须回报 recall 1.0、rel_l2 0。回报别的数值说明测量本身坏了，
  而不是方案坏了。
- **LSE 一律用精确值**。top-k 对每行常数不变，所以 LSE 不可能改变选择；统一用精确 LSE 只是
  让各方案的热力误差落在同一把尺子上。
- **指标**：
  - `recall`：和 bf16 oracle 的 top-k 重合度（强制对角线不计，它是 kernel 规则不是打分结果）；
  - `heat_kept` / `heat_ceiling`：量化选择留住的**真实**热量占比，以及 bf16 选择留住的占比。
    recall 数块，这一对按块的实际注意力质量加权，是更接近画质的指标；
  - `rel_l2`、`max_abs`：热力图本身的误差，用来分辨\"分数没怎么动\"和\"分数动了但排序没翻\"。
- 汇总按**头数加权**：一层里的头组大小不同（2 头的组和 30 头的组），不加权的平均会把小组
  放大。

## 打分器权重的存储精度（bf16 / fp8）

上面测的是**激活**（q/k）的精度。另一条正交的线是**打分器权重**的存储精度：bundle 里
50 层 × 56 头 × 128 的 `proj_q/proj_k/embed`，bf16 是 525 MiB。

- `bundle.save(..., dtype=torch.float8_e4m3fn)` 每个张量按**第 0 维（头）**取 amax，映射到
  e4m3 的 448.0，scale 以 `<key>.__scale` 的名字并排存 fp32。文件从 550,635,792 字节降到
  275,415,648 字节。
- **fp8 省的是文件和加载，不是常驻显存**：`load()` 读回来立刻反量化成 bf16 参数，因为
  `torch.bmm` 和 `LayerPredictor.embed` 都没有 e4m3 的路径。所以它真正回答的问题是
  **3 位尾数的权重舍入会不会改变选出来的块**，而不是省显存。
- 缺 scale 直接 `ValueError`，不静默当成 1.0 —— 一个被当成 1.0 的 fp8 权重看起来还是能跑，
  只是分数全错。
- 测量用 `PredictorProbe` / `scripts/predictor_precision.py`：同一 step 的若干 bundle
  （第一个是参考）在同一条教师轨迹上打分，除了 `recall` / `heat_kept` / `heat_ceiling`
  之外多报一个 **`agree`** —— 与参考 bundle 自己选择的重合度。recall 混着"打分器好不好"，
  `agree` 才单独回答"舍入动了多少"。脚本拒绝 step 不同的 bundle，否则舍入和训练会混在一起。

## 代码位置与接口

- `miowtion/veda/quant.py`
  - `SCHEMES`、`fake_quantize(x, scheme)`：对 `[N, H, D]` 的 tile 序 q/k 做量化-反量化，
    形状和 dtype 不变。
  - `QuantHeatProbe`：插在 `TeacherCollector` 的位置上的注意力函数。残差流走的仍然是稠密
    教师（逐位不变），额外在每个被测层上算一次精确热力图 + 每个方案一次，然后比较 top-k。
    `layer_every` / `q_tile_fraction` 控制成本（指标是对 query tile 的均值，子采样无偏）。
  - `base_scheme(scheme)`：去掉 `+smoothk` 后缀，拿到底层方案名。
  - `QuantRecord`、`summarize()`。
  - `PredictorProbe`、`PredictorRecord`、`summarize_predictor()`：同一位置的探针，比较的
    不是量化方案而是若干个打分器 bundle。
- `miowtion/veda/bundle.py`：`DTYPES` 增加 `float8_e4m3fn`，`save/load` 处理
  `.__scale` 伴随张量。
- `scripts/predictor_precision.py`：多个 `--bundle NAME=PATH`，输出 `records.jsonl` +
  `summary.json`（overall / per_step，含每个 bundle 的路径、dtype、字节数）。
- `scripts/quant_topk_error.py`：沿教师的少步轨迹滚一个 sample（全程稠密），把逐
  (step, layer, head group, scheme) 的记录写成 `records.jsonl`，汇总写成 `summary.json`。
  不训练、不需要打分器 —— 这测的是打分这一步本身。

## 测试

- `tests/unit/test_veda_quant.py`（45 个）：
  - 每个方案的形状 / dtype 不变，全零输入不产生 NaN；
  - 每个方案的逐元素误差落在该格式自己的舍入步长内（e4m3 2⁻⁴、e5m2 2⁻³、e2m1 1/3）；
  - e2m1 落在八个电平上、e8m0 scale 向上取到 2 的幂、mxfp8 在它能精确表示的块上逐位还原；
  - per-row e4m3 在尺度失衡的张量上严格优于 per-head e4m3；
  - `summarize` 的头数加权、NaN 跳过；
  - CPU 上的 probe 端到端：返回的输出与稠密逐位相等，`bf16` 对照组 recall 恰为 1.0、
    误差恰为 0，且 `heat_kept <= heat_ceiling`；
  - smooth-K：`bf16+smoothk` 是恒等（残差保持 fp32 才成立），平滑不改变精确分数的
    `argsort`，平滑后的 k 在量化前的动态范围更小，未知后缀直接报错。
- `tests/unit/test_veda_bundle.py`（16 个）：fp8 往返的 per-head 相对误差在 e4m3 的舍入
  步长内、文件比 bf16 小、载回来是 bf16 参数、缺 scale 报错、scale 张量不混进权重。
- `tests/unit/test_scripts_predictor_precision.py`（3 个）：`--bundle` 解析与重名报错。

## 踩坑记录

- **主机内存，不是显存**：再开一个 33B 教师进程需要约 66 GB 的 pinned 主机内存
  （`parallel.HostSlabs`），而两卡训练已经占掉了大部分。权重读完、刚开始 pin 的时候进程被
  静默 kill，日志里没有任何 traceback（不是 CUDA OOM，是内核的 OOM killer）。**判断依据：
  日志停在 `load weights ... done`，进程消失且无回溯。** 对策：这类离线探针要和训练错开跑，
  别和训练抢主机内存；空闲的 GPU 并不代表能再开一份教师。
- **平滑的残差不能先落回 bf16**：`(x - mean).to(bf16)` 之后再量化，`bf16+smoothk` 就不再是
  恒等 —— 减法本身丢掉的位被算进了平滑的账上。但真实硬件里这个减法发生在量化器的 fp32
  累加器中，不存在这次舍入。对策：`fake_quantize` 全程用 `x.float()`，最后才 `.to(x.dtype)`。
  现象是 `test_smoothing_is_the_identity_for_bf16` 失败。
- **fp8 bundle 的"减半"在小文件上看不出来**：safetensors 的 header 在玩具尺寸下比权重还
  大，单元测试里 fp8 文件只比 bf16 小 35%。对策：测试的阈值放到 0.75 并写明原因，真正的
  一半要在真实尺寸上看（550,635,792 → 275,415,648 字节）。

## 验证记录

- 2026-09-25 macOS CPU：`pytest tests/unit` 212 passed（含本功能 32 个）。
- GPU 实测待做：要在 24 GB 卡上跑，得等当前的 600 step 训练跑完再用空出来的主机内存跑
  `scripts/quant_topk_error.py`（`runs/quant_topk_t37`）。
- **2026-09-26，单张 RTX 4090（sm_89），`scripts/predictor_precision.py`**：打分器 bundle
  的 bf16 vs fp8，同一个 step 200 的权重（`stage1_refine_t102` 的 live 权重），
  16:9@102、104603 token、8 步、keep 0.1、`layer_every 1`、`q_tile_fraction 0.25`，
  19m20s：

  | bundle | 字节 | recall | heat_kept / ceiling | agree | rel_l2 |
  |---|---|---|---|---|---|
  | bf16 | 550,635,792 | 0.6017 | 0.6153 / 0.7273 | 1.0000 | 0 |
  | fp8 e4m3 | 275,415,648 | 0.6017 | 0.6153 / 0.7273 | 0.9944 | 5.2e-3 |

  **结论：fp8 存储把文件减半，选择基本不动。** recall 差 2e-5、`heat_kept` 差 6e-6，
  逐 step 的八个数四位小数全部相同。真正的差别只有 `agree` 0.9944 —— 每约 180 个选中的
  块换掉 1 个，而且换掉的显然在预算边界上、热量可以忽略（否则 `heat_kept` 会跟着动）。
  分数本身的 rel_l2 5.2e-3 没有按比例进到选择里，正是 top-k 对每行常数 / 单调缩放不敏感
  的结果。
  同一次测量也记下了打分器随噪声的表现：recall 0.5654（step 0）单调升到 0.6143（step 7），
  `heat_kept/heat_ceiling` 0.812 → 0.861，高噪声步最难选。
  （测量用的是 t2va prompt，`latent_t` 由 `--geometry` 决定、latent 从噪声起步，所以
  sample 自己缓存的 latent 长度与此无关。）

## 待办

- 真实数据上的 GPU 实测：一个 sample、8 个 step、16:9@37，之后补 latent_t 102。
- 如果某个方案的 recall 掉得可以忽略，下一步是让打分 kernel 真的以该精度跑
  （`kernels/block_heat_triton.py` 与 FA4 的打分路径），再测端到端加速。
- q 和 k 用不同精度（例如 k 用 fp8、q 保持 bf16）是否值得，本模块可以直接扩。
