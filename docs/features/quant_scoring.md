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

## 代码位置与接口

- `miowtion/veda/quant.py`
  - `SCHEMES`、`fake_quantize(x, scheme)`：对 `[N, H, D]` 的 tile 序 q/k 做量化-反量化，
    形状和 dtype 不变。
  - `QuantHeatProbe`：插在 `TeacherCollector` 的位置上的注意力函数。残差流走的仍然是稠密
    教师（逐位不变），额外在每个被测层上算一次精确热力图 + 每个方案一次，然后比较 top-k。
    `layer_every` / `q_tile_fraction` 控制成本（指标是对 query tile 的均值，子采样无偏）。
  - `QuantRecord`、`summarize()`。
- `scripts/quant_topk_error.py`：沿教师的少步轨迹滚一个 sample（全程稠密），把逐
  (step, layer, head group, scheme) 的记录写成 `records.jsonl`，汇总写成 `summary.json`。
  不训练、不需要打分器 —— 这测的是打分这一步本身。

## 测试

- `tests/unit/test_veda_quant.py`（32 个）：
  - 每个方案的形状 / dtype 不变，全零输入不产生 NaN；
  - 每个方案的逐元素误差落在该格式自己的舍入步长内（e4m3 2⁻⁴、e5m2 2⁻³、e2m1 1/3）；
  - e2m1 落在八个电平上、e8m0 scale 向上取到 2 的幂、mxfp8 在它能精确表示的块上逐位还原；
  - per-row e4m3 在尺度失衡的张量上严格优于 per-head e4m3；
  - `summarize` 的头数加权、NaN 跳过；
  - CPU 上的 probe 端到端：返回的输出与稠密逐位相等，`bf16` 对照组 recall 恰为 1.0、
    误差恰为 0，且 `heat_kept <= heat_ceiling`。

## 踩坑记录

- **主机内存，不是显存**：再开一个 33B 教师进程需要约 66 GB 的 pinned 主机内存
  （`parallel.HostSlabs`），而两卡训练已经占掉了大部分。权重读完、刚开始 pin 的时候进程被
  静默 kill，日志里没有任何 traceback（不是 CUDA OOM，是内核的 OOM killer）。**判断依据：
  日志停在 `load weights ... done`，进程消失且无回溯。** 对策：这类离线探针要和训练错开跑，
  别和训练抢主机内存；空闲的 GPU 并不代表能再开一份教师。

## 验证记录

- 2026-09-25 macOS CPU：`pytest tests/unit` 212 passed（含本功能 32 个）。
- GPU 实测待做：要在 24 GB 卡上跑，得等当前的 600 step 训练跑完再用空出来的主机内存跑
  `scripts/quant_topk_error.py`（`runs/quant_topk_t37`）。

## 待办

- 真实数据上的 GPU 实测：一个 sample、8 个 step、16:9@37，之后补 latent_t 102。
- 如果某个方案的 recall 掉得可以忽略，下一步是让打分 kernel 真的以该精度跑
  （`kernels/block_heat_triton.py` 与 FA4 的打分路径），再测端到端加速。
- q 和 k 用不同精度（例如 k 用 fp8、q 保持 bf16）是否值得，本模块可以直接扩。
