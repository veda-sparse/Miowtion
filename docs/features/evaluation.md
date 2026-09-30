# 评测集与评测流程（v1）

## 目标

给"这条注意力路径 / 这个打分器比那个好吗"一个**固定的、可重复的**答案。固定的意思是：
同一批 prompt、同一批几何、同一个 seed，换的只有被评的那一项。不固定的话，换一条片子
带来的差异就能盖过改动本身——实测同一几何换片子，`heat_ceiling` 能差 4%，和我们想
测的效果同量级。

两层评测，服务两个不同的问题：

| 层 | 问的问题 | 工具 | 产物 |
|---|---|---|---|
| 数值 | 掩码选得对不对（recall / 保住的注意力质量） | `scripts/predictor_precision.py` | `summary.json` + 每层 `records.jsonl` |
| 人眼 | 片子看起来有没有变差 | `scripts/generate.py` + `scripts/visual_check.py` | 每路一个 mp4 + 一张拼起来的对比视频 |

AGENTS.md 1.5.1 的规矩在这里落地：**质量由人眼判定**，PSNR / SSIM 只是可选的辅助
（`--metrics`），不作为结论；每个样本既要**每路各自的单独视频**，也要**一张带标题栏的
拼接视频**，顺序固定为 原始/Baseline → 生成结果。

## 评测集 v1：`moviegen_holdout20`

20 条 t2va，**三种时长 × 四种纵横比混合**：

| 时长 | latent_t | 条数 | id |
|---|---|---|---|
| 5.17 s | 37 | 5 | `holdout5s_0000..0004` |
| 10.1 s | 72 | 8 | `holdout10s_0000..0007` |
| 14.4 s | 102 | 7 | `holdout14s_0000..0006` |

- **几何在配置里钉死**，不是随机的：`configs/infer_holdout20_step600.yaml` 的
  `geometry` 列表与 `sample_id` 一一对应（4:3 / 1:1 / 9:16 / 16:9 都有）。改评测集可以，
  但不要改这个对应关系，否则和历史视频不可比。
- 样本 `aspect` 是 `None`（t2va 记录不声明纵横比），所以**任何纵横比都能配**，只有
  `latent_t` 必须匹配；prompt 本身是按 16:9 扩写的。
- `split` 字段全是 `train`——这个 cache 是**专门为评测编码的**，不是某个训练集的切分，
  所以 split 没有意义。它与训练用的 cache 相互独立，不会被训练看到。

### 它为什么是 v1（换集合前先读这段）

1. **dense 基线已经存在**，而且很贵：t102 稠密一条是 8×56 s 去噪 + 约 200 s 解码，20 条
   重算一遍要小时级。换集合意味着 dense 全部重算。
2. 时长和纵横比同时混合，能同时暴露"长序列退化"和"某个纵横比退化"这两类问题。
3. 历史结论都挂在这个集合上（见下面的基线清单），换集合会让它们失去参照。

### 不在 git 里：怎么拿到 / 怎么再生

cache 是 89 MB（`index.json` + `text.safetensors`），按 AGENTS.md 放在
`artifacts/samples/moviegen_holdout20/`（gitignore）。两条路：

- **拷**：从任何跑过评测的机器上把这两个文件复制过来即可，与机器无关。
- **再生**：20 条的 prompt 清单（schema 与 `data/prompts/moviegen_video_bench_h3.jsonl`
  相同）过一遍编码：

  ```bash
  python scripts/encode_samples.py --root weights/MiniMax-H3 \
      --manifest <holdout20.jsonl> --out artifacts/samples/moviegen_holdout20 \
      --num-test 0 --max-memory "0=80GiB,cpu=200GiB"
  ```

  这 20 条的 `source_prompt` 全部是 `data/prompts/moviegen_video_bench_h3.jsonl` 里
  1003 条的子集，只是扩写成了 5.17 / 10.1 / 14.4 s 三种时长；扩写本身（`prompt` 字段）
  目前不在仓库里。**清单入库需要用户同意**（AGENTS.md 3：`data/` 只放经同意发布的文本）。
  在一张 96 GB 卡上编码 20 条约 1 分钟，文本塔加载另算 8 分钟。

## 怎么跑一次对比

### 1. 每路各自生成（dense 只付一次）

```bash
# 基线两路：dense + 已发布打分器（历史视频已存在，通常不需要重跑）
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py \
  --config configs/infer_holdout20_step600.yaml

# 新的一路：只换 attention / predictor / out_dir，其余一字不改
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py \
  --config configs/infer_holdout20_continued.yaml
```

`generate.py` 对每个 `(sample, geometry)` 写一个目录
`<out_dir>/<sample_id>_<geometry>/`，里面每个 attention 模式一个 `<mode>.mp4`；
dense 和 veda 同时跑时还会有 `dense_vs_veda.mp4` 和带计时的 `summary.json`
（每步、只算注意力、加速比；第 0 步含 kernel 编译，已排除）。

**不变量：可比的前提是 seed、sample、geometry、few-step LoRA 完全一致。** 只要换了
cache 或 id，新旧视频就不能拼在一起——初始噪声都不一样。

### 2. 拼成对比视频

```bash
python scripts/visual_check.py --feature holdout20_<note> \
  --video "Dense"=artifacts/generate/holdout20_step600/<clip>/dense.mp4 \
  --video "Veda t37 step600"=artifacts/generate/holdout20_step600/<clip>/veda.mp4 \
  --video "Veda <新的一路>"=artifacts/generate/<新>/<clip>/veda.mp4 \
  --out-dir artifacts/visual_checks/<feature>/<date>/<clip>
```

pane 顺序就是命令行顺序，baseline 在前。产物是每路的副本 + 一个 N-up 视频 +
`report.json`（记录每个 pane 的来源路径）。

### 3. 数值对比（不需要解码）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predictor_precision.py \
  --root weights/MiniMax-H3 --adapter weights/turbo_lora/<lora> \
  --sample-cache artifacts/samples/moviegen_holdout20 \
  --sample-id holdout14s_0000 --geometry 16:9@102 \
  --bundle release=<bundle A> --bundle continued=<bundle B> \
  --across-steps --keep-ratio 0.1 --offload-blocks 0 \
  --out-dir runs/prec_<note>
```

一次滚一条轨迹，所有 bundle 在**同一批输入**上打分，所以它是读训练趋势唯一干净的方式
——训练日志里的 recall 每个 update 换一条片子，噪声和效果同量级。`--across-steps`
是显式开关：这个脚本默认只比同一 step 的不同精度，跨 step 比较需要自己声明（否则
把"量化误差"和"训练进展"混在一起）。

## 成本（实测）

1×RTX PRO 6000 Blackwell（offload 0、`mlp_chunk_rows 8192`、`veda_collect_mib 2048`）：

| | 稠密去噪（8 步） | Veda 去噪（8 步） | 视频解码 |
|---|---:|---:|---:|
| t37 | 82 s | 44 s | 约 35 s |
| t72 | — | 110 s | 约 100 s |
| t102 | 448 s | 147 s | 约 200 s |

- 20 条**只跑 Veda 一路**约 78 分钟；**加上 dense** 要再多约 3 小时。所以 dense 基线
  值得留着复用。
- 每个进程的启动开销约 14 分钟（教师权重 + LoRA 合并 + AdaLN 表 + VAE），与条数无关。
- 长片子里**解码已经和去噪同量级**（t102 Veda：147 s 去噪 vs 约 200 s 解码），这也是
  `docs/benchmark/performance.md` §9 把解码列为下一个优化点的原因。

## 已有的基线视频（v1 集合上）

| 路 | 目录 | 可用 |
|---|---|---|
| Dense | `artifacts/generate/holdout20_step600_raw/<clip>/dense.mp4` | 是 |
| Veda t37 step600（已发布的打分器） | `artifacts/generate/holdout20_step600_raw/<clip>/veda.mp4` | 是 |
| Veda t102 step200 fp8 | `artifacts/generate/holdout20_step200fp8/<clip>/veda.mp4` | **否：花屏**，不要拿它当基线 |

注意 `holdout20_step600/<clip>/` 里只剩 `dense_vs_veda.mp4`（单独视频被清过），**单独的
`dense.mp4` / `veda.mp4` 在 `holdout20_step600_raw/`**。拼接需要单独视频。

## 拼接方向：横版竖着拼，竖版横着拼

三路 16:9 并排是 4032 px 宽，屏幕上没法看；竖着堆是 1344 宽。三路 9:16 竖着堆有 4000 px 高，
所以要并排。`visual_check.py --stack` 默认 `auto`：用 ffprobe 读参考那一路的宽高，
**横版（含正方形）竖堆、竖版横排**。拿不到 ffprobe 就直接报错要求显式 `--stack h/v`，
不猜——猜错了产出的是一个没人能看的对比。

## 另一个集合：OpenVDN 的 16 条（`artifacts/compare/`）

别人放出来的 prompt + 别人自己的视频，用来回答"在**不是我们挑的** prompt 上，我们的稀疏
路是不是也站得住"。和 v1 不同，它**不是控制变量的**：他们的视频是他们的模型、他们的
seed、他们的步数，只有 prompt 是共享的。所以它读的是"整条路线的观感"，不是"某个改动的
效果"——后者仍然只能在 v1 上读。

- 6 条 `cmp_*`：他们同时放了 50NFE 稠密和 8NFE VDN 两版，拼成 **2×2**
  （`Dense H3 50NFE (theirs)` / `VDN-H3 8NFE (theirs)` / `Dense 8NFE (ours)` /
  `Veda 8NFE 90% (ours)`）。
- 10 条 `only_*`：他们只放了一版，拼成 **3 路竖堆**
  （`OpenVDN (theirs)` / `Dense 8NFE (ours)` / `Veda 8NFE 90% (ours)`）。
- 全部 16:9@102，seed 0，我们这两路用
  `artifacts/init/accum8_100_leap08_fp8.safetensors`（leap 0.8，fp32 上混完再量化到 fp8）。
- 产物：每路单独 mp4 在 `artifacts/generate/openvdn16/<key>/`，拼接视频集中在
  `artifacts/visual_checks/joined_openvdn16/`（16 个文件，2688×1632 或 1344×2448）。

这 16 条同时是我们自己稠密 vs 稀疏计时的一个独立样本（同机同几何，非我们挑的 prompt）：
1×RTX PRO 6000 Blackwell 上稳态 step **56.4 s → 18.8 s**，其中注意力
**44.3 s → 6.8 s**，即注意力 **6.50×**、端到端 **3.00×**。

## 另一个集合：训练语料自己的 holdout

`artifacts/samples/moviegen_video_bench`（1003 条 MovieGen Video Bench 扩写，
`data/prompts/moviegen_video_bench_h3.jsonl`）按 seed 0 切出 20 条 test
（11 条 t37 + 9 条 t102，全 16:9）。它和训练同源、训练不会看到，**适合数值评测**
（`predictor_precision`），不适合和 v1 的历史视频拼对比——prompt 与几何都不同。

## 踩坑记录
- **`visual_check.py` 的标题栏依赖 ffmpeg 的 `drawtext`**，而这个滤镜要 ffmpeg 链接
  libfreetype——工作站的 homebrew 9.0.2 和 GPU 机器的静态 7.0.2 **都没有**。现在标题用
  Pillow 预渲染成紧凑 PNG 再 `overlay` 居中：不缩放所以字清晰，不用探测每路宽高，标签里
  的特殊字符也不再需要为 drawtext 转义。字体策略与 `infer.decode.title_bar` 共用一份，
  所以 `generate.py` 的并排图和这里的拼接图标题长得一样。

- **别用 mp4 的总数判断一批生成拉全了没有**：`generate.py` 在 dense 和 veda 同跑时会多写
  一个 `dense_vs_veda.mp4`，所以 16 条的目录里应该有 48 个文件而不是 32 个。我按 32 收工，
  结果 9 条的 `veda.mp4` 根本没拉下来，拼接脚本安静地跳过它们（只拼出 8 条）才暴露。
  按目录逐个核对 `dense.mp4` + `veda.mp4` 是否都在，别核对总数。

- **拿新 cache 去和旧视频拼**：id、几何、初始噪声全都不同，拼出来的对比毫无意义。要么
  用同一个 cache，要么把每一路都重跑一遍（dense 也要重跑，代价是小时级）。
- **`predictor_precision.py` 默认拒绝跨 step 比较**：它是为"同一 step 的不同存储精度"
  写的，混入不同 step 会把量化误差和训练进展搅在一起。跨 step 要显式加 `--across-steps`。
- **同一几何换片子的噪声和效果同量级**：训练日志里某个纵横比"连续三次退化"不足以判定
  退化。固定片子重测过才算（2026-09-29 就是这么把一次"16:9 在退化"的误判纠回来的，见
  `docs/features/training.md` 的验证记录）。

## 验证记录

- 2026-09-30，8×RTX PRO 6000 Blackwell：OpenVDN 16 条 prompt 的对比已生成，拼接视频在
  `artifacts/visual_checks/joined_openvdn16/`（6 个 2×2 + 10 个三路竖堆），单独视频在
  `artifacts/generate/openvdn16/`。同批测得注意力 6.50×、端到端 3.00×。
  **人工确认尚未进行。**
- 2026-09-29，1×RTX PRO 6000 Blackwell：v1 集合 20 条的三路对比
  （Dense / Veda t37 step600 / Veda 续训 step15）已生成，在
  `artifacts/visual_checks/holdout20_step200/2026-09-29/<clip>/`：每路一个单独 mp4 +
  一个 `side_by_side.mp4`（横版竖堆、竖版横排）。**人工确认尚未进行**（AGENTS.md 1.5
  要求确认人 / 日期 / commit / 结论）。t102 step200 fp8 那一路因为花屏没有参与。
- 2026-09-29，1×RTX PRO 6000 Blackwell：v1 集合的 cache 从 4090 机器复制过来后，
  `predictor_precision.py --across-steps` 在 `holdout14s_0000`（16:9@102）上比较
  已发布 step600 与续训 step5 / step15，8 个去噪步全部 step15 > step5 > release
  （recall 0.5592 → 0.5755 → 0.5768，heat_kept 0.5745 → 0.5837 → 0.5840，
  ceiling 0.7077）。视频一路（续训打分器）正在生成。

## 待办

- v1 的 prompt 清单（20 条扩写）是否入 `data/prompts/`，需要用户决定；不入库的话评测集
  只能靠复制传播。
- t72 的稠密去噪没有单独测过（表里留空），因为历史 dense 视频是在 4090 上跑的。
- 20 条的人工确认结论还没有记录到任何 feature 文档里（AGENTS.md 1.5 要求确认人 / 日期 /
  commit / 结论）。
