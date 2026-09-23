# docs 目录

本目录是项目知识库，组织方式类似 skill：本文件是目录，每个子功能一个文档。改代码时同步改对应
文档；踩过的坑写进 feature 文档的「踩坑记录」，并在 `pitfalls.md` 加一行索引。

| 文档 | 内容 | 状态 |
|---|---|---|
| [features/h3_model.md](features/h3_model.md) | H3 DiT 训练侧实现：请求几何、打包布局、调度、噪声、模型、权重加载 | CPU 测试通过；GPU 待验证 |
| [features/veda_tiling.md](features/veda_tiling.md) | tile 排列（按头动态 tile 形状 + 按 tile 变长）、方案表 | CPU 测试通过 |
| [features/veda_predictor_mask.md](features/veda_predictor_mask.md) | 打分器、预算与掩码规则、教师热力图、KL、recall | CPU 测试通过 |
| [features/veda_kernel.md](features/veda_kernel.md) | FA4 CuTe 块稀疏整合、架构支持、测速结果 | SM89 不支持（已实测）；SM90/100 待验证 |
| [features/tile_search.md](features/tile_search.md) | oracle 评分、投票、每层 ≤2 种形状、搜索驱动 | CPU 测试通过；GPU 待验证 |
| [features/training.md](features/training.md) | FSDP2 训练框架：阶段 1/2、AdaLN 表、数据、checkpoint | CPU 端到端测试通过；GPU 待验证 |
| [features/prompt_expansion.md](features/prompt_expansion.md) | prompt 扩写：短 prompt → H3 T2VA 结构化 prompt（DeepSeek；system prompt 由 H3 skill 拼成；校验 + repair 重试） | CPU 测试通过；10 条冒烟通过 |
| [dependencies.md](dependencies.md) | 外部依赖、锁定版本、依赖的内部接口 | 当前 |
| [pitfalls.md](pitfalls.md) | 踩坑总表 | 持续更新 |

## feature 文档模板

```markdown
# <功能名>

## 目标
## 设计与不变量        （为什么这样做；哪些性质不能被破坏）
## 代码位置与接口
## 测试               （unit / gpu，各测什么）
## 踩坑记录           （现象 / 原因 / 对策）
## 验证记录           （日期、机器、commit、结果；无法位级对齐时的人工确认）
## 待办
```
