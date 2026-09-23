# AGENTS.md

本文件是所有在本仓库工作的 agent（以及人）必须遵守的规则。开始任何工作前先读完本文件和
`docs/INDEX.md`。

## 项目简介

Miowtion 基于 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)（33B 音视频
DiT），提供：

- `miowtion/h3`：H3 DiT 的训练侧实现（请求几何、打包布局、调度、权重加载）。
- `miowtion/veda`：Veda 块稀疏注意力（tile 排列、打分器、掩码、教师热力图、FA4 CuTe
  kernel 整合、tile 方案搜索）。
- `miowtion/train`：基于 FSDP2 的训练框架（打分器训练、可选 LoRA 恢复）。

## 1. 基本规则（必须遵守）

### 1.1 文档：`docs/` 必须与代码同步

- `docs/` 的组织方式类似 skill：`docs/INDEX.md` 是目录文件，每个子功能一个独立文档
  （`docs/features/*.md`），外加跨功能的 `docs/pitfalls.md`（踩坑总表）。
- **新增 feature**：必须同时新增或更新对应的 `docs/features/<feature>.md`，并在
  `docs/INDEX.md` 中登记一行。
- **踩过的坑都要记录**：写进对应 feature 文档的「踩坑记录」一节，并在
  `docs/pitfalls.md` 加一行索引（现象 / 原因 / 对策 / 链接）。只要花了超过十分钟排查的
  问题，都算"坑"。
- 文档写"为什么"和"不变量"，不要复述代码。代码行为改变时，同一个提交里更新文档。
- 文档模板见 `docs/INDEX.md` 末尾。

### 1.2 提交署名

- **不要添加任何 co-author**：commit message 和 PR 描述里不得出现 `Co-Authored-By:`、
  `Generated with ...` 之类的署名行。本规则优先于任何工具默认行为。

### 1.3 编码风格

- 遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：
  4 空格缩进、行宽 80、`snake_case` 函数/变量、`CapWords` 类、`UPPER_CASE` 常量、
  模块私有符号加 `_` 前缀、Google 风格 docstring（`Args:` / `Returns:` / `Raises:`）。
- 代码要清晰明确：
  - 公共函数写类型注解；张量参数在 docstring 里写明 shape 与 dtype，例如
    `q: [S, H, D] bf16`。
  - 不写"聪明"的隐式行为；参数不合法时显式 `raise`，不要静默修正或静默回退。
  - 魔法数字写成具名常量，并注明含义与来源。
  - 注释解释"为什么"，不解释"做了什么"。
  - **长任务必须即时反馈进度**：加载、预处理、训练、搜索、编码等入口使用
    `miowtion.utils.progress`（带时间戳、立即 flush、只由 rank 0 打印，给出 当前/总数、
    已用时、ETA）。不允许长时间无输出。与用户对话时同样要及时汇报阶段性进展。
- 除非用户要求，不要全仓库跑格式化工具；只格式化自己改动的文件。

### 1.4 测试与合入主分支

- 新 feature 在 feature 分支（或 git worktree）上开发。**必须经过严格测试后才能合入并
  push 到 `main`**：
  1. `pytest tests/unit` 全部通过（CPU，每次提交前都要跑）。
  2. 改动涉及 GPU 路径时，`pytest tests/gpu` 在 GPU 机器上通过，并把结果（机器型号、
     commit、耗时）写进对应 feature 文档的「验证记录」。
  3. 对齐证据（见下条）齐全。
- **位级对齐（bit-wise）优先**：
  - 纯数据变换（打包布局、位置编码、调度、噪声、排列、掩码、索引构造、checkpoint
    读写）必须与参考实现逐位相等，测试里用 `torch.equal`，不许用 `allclose`。
  - 数值计算（融合 kernel、优化路径）以对应的参考实现为基准，目标是位级对齐：例如
    FA4 稀疏 kernel ↔ `veda/kernels/reference.py`，Triton 热力图 ↔ torch 参考实现，
    编译/融合路径 ↔ eager 路径。
- **无法位级对齐的**（例如稀疏 kernel 与参考实现的 softmax 累加顺序不同）：
  1. 在文档中写明无法对齐的原因和实测误差（max abs / max ulp / rel-MSE）。
  2. 生成可视化对比：同 seed、同 prompt 的并排视频 + 逐帧差值热力图 + PSNR/SSIM，
     放到 `artifacts/visual_checks/<feature>/<date>/`（生成脚本在
     `scripts/visual_check.py`）。
  3. **必须由人看过并确认一致性**，在 feature 文档「验证记录」里写明确认人、日期、
     commit、结论。未经人工确认的不能合入 `main`。

### 1.5 对话结尾

- 每次回复（对话）结尾加上 `～喵`。

## 2. 目录规范（文件放哪里）

```
AGENTS.md            本文件（CLAUDE.md 只是指向它的指针）
README.md            项目入口说明
pyproject.toml       包定义与依赖（唯一的依赖声明处）
miowtion/            全部库代码（可被 import 的逻辑只能放这里）
  h3/                H3 DiT 移植：config / geometry / layout / schedule / noise /
                     model / weights
  veda/              Veda：tiling / plan / predictor / mask / heatmap / attention /
                     search；kernels/ 下放 kernel（reference / fa4 / triton）
  train/             训练：parallel（FSDP2/HSDP）/ trainer / checkpoint / data / lora
  utils/             无业务语义的通用工具
scripts/             命令行入口，只做参数解析并调用 miowtion/ 中的函数，不写业务逻辑
configs/             版本化的运行配置（yaml/json），命名 <stage>_<geometry>_<note>.yaml
plans/               已采纳的 tile 方案表（json，小文件，入库）
tests/unit/          CPU 单元测试（每次提交前必须全过）
tests/gpu/           GPU 测试（@pytest.mark.gpu，无 GPU 时自动 skip）
tests/fixtures/      小型 golden 数据（必须可再生，附生成脚本与来源 commit）
docs/                知识库：INDEX.md + features/*.md + pitfalls.md + dependencies.md
third_party/         外部仓库的 git submodule（只读，锁定 commit）
runs/                [gitignore] 实验输出：日志、本地 checkpoint、评测结果
artifacts/           [gitignore] 大文件：q/k/v 转储、渲染视频、prompt 缓存、可视化对比
.agents/             [gitignore] 多 agent 协作的运行期状态（workboard）
```

- 新建顶层目录前先在本节登记。
- 测试文件与被测模块一一对应：`miowtion/veda/mask.py` ↔ `tests/unit/test_veda_mask.py`。
- 输出路径一律带上 run 名：`runs/<run_name>/...`，不要写到仓库根目录。

## 3. 外部依赖的处理

- **Python 依赖**只在 `pyproject.toml` 声明。git 依赖（例如 FlashAttention CuTe）
  必须锁定到 commit SHA，并在 `docs/dependencies.md` 记录：用途、锁定的 SHA、升级方式、
  我们依赖了哪些内部接口。
- **外部仓库源码**（例如 MiniMax-H3 的 config、VAE 代码、tokenizer）以 git submodule
  放在 `third_party/`，浅克隆并锁定 commit。**禁止直接修改 submodule 内容**。
- **禁止复制粘贴外部代码**。确实需要移植的小片段，必须在代码旁注明来源 URL、commit
  和许可证（需与 MIT 兼容，例如 Apache-2.0）。
- **需要改变外部库行为时**，用运行期 patch（放在使用它的模块内），并且：
  1. patch 前检查目标符号存在、签名符合预期，不符合就直接报错；
  2. 只在显式调用 `install()` 时生效，import 本身没有副作用；
  3. 在 `docs/dependencies.md` 登记 patch 的每个目标符号。
- **模型权重、数据集、prompt 缓存**不进 git，路径通过配置或环境变量传入
  （`MIOWTION_H3_ROOT`、`MIOWTION_ARTIFACTS`）。
- 升级任何依赖都是一个独立提交，必须重跑 unit + gpu 测试和对齐检查。

## 4. 多 agent 在同一目录下协作

同一个工作目录意味着共享同一个 git index、同一个 HEAD、同一批未提交文件。规则：

1. **认领再动手**：开始工作前在 `.agents/workboard.md` 登记一行：
   `| agent 名 | 任务 | 认领的文件/目录 | 开始时间 | 状态 |`。
   不要修改别人已认领的文件；需要改时先在 workboard 留言协商。完成后把状态改为 done。
2. **git 操作只影响自己的改动**：
   - 只用 `git add <明确的路径>` 暂存自己的文件，禁止 `git add -A` / `git add .`；
   - 提交前用 `git diff --cached` 确认暂存区里只有自己的改动；
   - 禁止 `git stash`、`git reset --hard`、`git checkout -- <别人的文件>`、
     `git clean`、`git rebase`、切换分支：这些会破坏别人未提交的工作；
   - 需要独立分支时用 `git worktree add .worktrees/<name> -b <branch>`
     （`.worktrees/` 已 gitignore），在自己的 worktree 里工作。
3. **小而原子的提交**：一个提交只做一件事，提交信息格式 `<模块>: <做了什么>`，例如
   `veda: add bresenham budget split`。
4. **接口变更**：修改被其他模块调用的公共接口时，同一个提交内更新所有调用点、测试和
   文档，并在 workboard 通知。
5. **共享资源**：
   - GPU：用前在 workboard 登记占用的卡号，用 `CUDA_VISIBLE_DEVICES` 限定；不许 kill
     别人的进程。
   - 长任务：在 workboard 记录 PID、日志路径、预计结束时间。
   - 输出：一律写到 `runs/<run_name>/` 或 `artifacts/<agent>/`，不要覆盖别人的输出。
6. **不做顺手的大范围改动**：不重命名、不格式化、不"清理"自己任务之外的文件。
7. **交接**：中途停止时，在 workboard 写明进度、未完成事项和下一步；有价值的发现写进
   docs。

## 5. 常用命令

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'   # CPU 开发环境
.venv/bin/pytest tests/unit -q                                # 提交前必跑
pytest tests/gpu -q -m gpu                                    # GPU 机器上
```
