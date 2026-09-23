# ThermoAgent

> [English](readme.md) | **中文**

ThermoAgent 是一个对话式热力学工程工作台。它把一句自然语言描述的问题，转化为可复现的
VLE / LLE 计算，并在需要时生成可直接用 DWSIM 打开的流程文件。

设计目标是**可追溯**：输出中的每一个数值都能归因于实验来源、ThermoFormer 预测或 DWSIM
计算三者之一。语言模型只负责路由与解释——**它绝不编造相平衡数值**。

## 能做什么

- **二元与三元 VLE** —— 泡点、露点、等压与等温 VLE、TP 闪蒸、共沸点搜索。
- **二元与三元 LLE** —— 液液共存端点与分相计算。
- **精馏塔设计** —— 短节法 Fenske / Underwood / Gilliland 设计（理论板数、回流比、
  进料板、产品温度），覆盖普通精馏与萃取精馏。
- **DWSIM 导出** —— 生成 `.dwxmz` 流程文件，覆盖 TP 闪蒸、二元精馏、萃取精馏与液液萃取。
- **知识图谱推理** —— 面向模型、任务、参数与体系类型的相平衡知识图谱，支持带排除关系的
  多跳查询。
- **检索增强问答** —— 概念与流程类问题基于带版本的知识文档作答，并附来源归属。
- **三源校验** —— 把实验数据、ThermoFormer 预测与 DWSIM 计算并排比较，让模型可靠性可被
  判断，而不是被假定。

v0.1 明确不支持（会被显式拒绝）：电解质、反应平衡、固液平衡（SLE）、聚合物、水合物、
石油假组分、多晶型。

## 工作原理

```
自然语言
      │
      ▼
  意图分流 ──► 组分解析 ──► 模型选择
      │                                            │
      │                        ┌───────────────────┴───────────────────┐
      ▼                        ▼                                       ▼
  缺参数报告              thermo_engine                        ThermoFormer
  （绝不猜测）            （经典模型）                          （神经网络 VLE / LLE）
                               └───────────────┬───────────────────────┘
                                               ▼
                                          相平衡校验
                                               ▼
                                设计 / 报告 / DWSIM .dwxmz
```

模型选择是确定性的，发生在语言模型之外：

| 任务 | protocol | 权值文件 |
|---|---|---|
| 二元 VLE | `vle_overall_binary` | `models/vle/prediction/vle_overall_binary/seed_2/best_model.pt` |
| 三元 VLE | `vle_overall_ternary` | `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt` |
| 二元 LLE | `binary-system` | `models/lle/prediction/binary-system/seed_0/best.pt` |

当必要信息缺失时，系统返回结构化的 `missing_parameters` 失败，**而不是用假设填补空白**。

## 知识图谱

`knowledge_graph/` 存放用于模型适用性推理的相平衡知识图谱。它与驱动路由的模型卡同源构建，
并持久化到 `data/knowledge_graph.json`。

**实体**由 `ThermoEntityExtractor` 从文本中抽取，把热力学术语解析为带类型的节点：`model`、
`task`、`component`、`system_type`、`parameter` 与 `property`。实体匹配采用**精确匹配**，
因此 `propane` 绝不会误配到 `propanol` 节点。

**关系**以固定词汇表连接这些节点（`knowledge_graph/graph.py` 中的 `RelationshipType`）：

| 关系 | 含义 |
|---|---|
| `supports_task` | 某模型能执行某任务（VLE、FLASH、LLE） |
| `excludes` | 某模型对某体系类型明确不适用 |
| `requires_parameter` | 某模型执行时需要二元参数集 |
| `uses_model` | 某任务或报告由某模型支撑 |
| `has_property` | 某组分或模型具有某性质 |
| `has_relationship` | 两个节点存在关联，但无更具体类型 |
| `belongs_to` | 某节点隶属于某族或某组 |
| `describes` | 某文档或模型卡描述某节点 |
| `applies_to` | 某参数或规则适用于某体系类型 |

有两点设计值得说明。节点 ID 带类型前缀（`model:peng-robinson` 与 `task:pr`），因此不同
命名空间下的同名短名不会冲突。**排除关系被建模为显式的 `excludes` 边**，而不是"缺少
`belongs_to` 边"，这正是否定式问题能够被回答的原因。

**查询**通过 `GraphQueryEngine` 完成：从问题中抽取实体、定位匹配节点，再走一跳收集其关系。
这支持诸如"NTRL 不适用于哪些体系"这类多跳问题——答案来自沿 `excludes` 边走图，而非对文本
做模式匹配。`GraphQuerySkill` 为该引擎提供面向 Agent 的封装，在图结果之上叠加一层可选的
LLM 解释：**图谱提供事实，LLM 只负责措辞**。

图谱可用 `KnowledgeGraph.export_graphml()` 导出为 GraphML，便于在标准图可视化工具中查看；
`build_graph_from_kb()` 则可由 `knowledge/` 下的 YAML 模型卡重建图谱。

## 检索与技能

- `rag/` —— 知识库的文档加载、切分、向量化与向量检索。
- `skills/` —— Agent 可调用的带类型能力：热力学计算、模型路由、校验、知识库问答、
  图谱查询，以及相平衡架构与评测技能。每个技能都声明其输入、输出与失败行为。
- `evals/` —— Agent 行为评测，覆盖编造、歧义、提示注入、遗漏与越权。

## 完整案例

一个端到端的完整案例单独成文：
**[docs/heptane-nonane-case.en.md](docs/heptane-nonane-case.en.md)**。

该案例覆盖正庚烷 / 正壬烷直接二元精馏参考体系：对照 NIST ThermoML 数据的三源泡点比较、
塔设计、DWSIM 导出，以及从工作台复现该案例的准确输入语句。它是**推荐的首次运行对象**，
因为无需萃取剂筛选。

完整操作录屏见
[`demo_video/heptane-nonane-dwsim-demo.mp4`](demo_video/heptane-nonane-dwsim-demo.mp4)
（3.8 MB）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `agent/` | 意图分流、编排、有界执行图 |
| `thermo_engine/` | 经典模型、塔设计、DWSIM 导出 |
| `knowledge_graph/` | 热力学知识图谱与查询引擎 |
| `knowledge/` | 模型卡、参数数据、校验基准 |
| `rag/` | 检索增强生成流水线 |
| `skills/` | Agent 技能及其契约 |
| `apps/` | FastAPI 后端与 React 工作台 |
| `schemas/` | Pydantic 领域模型与 API 契约 |
| `database/` | 持久化模型与会话 |
| `lab_models/` | ThermoFormer 源码、配置、数据集与权值 |
| `scripts/` | 复现入口脚本 |
| `demo_video/` | 工作台操作录屏 |
| `tests/`、`evals/` | 行为测试与 Agent 评测 |
| `docs/` | 中英双语的架构、DWSIM 与方法学文档 |

## 科学规则

以下规则在**代码中强制执行**，而不只是写在文档里：

1. 语言模型可以分类、检索、编排与解释，**绝不计算或编造相平衡数值**。
2. 每一个数值结果都来自 `thermo_engine`，并通过校验。
3. 求解器状态**不等于**物理校验：组成、物料衡算、相平衡残差、收敛性与参数适用性都会被检查。
4. 参数缺失时产生结构化的 `missing_parameters` 失败。二元参数、实验数据与文献引用**绝不编造**。

## 安装

### 环境要求

| 工具 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 以上（推荐 3.12） | Phasepy 目前尚无 Python 3.13 的 Windows wheel |
| Node.js | 22 以上 | 前端基于 Next.js 16 |
| pnpm | 11.9.0 | 由 `packageManager` 字段声明，可用 Corepack 激活 |
| DWSIM | 9.x，仅 Windows | 仅在打开导出的 `.dwxmz` 文件时需要 |

### 1. 后端

创建虚拟环境并安装 Python 依赖：

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1

# 方式 A —— 安装固定版本的依赖清单
python -m pip install -r requirements.txt
```

更推荐把项目本身装上，这样会同时注册 `thermoequi` 与 `thermoequi-seed` 两个命令行入口，
并且可以按需选择依赖组：

```powershell
# 方式 B —— 仅核心运行时
python -m pip install -e .

# ……再加上全部可选组
python -m pip install -e ".[dev,phase-engines,thermoformer,dwsim]"
```

各可选组与 `requirements.txt` 中的分段一一对应：

| 依赖组 | 追加安装 | 用途 |
|---|---|---|
| `dev` | pytest、mypy、ruff | 运行测试与静态检查 |
| `phase-engines` | Phasepy、pyclapeyron | Phasepy 与 Clapeyron 后端 |
| `thermoformer` | torch、rdkit、unimol-tools、scikit-learn | ThermoFormer 神经网络后端 |
| `dwsim` | pythonnet | DWSIM 自动化（Windows，需 .NET 运行时） |
| `skills` | sentence-transformers | 知识库检索 |

随后生成本地环境文件并初始化参数库：

```powershell
Copy-Item .env.example .env
thermoequi-seed          # 把 knowledge/parameters/*.yaml 载入数据库
```

`.env` 只保留在本机（已被 git 忽略）。若没有私有模型权值，把对应的 checkpoint 变量留空即可：
相关后端仍保持注册状态，会返回结构化的 `missing_parameters` 失败，而不会编造数值。

### 2. 前端

前端是 `apps/web` 下的 pnpm 工作区成员。如果本机没有 pnpm，先用 Corepack 启用，再安装依赖：

```powershell
corepack enable          # 使 packageManager 声明的 pnpm 11.9.0 可用
corepack prepare pnpm@11.9.0 --activate

pnpm --dir apps/web install --frozen-lockfile
```

`--frozen-lockfile` 会严格按 `apps/web/pnpm-lock.yaml` 安装，与 CI 一致。只有在确实要修改依赖时
才去掉该参数。

也可以在仓库根目录执行：

```powershell
pnpm install --frozen-lockfile
```

### 3. 启动

需要两个进程，各自一个终端：

```powershell
# 终端 1 —— 后端，http://localhost:8000
python -m uvicorn apps.api.main:app --reload --port 8000

# 终端 2 —— 前端，http://localhost:3000
pnpm --dir apps/web dev
```

浏览器打开 `http://localhost:3000` 进入工作台，`http://localhost:8000/docs` 查看 OpenAPI，
`http://localhost:8000/health` 确认服务与 Provider 状态。

默认 `LLM_PROVIDER=deterministic`，**不需要任何 API Key**：意图分流、模型选择与计算完全由本地
规则完成。如需让模型参与措辞与解释，在 `.env` 中设置 `LLM_PROVIDER=deepseek` 与 `DEEPSEEK_API_KEY`。

### 4. 容器化方式

```powershell
docker compose up --build     # 后端 8000，前端 3000
```

默认数据库为 SQLite。如需具备迁移能力的部署，在 `.env` 中设置 PostgreSQL 的 SQLAlchemy URL。

## 开发

```powershell
python -m pytest                      # 后端测试
ruff check . && ruff format --check . # 静态检查与格式检查
mypy .                                # 严格类型检查
pnpm --dir apps/web test              # 前端测试
pnpm --dir apps/web lint              # 前端 lint
pnpm --dir apps/web build             # 前端生产构建
```

`python -m pytest` 从 `pyproject.toml` 读取配置，会同时运行 `tests/` 与 `evals/`。

## 文档

`docs/` 下的技术文档以英文维护，文件后缀为 `.en.md`；本 README 另提供中文版
（[readme.zh-CN.md](readme.zh-CN.md)）。

常用入口：

| 文档 | 内容 |
|---|---|
| `docs/heptane-nonane-case.en.md` | 端到端完整案例 |
| `docs/repository-guide.en.md` | 目录与文件说明 |
| `docs/agent-architecture.en.md` | Agent 编排细节 |
| `docs/model_applicability.en.md` | 模型适用范围与筛选规则 |
| `docs/dwsim-dwxmz-export-guide.en.md` | DWSIM 文件生成与格式 |
| `docs/ThermoFormer.en.md` | ThermoFormer 模型与其结果 |
