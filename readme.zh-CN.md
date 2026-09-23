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
**[docs/heptane-nonane-case.zh-CN.md](docs/heptane-nonane-case.zh-CN.md)**（英文版：
[docs/heptane-nonane-case.en.md](docs/heptane-nonane-case.en.md)）。

该案例覆盖正庚烷 / 正壬烷直接二元精馏参考体系：对照 NIST ThermoML 数据的三源泡点比较、
塔设计、DWSIM 导出，以及从工作台复现该案例的准确输入语句。它是**推荐的首次运行对象**，
因为无需萃取剂筛选。

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
| `tests/`、`evals/` | 行为测试与 Agent 评测 |
| `docs/` | 中英双语的架构、DWSIM 与方法学文档 |

## 科学规则

以下规则在**代码中强制执行**，而不只是写在文档里：

1. 语言模型可以分类、检索、编排与解释，**绝不计算或编造相平衡数值**。
2. 每一个数值结果都来自 `thermo_engine`，并通过校验。
3. 求解器状态**不等于**物理校验：组成、物料衡算、相平衡残差、收敛性与参数适用性都会被检查。
4. 参数缺失时产生结构化的 `missing_parameters` 失败。二元参数、实验数据与文献引用**绝不编造**。

## 开发

```powershell
python -m pytest                      # 后端测试
ruff check . && mypy .                # 静态检查与类型检查
pnpm --dir apps/web test              # 前端测试
docker compose up --build             # 完整栈
```

## 文档

全部文档在 `docs/` 下以中英双语维护，文件后缀为 `.zh-CN.md` 或 `.en.md`，每个文档都有另一
语种的对应版本。

常用入口：

| 文档 | 内容 |
|---|---|
| `docs/heptane-nonane-case.zh-CN.md` | 端到端完整案例 |
| `docs/repository-guide.zh-CN.md` | 目录与文件说明 |
| `docs/agent-architecture.zh-CN.md` | Agent 编排细节 |
| `docs/model_applicability.zh-CN.md` | 模型适用范围与筛选规则 |
| `docs/dwsim-dwxmz-export-guide.zh-CN.md` | DWSIM 文件生成与格式 |
| `docs/ThermoFormer.zh-CN.md` | ThermoFormer 模型与其结果 |
