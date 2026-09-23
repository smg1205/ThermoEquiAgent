# 工程目录与文件说明

本文是当前仓库的导航图。原则是：源码、契约、测试、知识证据和协作配置纳入 Git；本地环境、密钥、
数据库、依赖目录和构建缓存留在开发机。

## 根目录

| 路径 | 作用 |
|---|---|
| `.dockerignore` | 控制 Docker 构建上下文，排除本地依赖、缓存和敏感文件。 |
| `.env.example` | 环境变量模板；只放变量名和安全默认值，不放真实 Key。 |
| `.gitattributes` | 统一 Git 文本和换行行为。 |
| `.gitignore` | 排除虚拟环境、缓存、数据库、日志、前端依赖和构建产物。 |
| `AGENTS.md` | 给 AI 编程代理的仓库级科学边界和工程规则。 |
| `CONTRIBUTING.md` | 团队分支、测试、提交、PR 和科学评审规范。 |
| `PLANS.md` | 架构决策、版本阶段和交付状态。 |
| `README.md` | 项目入口：能力、架构、启动、测试、API 与已知限制。 |
| `docker-compose.yml` | API、Web 和持久化卷的一键容器编排。 |
| `package.json` | 根目录 pnpm 快捷命令，转发到 `apps/web`。 |
| `pyproject.toml` | Python 包信息、依赖、CLI、pytest、Ruff 和 mypy 配置。 |

## GitHub 配置

| 路径 | 作用 |
|---|---|
| `.github/workflows/ci.yml` | 后端、前端和可选 Phasepy/Clapeyron 集成的 CI。 |
| `.github/pull_request_template.md` | PR 的科学安全、契约同步和验证证据检查表。 |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | 可复现 Bug 的结构化 Issue 表单。 |
| `.github/ISSUE_TEMPLATE/model_integration.yml` | 新模型、参数源或后端集成的证据与验收表单。 |
| `.github/ISSUE_TEMPLATE/config.yml` | 禁止无结构空白 Issue，引导团队使用模板。 |

## Agent 编排：`agent/`

| 路径 | 作用 |
|---|---|
| `agent/__init__.py` | Agent 包边界和公共导出。 |
| `agent/providers.py` | 确定性、DeepSeek、OpenAI Provider；约束结构化输出并过滤无依据内容。 |
| `agent/orchestrator.py` | 对话状态、组分识别、意图分类、任务清单构建和多轮上下文合并。 |
| `agent/graph_workflow.py` | 有界 LangGraph：`plan → execute → validate → respond`。 |
| `agent/tools.py` | Agent 可调用工具的白名单；当前只暴露受控相平衡工具。 |
| `agent/router.py` | 读取模型卡，构造物系画像并按适用性推荐模型。 |
| `agent/executor.py` | 调用确定性服务并强制通过独立验证门。 |

这里是“思考 + 执行”的连接层，不允许出现自行计算相平衡数值的 LLM 逻辑。

## API 与前端：`apps/`

| 路径 | 作用 |
|---|---|
| `apps/__init__.py` | 应用包标记。 |
| `apps/api/__init__.py` | FastAPI 子包标记。 |
| `apps/api/main.py` | 应用入口、Provider 配置、中间件、异常映射、聊天/计算/模型/参数/运行导出端点。 |
| `apps/api/Dockerfile` | 后端生产镜像构建和 Uvicorn 启动。 |
| `apps/web/Dockerfile` | Next.js 前端生产镜像构建。 |
| `apps/web/package.json` | 前端依赖和 dev/test/lint/build 命令。 |
| `apps/web/pnpm-lock.yaml` | 精确锁定前端依赖树，保证 CI 与成员环境一致。 |
| `apps/web/pnpm-workspace.yaml` | pnpm 工作区配置。 |
| `apps/web/next.config.ts` | Next.js 配置。 |
| `apps/web/next-env.d.ts` | Next.js 自动类型声明入口。 |
| `apps/web/tsconfig.json` | TypeScript 严格模式、路径别名和编译设置。 |
| `apps/web/eslint.config.mjs` | ESLint 与 Next.js 规则。 |
| `apps/web/vitest.config.ts` | Vitest、jsdom 和测试初始化配置。 |
| `apps/web/src/app/layout.tsx` | 全局 HTML 布局和页面元数据。 |
| `apps/web/src/app/page.tsx` | 首页入口，挂载工程工作台。 |
| `apps/web/src/app/globals.css` | 全局布局、配色、表单、图表与响应式样式。 |
| `apps/web/src/components/Workbench.tsx` | 主交互界面：对话、任务编辑、运行、结果页签和导出。 |
| `apps/web/src/components/VleChart.tsx` | 把统一平衡点结果绘制为 Plotly 相图。 |
| `apps/web/src/components/Workbench.test.tsx` | 工作台聊天、计算、失败提示和重算交互测试。 |
| `apps/web/src/components/VleChart.test.tsx` | 相图组件和数据映射测试。 |
| `apps/web/src/lib/api.ts` | 浏览器到 FastAPI 的类型化 HTTP 客户端和统一错误处理。 |
| `apps/web/src/lib/types.ts` | 与 Pydantic API 契约同步的 TypeScript 类型。 |
| `apps/web/src/test/setup.ts` | Vitest DOM 匹配器和浏览器测试初始化。 |
| `apps/web/src/types/react-plotly.d.ts` | `react-plotly.js` 的本地类型补充。 |

## 确定性热力学内核：`thermo_engine/`

| 路径 | 作用 |
|---|---|
| `thermo_engine/__init__.py` | 对外暴露计算、路由、验证和注册表入口。 |
| `thermo_engine/backend.py` | 所有计算后端必须实现的 `ThermodynamicBackend` 协议。 |
| `thermo_engine/registry.py` | 后端别名、支持任务和构造器的中央注册表。 |
| `thermo_engine/service.py` | 公共计算边界：范围拒绝、身份核验、模型路由、后端调用和验证入口。 |
| `thermo_engine/ideal.py` | 内置 Ideal/Raoult 泡点、露点、VLE、Flash 与共沸候选求解。 |
| `thermo_engine/thermo_backend.py` | CalebBell/thermo 的 Peng–Robinson 适配器。 |
| `thermo_engine/phasepy_backend.py` | 可选 Phasepy/Peng–Robinson 适配器。 |
| `thermo_engine/clapeyron_backend.py` | 通过 pyclapeyron 调用 Clapeyron.jl/Peng–Robinson。 |
| `thermo_engine/properties.py` | 纯组分身份、Antoine 关联式和来源记录。 |
| `thermo_engine/identity.py` | 组分名称/CAS 解析、别名核验、电解质和歧义识别。 |
| `thermo_engine/parameters.py` | 有方向二元参数的显式反向转换。 |
| `thermo_engine/units.py` | 压力、温度和摩尔组成的显式归一化。 |
| `thermo_engine/validation.py` | 组成、物料衡算、平衡残差、收敛、适用性与稳定性检查。 |
| `thermo_engine/errors.py` | 缺参、不支持范围等结构化领域异常。 |
| `thermo_engine/cli.py` | 从 JSON 任务运行确定性计算的命令行入口。 |

## 公共契约：`schemas/`

| 路径 | 作用 |
|---|---|
| `schemas/__init__.py` | Schema 包导出。 |
| `schemas/domain.py` | 任务、条件、模型卡、参数、结果、验证、聊天和错误的 Pydantic 真源。 |

修改这里时必须同步 `apps/web/src/lib/types.ts`、API 端点和契约测试。

## 数据持久化：`database/`

| 路径 | 作用 |
|---|---|
| `database/__init__.py` | 数据库包导出。 |
| `database/models.py` | 会话、消息、任务、参数、运行、平衡点、验证和导出的 SQLAlchemy 表。 |
| `database/session.py` | 数据库引擎、初始化、事务上下文和仓储读写。 |

本地默认生成 `thermoequi.db`；它是运行数据，不进入 Git。

## 模型与知识资产：`knowledge/`

| 路径 | 作用 |
|---|---|
| `knowledge/model_cards/ideal.yaml` | Ideal/Raoult 的适用性、风险和可执行状态。 |
| `knowledge/model_cards/peng-robinson.yaml` | Peng–Robinson 的适用性和参数要求。 |
| `knowledge/model_cards/wilson.yaml` | Wilson 模型卡；当前为契约/规划资产。 |
| `knowledge/model_cards/nrtl.yaml` | NRTL 模型卡；当前为契约/规划资产。 |
| `knowledge/model_cards/uniquac.yaml` | UNIQUAC 模型卡；当前为契约/规划资产。 |
| `knowledge/model_selection_rules/core.yaml` | 按相态、压力、物系和任务筛选/排除模型的规则。 |
| `knowledge/fundamentals/vle.md` | VLE 基础知识与术语。 |
| `knowledge/engineering_cases/benzene-toluene.md` | 苯–甲苯理想体系演示案例和边界。 |
| `knowledge/parameter_guides/import.md` | 参数导入、方向、来源和审核要求。 |
| `knowledge/validation_guides/vle.md` | VLE 结果的工程验证清单。 |
| `knowledge/software_mappings/README.md` | 不同第三方软件/库与统一模型契约的映射说明。 |

模型卡即使暂时不可执行，也参与显式推荐、拒绝和路线图，不属于废弃文件。

## 项目内 AI 工作规范：`skills/`

| 路径 | 作用 |
|---|---|
| `skills/agent-tool-contract/SKILL.md` | Agent 工具白名单、输入输出和安全边界。 |
| `skills/frontend-engineering-workbench/SKILL.md` | 工程工作台前端开发约束。 |
| `skills/phase-equilibrium-architecture/SKILL.md` | 相平衡系统分层和确定性边界。 |
| `skills/phase-equilibrium-evals/SKILL.md` | Agent 科学评测与失败用例要求。 |
| `skills/thermodynamic-calculation/SKILL.md` | 确定性计算实现与测试规则。 |
| `skills/thermodynamic-model-routing/SKILL.md` | 模型适用性路由规则。 |
| `skills/thermodynamic-validation/SKILL.md` | 独立物理验证门要求。 |
| `skills/*/agents/openai.yaml` | 对应本地 Skill 的代理发现与调用元数据。 |

这些文件是后续使用 Codex/其他代理协作时的项目知识，不参与生产运行，但应保留在仓库。

## 文档：`docs/`

| 路径 | 作用 |
|---|---|
| `docs/architecture.md` | 系统分层、数据流和安全边界。 |
| `docs/agent-thinking-execution.zh-CN.md` | 当前“思考 + 执行”落地状态的中文简述。 |
| `docs/api.md` | FastAPI 路由、请求响应和错误契约。 |
| `docs/deployment.md` | 本地和容器部署说明。 |
| `docs/frontend.md` | 前端布局、状态和接口行为。 |
| `docs/github-collaboration.zh-CN.md` | GitHub 团队、分支保护、Issue、PR、密钥和发布设置。 |
| `docs/integrations.md` | CAi_copilot 思路、LangGraph、thermo、Phasepy、Clapeyron 的真实集成矩阵。 |
| `docs/methodology.md` | 工程方法、假设和计算流程。 |
| `docs/model_routing.md` | 模型筛选、评分和硬排除逻辑。 |
| `docs/parameter_evidence.md` | 参数证据、可追溯性和测试夹具隔离。 |
| `docs/repository-guide.zh-CN.md` | 本文件；仓库地图和保留策略。 |
| `docs/roadmap.md` | 后续阶段与未完成能力。 |
| `docs/team-responsibilities.zh-CN.md` | 团队岗位职责、Issue 拆分、交付物、验收标准和后续领域扩展。 |
| `docs/thermodynamic_scope.md` | v0.1 支持与明确拒绝的科学范围。 |
| `docs/validation.md` | 独立物理验证规则和状态解释。 |

## 示例、测试与评测

| 路径 | 作用 |
|---|---|
| `examples/benzene_toluene_isobaric.json` | CLI 可直接运行的苯–甲苯等压 VLE 示例输入。 |
| `tests/fixtures/synthetic_nrtl.json` | 仅供测试的合成 NRTL 参数，严禁生产代码导入。 |
| `tests/test_api.py` | HTTP、OpenAPI、聊天、DeepSeek 修正、运行持久化和导出测试。 |
| `tests/test_database.py` | 数据库及测试夹具隔离测试。 |
| `tests/test_deepseek_provider.py` | DeepSeek 协议、结构化输出、安全过滤和编排边界测试。 |
| `tests/test_external_backends.py` | Phasepy/Clapeyron 可选依赖、参数和验证门测试。 |
| `tests/test_frontend_contract.py` | Python API 与 TypeScript 类型字段同步测试。 |
| `tests/test_schemas_validation.py` | Pydantic 约束和不合格结果验证测试。 |
| `tests/test_thermo_engine.py` | 单位、泡露点、VLE、Flash、路由、缺参、范围和物理不变量测试。 |
| `evals/test_agent.py` | 面向自然语言任务的 Agent 行为评测，包括歧义、越界和防虚构。 |

## 不进入 Git 的本地内容

| 路径 | 处理方式 |
|---|---|
| `.venv312/`、`.venv/` | Python 虚拟环境；保留在本机，可按依赖清单重建。 |
| `apps/web/node_modules/` | 前端依赖；保留在本机，可由 lockfile 重建。 |
| `apps/web/.next/` | Next.js dev/build 产物；可删除并自动重建。 |
| `.mypy_cache/`、`.ruff_cache/`、`__pycache__/` | 静态检查和 Python 缓存；可随时删除。 |
| `.coverage`、`htmlcov/`、`coverage/` | 测试覆盖率产物；可随时删除。 |
| `thermoequi_agent.egg-info/` | 可编辑安装元数据；重新安装会生成。 |
| `thermoequi.db`、`*.db`、`*.sqlite3` | 本地运行数据；不提交，删除前应确认是否需要历史记录。 |
| `.env` | 本地密钥和环境配置；永不提交。 |

## 核心调用链

```text
apps/web
  → apps/api/main.py
  → agent/orchestrator.py
  → agent/graph_workflow.py
  → agent/tools.py
  → agent/executor.py
  → thermo_engine/service.py
  → thermo_engine/registry.py
  → 具体后端
  → thermo_engine/validation.py
  → API / 数据库 / 前端
```

团队成员先按这个调用链定位问题，再在 Issue 中明确负责层，能显著减少跨层修改和契约冲突。
