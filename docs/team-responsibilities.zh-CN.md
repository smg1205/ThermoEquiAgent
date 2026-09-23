# ThermoEqui-Agent 团队分工与交付规范

版本：v0.1

适用阶段：相平衡工程工作台第一阶段

维护方式：每个 Milestone 开始前更新负责人、交付日期和验收状态

> 本文中的“RAG”指检索增强生成。如果团队内部使用“RAD”表示其他技术，请在开始开发前补充明确定义。

## 1. 第一阶段共同目标

第一阶段先完成相平衡领域的端到端闭环：

```text
用户问题
→ DeepSeek + RAG + Skill 约束
→ 结构化任务 TaskManifest
→ 模型选择与参数检查
→ 确定性相平衡计算
→ 独立物理验证
→ 前端展示、溯源与导出
```

第一阶段完成前，不同时启动逆合成、催化剂筛选、塔板监测和搅拌桨设计的生产实现。可以预留领域插件接口，
但不能让后续领域需求破坏当前相平衡闭环。

## 2. 不可违反的共同规则

1. DeepSeek 只负责识别、规划、检索、调用工具和解释，不能直接生成相平衡数值。
2. 所有工程数值必须来自确定性计算工具或明确标记的 AI 预测模型。
3. 确定性相平衡结果必须通过 `validate_equilibrium_result`。
4. 禁止编造二元参数、实验数据、物性、适用范围和文献引用。
5. 缺少参数时返回结构化 `missing_parameters`，不能填入默认猜测值。
6. AI 预测值必须标记模型版本、训练数据版本、适用域、不确定度和验证状态。
7. 测试夹具只能放在 `tests/fixtures`，生产代码不得导入。
8. API Key 只通过环境变量或 GitHub Secrets 注入，不进入 Git、日志、数据库快照或前端。
9. 每项工作都通过 GitHub Issue、独立分支和 Pull Request 完成。

## 3. 成员认领表

| 工作组 | 主负责人 | 备份/评审人 | 当前 Milestone |
|---|---|---|---|
| 项目架构与产品协调 | 待填写 | 待填写 | M1 |
| 前端界面 | 待填写 | 待填写 | M1 |
| 后端与数据契约 | 待填写 | 待填写 | M1 |
| DeepSeek、RAG 与 Skills | 待填写 | 待填写 | M1–M2 |
| 相平衡模型与计算后端 | 待填写 | 待填写 | M1–M3 |
| 科学验证与质量保证 | 待填写 | 待填写 | M1–M4 |
| CI、部署与版本发布 | 待填写 | 待填写 | M1 |

一个人可以承担多个角色，但科学模型的实现者不能作为该模型唯一的验收人。

## 4. 项目架构与产品协调

### 职责

- 维护第一阶段范围、路线图、Milestone 和优先级。
- 定义前端、API、Agent、计算后端和数据库之间的边界。
- 主持公共 Schema、插件接口和重大架构变更评审。
- 把大任务拆成可以独立验收的 GitHub Issues。
- 处理跨工作组依赖和接口冲突。
- 确保当前版本不悄悄扩大到电解质、反应平衡、SLE、聚合物、VLLE 或流程模拟。

### 交付物

- 架构图和接口说明；
- Milestone 与 Issue 清单；
- 领域插件接口草案；
- 版本发布说明；
- 重大决策记录。

### 验收标准

- 每个 Issue 都有负责人、范围、依赖和验收标准；
- 公共接口改动经过相关工作组共同评审；
- 每个 Milestone 都能形成可运行的纵向闭环。

## 5. 前端界面组

负责目录：`apps/web/`

### 职责

- 建设工程对话界面和任务输入区。
- 展示并允许编辑组分、温度、压力、组成、任务类型和模型。
- 展示模型推荐、适用性评分、缺参原因和不可执行原因。
- 展示相图、数据表、相组成、相分率和计算状态。
- 展示参数来源、模型版本、验证报告和运行记录。
- 处理加载、超时、API 失败、缺少输入、缺少参数、求解失败和验证失败。
- 支持 JSON/CSV 导出。
- 与 `schemas/domain.py` 保持字段同步。

### 第一阶段 Issues

| 编号 | 任务 | 交付物 |
|---|---|---|
| FE-01 | 工作台页面骨架 | 对话区、任务区、结果区、响应式布局 |
| FE-02 | 类型化 API 客户端 | 统一请求、错误解析、request ID |
| FE-03 | 任务清单编辑 | 工况、组成、模型编辑与重新计算 |
| FE-04 | 结果可视化 | T-x-y、P-x-y、Flash、表格 |
| FE-05 | 证据与验证视图 | 参数来源、模型卡、验证检查 |
| FE-06 | 失败状态 | 缺参、不适用、不收敛、网关错误 |
| FE-07 | 前端自动化测试 | 关键用户流程和错误流程 |

### 验收标准

- 用户不查看终端也能判断任务处于哪个阶段；
- 不把 LLM 回答成功显示成热力学计算成功；
- 每个数值结果都能查看模型、后端、参数来源和验证状态；
- `pnpm --dir apps/web test`、`lint` 和 `build` 全部通过。

## 6. 后端与数据契约组

负责目录：`apps/api/`、`schemas/`、`database/`

### 职责

- 维护 FastAPI、OpenAPI 和 Pydantic 公共契约。
- 维护聊天、任务解析、模型推荐、计算、验证、运行查询和导出接口。
- 统一错误响应和 request ID。
- 保存原始问题、任务清单、参数快照、结果、验证报告和软件版本。
- 隔离第三方计算库对象，API 只返回统一 Schema。
- 维护数据库初始化、事务和 Repository。
- 为后续多领域插件设计统一任务和运行元数据。

### 第一阶段 Issues

| 编号 | 任务 | 交付物 |
|---|---|---|
| BE-01 | 公共任务与结果 Schema | Pydantic 真源和前端契约 |
| BE-02 | 聊天与任务接口 | `/api/chat`、`/api/tasks/parse` |
| BE-03 | 计算接口 | bubble、dew、VLE、Flash、azeotrope |
| BE-04 | 错误协议 | 统一错误码、request ID、脱敏 |
| BE-05 | 运行持久化 | 会话、任务、计算、验证和导出记录 |
| BE-06 | OpenAPI 与契约测试 | 路由覆盖、字段同步、错误测试 |
| BE-07 | 领域插件边界 | `domain`、`tool`、`result_kind` 元数据 |

### 验收标准

- 所有公共输入输出都通过 Pydantic 校验；
- Python Schema 与 TypeScript 类型同步；
- API 不返回第三方库内部对象；
- 运行记录可以查询、导出并追溯；
- `python -m pytest` 和 `mypy .` 通过。

## 7. DeepSeek、RAG 与 Skills 组

负责目录：`agent/`、`skills/`，以及后续检索模块

### 职责

- 维护 DeepSeek Provider、超时、重试和错误脱敏。
- 识别知识问答、计算、参数查询、模型比较和超范围任务。
- 从用户文本提取组分、CAS、工况、组成和任务类型。
- 使用 JSON Schema/Pydantic 约束 DeepSeek 输出。
- 建立 RAG 文档入库、检索、版本和审核状态。
- 建立 Skill 注册表、输入输出、工具白名单和失败协议。
- 防止提示注入、组分遗漏、条件篡改、参数虚构和越权调用。
- 维护 Agent 行为评测。

### DeepSeek 结构化输出

建议统一为：

```json
{
  "intent": "calculation",
  "domain": "phase_equilibrium",
  "task_manifest": {},
  "selected_tool": "phase_equilibrium",
  "missing_inputs": [],
  "assumptions": [],
  "answer_mode": "execute_then_explain"
}
```

### 第一阶段 Skills

| Skill | 作用 |
|---|---|
| `task-classification` | 判断问答、计算、参数、比较或超范围请求 |
| `component-grounding` | 将自然语言组分绑定到经过核验的身份 |
| `thermodynamic-model-routing` | 根据物系、相态和工况选择候选模型 |
| `parameter-retrieval` | 查找经过审核的参数与来源 |
| `phase-equilibrium-calculation` | 调用唯一允许的相平衡计算工具 |
| `thermodynamic-validation` | 调用独立验证器并解释检查结果 |
| `engineering-explanation` | 基于真实工具结果生成工程说明 |
| `scope-rejection` | 明确拒绝当前版本不支持的任务 |

### 第一阶段 Issues

| 编号 | 任务 | 交付物 |
|---|---|---|
| AI-01 | DeepSeek Provider | 配置、超时、重试、错误脱敏 |
| AI-02 | 结构化任务输出 | JSON Schema、修复一次、失败协议 |
| AI-03 | RAG 文档规范 | 来源、版本、审核状态、适用范围 |
| AI-04 | 检索与回答 | 模型卡、参数规范、验证规则检索 |
| AI-05 | Skill 工具白名单 | 禁止任意 Shell、Python 和数据库执行 |
| AI-06 | 安全与行为评测 | 虚构、歧义、注入、遗漏、越权测试 |

### 验收标准

- 非法输出不能进入计算层；
- 明确给出的组分和组成不能被遗漏或替换；
- 知识问答不能误触发计算；
- 缺少必要输入时必须提问或结构化失败；
- DeepSeek 不产生最终热力学数值；
- `evals/test_agent.py` 和 Provider 测试通过。

## 8. 相平衡模型与计算后端组

负责目录：`thermo_engine/`、`knowledge/model_cards/`、`knowledge/model_selection_rules/`

### 职责

- 维护统一 `ThermodynamicBackend` 接口和后端注册表。
- 集成确定性热力学框架，不把第三方对象泄漏到上层。
- 建立模型能力矩阵和适用性路由。
- 管理纯物性、二元参数、来源、单位、方向和适用范围。
- 实现泡点、露点、VLE、TP Flash、共沸搜索和计划中的 LLE。
- 为每个数值路径提供独立验证和基准测试。
- 建立 AI 物性/参数/代理模型的隔离接口。

### 模型集成优先级

| 优先级 | 模型或框架 | 目标 |
|---|---|---|
| P0 | Ideal/Raoult | 低压基线和端到端闭环 |
| P0 | CalebBell/thermo + Peng–Robinson | 烃类和轻气体确定性计算 |
| P0 | Phasepy/Peng–Robinson | 可替换 Python 后端 |
| P0 | Clapeyron.jl/Peng–Robinson | Julia 后端和交叉验证 |
| P1 | SRK | 第二种立方状态方程 |
| P1 | Wilson、NRTL、UNIQUAC | 非理想液相 VLE |
| P1 | UNIFAC/Modified UNIFAC | 基团贡献预测和缺参候选 |
| P2 | PC-SAFT、SAFT-VR Mie、CPA | 缔合和复杂分子体系 |
| P2 | LLE、VLLE、相稳定性 | 多液相和多相扩展 |
| 独立后续版本 | 电解质、反应、SLE、聚合物、水合物 | 不混入当前 v0.1 |

“整合所有模型”在工程上定义为：先登记模型能力与适用性，再对有参数证据、验证案例和维护责任的模型开放执行。

### 每个模型的标准交付包

1. 后端适配器；
2. 能力声明；
3. 模型卡；
4. 参数 Schema 和来源；
5. 缺参行为；
6. 不适用行为；
7. 至少一个公开基准案例；
8. 物理验证测试；
9. API 行为测试；
10. 使用说明和已知限制。

### AI 相平衡模型要求

AI 可以用于：

- 物性预测；
- 二元参数候选；
- 模型选择；
- Flash 或相图代理；
- 初值生成；
- 不确定度估计。

AI 结果必须额外包含：

```text
result_kind
model_version
training_data_version
applicability_domain
uncertainty
validated_against
review_status
```

未经审核的 AI 预测不能伪装成数据库参数或确定性工程结果。

### 第一阶段 Issues

| 编号 | 任务 | 交付物 |
|---|---|---|
| TH-01 | 后端协议和注册表 | 统一接口、别名和能力声明 |
| TH-02 | 参数仓库 | 来源、单位、方向、适用范围和哈希 |
| TH-03 | Ideal/Raoult 闭环 | bubble、dew、VLE、Flash |
| TH-04 | thermo/Peng–Robinson | PR 适配和 ChemSep 参数证据 |
| TH-05 | Phasepy 适配 | 统一结果和验证门 |
| TH-06 | Clapeyron 适配 | Julia 桥接、参数快照和验证 |
| TH-07 | 模型选择矩阵 | 适用、排除、缺参和风险规则 |
| TH-08 | 独立物理验证 | 衡算、残差、稳定性和适用域 |

### 验收标准

- 所有数值都来自注册后端；
- 缺参时不产生结果；
- 每个可执行模型都有基准与行为测试；
- 所有结果经过独立验证门；
- 第三方框架升级不会改变公共 API。

## 9. 科学验证与质量保证组

负责目录：`tests/`、`evals/`、`knowledge/validation_guides/`

### 职责

- 建立公开基准案例和允许误差。
- 对照实验数据、论文、手册或上游软件测试。
- 审核模型适用范围、参数来源和许可证。
- 维护数值回归、物理不变量和失败行为测试。
- 独立审核 AI 模型的适用域和不确定度。
- 对模型、参数和验证相关 PR 进行科学评审。

### 验收标准

- 模型实现者不是唯一审核人；
- 基准数据来源可追溯；
- 求解器收敛不等同于验证通过；
- 无法验证的结果必须标记 warning 或 failed；
- 测试参数不会进入生产参数仓库。

## 10. CI、部署与版本发布

负责目录：`.github/`、Docker 配置和环境模板

### 职责

- 维护后端、前端和可选计算后端 CI。
- 维护 Python、Node、pnpm、Julia 和依赖版本。
- 维护 Dockerfile、Compose 和部署文档。
- 管理 GitHub Secrets、分支保护和发布 Tag。
- 保证默认 CI 不调用付费外部 LLM API。

### 验收标准

- PR 必须通过 backend、frontend 和适用的 external-engines 检查；
- `main` 禁止直接 Push 和强制 Push；
- 发布说明包含模型、参数、验证、限制和迁移信息；
- 仓库、镜像和日志中没有密钥。

## 11. 工作依赖和合并顺序

推荐顺序：

```text
公共 Schema
→ 后端协议与错误格式
→ 确定性后端和验证器
→ Agent 工具契约
→ DeepSeek/RAG/Skills
→ 前端交互与结果展示
→ 端到端测试和发布
```

可以并行进行的工作：

- 前端先基于固定 JSON Mock 开发；
- 相平衡组基于 Python 公共接口开发；
- RAG 组先整理模型卡和文档元数据；
- 后端组维护最终契约并负责合流；
- 验证组独立准备基准数据。

禁止并行修改而没有协调的高冲突文件：

- `schemas/domain.py`
- `apps/api/main.py`
- `apps/web/src/lib/types.ts`
- `thermo_engine/service.py`
- `thermo_engine/registry.py`

## 12. GitHub 工作流程

1. 每项任务先创建 Issue。
2. Issue 写清范围、依赖、输入、输出和验收标准。
3. 从最新 `main` 创建短分支：

```text
feat/123-phasepy-flash
fix/234-component-grounding
docs/345-model-card
```

4. 一个 PR 只解决一个主要问题。
5. PR 必须关联 Issue，并填写仓库 PR 模板。
6. 科学模型 PR 至少需要：
   - 一名软件评审人；
   - 一名热力学/数据评审人。
7. CI 全绿、讨论解决后才能合并。
8. 优先使用 squash merge。

## 13. Milestone 规划

### M1：相平衡平台闭环

- 前端工作台；
- FastAPI 和公共 Schema；
- DeepSeek Provider；
- LangGraph 受控工具链；
- Ideal/Raoult；
- thermo/Peng–Robinson；
- 物理验证；
- 运行记录和导出。

完成定义：至少一个知识问答、一个 VLE 和一个 TP Flash 任务可以从网页完整执行并追溯。

### M2：RAG 与 Skill 约束

- 知识文档元数据；
- 检索和引用；
- Skill 注册表；
- JSON Schema 输出；
- 防虚构和提示注入评测；
- 模型选择解释。

完成定义：DeepSeek 的任务、工具和回答均被可测试的契约约束。

### M3：相平衡模型矩阵

- Phasepy；
- Clapeyron；
- SRK；
- NRTL、Wilson、UNIQUAC；
- UNIFAC；
- 统一参数仓库；
- 统一验证基准。

完成定义：每个标记为“可执行”的模型都有参数证据、基准案例和独立验证。

### M4：AI 辅助模型

- 物性预测；
- 参数候选；
- 相平衡代理模型；
- 适用域；
- 不确定度；
- AI 与确定性结果隔离。

完成定义：AI 结果不会被误认作经过审核的确定性工程结果。

## 14. 第二阶段领域扩展

第一阶段完成后，将系统扩展为公共平台加领域插件：

```text
platform/
  agent_runtime/
  rag/
  skills/
  api/
  database/
  validation/

domains/
  phase_equilibrium/
  retrosynthesis/
  catalyst_screening/
  tray_monitoring/
  impeller_design/
```

### 分子逆合成

- 分子表示与身份；
- 反应模板和路线搜索；
- 可合成性、成本、安全和绿色化学评分；
- 文献与供应商证据；
- 路线验证和专家审核。

### 催化剂筛选

- 催化剂和反应数据模型；
- 描述符与特征；
- 机器学习筛选；
- 多目标优化；
- 不确定度和实验反馈闭环。

### 塔板监测

- 时序数据接入；
- 工况标签和数据质量；
- 异常检测；
- 软测量；
- 故障诊断；
- 报警依据与人工确认。

塔板监测不等于流程设计。完整精馏塔设计应作为独立后续范围。

### 搅拌桨设计

- 流体性质和操作条件；
- 桨型选择；
- 功率准数和雷诺数；
- 混合时间、传质和悬浮判据；
- 经验关联式和 CFD 接口；
- 设计安全裕量和适用范围。

五个领域共享身份、RAG、Skills、权限、运行记录和前端框架，但必须拥有独立的工具、Schema、知识库、
计算方法、验证标准和负责人。

## 15. Pull Request 完成定义

一个任务只有同时满足以下条件才算完成：

- 代码已实现；
- 自动化测试已增加；
- 科学假设和适用范围已记录；
- 参数和数据来源已记录；
- 错误与失败行为已覆盖；
- 公共 Schema 已同步；
- 文档已更新；
- CI 全绿；
- 至少一名其他成员批准；
- 没有密钥、数据库、缓存或测试产物进入 Git。
