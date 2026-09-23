# Agent 与热力学引擎集成

## 决策

ThermoEqui-Agent 保留其 FastAPI/Next.js 产品外壳，并**有选择地**采用 [datamllab/CAi_copilot](https://github.com/datamllab/CAi_copilot) 所展示的面向工具的 Agent 边界。CAi_copilot 当前的 Agent 内核使用 LangGraph、LLM 与 REPL，因此属于 LangChain 生态，并非单纯的经典 LangChain `AgentExecutor`。本仓库**未内嵌**任何 CAi_copilot 源代码。

这是一次**架构层面的集成**，而非源码搬运。其不受限的 REPL/Bash 执行模型与本仓库的规则相冲突——LLM 可以编排，但绝不可计算相平衡数值。生产计算回路现已改为有界的 LangGraph `StateGraph`，节点为 `plan -> execute -> validate -> respond`，直接沿用 CAi_copilot 面向图的 Agent 外壳，同时**只暴露白名单内的 `phase_equilibrium` 工具**。数值工作仍留在经审核的 `ThermodynamicBackend` 适配器内部；图中不提供任意 Python、Bash 与 notebook 执行能力。

确定性 Peng-Robinson 计算通过本地适配器使用固定版本的 [`thermo==0.6.1`](https://github.com/CalebBell/thermo) 包。可选适配器直接集成 Phasepy，并通过官方 `pyclapeyron` 桥接 Clapeyron.jl。**上游对象绝不跨越应用边界**：每个适配器都把结果映射为 `CalculationResult`，记录参数证据，并送入 `validate_equilibrium_result`。

安装可选引擎：

```bash
python -m pip install -e ".[phase-engines]"
```

请使用 Python 3.11 或 3.12。Phasepy 目前尚未发布 Python 3.13 的 Windows wheel。首次 `import pyclapeyron` 可能会下载 Julia 并对 Clapeyron 做一次预编译。

## 思考—执行契约

1. **Plan（规划）** — DeepSeek 返回符合所提供 JSON Schema 的 `TaskManifest`。它可以把 `model_name` 留空以交由确定性路由处理，且**不得计算相平衡数值**。
2. **Execute（执行）** — LangGraph 工作流调用 `EngineeringToolRegistry`，后者仅允许 `phase_equilibrium` 工具。后端注册表解析出经审核的模型适配器。
3. **Validate（校验）** — 独立检查组成、相分率、物料衡算、相平衡残差、收敛性、适用性以及可得的相稳定性证据。
4. **Respond（回应）** — DeepSeek 只能解释工具结果与校验报告。**缺乏依据的数字与引用会被扣留。**

UI 只暴露以上四个高层审计步骤，**不暴露模型的思维链**。

## 生产实现矩阵

| 模型 | 后端 | 当前可执行范围 | 参数策略 | 状态 |
|---|---|---|---|---|
| Ideal/Raoult | 内置确定性求解器 | 二元泡点 / 露点、等压 / 等温 VLE、TP 闪蒸、共沸点候选搜索 | 经审核的本地纯组分性质关联式 | 可用，低压基准 |
| Peng-Robinson | CalebBell/thermo 适配器 | 烃类 / 白名单轻气体泡点 / 露点、二元 VLE 曲线、TP 闪蒸、相态分类、共沸点候选搜索 | 纯组分性质取自 thermo；每对二元组分必须存在于 ChemSep PR；精确的矩阵 / 顺序 / 形式 / 单位 / 版本会被快照并哈希 | 可用 |
| Phasepy/Peng-Robinson | Phasepy 0.0.56 可选适配器 | 烃类 / 白名单轻气体泡点 / 露点、二元 VLE 曲线、两相 TP 闪蒸、共沸点候选搜索 | 纯组分性质取自 thermo；每对二元组分必须存在于 ChemSep PR；返回的相态再由逸度残差复检 | 安装 `phase-engines` 后可用 |
| Clapeyron/Peng-Robinson | pyclapeyron 0.1.1 + Clapeyron.jl 可选适配器 | 泡点 / 露点、二元 VLE 曲线、单 / 两相 Gibbs TP 闪蒸、相态分类、共沸点候选搜索 | CAS 标识通过 thermo 落实；Clapeyron 打包的纯组分参数与参考文献，加上经审核的 ChemSep PR `kij` 矩阵，会被快照并哈希；缺失的组分对返回 `missing_parameters` | Julia 初始化成功时可用 |
| Wilson | 规划中的活度系数适配器 | VLE 契约 | 需要经审核的有方向二元参数 | 仅契约 |
| NRTL | 规划中的活度系数适配器 | VLE/LLE 契约 | 需要经审核的有方向二元参数 | 仅契约 |
| UNIQUAC | 规划中的活度系数适配器 | VLE/LLE 契约 | 需要经审核的二元参数与结构常数 | 仅契约 |

"仅契约"的模型仍会出现在推荐与 schema 设计中，但**不能产出计算结果**。它们以结构化的 `missing_parameters` 失败，而不会使用合成的默认值。

## 扩展接缝

新增模型或引擎的方式：实现 `ThermodynamicBackend`，在 `ThermodynamicBackendRegistry` 中注册其别名与支持的计算类型，提供带证据的参数来源，并在 `calculate_equilibrium` 或 HTTP 计算端点处补充行为测试。**不要**把 DeepSeek 提示词或前端组件与上游引擎对象耦合。

## v0.1 有意划定的边界

电解质、反应平衡、固液平衡（SLE）、聚合物、水合物、石油假组分、多晶型、汽液液平衡（VLLE）与流程设计均被显式拒绝。LLE 目前只有带类型的契约，尚无生产数值后端。
