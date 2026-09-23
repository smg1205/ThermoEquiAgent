# 路线图

## P0 模型广度（2026-08）

模型集成现已优先于待补的实验数据集：

- **`UNIFAC` 原始试点**：基于 CalebBell/thermo DDBST 基团归属的预测型基团贡献后端；不需要二元 `ParameterSet`，缺失基团归属时返回结构化的 `missing_parameters`。
- **`RK` 试点**：基于 `thermo.RKMIX` 的显式 kij 二元适配器，与既有 SRK 试点共用新的 `PilotKijCubicEosBackend` 基类。
- **多模型对比**：`POST /api/calculations/compare` 对同一任务运行所有可执行模型，返回各模型结果、校验报告、参数来源与结构化失败。该端点为纯后端；前端接入由 UI 集成任务负责。
- **注册表契约测试**现已覆盖每一个已注册后端，包括新增的 UNIFAC 与 RK 条目，对照目录 / 模型卡元数据与公开校验闸门进行验证。

两个试点均保持 `production_ready=false`。其毕业检查清单为：参数溯源审核、至少一项实验或软件参考基准、适用范围检查，之后才置 `production_ready=true`。

## PGSSI 一等试点（2026-08）

PGSSI（本组的 Physics-Guided 3D Solute-Solvent Interaction 框架）现已作为一等预测后端集成，与 NRTL/UNIQUAC/Wilson **并列**，而非作为任何 VLE 模型的依赖：

- `PgssiBackend`（`thermo_engine/pgssi_backend.py`）实现了完整的 `ThermodynamicBackend` 协议，并以 `PGSSI` 之名注册于 `DEFAULT_BACKEND_REGISTRY`，支持新的 `infinite_dilution_activity` 计算类型。
- 它由溶质 / 溶剂的 SMILES 与温度预测**随温度变化的无限稀释活度系数**（`log-gamma_inf = K1 + K2/T`）。它不需要二元 `ParameterSet`；检查点、SMILES、源码树或可选依赖缺失，都属于结构化的 `missing_parameters` 失败。
- VLE / 闪蒸操作以 `unsupported_model` **显式失败**；PGSSI 绝不假装自己是相平衡求解器，也没有任何 VLE 后端依赖 PGSSI。
- 一条 gamma-infinity 到 NRTL 的回归桥（`thermo_engine/pgssi_params.py`）把 gamma-infinity 数据转换为生产用 NRTL 的 `a+b/T` 参数形式；完整链路（gamma-infinity 数据 → NRTL 参数 → 等压泡点 → 校验闸门）已用真实乙醇 / 水数据跑通。
- 新端点：`POST /api/calculations/infinite-dilution-activity`。

PGSSI 仍为 `production_ready=false`。其毕业要求：经审核的训练检查点、实验 gamma-infinity 或有限浓度 VLE 基准闭合，以及适用性审核。本组数据集（`39,840` 行 gamma-infinity）是候选基准来源；任何专有数据都必须留在公开仓库之外。

## P0 加固（2026-08）

参数流水线现已闭合，无需等待新的实验数据：

- 经审核的活度系数参数从硬编码模块迁入 `knowledge/parameters/*.yaml`，并通过 `thermoequi-seed` 幂等播种。
- 后端与路由只消费 `ParameterSet` 记录；参数缺失会产生结构化的 `missing_parameters` 失败。
- 注册表契约测试对照模型目录 / 模型卡、参数来源报告与公开校验闸门，验证能力声明。
- Peng-Robinson 与 SRK 现在共用一份 `CubicEosBackend` 实现；既有 SRK/PR 行为测试加上新的注册表契约测试覆盖了此次重构。

对于经 ChemSep 验证、并以实验等压 VLE 数据为基准的乙醇 / 水与乙醇 / 苯二元体系，NRTL、UNIQUAC 与 Wilson 的 `production_ready` 现为 `true`。SRK 在经审核的 kij 覆盖与基准闭合完成之前保持 `false`。

**第四阶段**将加入有证据支撑的参数回归、生产级 NRTL/UNIQUAC 二元 LLE、前端对多模型对比 API 的接入、敏感性分析、PDF 报告，以及 DWSIM/Aspen 配置支持。

SRK 现已有试点 `thermo` 适配器；将其启用为 `production_ready` 需要经审核的 kij 数据与基准闭合。后续研究可能评估 Dortmund-UNIFAC、CPA、PC-SAFT、eNRTL 与 Pitzer 适配器。**范围扩展只在验证覆盖到位之后进行**；电解质、固液平衡（SLE）、汽液液平衡（VLLE）与完整流程模拟在当前版本中仍在范围之外。

Phasepy 与 Clapeyron.jl 现已具备面向经审核非电解质 VLE 边界的可选 Peng-Robinson 适配器。下一步集成工作是有证据支撑的 Phasepy NRTL/UNIQUAC 支持，以及 Clapeyron 的缔合 / SAFT 模型；在参数溯源与行为验证用例就位之前，二者均不得启用。NeqSim 仍是潜在的工业级 JVM 适配器。**所有引擎都隔离在 `ThermodynamicBackend` 之后，并必须通过相同的证据与校验闸门。**
