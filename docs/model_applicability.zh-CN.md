# 模型适用性

## 目标

当前的模型适用性层提供两处集中能力：一处用于描述经审核的热力学模型范围，另一处用于在执行逻辑之前施加最小化的候选筛选。

其当前目标为：

- 统一管理模型适用性元数据。
- 依据结构化任务工况筛选候选模型。
- 为每一条目录条目返回明确的保留或排除理由。

本层当前**并不**自行执行自动路由。最终模型路由仍与这一元数据及筛选层相互独立。

## 当前选择流程

当前项目流程为：

- `knowledge/model_cards/*.yaml`
- `agent/router.py`
- `thermo_engine.model_applicability`
- `agent/executor.py`
- `thermo_engine.service`

实际执行时：

1. `agent/router.py` 加载 `model_cards` 并生成候选推荐。
2. 在候选推荐过程中，路由会调用模型适用性层做**逐模型的硬约束检查**。
3. 适用性层可以把某个候选标记为不可执行，并附加明确的排除理由。
4. 路由仍保留推荐对象与原有评分结构。
5. `agent/executor.py` 仍负责经由 `thermo_engine.service` 完成实际执行。

这意味着：

- **适用性层**负责模型范围检查、执行约束与排除理由。
- **路由**负责候选生成与最终推荐排序。
- **执行器与 `thermo_engine`** 负责后端解析与计算。

## 已实现内容

当前已就位的部分：

- `knowledge/model_catalog/*.yaml`
  - 仓库当前跟踪的经审核模型的静态目录条目。
- `schemas/model_catalog.py`
  - 目录条目的 Pydantic schema，含严格的 `extra="forbid"` 校验。
- `thermo_engine/model_catalog.py`
  - 模型目录的 YAML 加载器。
- `schemas/model_applicability.py`
  - 适用性筛选的请求与结果 schema。
- `thermo_engine/model_applicability.py`
  - 针对当前目录的最小化基于规则的候选筛选。
- 测试
  - `tests/test_model_catalog.py`
  - `tests/test_model_applicability.py`

## 当前模型状态

| 模型 | 后端标签 | implementation_status | production_ready | 范围摘要 |
|---|---|---|---|---|
| `Ideal/Raoult` | `internal` | `available` | `true` | 本地纯组分性质注册表范围内的低压经审核非电解质 VLE 与闪蒸基准 |
| `Peng-Robinson` | `thermo` | `available` | `true` | 烃类与经审核轻气体体系的中高压 VLE 与闪蒸 |
| `Phasepy/Peng-Robinson` | `phasepy` | `available` | `false` | 面向 VLE 与闪蒸的可选外部 Peng-Robinson 后端 |
| `Clapeyron/Peng-Robinson` | `clapeyron` | `available` | `false` | 面向 VLE 与闪蒸的可选外部 Peng-Robinson 后端 |
| `SRK` | `thermo` | `available` | `false` | 试点二元 SRK VLE/闪蒸，需要经审核或用户具证的显式 kij `ParameterSet`；基准闭合待完成 |
| `RK` | `thermo` | `available` | `false` | 试点二元 Redlich-Kwong VLE/闪蒸，需要经审核或用户具证的显式 kij `ParameterSet`；基准闭合待完成 |
| `UNIFAC` | `thermo` | `available` | `false` | 预测型原始 UNIFAC 试点，使用 DDBST 基团归属；不需要二元 `ParameterSet`；基准闭合待完成 |
| `NRTL` | `internal` | `available` | `true` | 经 ChemSep 验证的乙醇 / 水与乙醇 / 苯二元体系的中低压非理想 VLE 与闪蒸；遗留 DECHEMA 参数集仍属原型 |
| `UNIQUAC` | `internal` | `available` | `true` | 经 ChemSep 验证的乙醇 / 水与乙醇 / 苯二元体系的中低压非理想 VLE 与闪蒸；UNIQUAC 基准不含纯组分端点 |
| `Wilson` | `internal` | `available` | `true` | 经 ChemSep 验证的乙醇 / 水与乙醇 / 苯二元体系的中低压 VLE 与闪蒸；**显式拒绝 LLE** |
| `PGSSI` | `pgssi` | `available` | `false` | 由 SMILES 预测随温度变化的无限稀释活度系数（gamma-infinity）的一等预测后端；需要训练好的检查点；基准闭合待完成 |

## 当前筛选规则

当前的适用性筛选**有意保持最小化**。它只施加以下规则：

1. `calculation_type`
   - 若任务 `calculation_type` 未列在该目录条目的 `supported_calculation_types` 中，则排除该模型。
2. `equilibrium_type`
   - 若任务 `equilibrium_type` 未列在该目录条目的 `supported_equilibrium_types` 中，则排除该模型。
3. `implementation_status`
   - 若模型的 `implementation_status` 为 `contract_only`，则排除该模型。
4. `production_ready`
   - 当 `production_only=True` 时，若 `production_ready=false`，则排除该模型。
5. 二元参数可得性
   - 若 `requires_binary_parameters=true` 且其模型名未出现在 `available_parameter_models` 中，则排除该模型。

筛选会为每一条目录条目返回一个结果，并累积所有适用的排除理由。若未触发任何排除规则，结果为 `keep` 并附一条简短的正向理由。

## 模型适用性规则

仓库现在还提供一个小型单模型适用性检查，用于回答：

- 某个具名模型是否允许用于某一请求的问题形态
- 允许或拒绝的原因

该检查使用 `thermo_engine.model_applicability.is_model_allowed(...)`，当前需要：

- `model_name`
- `calculation_type`
- `equilibrium_type`
- `available_parameters`

它返回：

- `allowed: bool`
- `reason: str`

当前各模型的具体规则**有意保持保守**，且不改变目录的执行状态。本适用性层只做候选模型筛查：它不实现新的热力学后端，不修改 `registry.py`、`service.py` 或 `router.py`，也不提升 `production_ready`。

### NRTL

适用于：

- 非理想液相 VLE
- 已由共享活度系数后端实现的闪蒸类计算
- 具备有效二元交互参数的场景

以下情况拒绝：

- 所需参数不可得
- 所请求的计算超出支持范围
- 请求 `LLE`
- 所请求的相平衡类型不受支持

当前状态：

- 后端代码已实现并注册在当前代码包中
- 经审核的二元参数通过生产参数库管理，并以 `thermoequi-seed` 播种
- 对于经 ChemSep 验证、并以实验等压 VLE 数据为基准的乙醇 / 水与乙醇 / 苯二元体系，`production_ready` 为 `true`
- 其他二元体系的遗留 DECHEMA 参数集仍可用，但尚未经生产验证

### UNIQUAC

适用于：

- 非理想液相 VLE
- 已由共享活度系数后端实现的闪蒸类计算
- 具备有效二元交互参数的场景

以下情况拒绝：

- 所需参数不可得
- 所请求的计算超出支持范围
- 请求 `LLE`
- 所请求的相平衡类型不受支持

当前状态：

- 后端代码已实现并注册在当前代码包中
- 经审核的二元参数通过生产参数库管理，并以 `thermoequi-seed` 播种
- 对于经 ChemSep 验证、并以实验等压 VLE 数据为基准的乙醇 / 水与乙醇 / 苯二元体系，`production_ready` 为 `true`
- UNIQUAC 的实验基准**不含纯组分端点**，因为当前后端在 x=0/1 处组合项无定义
- 其他二元体系的遗留 DECHEMA 参数集仍可用，但尚未经生产验证

### Wilson

适用于：

- VLE
- 已由共享活度系数后端实现的闪蒸类计算

以下情况拒绝：

- `LLE`
- 所请求的相平衡类型不受支持
- 所请求的计算超出支持范围
- 所需二元参数不可得

当前状态：

- 后端代码已实现并注册在当前代码包中
- 经审核的二元参数通过生产参数库管理，并以 `thermoequi-seed` 播种
- 对于经 ChemSep 验证、并以实验等压 VLE 数据为基准的乙醇 / 水与乙醇 / 苯二元体系，`production_ready` 为 `true`
- 其他二元体系的遗留 DECHEMA 参数集仍可用，但尚未经生产验证

### UNIFAC

适用于：

- 中低压非电解质 VLE 与闪蒸筛选
- 可获得 DDBST UNIFAC 基团归属的多组分体系
- 不存在经审核二元交互参数集的场景

以下情况拒绝：

- 某组分没有 DDBST 基团归属或没有 CAS 号
- 所请求的计算超出支持范围
- 当前试点中请求 `LLE`
- 所请求的相平衡类型不受支持

当前状态：

- 预测型原始 UNIFAC 后端已实现并注册
- 在实验基准与适用性审核完成之前，`production_ready` 为 `false`

### RK

适用于：

- 中高压烃类 / 轻气体二元 VLE 与闪蒸
- 具备经审核或用户具证的显式 RK kij `ParameterSet` 的体系

以下情况拒绝：

- 所需二元 `kij` 参数不可得
- 请求的组分数超过两个
- 请求 `LLE`
- 所请求的相平衡类型不受支持

当前状态：

- 试点 `thermo.RKMIX` 后端已实现并注册
- 在经审核的 kij 覆盖与基准闭合完成之前，`production_ready` 为 `false`

### PGSSI

适用于：

- 由溶质 / 溶剂 SMILES 与温度预测随温度变化的无限稀释活度系数（gamma-infinity）
  （`infinite_dilution_activity` 计算类型）
- 各组分带有 SMILES 标识的体系

以下情况拒绝：

- 所请求的计算类型不是 `infinite_dilution_activity`
- 未配置训练好的 PGSSI 检查点（`PGSSI_CHECKPOINT`）
- PGSSI 源码树不可达（`PGSSI_SRC`）
- 未安装 torch / torch_geometric / rdkit
- 某组分缺少 SMILES 标识

当前状态：

- 一等 `PgssiBackend` 已实现，并与 NRTL/UNIQUAC/Wilson 并列注册；
  它独立于二元 `ParameterSet` 记录，且没有任何 VLE 后端依赖它
- 在实验基准闭合与适用性审核完成之前，`production_ready` 为 `false`
- gamma-infinity 到 NRTL 的参数回归桥（`thermo_engine/pgssi_params.py`）
  可作为可选辅助工具；回归得到的参数在投入生产前需要有限浓度 VLE 验证

## 当前边界

本功能的当前边界**有意保持狭窄**：

- 适用性筛选**尚未接入** `executor`、`router` 或后端解析。
- 即使代码库中已存在某个后端，本层也不改变其后端执行逻辑。
- 本文档不评估、不总结外部传统模型代码的质量。
- 本文档不描述未经确认的 AI 模型能力。
- `SRK`、`RK` 与 `UNIFAC` 作为试点适配器跟踪；在经审核的参数、基准闭合与适用性审核完成之前，`production_ready` 保持 `false`。

当前实现中有意排除的内容同样包括：

- 压力阈值规则
- 组分性质规则
- 排序或评分
- 自动路由
- 注册表同步改由 `tests/test_backend_registry_contract.py` 覆盖

## 最小示例

```python
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from schemas.model_applicability import ModelApplicabilityRequest
from thermo_engine.model_applicability import filter_applicable_models

task = TaskManifest(
    equilibrium_type="VLE",
    calculation_type="isobaric_vle",
    components=[
        ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
        ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
    ],
    conditions=ThermodynamicConditions(pressure_kPa=101.325),
)

report = filter_applicable_models(
    ModelApplicabilityRequest(
        task=task,
        production_only=True,
        available_parameter_models={"Peng-Robinson"},
    )
)

for item in report.results:
    print(item.model_name, item.decision, item.reasons)
```

本示例只执行基于目录的筛选。它**不执行**任何热力学后端，也**不自动选择**最终模型。
