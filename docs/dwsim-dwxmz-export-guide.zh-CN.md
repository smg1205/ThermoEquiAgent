# DWSIM 工程文件（`.dwxmz`）的生成、格式与导出数据要求

## 摘要

本文说明 ThermoAgent 生成 DWSIM 工程文件的机制，剖析 `.dwxmz` 的格式，并列出各导出路径所需的数据与校验约束。系统不直接序列化工程文件，而是经 .NET Automation 接口驱动本机 DWSIM 完成流程构建与落盘：文件格式的权威性由 DWSIM 保证，本项目的职责限于提供组分、状态与设计参数，并规避自动化接口的版本差异。格式描述基于对实际导出文件的检查，接口与校验规则以 `thermo_engine/dwsim_export.py`、`agent/extractive_distillation.py`、`schemas/column_design.py` 的当前实现为准。

> 口语中的“dwism 文件”即 DWSIM 文件；`dwism` 是代码中显式接受的拼写变体（`_DWSIM_FILE_MARKERS`）。

---

## 1　生成机制

### 1.1　总体架构

**本系统不生成 `.dwxmz` 的字节内容，而是调用本机 DWSIM 的 .NET Automation 接口，由 DWSIM 自身完成流程构建与序列化。** Python 侧仅承担意图识别、参数解析、确定性塔设计与接口调用。其工程含义是：涉及格式细节的问题，权威来源是 DWSIM 安装本身；本项目需处理的是自动化接口在各版本间暴露的方法名、属性可写性与静默失败行为的差异。

```
自然语言请求
  └─ agent/extractive_distillation.py      意图识别、参数解析、塔设计编排
       └─ thermo_engine/column_design.py   确定性短节法（Fenske–Underwood–Gilliland）
            └─ thermo_engine/dwsim_export.py    调用 DWSIM Automation API
                 └─ DWSIM .NET 程序集（pythonnet / clr） → .dwxmz
```

### 1.2　程序集加载与流程构建

程序集采用惰性加载（`_automation_factory`），仅在发起导出时才 `import clr`，故未安装 DWSIM 的主机仍可运行计算核心。加载序列为：读取 `.env` 取得 `DWSIM_HOME`；校验 `<DWSIM_HOME>/DWSIM.Automation.dll` 存在；将 `TEMP`/`TMP` 指向 `DWSIM_TEMP_DIR`（默认 `<cwd>/.tmp/dwsim`）；`import clr` 并将安装目录追加至 `sys.path`；`clr.AddReference` 加载自动化程序集；导入 `Automation3` 与 `ObjectType`。

需注意该函数返回的是**两个类**而非实例，调用方须再执行 `factory()` 完成实例化。以 TP 闪蒸为例，构建过程为：

```python
automation = factory()                     # 实例化 Automation3
flowsheet  = automation.CreateFlowsheet()  # 创建空流程
flowsheet.AddCompound("n-Heptane")         # 添加组分（键名大小写严格）
_add_property_package(flowsheet, "NRTL")   # 添加物性包
feed  = flowsheet.AddObject(ObjectType.MaterialStream, 0,   0, "Feed")
sep   = flowsheet.AddObject(ObjectType.Vessel,        250,  0, "Equilibrium Flash")
vapor = flowsheet.AddObject(ObjectType.MaterialStream, 500, -80, "Vapor Product")
fs = _simulation_object(feed)              # 经 GetAsObject() 解包为具体类型
fs.SetTemperature(temperature_K)           # K
fs.SetPressure(pressure_kPa * 1000.0)      # kPa → Pa（DWSIM 内部以 Pa 接收）
fs.SetMolarFlow(1.0)                       # 摩尔流量，TP 闪蒸归一化为 1.0
fs.SetOverallComposition(Array[Double](composition))
flowsheet.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
_save_flowsheet_via_temp(automation, flowsheet, destination)
```

三处细节须注意：`AddCompound` 键名大小写严格且须与 DWSIM 化合物字典一致，本项目由 `_DWSIM_COMPOUND_MAP` 将模型侧名称（`heptane`、`2-propanol`、`DMSO`）规范化为字典键（`n-Heptane`、`Isopropanol`、`Dimethyl sulfoxide`），键名不符将抛 `KeyNotFoundException` 并中止导出；DWSIM 9 返回通用接口对象，须经 `GetAsObject()` 解包后方可设置状态；压力须换算为 Pa，组成须转为 `System.Array[Double]`。

各版本保存方法不一致，`_save_flowsheet` 依序尝试 `SaveFlowsheet2`、`SaveFlowsheet(_, _, True)`、`SaveFlowsheet(_, _)`、`SaveToXML`、`SaveToFile`；外层 `_save_flowsheet_via_temp` 在遇到 `UnauthorizedAccessException` 时改为先存至临时目录再复制回目标路径。

### 1.3　求解与落盘的时序

DWSIM 保存的是**当前状态**，由此产生两类产物。其一为**仅含结构骨架的文件**：保存前未触发求解时，文件中每个对象的 `<Calculated>` 均为 `false`，界面中表现为相分率为空、产品流量为零（本文样本即属此类，其 8 处 `<Calculated>` 与 3 处 `<AtEquilibrium>` 皆为 `false`）。TP 闪蒸与塔类导出默认采用此模式。其二为**携带求解结果的文件**：LLE 分相类导出在保存前显式调用 `CalculateFlowsheet4` 并读回两相流量与组成，故落盘文件已含真实结果。

求解失败时文件仍会写出（结构正确，可在界面重算），但系统输出警告，不将未分相的结果表述为已收敛。另需注意 `CalculateFlowsheet2` **会抑制求解错误**，仅 `CalculateFlowsheet4` 回报错误——判定“塔是否算成功”必须依据后者。

### 1.4　严格精馏塔的附加装配

严格塔的装配复杂度高于闪蒸，除通用步骤外还须写入下列内容，每项均对应一处已被规避的接口缺陷。

| 装配项 | 所用 API | 未执行的后果 |
|:--|:--|:--|
| 塔板数 | `SetNumberOfStages(n)` | 必须用该方法；仅赋值 `NumberOfStages` 不重建 `Stages` 列表，DWSIM 以新数量索引旧列表而抛 `ArgumentOutOfRangeException: 索引超出范围` |
| 进料绑定塔板 | 先 `ConnectFeed`，后 `SetStreamFeedStage` | 顺序固定不可互换；仅前者则塔板号读回 `-1`，仅后者抛 `NullReferenceException` |
| 两个塔规格 | `SetCondenserSpec` / `SetReboilerSpec`，或写 `Specs["C"]`/`Specs["R"]` | 严格塔欠定；再沸器规格默认为 0，报 `Failed to fulfill mass balance ... Relative Error = ~1.0` |
| 求解器配置 | `SolvingMethodName` / `MaxIterations` | 默认以 Wang–Henke 法 100 次迭代冷启动，高塔必然报迭代上限 |
| 初值剖面 | `SetInitialTemperature/LiquidMolarFlow/VaporMolarFlowEstimates` | 缺少初值剖面是塔不收敛的首要原因 |
| 压降 | `ColumnPressureDrop`（Pa） | 未写入时沿用 DWSIM 默认值 |
| 产品连接 | `ConnectDistillate` / `ConnectBottoms` | 用通用 `ConnectObjects` 不报错但连接未建立，求解显示成功而产品流量为零 |

`SolvingMethodName` 的 setter 不作校验，赋值错误仅在求解时以 `Unable to find column solver with name '...'` 暴露。本机实测可解析的名称仅两个：`Wang-Henke Bubble-Point (BP) Solver` 与 `Modified Wang-Henke Bubble-Point (MBP) Solver`。

### 1.5　生成流程总览

```
① 意图识别    含 dwsim/dwism/dwxmz/导出/下载/export/download，且能识别出体系
② 参数解析    组分、组成、流量、压力、纯度、回收率等；缺失项取默认值
③ 组件映射    模型名 → DWSIM 化合物字典键；无映射则报 missing_parameters
④ 塔设计      确定性短节法给出板数、回流比、进料板、塔顶与塔釜温度
⑤ 程序集加载  DWSIM.Automation.dll + Automation3 + ObjectType
⑥ 流程构建    CreateFlowsheet → AddCompound → 物性包 → AddObject → 设状态 → 连线
⑦ 塔参数写入  塔板数、进料板、两个规格、求解器、初值剖面、压降
⑧ LLE 先求解  CalculateFlowsheet4，读回两相结果
⑨ 保存        依序尝试五种保存方法，必要时经临时目录中转
⑩ 提供下载    /api/export/extractive/<file-id>.dwxmz
```

---

## 2　文件格式

### 2.1　容器与顶层结构

`.dwxmz` 是 DWSIM 9 的原生工程文件，容器为 **ZIP 压缩包**（魔数 `50 4B 03 04`），内部仅含**一个以 GUID 命名的 XML**。对一份实际导出的二元精馏塔文件检查如下：

```
文件        hep_nonane_web.dwxmz                          32,944 字节
压缩包内容  4b95689b-7955-40a2-adf3-46bc8b5333c0.xml     265,082 字节
```

即 `.dwxmz` = ZIP(单个 `{GUID}.xml`)。XML 未压缩，可用任意 ZIP 工具解开直接阅读。旧格式 `.dwrsd` 不予使用；所有导出强制 `.dwxmz` 扩展名，否则抛 `ValueError`。

解压后根节点为 `<DWSIM_Simulation_Data>`，子节点分组如下。其中 `SimulationObjects` 为核心，其余多为界面布局、结果缓存或本项目留空的节点（`ReactionSets`、`Reactions`、`DynamicsManager` 等）。

| 顶层节点 | 作用 |
|---|---|
| `SimulationObjects` | 全部单元操作与流股的物性与状态数据 |
| `Compounds` | 组分列表，顺序即全局组成向量顺序 |
| `PropertyPackages` | 物性包及其内部选项 |
| `GraphicObjects` | 画布位置、尺寸与连线 |
| `GeneralInfo` / `Settings` / `Results` | 元信息、全局设置、上次求解结果缓存 |

### 2.2　关键节点

`Compounds` 下每个组分对应一个同名元素，其名称即 DWSIM 化合物字典键，顺序与组成向量一致。`PropertyPackages` 中，`AutoEstimateMissingNRTLUNIQUACParameters` 尤须留意：内置库缺少某对二元交互参数时该选项触发自动估算，结果应当复核。样本所记为 UNIQUAC：

```xml
<PropertyPackage>
  <Type>DWSIM.Thermodynamics.PropertyPackages.UNIQUACPropertyPackage</Type>
  <Tag>UNIQUAC</Tag>
  <AutoEstimateMissingNRTLUNIQUACParameters>true</AutoEstimateMissingNRTLUNIQUACParameters>
</PropertyPackage>
```

`SimulationObjects` 下每个对象为以“类型前缀–GUID”命名的子节点，如 `DC-3b76714b-...`（塔）与 `MAT-40aadbdb-...`（流股）。物料流股的主要字段为 `Type`（完全限定类名）、`SpecType`（如 `Temperature_and_Pressure`）、`CompositionBasis`（如 `Molar_Fractions`）、`DefinedFlow`（如 `Mole`）、`ForcePhase`、`PreferredFlashAlgorithmTag`、`Calculated`、`AtEquilibrium`。塔对象另含 `Specs`（键 `C` 为冷凝器、`R` 为再沸器）、`SType`、`SpecValue`、`SpecUnit`，以及初值数组 `T0`/`Tf`、`V0`/`Vf`、`L0`/`Lf`、`P0`。

样本中塔配置的落盘值与设计值一致，可作为装配正确性的判据，亦印证第 3.2.2 节要求的两个独立规格：

```
NumberOfStages        20
SType                 Stream_Ratio  |  Product_Molar_Flow_Rate
SpecValue             1.937         |  0.5
SpecUnit              （空）         |  mol/s
SolvingMethodName     Naphtali-Sandholm      MaxIterations  500
ColumnPressureDrop    5000（Pa）
```

### 2.3　属性包名映射

| 模型侧名称 | DWSIM 属性包名 |
|---|---|
| `ideal/raoult` | `Raoult's Law` |
| `peng-robinson` / `phasepy/peng-robinson` / `clapeyron/peng-robinson` | `Peng-Robinson (PR)` |
| `nrtl` / `wilson` / `uniquac` | 同名字符串 |

塔类导出经 `_COLUMN_PROPERTY_PACKAGES` 映射，支持 `NRTL`、`UNIQUAC`、`Wilson`、`Peng-Robinson (PR)`、`Ideal`（→ `Raoult's Law`）。

---

## 3　导出数据要求

### 3.1　环境前置条件

| 条件 | 说明 | 缺失时的报错 |
|---|---|---|
| 本机安装 DWSIM | 目录须含 `DWSIM.Automation.dll` | `DWSIM_HOME is not configured.` / `... dll was not found ...` |
| `pythonnet` | 须能 `import clr` | `pythonnet is not installed.` |
| 完整进程权限 | 不可运行于受限沙箱，须用普通终端 | `Python.Runtime ... 拒绝访问` |

```dotenv
DWSIM_HOME=C:\Users\<user>\AppData\Local\DWSIM   # 必填
DWSIM_TEMP_DIR=E:\temp\dwsim                     # 可选，默认 <cwd>/.tmp/dwsim
EXTRACTIVE_EXPORT_DIR=E:\exports\dwsim           # 可选，默认 data/exports/extractive
```

首次使用自动化接口，须于 DWSIM 安装目录执行 `automation_reg.bat` 注册程序集。

### 3.2　各导出路径的拓扑与数据

| 导出函数 | 体系 | 流程结构 | 默认物性包 |
|---|---|---|---|
| `export_dwsim_flowsheet` | 任意 | `Feed → Vessel(TP flash) → Vapor / Liquid` | 由 run 的 `model_name` 决定 |
| `export_dwsim_binary_column` | 二元 | 严格精馏塔 + 单进料 + 塔顶/塔釜 | UNIQUAC |
| `export_dwsim_extractive_column` | 乙醇/水/夹带剂 | 严格塔 + 原料 + 萃取剂 + 两产品 | 由设计指定 |
| `export_generic_extractive_column` | 任意三元 | 严格塔 + 原料 + 萃取剂 + 两产品 | NRTL |
| `export_dwsim_binary_lle_flowsheet` | 二元 | `Feed → Vessel_LLE → Vapor / Light_Liquid / Heavy_Liquid` | NRTL |
| `export_dwsim_ternary_lle_flowsheet` | 三元 | 同二元 LLE | NRTL |
| `export_dwsim_lle_extraction` | 三元 | `Feed + Solvent → Liquid-Liquid Extractor → Raffinate / Extract` | NRTL |

**TP 闪蒸**（`export_dwsim_flowsheet`）自 `RunRecord` 取数，不接受额外参数，取值遵循优先级：组分为 `input_snapshot.components[].name`；进料组成为 `conditions.feed_composition` → `conditions.liquid_composition` → `conditions.vapor_composition` → `points[0]` 的液相或汽相组成；温度取自 `result.temperature_K` → `conditions` → `points[0]`；压力与模型名同理。约束为组成向量长度等于组分数，摩尔流量归一化为 1.0。

**二元精馏塔**（`export_dwsim_binary_column`）必需参数全部由确定性短节法提供：`components`(2)、`feed_composition`(2)、`feed_flow_mol_s`、`feed_temperature_K`、`operating_pressure_kPa`、`stages`、`minimum_stages`、`reflux_ratio`、`minimum_reflux_ratio`、`feed_stage`、`condenser_temperature_K`、`reboiler_temperature_K`、`destination`；可选 `distillate_flow_mol_s`、`bottoms_flow_mol_s`、`pressure_drop_kPa`、`solving_method`（默认 `Naphtali-Sandholm`）、`max_iterations`（默认 500）。校验为 `feed_stage ∈ [1, stages)`，且严格塔须有恰好两个规格。聊天入口默认流量 1 mol/s、压力 101.325 kPa、塔顶纯度 0.995、回收率 0.98、压降 5 kPa，进料板取 `max(2, 理论板数 − 1)`，塔釜取进料之半。

**萃取精馏塔**（`export_dwsim_extractive_column`）仅需一个 `ExtractiveColumnDesign`，其余自该设计读取：夹带剂、物性包、进料温度/压力/流量与组成，以及 `theoretical_stages`、`reflux_ratio`、`feed_stage`、`entrainer_stage`。**通用三元萃取精馏**（`export_generic_extractive_column`）必需 `light`、`heavy`、`entrainer`、`feed_composition`(2) 及各状态与设计参数，可选 `property_package`（默认 NRTL）等；校验为 `1 ≤ feed_stage < stages`、`1 ≤ entrainer_stage < feed_stage`（萃取剂须在原料板上方），以及 `D + B = feed + entrainer`（容差 1e-6）——后者最易出错，D、B 若取自其他进料基准，DWSIM 将报出难以直接解读的迭代质量守恒失败。

**二元与三元 LLE 分相**（`export_dwsim_*_lle_flowsheet`）必需 `components`(2 或 3)、`feed_composition`、`temperature_K`（须为正）、`destination`；可选 `pressure_kPa`（默认 101.325）、`feed_flow_mol_s`（默认 1.0）、`property_package`（默认 NRTL）、`split_out`。校验为组分数与组成向量长度严格匹配、分数和为 1（容差 1e-6）、三元不允许负分数、组分互不相同。温度须**同时**写入进料流股与容器的 `FlashTemperature`/`FlashPressure`，否则 `Vessel` 将以其自带的 298.15 K 默认值闪蒸，在错误温度上计算分相。

**液液萃取器**（`export_dwsim_lle_extraction`）必需 `components`（恰为 3，两溶质加一溶剂）、`feed_composition`（恰为 2）、`solvent`（须属 `components`）、`solvent_ratio`、`feed_flow_mol_s`、`feed_temperature_K`、`feed_pressure_kPa`、`destination`；可选 `property_package`（默认 NRTL）与 `raffinate_*`/`extract_*` 预设产品参数。校验为组分数须为 3、进料分数须为 2、分数和为 1（容差 1e-6）。

### 3.3　界面导出路径

触发条件为「识别出体系」与「明确的导出意图」同时满足，意图关键词为 `dwsim`、`dwism`、`dwxmz`、`导出`、`下载`、`export`、`download`。示例：

```
2-丙醇-水二元 VLE 精馏塔，x_IPA=0.3，1 atm，导出 DWSIM 文件
正丁醇-水二元液液平衡，组成 0.3/0.7，313.15 K，导出 DWSIM
乙醇、乙酸乙酯、水三元 LLE，组成 0.129/0.188/0.683，298.15 K，导出 DWSIM
乙酸正丙酯 40%、乙酸乙酯 60%，用二甲基亚砜做溶剂，溶剂比 1.5，导出 DWSIM 液液萃取文件
```

若体系中存在无 DWSIM 名称映射的组分，系统返回 `status=missing_parameters` 与 `missing_parameters=["dwsim_compound_mapping"]`，不以推测名称强行写入。导出成功后返回 `dwsim_file_uri` 与 `file_id`，经 `GET /api/export/extractive/<file-id>.dwxmz` 下载；通用 run 导出为 `GET /api/runs/{run_id}/export?format=dwsim`。

### 3.4　失败模式

| 现象 | 原因 | 处理 |
|---|---|---|
| `DWSIM_HOME is not configured.` | 未配置环境变量 | 在 `.env` 中指向真实安装目录 |
| `DWSIM.Automation.dll was not found` | 路径有误 | 确认该目录下确有此 DLL |
| `Python.Runtime ... 拒绝访问` | 运行于受限沙箱 | 改用普通终端，非代码问题 |
| `AddCompound` 抛 `KeyNotFoundException` | 组分名大小写或映射不符 | 在 `_DWSIM_COMPOUND_MAP` 中补充映射 |
| `Failed to fulfill mass balance ... Relative Error = ~1.0` | 严格塔缺第二个规格，或进料未绑定塔板 | 写入塔釜产品流量规格；以 `ConnectFeed` 加 `SetStreamFeedStage` 绑定塔板 |
| `Solver reached the maximum number of iterations` | 冷启动且无初值 | 写入初值剖面并提高迭代上限 |
| `ArgumentOutOfRangeException: 索引超出范围` | 仅改 `NumberOfStages`，`Stages` 未重建 | 必须使用 `SetNumberOfStages(n)` |
| 产品流量为零、结果为默认值 | 仅保存未求解，或求解静默失败 | 使用 `CalculateFlowsheet4` |
| 三元 LLE 重相为空 | 自动化路径未启用不互溶闪蒸内核 | 在界面切换至 `Nested Loops (Immiscible)`、`(VLLE)` 或 `Simple LLE` 后重算 |

### 3.5　验收要点

组分名称与顺序正确且分数和为 1；温度单位为 K，压力由接口自 kPa 换算为 Pa；物性包与任务类型匹配，LLE 优先采用有证据支持的 NRTL 或 UNIQUAC 参数；进出口连接完整、进料进入正确端口或塔板；VLE 塔具备两个独立规格并已记录收敛算法；LLE 文件中 `Heavy_Liquid` 或 `Extract` 流量不应在未加说明时为零；DWSIM 状态应显示已计算或已收敛且物料衡算处于允许误差内。用于工程设计之前，须与实验数据或 ThermoFormer 结果交叉验证。

---

## 4　科学边界

本项目**不计算 LLE 数值，也不构造二元交互参数**。LLE 类导出仅负责流程结构、进料状态与物性包，分相由 DWSIM 依据其内置二元参数（或参数估算机制）求解。全部平衡数据须源自 `thermo_engine` 并通过 `validate_equilibrium_result`；语言模型仅承担分类、解析、编排与解释职能。

## 参考出处

`thermo_engine/dwsim_export.py`（流程构建、映射与保存）、`agent/extractive_distillation.py`（请求识别与编排）、`schemas/column_design.py`（输入结构）、`apps/api/main.py`（HTTP 端点）；配套文档 `docs/dwsim-automation-api.zh-CN.md`、`docs/dwsim-extraction-export.zh-CN.md`；测试 `tests/test_dwsim_export.py`、`test_binary_vle_dwsim.py`、`test_binary_lle_dwsim.py`、`test_lle_extraction_export.py`、`test_dwsim_extractive_export.py`。
