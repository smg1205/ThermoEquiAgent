# §1.5 案例二（1-丁醇 / 水 / 甲苯）DWSIM 萃取精馏塔 — 接线与收敛诊断

本文汇总对 `water_toluene_butanol_extractive_thermoformer.dwxmz` 生成链路的排查结论。
所有结论均由本机 DWSIM（`C:\Users\34861\AppData\Local\DWSIM`）实跑验证，非推断。

## 1. 已修复并验证的缺陷

### 1.1 塔釜流量规格丢失 `recovery`（用户报错的直接原因）

`thermo_engine/dwsim_export.py::export_generic_extractive_column` 原先自行重算塔顶流量，
**忽略回收率**：

```python
d_light = feed_composition[0] * feed_flow_mol_s   # 0.5，隐含 100% 回收
d_total = d_light / purity                        # 0.526316（应为 0.473684）
bottoms = total_in - d_total                      # 2.473684（应为 2.526316）
```

设计回收率为 0.90，塔顶轻组分应为 `0.5 × 0.9 = 0.45`。塔釜因此少写 0.0526 mol/s，
两条规格（回流比 + 塔釜流量）无法同时满足，DWSIM 报：

```
Failed to fulfill mass balance for Water: Relative Error = 0.872855481143158
```

**修复**：新增 `recovery` 参数；允许调用方直接传入设计算出的
`distillate_flow_mol_s` / `bottoms_flow_mol_s`（优先采用，保持单一事实源）；
并加 `D + B == total_in` 一致性断言。

注意：该断言**无法**捕获原始缺陷 —— 错误算法给出的 D + B 同样等于 3.0，
它只是内部自洽但基于错误回收率。

### 1.2 进料未绑定到塔板

`ConnectFeed` 与 `SetStreamFeedStage` 的调用关系，实测结果如下
（读取 `GetStreamFeedStageIndex` 回验）：

| 调用方式 | 回读的 stage index |
|---|---|
| 仅 `ConnectFeed(feed, 5)` | **-1**（未绑定） |
| 仅 `SetStreamFeedStage(feed, 5)` | `NullReferenceException`（流股尚未连接） |
| `ConnectFeed` 后再 `SetStreamFeedStage` | **5** ✅ |

即**两者都必须调用，且顺序不可颠倒**。修复前该函数虽按正确顺序调用两者，
但外层 `export_generic_extractive_column` 的其它改动使绑定未能生效；
现已恢复为「先 ConnectFeed，再 SetStreamFeedStage」并加注释说明顺序敏感性。

**验证结果**（重新生成的 TF 文件）：

```
GetSolverInputData: OK (all connections present)
feed@2   T=325.63 K  F=2.0000 mol/s  x=[0.0, 0.0, 1.0]     ← 萃取剂 1-丁醇
feed@8   T=327.09 K  F=1.0000 mol/s  x=[0.5, 0.5, 0.0]     ← 水/甲苯进料
```

与 §1.5 表 1.5-2 设计（进料板 8、萃取剂板 2）完全一致。

### 1.3 产品出口接线

实测确认：

| 接线方式 | 结果 |
|---|---|
| `ConnectDistillate` / `ConnectBottoms` | **正确**，`GetSolverInputData` 通过 |
| 旧的 graphic `ConnectObjects(col, s, 0\|1, 0)` | **抛异常**，连接缺失 |

故专用连接器是必需且正确的；旧 graphic 调用无效。已替换。

### 1.4 导出脚本从未配置求解器

`_set_column_solver` / `_set_column_initial_estimates` 原**仅**被二元塔导出路径调用
（旧行号 1429/1436），通用萃取塔路径从未调用。生成文件实测为 DWSIM 默认值：

| 设置 | 生成文件 | 期望 |
|---|---|---|
| SolvingMethodName | `Wang-Henke (Bubble Point)` | 见 §2.2 |
| MaxIterations | 100 | 1000 |
| UseTemperatureEstimates | False | True |
| UseLiquidFlowEstimates | False | True |
| UseVaporFlowEstimates | False | True |

已补上调用。

## 2. 关键测量陷阱（影响所有后续结论的解读）

### 2.1 `CalculateFlowsheet2` 会吞掉错误

`CalculateFlowsheet2` 在塔收敛失败时**正常返回、不抛异常**。因此
「求解无异常」不构成任何成功证据 —— 本次排查中一度据此得出错误的
「接线有误」结论。**必须使用 `CalculateFlowsheet4`**，它返回错误列表：

```python
errors = automation.CalculateFlowsheet4(flowsheet)   # List[str]
```

### 2.2 `SolvingMethodName` 不做校验

该属性为 `String`，setter 接受任意值；错误名称只在求解时才报
`Unable to find column solver with name '...'`。

实测本机**可解析**的名称仅两个：

- `Wang-Henke Bubble-Point (BP) Solver`
- `Modified Wang-Henke Bubble-Point (MBP) Solver`

`Sum-Rates (SR) Method`、`Simultaneous Correction (SC) Method` 等
在程序集中仅作为**描述文本**出现，不能解析。默认值已设为可解析者。

### 2.3 其它 API 事实

- `ConnectDistillate(ISimulationObject)` / `ConnectBottoms(ISimulationObject)`
- `SetInitial{Temperature,LiquidMolarFlow,VaporMolarFlow}Estimates` 要求数组长度
  **恰好等于 `NumberOfStages`**（`RigorousColumn.vb:2317`），否则报
  「value vector needs to have N elements」。
- `column.MaterialStreams` 是权威的已连接流股注册表。
- DWSIM 命名空间（`DWSIM.*`、`System`）仅在创建 flowsheet 之后才可导入。

## 3. 严格塔求解状态

> **交付标准（已经用户确认）**：本文件以「可在 DWSIM GUI 中打开、接线与设计
> 数据正确」为准，与仓库既有标准一致 —— 报告 §1.4 的 DWSIM 证据即为「打开文件
> 后在 GUI 中读取结果页」，同类归档文件（如
> `water_toluene_butanol_extractive_unifac.dwxmz`）亦未要求严格塔收敛。
> 严格塔收敛作为**已知未决项**记录于此，不影响本次交付。

### 3.0 根因一：级数列表未同步（已修复，影响所有塔）

**这是本轮找到的最重要缺陷，且与 1-丁醇/水/甲苯无关。**

DWSIM 的塔把级数存了两份：属性 `NumberOfStages` 与列表 `Stages`。
**赋值属性不重建列表**，列表恒保持构造时的长度 12：

| 写法 | NumberOfStages | Stages.Count | 同步 |
|---|---:|---:|:--:|
| `col.NumberOfStages = 18` | 18 | 12 | ✗ |
| `col.set_NumberOfStages(18)` | 18 | 12 | ✗ |
| `col.SetNumberOfStages(18)` | 18 | 18 | ✓ |

DWSIM 随后按 18 去索引只有 12 个元素的列表，抛
`ArgumentOutOfRangeException: 索引超出范围`（源自 `Column.GetSolverInputData`）。

仓库导出代码**两种写法都调用了**，且顺序有害：

```python
("SetNumberOfStages", stages),   # 正确：列表重建为 n
("set_NumberOfStages", stages),  # 属性 setter：把列表again改回 12
("NumberStages", stages),        # 同上
```

**对照验证**（教科书体系 乙醇/水 10 级 R=3）：

| 写法 | Stages.Count | 结果 |
|---|---:|---|
| 现行代码（两种都调） | 12 | 未收敛：索引超出范围 |
| 只用 `SetNumberOfStages` | 10 | **收敛**，塔顶乙醇 0.787 / 塔釜 0.787 |

这解释了为何**乙醇/水、水/甲苯、三元体系全部以同一异常失败** —— 它们走同一条
建塔路径。此前把问题归因于三元体系或萃取剂，方向有误。

**修复**：新增共享辅助 `_apply_stage_count()`（`thermo_engine/dwsim_export.py`），
统一走 `SetNumberOfStages` 并校验列表长度；`_set_column_design`、
`_set_binary_column_design` 与通用萃取塔路径均已改用它。
修复后 `Stages.Count` 与级数一致，越界错误消失。

### 3.1 根因二：残余不收敛（未解决）

级数修复后，三元塔报完整错误（此前被截断）：

```
A convergence error was found while trying to solve the column.
Possible reasons are: unfeasible specs, initial estimates far from solution
and/or very non-ideal (wide-boiling or azeotropic) mixtures being fed.
   在 WangHenkeMethod.Solve_Internal ... BubblePoint.vb:行号 1728
```

DWSIM 给出三个疑因，已排除其中一个：

| 疑因 | 排除依据 |
|---|---|
| 初值远离解 | **已排除**：注入两类温度初值（TF 口径 325.63→381.28 K、UNIFAC 口径 282.47→378.73 K）均无效 |
| 规格不可行 | 未排除：换水回收率 0.90 / 塔顶水纯度 0.95 均同样失败 |
| 体系强非理想 | 未排除：该体系选择性 0.8194 < 1，属强非理想 |

同时**对以下扰动均不敏感**（一律失败）：两个可解析求解器、
回流比 4/6/10、萃取剂板位 0/1/3、进料板 10、塔釜流量 2.0/2.8、
进料温度改为泡点 338.97 K。

> 注：早先「误差恒为 0.99999 且不随迭代变化」的读数，实际来自级数未同步
> 时的越界路径；修复后的错误语义已不同（真正的迭代收敛失败）。

### 3.2 设计自洽性已验证（已完成）

`scripts/check_wtb_design_consistency.py` 用报告 §1.5 的 alpha 独立复算 FUG：

| 来源 | alpha_base | alpha_ext | alpha_avg | N_min | R_min | 算得 N | 报告值 |
|---|---:|---:|---:|---:|---:|---:|---|
| ThermoFormer (seed_2) | 2.3501 | 1.9256 | 2.1273 | 9.071 | 1.497 | **18** | N=18, R=2.095 ✅ |
| UNIFAC | 1.9237 | 1.4364 | 1.6623 | 13.473 | 2.618 | **25** | N=25, R=3.665 ✅ |

两组**完全复现**，设计数据自洽、无算术错误。

关键判据：`alpha_avg = 2.1273 > 1`，即水/甲苯在无萃取剂时本就可分离 ——
这是塔能成立的热力学前提，分离本身并非不可行。

同时确认报告自述：选择性 `alpha_ext / alpha_base = 0.8194 < 1`，
萃取剂并未增强关键对相对挥发度，故该塔宜表述为
「带萃取剂进料的三元精馏塔」。

### 3.3 已排除的原因

- **不是缺失二元交互参数**：三对二元（水/甲苯、水/丁醇、甲苯/丁醇）的
  TP flash 均正常收敛（错误数 0）。日志中仅水/甲苯与甲苯/丁醇两对需要
  DWSIM 估算，水/丁醇使用内置参数。
- **不是接线问题**：`GetSolverInputData` 通过，四股流股全部正确注册。
- **不是迭代次数不足**：见 §3.1。

### 3.4 尚未验证的方向

规格对为 `C: Stream_Ratio`（回流比）+ `R: Product_Molar_Flow_Rate`（塔釜流量）。
`SpecType` 另可选 `Component_Recovery`、`Component_Fraction`、`Feed_Recovery`、
`Temperature`、`Heat_Duty` 等。质量平衡误差恒为 ~1.0 提示规格对可能不适定，
下一步可试以「水回收率 0.90」或「塔顶水纯度 0.95」作为第二条规格
（`scripts/probe_wtb_spec_pairs.py` 已备好该试验）。

## 4. 复现命令

```powershell
conda activate thermo
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# 1) 用 seed_2 实跑 §1.5 表 1.5-2 的 ThermoFormer 列（两次运行均可复现）
python scripts\design_1p5_wtb_thermoformer_seed2.py

# 2) 导出 .dwxmz
python scripts\export_wtb_extractive_tf_seed2.py

# 3) 求解并检查物料平衡（务必用 CalculateFlowsheet4）
python scripts\solve_and_report_wtb_column.py

# 4) 接线/构件校验（交付核验，应全部通过）
python scripts\verify_wtb_extractive_tf_seed2.py

# 5) 设计自洽性独立复核（FUG 复算，不依赖 DWSIM）
python scripts\check_wtb_design_consistency.py

# 6) 备选：试验不同的第二条规格（严格塔收敛排查用）
python scripts\probe_wtb_spec_pairs.py
```

## 5. 涉及文件

| 文件 | 状态 |
|---|---|
| `thermo_engine/dwsim_export.py` | 已修：recovery、进料绑定、产品连接、求解器配置 |
| `scripts/design_1p5_wtb_thermoformer_seed2.py` | 新增：seed_2 设计复算 |
| `scripts/export_wtb_extractive_tf_seed2.py` | 新增：导出入口 |
| `scripts/solve_and_report_wtb_column.py` | 新增：求解 + 物料平衡诊断（用 CalculateFlowsheet4） |
| `scripts/verify_wtb_extractive_tf_seed2.py` | 新增：构件校验（交付核验） |
| `scripts/check_wtb_design_consistency.py` | 新增：FUG 设计自洽性复核 |
| `scripts/probe_wtb_spec_pairs.py` | 新增：规格对试验（未决项排查用） |
| `scripts/probe_column_baseline.py` | 新增：最小对照案例 |
| `scripts/probe_solver_names.py` | 新增：求解器名称/收敛探测 |
| `report/dwsim/design_1p5_wtb_thermoformer_seed2.json` | 设计值留痕 |
| `data/exports/flow_examples/water_toluene_butanol_extractive_thermoformer.dwxmz` | **交付文件** |

## 6. 交付核验结果（全部通过）

```
compounds              ['Water', 'Toluene', '1-butanol']        OK
NumberOfStages         18                                        OK
RefluxRatio            2.095                                     OK
ColumnPressureDrop     5000.0 Pa                                 OK
streams attached       4 / 4                                     OK
number of bound feeds  2                                         OK
product outlet count   2                                         OK

萃取剂 (1-丁醇) @ 第 2 板:  x=[0,0,1]   2.0 mol/s  325.63 K      OK
进料   (水/甲苯) @ 第 8 板:  x=[.5,.5,0] 1.0 mol/s  327.09 K      OK
```

