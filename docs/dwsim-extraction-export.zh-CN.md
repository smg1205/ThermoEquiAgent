# DWSIM 相平衡与萃取文件导出指南

本文档汇总 ThermoAgent 当前提供的 DWSIM `.dwxmz` 导出能力，包括二元/三元 VLE、二元/三元 LLE，以及液液萃取和萃取精馏。文中的接口和限制以 `thermo_engine/dwsim_export.py` 与 `agent/extractive_distillation.py` 的当前实现为准。

## 1. 环境与输出位置

导出依赖本机 DWSIM 和 `pythonnet`。项目 `.env` 至少需要配置：

```dotenv
DWSIM_HOME=C:\path\to\DWSIM
```

`DWSIM_HOME` 必须指向包含 `DWSIM.Automation.dll` 的目录。可选变量：

```dotenv
DWSIM_TEMP_DIR=E:\temp\dwsim
EXTRACTIVE_EXPORT_DIR=E:\exports\dwsim
```

未配置 `EXTRACTIVE_EXPORT_DIR` 时，聊天入口生成的文件默认写入：

```text
data/exports/extractive/
```

文件扩展名必须为 `.dwxmz`。导出成功后，接口通过 `/api/export/extractive/<file-id>.dwxmz` 提供下载。

## 2. 当前支持范围

| 类型 | 组分数 | DWSIM 结构 | 默认物性包 | 主要入口 |
|---|---:|---|---|---|
| VLE 精馏 | 二元 | 严格精馏塔 + 单股进料 + 塔顶/塔釜产品 | UNIQUAC | `run_binary_distillation` / `export_dwsim_binary_column` |
| VLE 萃取精馏 | 三元 | 严格精馏塔 + 原料 + 萃取剂 + 两股产品 | 设计指定，常用 NRTL/UNIQUAC | `run_extractive_export` / `export_dwsim_extractive_column` |
| LLE 相分离 | 二元 | `Vessel_LLE` + 轻液相/重液相/气相 | NRTL | `run_binary_lle_export` / `export_dwsim_binary_lle_flowsheet` |
| LLE 相分离 | 三元 | `Vessel_LLE` + 轻液相/重液相/气相 | NRTL | `run_generic_ternary_lle_export` / `export_dwsim_ternary_lle_flowsheet` |
| 液液萃取 | 三元 | 原料 + 溶剂 + `Liquid-Liquid Extractor` + 萃余相/萃取相 | NRTL | `run_lle_extraction_export` / `export_dwsim_lle_extraction` |

VLE 指气液平衡，LLE 指液液平衡。萃取精馏属于 VLE 塔流程；液液萃取属于 LLE 流程，两者不能混用。

## 3. 二元 VLE 导出

### 3.1 支持的聊天体系

当前聊天入口可识别：

- 2-丙醇/水
- 甲醇/水
- 乙醇/水
- 乙醇/甲苯

请求中必须给出二元进料组成；只给一个组分的摩尔分数时，另一组分按 `1 - x` 计算。未给出的参数采用当前设计默认值：进料流量 `1 mol/s`、压力 `101.325 kPa`、塔顶纯度 `0.995`、回收率 `0.98`、全塔压降 `5 kPa`。

示例：

```text
2-丙醇-水二元 VLE 精馏塔，x_IPA=0.3，1 atm，导出 DWSIM 文件
```

输出文件包含严格精馏塔、进料、塔顶产品和塔釜产品。冷凝器规格为回流比，塔釜规格为塔釜产品摩尔流量；进料板、塔板数、温度初值和流量初值均写入文件。

### 3.2 手动选择收敛算法

二元 VLE 导出器不会锁死收敛算法。代码调用时通过 `solving_method` 手动选择：

```python
from pathlib import Path

from thermo_engine.dwsim_export import export_dwsim_binary_column

export_dwsim_binary_column(
    components=["isopropanol", "water"],
    feed_composition=[0.3, 0.7],
    feed_flow_mol_s=1.0,
    feed_temperature_K=354.60,
    operating_pressure_kPa=101.325,
    stages=19,
    minimum_stages=10.069,
    reflux_ratio=2.694,
    minimum_reflux_ratio=1.924,
    feed_stage=18,
    condenser_temperature_K=355.26,
    reboiler_temperature_K=367.46,
    distillate_flow_mol_s=0.5,
    bottoms_flow_mol_s=0.5,
    pressure_drop_kPa=5.0,
    solving_method="Naphtali-Sandholm",
    max_iterations=500,
    destination=Path("data/exports/extractive/ipa-water-binary-vle.dwxmz"),
)
```

常用选择：

| 收敛算法 | 建议用途 |
|---|---|
| `Naphtali-Sandholm` | 当前默认值；适合带温度/流量初值的严格塔，优先尝试 |
| `Wang-Henke (Bubble Point)` | 泡点法；简单体系可用，对冷启动和初值更敏感 |
| `Burningham-Otto (Sum Rates)` | 求和速率法；前两者不收敛时可作为替代方案 |

算法名称必须与所安装 DWSIM 版本公开的 `SolvingMethodName` 完全一致。若版本中的显示名称不同，应以 DWSIM GUI 下拉框为准。

也可在导出后手动修改：

1. 用 DWSIM 打开 `.dwxmz`。
2. 双击 `Binary Distillation Column`。
3. 打开求解器或 Convergence/Solver 页面。
4. 在算法下拉框中选择收敛算法，并将最大迭代次数设为合适值，例如 `500`。
5. 保留已写入的温度、液相流量和气相流量初值，重新运行计算。
6. 检查塔顶、塔釜流量非零，组分守恒，且求解状态为 converged。

聊天入口当前使用默认的 `Naphtali-Sandholm`。需要精确控制算法时，应直接调用 `export_dwsim_binary_column`，或导出后在 DWSIM GUI 中手动切换。

## 4. 三元 VLE 与萃取精馏导出

三元 VLE 导出在项目中以萃取精馏塔表示：两种待分离组分构成原料，第三种组分作为高沸点萃取剂。文件包含原料流、萃取剂流、严格塔、塔顶产品和塔釜产品。

固定验证体系：

```text
乙酸乙酯/乙酸正丙酯/DMSO 三元 VLE 萃取精馏，导出 DWSIM
```

通用请求示例：

```text
乙腈和甲苯用四氢呋喃做萃取精馏，给出进料组成、流量、压力和萃取剂比，导出 DWSIM
```

乙醇/水流程还支持乙二醇或甘油萃取剂。可在请求中明确要求 ThermoFormer 作为相对挥发度来源；未明确要求时使用项目默认的确定性设计路径。

严格塔必须具有两个独立规格。打开导出文件后应核对：

- 原料与萃取剂是否进入预期塔板；
- 冷凝器规格是否为回流比或塔顶产品流量；
- 再沸器规格是否为塔釜流量、再沸比或热负荷；
- 物性包与组分映射是否正确；
- 求解器、最大迭代次数和初值是否适合该体系。

部分 DWSIM 版本无法通过 Automation API 稳定写入所有塔内部规格。此时文件结构和进料连接仍会保留，需要在 GUI 中补齐冷凝器/再沸器规格后再计算。

## 5. 二元 LLE 导出

二元 LLE 使用如下结构：

```text
Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid
```

示例：

```text
正丁醇-水二元液液平衡，组成 0.3/0.7，313.15 K，导出 DWSIM
```

温度同时写入进料和容器闪蒸条件，压力默认 `101.325 kPa`，物性包默认 NRTL。导出器在保存前调用 DWSIM 求解，并读取轻液相、重液相流量与组成。若 DWSIM 只得到单液相，文件仍会生成，但返回信息会明确标记 `separated=false`，不能把它当作已经完成的两液相分离设计。

科学边界：项目只负责流程结构、进料状态和导出；LLE 分相数值来自 DWSIM 自带参数或其参数估算机制。用于工程设计前必须复核二元交互参数和实验数据。

## 6. 三元 LLE 导出

三元 LLE 使用与二元 LLE 相同的 `Vessel_LLE` 拓扑，但进料含三个组分。

示例：

```text
乙醇、乙酸乙酯、水三元 LLE，组成 0.129/0.188/0.683，298.15 K，导出 DWSIM
```

当前 DWSIM 9.0.5 Automation 路径存在已知限制：已测试的 `NestedLoops`、`InsideOut`、`GibbsMinimization` 以及多个 `PreferredFlashAlgorithmTag` 可能都返回单液相，导致 `Heavy_Liquid` 为空。导出器会保留结构正确的文件并发出警告。

遇到该情况时：

1. 在 DWSIM GUI 中打开文件。
2. 选择支持不互溶液相或 VLLE 的闪蒸算法，例如版本中提供的 `Nested Loops (Immiscible)`、`Nested Loops (VLLE)` 或 `Simple LLE`。
3. 检查物性包中的二元交互参数。
4. 重新计算并确认 `Heavy_Liquid` 流量大于零。

若 GUI 重新计算后仍为单液相，应视为该模型/参数在给定温压和组成下未预测出分相，不应强制解释为萃取成功。

## 7. 三元液液萃取器导出

当请求明确包含两种原料组分和一种独立溶剂时，可导出液液萃取器：

```text
Feed + Solvent -> Liquid-Liquid Extractor -> Raffinate + Extract
```

已验证示例：

```text
乙酸正丙酯 40%、乙酸乙酯 60%，用二甲基亚砜做溶剂，溶剂比 1.5，导出 DWSIM 液液萃取文件
```

该例中：

- 原料组成写为 `[0.4, 0.6, 0.0]`；
- 溶剂流为纯 DMSO，组成 `[0.0, 0.0, 1.0]`；
- 溶剂流量为 `solvent_ratio × feed_flow`；
- 两股出口分别为 `Raffinate` 和 `Extract`。

这是液液萃取单元，不是精馏塔。若请求的是高沸点溶剂改变相对挥发度，应使用第 4 节的三元 VLE/萃取精馏路径。

## 8. 导出文件验收

每个 `.dwxmz` 文件至少应完成以下检查：

1. 组分名称与顺序正确，摩尔分数和为 1。
2. 温度单位为 K，压力由接口写入 DWSIM 时转换为 Pa。
3. 物性包与任务类型匹配，LLE 优先使用有证据支持的 NRTL/UNIQUAC 参数。
4. 所有进出口连接完整，进料进入正确端口或塔板。
5. VLE 塔具有两个独立规格，并明确记录所选收敛算法。
6. LLE 文件中 `Heavy_Liquid` 或 `Extract` 流量不应在未说明的情况下为零。
7. DWSIM 状态显示已计算/收敛，物料衡算满足项目允许的误差范围。
8. 工程使用前，将 DWSIM 结果与实验数据或 ThermoFormer/其他模型结果交叉验证。

## 9. 常见失败

| 现象 | 常见原因 | 处理方式 |
|---|---|---|
| 找不到 `DWSIM.Automation.dll` | `DWSIM_HOME` 错误 | 指向真实 DWSIM 安装目录 |
| `AddCompound` 失败 | DWSIM 名称区分大小写或缺少映射 | 在 `_DWSIM_COMPOUND_MAP` 中补充明确映射 |
| 二元 VLE 达到迭代上限 | 算法或初值不适合 | 手动切换 Naphtali-Sandholm/Wang-Henke/Burningham-Otto，增加迭代次数并保留初值 |
| 严格塔产品流量为零 | 缺少第二个独立规格 | 补塔釜产品流量、再沸比或热负荷规格 |
| 三元 LLE 重相为空 | Automation 闪蒸内核未启用不互溶计算，或模型确实预测单相 | 在 GUI 切换 LLE/VLLE 内核并检查二元参数 |
| 文件生成但未计算 | DWSIM 求解失败或只保存了结构 | 打开 GUI 查看错误，修正规格后重新计算并保存 |

## 10. 代码位置

- DWSIM 文件构建与保存：`thermo_engine/dwsim_export.py`
- 请求识别、参数解析与下载信息：`agent/extractive_distillation.py`
- 二元 VLE 测试：`tests/test_binary_vle_dwsim.py`
- 二元 LLE 测试：`tests/test_binary_lle_dwsim.py`
- 三元 LLE 测试：`tests/test_ternary_lle_dwsim.py`
- 液液萃取器测试：`tests/test_lle_extraction_export.py`
- 萃取精馏塔测试：`tests/test_dwsim_extractive_export.py`

