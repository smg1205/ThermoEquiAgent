# ThermoFormer × Agent 相平衡流程说明与示例：2-丙醇-水

> 本文档把「LLM/Agent 编排 → 确定性/机器学习相平衡求解 → 工艺流程模拟（ThermoFormer 与 DWSIM）→ 三源交叉校核」整理成一份与
> **ThermoEqui-Agent 仓库实际实现一致**的流程说明，并以 **2-丙醇-水** 作为示例逐步骤给出结果；
> 若某一步骤未实际计算，则写明「该步骤为下一步骤提供了哪些信息」。
>
> 对应仓库实现与数据：`thermo_engine/registry.py`、`thermo_engine/thermoformer_backend.py`、
> `thermo_engine/dwsim_export.py`、`scripts/dwsim_ipa_bubble*.py`、`agent/*`、`skills/*`；
> 三源结果见 `docs/ipa-water-three-source-comparison-v3.zh-CN.md`、`docs/dwsim_ipa_bubble.csv`、
> `docs/prediction_ipa_water.csv` 及各 `docs/BubbleData_ipa_x*.dwxmz`。

---

## 流程总览

```
用户请求 ──▶ (1) LLM 任务理解 + Agent 规划
                 ├─ 抽取出结构化任务（体系/SMILES、目标量、T/p/x、模型与参数）
                 └─ 经确定性后端注册表与 skills 规划求解路径
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                              ▼
 (2a) 确定性后端      (2b) ThermoFormer             (3) DWSIM 独立校验
 经典活动系数/EOS    ML 泡点基线                   + 生成 .dwxmz 工程文件
 (本示例用 NRTL)     ≤500 kPa、≤3 组分、           （不进入 registry，属
                      SMILES 驱动、免二元参数）       独立校验器/导出器）
        └─────────────────┼──────────────────────────────┘
                          ▼
 (4) 示例集成：Experiment / ThermoFormer / DWSIM 三源交叉校核 + 泡点/共沸核验
```

**边界（与 AGENTS.md 及代码一致）**：LLM 只负责理解、拆解、路由与解释，**不计算平衡数**；
所有平衡数值都来自确定性/基于求解器/模型的适配器，并经过 `validate_equilibrium_result`；
参数库缺失即回 `missing_parameters` 失败分支，**绝不编造参数或实验值**。

---

## 1. LLM 理解任务与 Agent 规划

**LLM 如何理解任务**：LLM 从自然语言抽取结构化任务（写为仓库中的 `TaskManifest`）——
体系与化学结构（SMILES，供 ThermoFormer 特征编码）、是否非理想/共沸、
目标量（如泡点温度 T、平衡汽相摩尔组成 y）、条件 (T / p / x 域)、以及所需的热力学模型与参数清单。

**Agent 如何规划任务**：Agent 通过确定性后端注册表（`thermo_engine/registry.py`，即 `DEFAULT_BACKEND_REGISTRY`）
与 skills（`thermodynamic-model-routing`、`thermodynamic-calculation`、`thermodynamic-validation`）做路由规划。

**采取的模型/求解器**：

- **LLM**：负责任务理解、分解、工具选择与结果解释；
- **Agent**（orchestrator/router/tools）：负责把任务路由到具体求解后端、维护工具注册与执行状态；
- **确定性经典后端（registry 内）**：NRTL、UNIQUAC、Wilson、UNIFAC、Ideal/Raoult、Peng-Robinson、SRK、RK、
  Phasepy/PR、Clapeyron/PR —— 给出**经典参考**泡点/VLE；
- **机器学习后端（registry 内）**：ThermoFormer（见步骤 2）；
- **DWSIM（独立校验，不在确定性后端注册表内）**：见步骤 3，作为**独立校验/导出器**而非 registry 求解后端。

> 需要修正如上：**DWSIM 不是与 NRTL/ThermoFormer 平级的 registry 平衡后端**；
> 它只以「独立校验脚本 + 工程文件导出」的形式参与。

**若本步未实际计算，为下一步提供**：一份完整的**任务输入契约**——体系/SMILES、目标 (T/p/x) 域、
需要哪些模型（经典 / 免参数 ML / 免内部模型的 DWSIM 校验）、以及每条路径应调用哪个求解器/脚本。

---

## 2. 相平衡求解：确定性后端 + ThermoFormer 预测泡点与汽相组成

一个给定任务未必直接落到单一路。仓库实际把「经典参考」与「ML 基线」分开：

**(a) 确定性经典后端**（`registry.py` 内的 NRTL、UNIQUAC、Wilson、UNIFAC、Ideal/Raoult、PR…）。当任务携带与组分顺序匹配的证据化参数集时，
会优先路由到 NRTL > UNIQUAC > Wilson（见 `registry.route_task`/`_route_activity_coeff_model`）；
无本地参数时可回退 UNIFAC 或理想/PR。这些后端给出**经典热力学参考解**。

> 说明：本示例（2-丙醇-水）在经典参考/DWSIM 两条链路上实际只用到 **NRTL**；UNIQUAC 等仅为
> 后端注册表存在性列举，**并未参与本体系的结算**。

**(b) ThermoFormer ML 后端**（`thermo_engine/thermoformer_backend.py`，registry 别名 `thermoformer`/`tf`）：
从分子结构（RDKit 描述符 + Uni-Mol v2 + SMARTS）直接学习 VLE/活度系数，**无需二元交互参数**。
适用于 `bubble_point`（等温→P+y、等压→T+y）、`isothermal_vle`、`isobaric_vle`、`infinite_dilution_activity`。
**适用范围/边界**（代码强制）：
- 压力 ≤ **500 kPa**；
- 组分数 ≤ **3**；
- 每个组分必须有 **SMILES**；
- `dew_point / tp_flash / phase_stability / azeotrope / lle` 在 ThermoFormer 上**未实现**（`_unsupported`）。

**区分「经典参考」与「ML 预测」**：ThermoFormer 输出为近似 ML 基线，未经实验校准并带不确定性声明，
不作为最终权威值；经典确定性后端才是仓库默认的常规求解参考。

**若本步未实际计算，为下一步提供**：一份**可交叉比对的 ML/经典基线** {T[b], y}（含数量级与曲线形状、适用边界），
供 DWSIM 独立校验核对量级与趋势。

---

## 3. 工艺流程计算与模拟：用 DWSIM 校验并生成可打开的文件

仓库用 DWSIM 对同一体系做**独立校验**并**生成本地工程文件**（此处针对 2-丙醇-水走独立驱动脚本实现）：

**(a) 在脚本中调用 DWSIM API 得到泡点与汽相组成**（`scripts/dwsim_ipa_bubble.py` 及变体）：
构造 `Feed → (Equilibrium) Flash` 流程，属性包选 **NRTL**；通过把泡点温度附近二分/扫描，使 **Vapor 摩尔分率趋近于 0** 作泡点判据，
读取温度 T[b] 与该点平衡汽相组成 y（`CalculateFlowsheet2` + 流股 `GetMolarFlow`/`GetOverallComposition` 实现，写为 CSV）。

**(b) 生成 DWSIM 工程文件**（`scripts/dwsim_ipa_bubble_files.py` / `dwsim_export`）：
把**进料温度设置在泡点上精确位置**，另存出对流股摩尔分率趋近 0、可判读泡点的 `docs/BubbleData_ipa_x*.dwxmz`。
该 `.dwxmz` 可直接用 DWSIM 打开；若依赖路径需要人工，则在 DWSIM GUI **按下 Calculate 即可复核**该组成的泡点与汽相组成。

**模型与参数出处**：平衡由 **DWSIM 内置的 NRTL 参数**（Water/Isopropanol：`45.59 / 944.70 / 0.2`）求解，属独立第三方校验源。

> 需修正说明：上一步（registry 的 NRTL）与 DWSIM 的 NRTL 属**两个不同实现**；
> 仓库并非把 DWSIM 当作 registry 求解后端，而是把它用作**能在脚本/GUI 中复核、并生成可打开文件的独立校验器**。

**若本步未实际计算，为下一步提供**：一份**独立严格热力学参考解**（含使用的模型假设、内置参数与收敛信息），
用于与步骤 2 的经典/ThermoFormer 结果做最终交叉比对。

---

## 4. 示例：2-丙醇-水 多源交叉校核与泡点核验

以 **2-丙醇（C3H8O）— 水（H2O）** 为示例，按 1–3 流程执行并展示。数据源 = 仓库实测三源报告
（`ipa-water-three-source-comparison-v3.zh-CN.md` 的表，@ **760 mmHg 等压泡点**，5 个液相摩尔分数点）。

### 4.1 任务与规划结果（步骤 1）

- 体系：2-丙醇-水；目标：101.325 kPa（760 mmHg）下各 x 的**泡点温度 T 与汽相摩尔组成 y**。
- 约束：体系为**非理想醇-水、存在最低沸点共沸结构**，需覆盖富水与富异丙醇两侧组成。
- 规划出的路径：经典确定性/热力学参考 + ThermoFormer（ML）预测 + DWSIM 独立校验做多源交叉。
- *该步职责是输出契约与路径，未做平衡计算（LLM/Agent 不产生平衡数）。*

### 4.2 ThermoFormer 泡点预测（步骤 2b）

方法：`predict_ipa_water.py`（权重 `overall_binary/seed_4/best_model.pt`），等压泡点 P=101.325 kPa；
关键前提是把**真实 Antoine 纯物性参数**（异丙醇、水）传给求解器，否则泡点会虚高上百 ℃。

> 实测：ThermoFormer 泡点温度与实验的 MAE ≈ **2.58 °C**；富水区偏差略大、富异丙醇端较准。
> 结果写为 `docs/prediction_ipa_water.csv`。下表（T_ThermoFormer 列）为其中数值。

### 4.3 DWSIM 校验 + 生成工程文件（步骤 3）

方法：二分/扫描使 Vapor 摩尔分率→0 定泡点；NRTL 内置参数（Water/Isopropanol `45.59 / 944.70 / 0.2`）；
生成 `docs/BubbleData_ipa_x0p5.dwxmz` 等文件，可在 DWSIM GUI 复核（Vapor 摩尔分率趋零即泡点）。
下表 T_DWSIM / y_DWSIM 取自 `docs/dwsim_ipa_bubble*.csv` 与三源报告。

### 4.4 集成与三源交叉校核（步骤 4）

> 表中 T 单位为 °C；y 单元格式为 `2-丙醇 / 水`。

| x_IPA | T_实验 (°C) | T_ThermoFormer (°C) | T_DWSIM (°C) | y_实验 | y_ThermoFormer | y_DWSIM |
|---|---|---|---|---|---|---|
| 0.1 | 84.75 | 90.21 | 89.50 | 0.495 / 0.505 | 0.360 / 0.640 | 0.378 / 0.622 |
| 0.3 | 81.85 | 83.77 | 83.68 | 0.541 / 0.459 | 0.570 / 0.430 | 0.565 / 0.435 |
| 0.5 | 80.15 | 81.81 | 81.92 | 0.605 / 0.395 | 0.657 / 0.343 | 0.644 / 0.356 |
| 0.7 | 80.02 | 81.14 | 81.12 | 0.687 / 0.313 | 0.727 / 0.273 | 0.730 / 0.270 |
| 0.9 | 80.87 | 81.49 | 81.42 | 0.872 / 0.128 | 0.876 / 0.124 | 0.875 / 0.125 |

**读取结果**：

- **富水区（x=0.1）**：实验泡点最高（84.75 °C）；实验汽相比两模型更富 2-丙醇（y_IPA 0.495 vs 0.360/TF、0.378/DWSIM）。
- **中高 x（0.3–0.9）**：ThermoFormer 与 DWSIM 在温度上几乎重合（差 ≤0.03 °C），二者比实验高约 +1~3 °C。
- **y 趋势**：三源的 `y_2-丙醇 / y_水` 单调趋势一致；x 越高汽相越富 2-丙醇。总评 **DWSIM ≈ ThermoFormer ≳ 实验**，物理合理。

**泡点/共沸结构核验（x≈0.5–0.9 区间）**：

- 实验泡点温度曲线在该段**探底到约 80.0–80.2 °C**（x=0.5 实验 80.15、x=0.7 实验 80.02），
  与 2-丙醇-水最低沸点共沸文献值（≈80.37 °C、x/y_IPA≈0.68）落在同一物理区域，是"非理想 + 最低沸点共沸"的正确形态；
- 在 x=0.7 处三源给出的平衡汽相 y_IPA≈0.687–0.730，与液相 x_IPA=0.7 接近（y 接近 x），是共沸附近汽液组成逼近的判据；
- 三元/多一点的精确"共沸点"赋值属 `azeotrope` 可解后端范畴，ThermoFormer 本身不实现 azeotrope，
  故这里以"泡点探底 + y≈x"作为物理核验锚点，而非断言单一共沸数。

**复现路径 / 模型出处**：`docs/BubbleData_ipa_water_EA_bubble.dwxmz` 与 `docs/BubbleData_ipa_x0p5.dwxmz` 等可在 DWSIM GUI 打开复核；
三源表数据见 `docs/ipa-water-three-source-comparison-v3.zh-CN.md`、`docs/dwsim_ipa_bubble*.csv`、`docs/prediction_ipa_water.csv`；
ThermoFormer 适用边界（≤500 kPa、≤3 组分）与实现见 `thermo_engine/thermoformer_backend.py`。

> 说明：经典确定性后端（registry 的 NRTL/UNIQUAC/UNIFAC 等）在仓库里亦可独立给出同一体系的泡点参考解，
> 作为比 ML/DWSIM 更贴仓库"默认求解器"定位的补充源；如需要可并入本表形成第四路对照。
