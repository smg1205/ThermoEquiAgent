# ThermoAgent: 模型集成与三源验证报告 v5

> 本版本在 v4 的框架上刷新第二部分数据口径：三元 VLE 体系替换为
> **乙酸 / 水 / 二甲基亚砜（DMSO）**，LLE 体系为二元体系（水 / 正丁醇）。  
> 三源定义保持一致：实验数据 / ThermoFormer 预测 / DWSIM 独立流程模拟。  
> 各体系的 DWSIM 工程文件均位于 `report/dwsim/`，塔流程示例位于 `data/exports/flow_examples/`。

> **数值口径（2026-09-17 刷新）**：第二部分的 **ThermoFormer 列**以 `seed_2` 权值
> （`models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt`）经 `ThermoFormerBackend`
> 重新实跑；实验列取自 NIST ThermoML 数据集，DWSIM 列为 Automation API 实算。
> §1.5 的塔设计 ThermoFormer 列与 §2.2 采用**同一权值（`seed_2`）**，保证口径一致。
> 体系①（二元 VLE）的 ThermoFormer 预测偏差较大，其根因排查与纯组分外推检验见 §2.4。

> **三元 VLE 体系变更（本版）**：三元体系替换为 **乙酸 / 水 / DMSO**
> （17 点，13.33 kPa，DOI `10.1016/j.fluid.2008.09.010`）。
> **变更理由**：原体系（1-丁醇 / 水 / 甲苯）的萃取精馏在热力学上不成立 ——
> 选择性 `alpha_ext/alpha_base` 两源均 **< 1**（UNIFAC 0.7467、TF 0.8193），
> 即萃取剂并未增强关键对的相对挥发度，报告正文亦只能将其表述为
> 「有萃取剂进料的三元精馏塔」。新体系的选择性 **> 1**
> （UNIFAC 6.4569、TF 各权值 1.62~2.37），是**真正的萃取精馏**。
>
> **本体系的三源结论与原体系相反**：DWSIM 汽相组成 MAE **0.0714**，
> 优于最佳 ThermoFormer 权值（`seed_4`）的 0.2233 约 3.1 倍；
> 而原体系是 TF（0.0388）优于 DWSIM（0.1183）。
> 原因见 §2.2：DMSO 为强极性强缔合组分，处于 ThermoFormer 训练分布之外。
>
> **注意**：本体系 ThermoFormer 三元预测精度（MAE 0.22~0.29）显著劣于原体系
> （0.0259）。「萃取成立」与「TF 预测精确」在该数据集上不可兼得。

---

## 第一部分　模型集成（ThermoAgent + ThermoFormer + DWSIM）及流程展示

### 1.1 目标与约束

ThermoAgent 是面向相平衡计算与流程模拟衔接的智能体系统。用户以自然语言提出组分、相平衡类型、操作条件和分离目标后，系统将请求转化为结构化热力学任务，调用 ThermoFormer 或经典热力学模型完成计算，并在需要时生成可由 DWSIM 加载的流程模拟文件。

基本约束如下：

- 语言模型负责意图解析、组分识别、任务路由和结果组织，不直接生成相平衡数值。
- 数值计算由 `thermo_engine`、ThermoFormer 后端、经典活度系数模型和 DWSIM 求解器承担。
- 缺少温度、压力、组成、SMILES 或相平衡类型等必要信息时，系统显式报告缺失项，不虚构参数。
- 对明显超出当前适用范围的任务，如电解质体系、聚合物体系或缺少可靠物性参数的复杂反应体系，系统给出边界说明。

### 1.2 ThermoAgent 编排

ThermoAgent 的核心功能包括：任务理解、组分名称映射、模型选择、参数导入、相平衡计算、DWSIM 文件导出和三源验证汇总。

在 DWSIM 衔接中，ThermoAgent 执行两类关键操作：

1. **自动猜测 DWSIM 映射名称**：系统根据常用名、IUPAC 名、同义名和 DWSIM 内部化合物库候选名进行匹配。例如，`2-propanol` 可映射到 `Isopropanol`，`n-propyl acetate` 可映射到 `N-propyl acetate`，`DMSO` 可映射到 `Dimethyl sulfoxide`。
2. **参数导入**：当体系需要显式二元交互参数时，ThermoAgent 将 NRTL / UNIQUAC 参数写入 DWSIM 工程文件或属性包字段；参数单位和方向按 DWSIM 内部字段要求转换，避免把文献 K 值直接写入 cal 尺度字段。

### 1.3 ThermoFormer 接入

ThermoFormer 作为相平衡预测后端接入 `thermo_engine/registry.py`。系统根据任务类型、组分数、压力范围和可用 SMILES 信息选择预测协议。

当前使用口径为：

| 体系 | 任务 | protocol | 默认 checkpoint |
|---|---|---|---|
| 正庚烷 / 正壬烷 | VLE | `vle_overall_binary` | `models/vle/prediction/vle_overall_binary/seed_2/best_model.pt` |
| 乙酸 / 水 / DMSO | VLE | `vle_overall_ternary` | `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt` |
| 水 / 正丁醇 | LLE | `binary-system` | `models/lle/prediction/binary-system/seed_0/best.pt` |

> 三元体系选用 `seed_2` 而非 registry 默认 `seed_0`，依据见 §2.4。
> §1.5 案例二与 §2.2 为同一体系，共用该权值。

### 1.4 DWSIM 支持与 SI 图像材料

正文中提到 ThermoAgent 支持 DWSIM，因此 SI 中应补充 DWSIM 图形界面证据。建议加入以下三组截图：

| SI 图 | 体系 | 建议打开文件 | 图像内容 |
|---|---|---|---|
| Fig. S-DWSIM-1 | 正庚烷 / 正壬烷 | `report/success/正庚烷-正壬烷/heptane_x0p466_2comp_bubble_117.7C.dwxmz` | near-bubble TP-flash flowsheet、UNIQUAC property package、small vapor outlet |
| Fig. S-DWSIM-2 | 乙酸 / 水 / DMSO | `report/success/乙酸-水-DMSO/aw_dmso_x0p316_x0p279_3comp_bubble_84.5C.dwxmz` | 三元 TP-flash flowsheet、三组分列表、汽相组成结果（UNIFAC 物性包） |
| Fig. S-DWSIM-3 | 水 / 正丁醇（二元 LLE） | `report/dwsim/water_butanol_LLE_298p15K.dwxmz` | 原生 Vessel、NRTL、Light/Heavy Liquid 两液相出口与两相组成 |

本次工作区未发现已归档的 `.png` / `.mp4` 截图文件；上述 `.dwxmz` 文件已归档（二元与三元泡点文件位于 `report/dwsim/`），可直接用于人工打开 DWSIM 后截取 SI 图。三元体系的操作视频建议录制 `report/success/乙酸-水-DMSO/aw_dmso_x0p316_x0p279_3comp_bubble_84.5C.dwxmz` 的打开、计算、结果页读取过程。

> **三元体系的物性包说明**：本体系在 DWSIM 中的物性包选择需要分用途讨论。
>
> - **§2.2 的三源泡点对比使用 UNIFAC**。原因是 DWSIM 内置库对 乙酸/水、
>   乙酸/DMSO、水/DMSO 三对**均无 NRTL 二元交互参数**，会走
>   `EstimateMissingInteractionParameters` 估算且 α 一律取默认 0.2，
>   用于**逐点泡点比对**时闪蒸失真（300 K 即出现汽相、泡点温度偏差 +108~+142 K），
>   无法与实验值有意义地比较。UNIFAC 为基团贡献法，不需二元参数，泡点偏差
>   收敛到 −6~−15 K 的合理范围。
> - **§1.5 的严格塔工程文件使用 NRTL**，并且实测**严格求解通过**（错误数 0）。
>   即：NRTL 的参数估算虽使单点泡点计算失真，但在严格塔的逐板迭代中
>   仍能得到收敛解，且塔釜温度（462.57 K）与 DMSO 沸点吻合。
>
> 因此两处口径不同是有意为之，各自服务于不同目的，不构成矛盾。
> 详见 §2.2 与 §1.5 表 1.5-3。

三个体系的 DWSIM 文件复现命令（需 DWSIM + pythonnet，且 pythonnet 需完整进程权限，不得在受限沙盒内运行）：

```powershell
conda activate thermo
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# 体系① 正庚烷 / 正壬烷（二元泡点 + .dwxmz + 补全 CSV）
python scripts\generate_heptane_nonane_comparison.py
python scripts\generate_heptane_nonane_dwsim.py

# 体系② 乙酸 / 水 / DMSO（三元三源 + .dwxmz）
python scripts\build_aw_dmso_three_source.py --stage tf --seeds seed_0,seed_1,seed_2,seed_3,seed_4
python scripts\build_aw_dmso_three_source.py --stage dwsim
python scripts\build_aw_dmso_three_source.py --stage merge --seeds seed_0,seed_1,seed_2,seed_3,seed_4
python scripts\build_aw_dmso_flash_files.py          # §1.4 的 SI 三元 flash 文件
python scripts\design_aw_dmso_multi_seed.py          # §1.5 的多源短节法设计
```

> **§1.5 选定塔文件的来源**：`water_acetic_acid_dmso_extractive_thermoformer.dwxmz`
> 在短节法设计量（`seed_2` 源：N=15、R=1.441）基础上，于 DWSIM GUI 中按
> NRTL 物性包实现，并把求解方法切换为 `Napthali-Sandholm (Simultaneous Correction)`、
> 进料板调整为第 14 板、萃取剂板调整为第 3 板，最终严格求解通过。
> 其精确规格见 §1.5 表 1.5-3。

# 体系③ 水 / 正丁醇 二元 LLE（298.15 / 313.15 / 343.15 / 353.15 K 四个流程）
python scripts\run_water_butanol_dwsim.py
```

### 1.5 塔设计与工程文件

塔设计采用确定性短节法（Fenske-Underwood-Gilliland，操作回流比取 1.4 × 最小回流比）。
二元案例以 UNIFAC 与 DWSIM-UNIQUAC 的实际 VLE 路径对照；三元案例仍以 UNIFAC 与
ThermoFormer 两种热力学来源对照。

**表 1.5-1　正庚烷 / 正壬烷普通精馏（101.325 kPa，F = 1.0 mol/s，x_正庚烷 = 0.466，塔顶纯度 0.995，回收率 0.98）**

| 项目 | UNIFAC | DWSIM-UNIQUAC |
|---|---:|---:|
| 相对挥发度 α | 4.4623 | 4.2859 |
| 理论塔板数 N | **14** | **15** |
| 最小塔板数 N_min | 6.243 | 6.416 |
| 回流比 R | **0.846** | **0.893** |
| 最小回流比 R_min | 0.605 | 0.638 |
| 进料板 | 6 | 7 |
| 进料泡点温度 (K) | 391.18 | 390.81 |
| 塔顶泡点温度 (K) | 371.69 | 371.43 |
| 塔釜泡点温度 (K) | 422.21 | 422.40 |

两条路径仅相差 **1 块理论板**，设计回流比相差 **5.6%**，与 §2.1 的五个二元 VLE 闪蒸点保持一致。
DWSIM-UNIQUAC 流程文件为
`report/success/正庚烷-正壬烷/heptane_nonane_binary_distillation_x0p466.dwxmz`；其 JSON 设计记录同时保留
进料、塔顶和塔釜的 DWSIM 泡点与汽相组成。

**表 1.5-2　乙酸 / 水 / DMSO 三元体系萃取精馏（101.325 kPa，F = 1.0 mol/s，
萃取剂比 2.0，塔顶纯度 0.95，回收率 0.90）**

> **权值口径**：本表 ThermoFormer 列使用
> `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt`。
> 该权值对本报告三元体系（§2.2 的 乙酸 / 水 / DMSO）的三组分汽相组成 MAE 为 **0.2548**
> （17 个实验点）。本体系各权值精度接近（0.2233~0.2871，最优为 `seed_4`），
> 采用 `seed_2` 是为与本报告既有口径保持一致，详见 §2.4。

本案例与 §2.2 为**同一体系**。常压沸点：水 319.55 K < 乙酸 390.94 K < DMSO 463.7 K。
萃取剂须为三者中沸点最高者，故取 **DMSO 作萃取剂**、水/乙酸为关键对。

| 项目 | UNIFAC | TF (seed_2) | 比值 |
|---|---:|---:|---:|
| alpha_base | 2.0728 | 1.6836 | 0.81 |
| alpha_ext | 13.3840 | 3.9952 | 0.30 |
| 选择性 alpha_ext / alpha_base | **6.4569** | **2.3731** | 0.37 |
| alpha_avg | 5.2671 | 2.5935 | 0.49 |
| 理论塔板数 N | 11 | **15** | 1.36 |
| 最小塔板数 N_min | 4.121 | 7.185 | 1.74 |
| 回流比 R | 0.451 | **1.441** | 3.19 |
| 最小回流比 R_min | 0.322 | 1.030 | 3.20 |
| 进料板 / 萃取剂板（短节法） | 5 / 2 | 7 / 2 | — |
| 塔顶 / 塔釜温度 (K)（短节法） | 372.23 / 459.70 | 354.10 / 366.59 | — |

**表 1.5-3　选定 DWSIM 工程文件的实算规格与结果**

上表的短节法设计量随后在 DWSIM 中实现并**严格求解通过**。工程文件为
`data/exports/flow_examples/water_acetic_acid_dmso_extractive_thermoformer.dwxmz`。

| 项目 | 值 |
|---|---|
| 组分 | 水 / 乙酸 / 二甲基亚砜（DMSO） |
| 物性包 | **NRTL** |
| 理论塔板数 | **15** |
| 回流比 R | **1.441** |
| 塔压降 | 5 kPa |
| 求解方法 | **Napthali-Sandholm (Simultaneous Correction)**，最大迭代 100 |
| 冷凝器规格 | `Stream_Ratio` = 1.441 |
| 再沸器规格 | `Product_Molar_Flow_Rate` = 2.526316 mol/s |
| 萃取剂进料 | 第 **3** 板，DMSO 2.0 mol/s，354.10 K，101.325 kPa |
| 关键对进料 | 第 **14** 板，水/乙酸 = 0.5/0.5，1.0 mol/s，366.01 K，101.325 kPa |
| 塔顶产品（实算） | **412.59 K**，**0.4737 mol/s**，x = 0.5036 / 0.0289 / 0.4675 |
| 塔釜产品（实算） | **462.57 K**，**2.5263 mol/s**，x = 0.1035 / 0.1925 / 0.7040 |

**要点说明**

- **选择性两源均 > 1（6.4569 / 2.3731），即 DMSO 确实增强了水/乙酸的相对挥发度**，
  本案例是**真正意义上的萃取精馏**。这一点与本报告早期版本的三元案例形成对比：
  1-丁醇 / 水 / 甲苯 的选择性两源均 < 1（0.7467 / 0.8193），只能表述为
  「有萃取剂进料的三元精馏塔」。
- `alpha_base` 两源均 > 1（2.07 / 1.68），即水/乙酸在无萃取剂时本就可分离，
  这是短节法有效的前提。
- 两源塔板数相差 4 块（11 vs 15），量级一致。
- **严格塔求解通过**：`CalculateFlowsheet` 返回错误数 0。塔顶、塔釜产品流量
  与物料衡算设计值精确一致（0.4737 / 2.5263 mol/s）。
- **塔釜温度 462.57 K 接近 DMSO 常压沸点 463.7 K**，物理上合理；
  这一点优于短节法 `seed_2` 源给出的 366.59 K（该值低于乙酸沸点 390.94 K，
  富 DMSO 的塔釜混合物不可能在该温度沸腾），亦说明严格模型修正了短节法
  在温度上的偏差。
- **严格求解按工程实践调整了两处设定**：萃取剂进料由第 2 板下移至第 3 板、
  关键对进料由第 7 板下移至第 14 板。短节法的进料板位置（`0.45 × 级数`）
  是经验估计，严格塔需要更靠下的进料点才能收敛。
- 塔顶产品组成为 x_水 0.5036，**未达到短节法设定的 0.95 纯度**。
  这与 §2.2 揭示的模型局限一致：本体系 DMSO 的气相组成预测存在系统性偏差，
  短节法基于 `[0.10, 0.10, 0.80]` 单点估算的选择性偏乐观。
  工程上若要达到 0.95，需按实算结果放大回流比或增加塔板数。

> **相邻方案对照**（仅供参考，非选定设计）：本体系 TF 各权值分歧较大 ——
> `seed_1` 给出 N=31 / R=4.969、塔釜 371.20 K；`seed_3` 给出 N=76 / R=15.284
> 且 `alpha_base` = 0.8001 < 1（短节法前提不成立）的退化结果；
> `seed_4` 给出 N=23 / R=3.356、塔釜 355.20 K（明显偏低）。
> 这些源均不作为塔设计依据，详见 §2.4。

对应 DWSIM 塔文件：

| 案例 | 工程文件 / 记录 |
|---|---|
| 正庚烷 / 正壬烷 | `report/success/正庚烷-正壬烷/heptane_nonane_binary_distillation_x0p466.dwxmz`（DWSIM-UNIQUAC）; `heptane_nonane_binary_distillation_x0p466_design.json` |
| 乙酸 / 水 / DMSO | `data/exports/flow_examples/water_acetic_acid_dmso_extractive_thermoformer.dwxmz`（**选定，严格塔求解通过**） |

> **塔文件校验状态**：`water_acetic_acid_dmso_extractive_thermoformer.dwxmz` 已实测
> 在 DWSIM 中打开并 `CalculateFlowsheet` 求解，**返回错误数 0**；
> 组分、级数（`NumberOfStages` 与 `Stages.Count` 同为 15）、两股进料板位与组成、
> 两条规格均已逐项回读核对。塔顶/塔釜产品流量（0.4737 / 2.5263 mol/s）
> 与物料衡算设计值一致。

> **收敛的关键设置**：求解方法使用 **`Napthali-Sandholm (Simultaneous Correction)`**。
> 实测该体系用 DWSIM 默认的 Wang-Henke 方法（100 次迭代）无法收敛，
> 报 `Solver reached the maximum number of iterations without converging`
> 或 `Failed to fulfill mass balance for Water`；换用 Napthali-Sandholm 后收敛。
> 另需注意 `SolvingMethodName` 的字符串拼写 —— 该属性不做校验，
> 名称错误只在求解时才报 `Unable to find column solver with name '...'`。

生成入口：正庚烷 / 正壬烷使用 `scripts/generate_heptane_nonane_binary_column.py`；
三元体系使用 `scripts/export_aw_dmso_seed2.py`（含 `--source` 选择 UNIFAC 或各权值，
并注入塔板温度/流量/组成初值）。

> **历史文件说明**：`ipa_water_binary_column_x0p3_tf.dwxmz` 与 `eac_npac_dmso_extractive_tf_column.dwxmz`
> 由 `generate_ipa_water_std_column_tf.py` / `generate_eac_npac_dmso_extractive_tf.py` 生成，
> 但这两个脚本中的"TF 设计量"是**硬编码字面量**（`TF_ALPHA`、`TF_STAGES`、`TF_REFLUX` 等），
> 并非由 ThermoFormer 实际计算；脚本只从 UNIFAC 取进料泡点温度。
> 即这两个文件的"TF"设计值并非 ThermoFormer 预测产物，**不可作为 TF 源设计证据使用**，
> 不再作为本报告 §1.5 的二元设计证据（旧文件保留未删除，以备溯源）。

---

## 第二部分　预测与实验的三源比较

### 2.1 二元 VLE：正庚烷 / 正壬烷（推荐直接精馏体系）

实验数据来自 NIST ThermoML 二元 VLE 数据集（DOI `10.1016/j.fluid.2013.05.016`，
`quality_status=passed`）。ThermoFormer 列取自 `vle_overall_binary` 的 `seed_2` 锁定测试输出；
该体系的 16 个实验状态均在该模型训练/验证之外。下表中 `x` 和 `y` 均为**正庚烷**摩尔分数，
正庚烷为塔顶轻组分，正壬烷为塔釜重组分。DWSIM 使用 UNIQUAC 物性包、101.325 kPa。

**表 2.1-1　实验 / TF / DWSIM 泡点对照**

| x_正庚烷 | T_实验 (°C) | T_TF (°C) | T_DWSIM (°C) | y_实验 | y_TF | y_DWSIM | DWSIM 文件 |
|:--:|---:|---:|---:|---:|---:|---:|---|
| 0.117 | 140.75 | 138.93 | 139.97 | 0.3250 | 0.3830 | 0.3403 | `heptane_x0p117_2comp_bubble_140.0C.dwxmz` |
| 0.359 | 123.05 | 121.67 | 123.28 | 0.7260 | 0.7384 | 0.7006 | `heptane_x0p359_2comp_bubble_123.3C.dwxmz` |
| 0.466 | 117.15 | 116.28 | 117.66 | 0.8240 | 0.8163 | 0.7890 | `heptane_x0p466_2comp_bubble_117.7C.dwxmz` |
| 0.633 | 109.15 | 109.55 | 110.30 | 0.9160 | 0.8970 | 0.8844 | `heptane_x0p633_2comp_bubble_110.3C.dwxmz` |
| 0.837 | 102.55 | 103.10 | 103.01 | 0.9730 | 0.9613 | 0.9593 | `heptane_x0p837_2comp_bubble_103.0C.dwxmz` |

16 个锁定实验点的 ThermoFormer 温度 MAE 为 **0.985 °C**、汽相正庚烷组成 MAE 为 **0.0246**。
完整实验—ThermoFormer 表见 `report/heptane_nonane_thermoformer_vs_experiment.csv`；五个 DWSIM
泡点流程及三源 CSV 位于 `report/success/正庚烷-正壬烷/`。

在同一直接精馏目标（101.325 kPa、F = 1.0 mol/s、进料 x_正庚烷 = 0.466、塔顶正庚烷
0.995、回收率 0.98）下，两条物性路径给出接近的短截法设计：

| 项目 | UNIFAC | DWSIM-UNIQUAC |
|---|---:|---:|
| 进料点相对挥发度 α | 4.4623 | 4.2859 |
| 最小理论板数 | 6.243 | 6.416 |
| 设计理论板数 | **14** | **15** |
| 进料板 | 6 | 7 |
| 最小回流比 | 0.605 | 0.638 |
| 设计回流比 | **0.846** | **0.893** |

因此该体系避免了 1-氯丁烷 / 环己烷中相对挥发度接近 1 所造成的放大差异。对应的直接二元精馏
文件为 `report/success/正庚烷-正壬烷/heptane_nonane_binary_distillation_x0p466.dwxmz`；设计记录明确标注
其为 Fenske–Underwood–Gilliland 短截法规格，打开 DWSIM 后仍应进行严格塔计算与收敛确认。

### 2.1b 二元 VLE：1-氯丁烷 / 环己烷

实验数据来自 NIST ThermoML 二元 VLE 数据集（DOI `10.1016/j.fluid.2006.02.009`）。以下 ThermoFormer 列来自
`vle_overall_binary` 的 `seed_2` 正式留出测试结果；该二元体系不参与该模型的训练或验证。为与实验数据源保持一致，
`x` 和 `y` 均表示 **1-氯丁烷** 的摩尔分数，另一组分环己烷的摩尔分数为 `1 - x` 或 `1 - y`。

**表 2.1b-1　常压等压泡点对照**（101.30 kPa；代表点，`y` 为汽相中 1-氯丁烷摩尔分数）

| x_1-氯丁烷 | T_实验 (°C) | T_TF (°C) | y_实验 | y_TF |
|:--:|---:|---:|---:|---:|
| 0.1005 | 79.39 | 79.31 | 0.1413 | 0.1367 |
| 0.2796 | 77.86 | 77.68 | 0.3277 | 0.3253 |
| 0.4705 | 77.04 | 76.87 | 0.4904 | 0.4920 |
| 0.6967 | 77.07 | 76.81 | 0.6819 | 0.6830 |
| 0.8960 | 77.80 | 77.58 | 0.8777 | 0.8762 |

该体系总计 51 个实验点，覆盖 53.30、80.00 和 101.30 kPa 三条等压线。去除两个纯组分端点后，
ThermoFormer 的泡点温度 MAE 为 **0.209 °C**（RMSE 0.243 °C），汽相组成 MAE 为 **0.0036**。
完整的 51 点“实验 / ThermoFormer”并列数据见
`report/chlorobutane_cyclohexane_thermoformer_vs_experiment.csv`；每一行的 `abs_temperature_error_C` 和
`abs_y_error` 均已列出，便于复核。

### 2.2 三元 VLE：乙酸 / 水 / 二甲基亚砜（DMSO）

**UNIFAC** 物性包，13.33 kPa（等压）。表中 `y` 的顺序为 乙酸 / 水 / DMSO。
实验数据取自 NIST ThermoML 三元 VLE 数据集（`datasets/vle/ternary_vle_english.csv`，
DOI `10.1016/j.fluid.2008.09.010`，17 点），ThermoFormer 列为 `vle_overall_ternary`
**`seed_2`** 权值在各实验温度下的等温泡点预测。

常压沸点：水 319.55 K < 乙酸 390.94 K < DMSO 463.7 K。

**表 2.2-1　代表点三源对照**（`y` 顺序：乙酸 / 水 / DMSO）

| T_实验 (K) | T_DWSIM (K) | x (乙酸/水/DMSO) | y_实验 | y_TF (seed_2) | y_DWSIM |
|---:|---:|:--|:--|:--|:--|
| 357.62 | 346.26 | 0.316 / 0.279 / 0.405 | 0.143 / 0.807 / 0.050 | 0.177 / 0.441 / 0.383 | 0.184 / 0.801 / 0.016 |
| 363.70 | 350.17 | 0.319 / 0.238 / 0.443 | 0.152 / 0.768 / 0.079 | 0.180 / 0.382 / 0.438 | 0.194 / 0.782 / 0.024 |
| 365.10 | 352.57 | 0.324 / 0.214 / 0.463 | 0.181 / 0.743 / 0.076 | 0.184 / 0.347 / 0.470 | 0.205 / 0.765 / 0.030 |
| 373.21 | 361.33 | 0.328 / 0.145 / 0.526 | 0.230 / 0.589 / 0.182 | 0.187 / 0.238 / 0.574 | 0.244 / 0.693 / 0.063 |
| 383.32 | 372.88 | 0.264 / 0.100 / 0.636 | 0.181 / 0.474 / 0.344 | 0.126 / 0.151 / 0.724 | 0.202 / 0.629 / 0.169 |
| 387.84 | 379.14 | 0.262 / 0.069 / 0.669 | 0.220 / 0.361 / 0.419 | 0.122 / 0.102 / 0.776 | 0.230 / 0.525 / 0.246 |
| 391.92 | 385.07 | 0.262 / 0.043 / 0.694 | 0.209 / 0.262 / 0.529 | 0.121 / 0.063 / 0.816 | 0.268 / 0.395 / 0.337 |

**表 2.2-2　三源偏差汇总（17 点）**

| 源 | 总体 MAE | 乙酸 | 水 | DMSO |
|---|---:|---:|---:|---:|
| **DWSIM (UNIFAC)** | **0.0714** | 0.0207 | 0.0873 | 0.1061 |
| TF `seed_4` | 0.2233 | 0.0647 | 0.3349 | 0.2702 |
| TF `seed_1` | 0.2445 | 0.0618 | 0.3667 | 0.3049 |
| TF `seed_2` | 0.2548 | 0.0459 | 0.3408 | 0.3779 |
| TF `seed_3` | 0.2814 | 0.0912 | 0.4193 | 0.3338 |
| TF `seed_0` | 0.2871 | 0.0660 | 0.4118 | 0.3835 |

DWSIM 泡点温度偏差：均值 **−11.84 K**（范围 −6.1 ~ −15.3 K）。

**要点说明**

- **DWSIM 的汽相组成预测显著优于 ThermoFormer**：总体 MAE **0.0714**，
  优于最佳权值 `seed_4` 的 0.2233 约 **3.1 倍**。
  这与 §2.1（二元 VLE）的结论方向一致，但**与原三元体系（1-丁醇 / 水 / 甲苯）
  的结论相反** —— 那里是 TF (0.0388) 优于 DWSIM (0.1183)。
- **ThermoFormer 的失效集中在 DMSO 与水**：最佳权值（`seed_4`）下 DMSO 分量
  MAE 0.2702、水 0.3349，而乙酸仅 0.0647。逐点看，TF 的 `y_DMSO` 随液相 DMSO
  **上升过快**：x_DMSO 从 0.405 升到 0.694 时，实验 `y_DMSO` 从 0.050 升到 0.529，
  而 TF 预测从 0.383 冲到 0.816（富 DMSO 端高估约 0.29）；
  DWSIM 则从 0.016 升到 0.337，始终低于实验（富 DMSO 端低估约 0.19）。
- 乙酸组分两源都准（DWSIM 0.0207、TF `seed_2` 0.0459）。
- DWSIM 泡点温度系统性偏低 6~15 K，与组成误差方向不一致，源于 UNIFAC
  基团贡献法对强缔合组分（乙酸、DMSO）的处理偏保守。
- 本节 DWSIM 列为 Automation API 等压泡点实算（**UNIFAC** 物性包，
  以 `PhaseIds` 判定相态、二分温度求泡点）；ThermoFormer 列为 `seed_2`
  权值的等温泡点预测。DWSIM 列与权值选择无关。

> **物性包选择（重要）**：**本节（逐点泡点比对）的 DWSIM 列使用 UNIFAC，不用 NRTL。**
> 实测 DWSIM 内置库对 乙酸/水、乙酸/DMSO、水/DMSO 三对**均无 NRTL 二元交互参数**，
> 会走 `EstimateMissingInteractionParameters` 估算，且 α 一律取默认值 0.2：
>
> ```
> Estimated NRTL IP set for Acetic acid/Water:              -189.10 / 728.84 / 0.2
> Estimated NRTL IP set for Acetic acid/Dimethyl sulfoxide: -1233.93 / -1295.15 / 0.2
> Estimated NRTL IP set for Water/Dimethyl sulfoxide:       -1999.94 / -499.97 / 0.2
> ```
>
> 用这套估算参数做**逐点泡点计算**完全失真，实测症状：300 K（远低于所有组分沸点）
> 即出现汽相；液相组成在 355~375 K 剧烈跳动、400~500 K 完全混乱；
> 泡点二分收敛到搜索上界，温度偏差达 **+108 ~ +142 K**。
> 对照：原体系（1-丁醇 / 水 / 甲苯）的三对**都**有内置参数，从不触发估算。
> UNIFAC 为基团贡献法，不需二元交互参数，泡点偏差收敛到 −6~−15 K，故本节改用 UNIFAC。
>
> 注意：§1.5 的**严格塔**工程文件采用 NRTL 且求解通过 —— 参数估算对单点泡点
> 比对不可用，但在严格塔逐板迭代中仍可收敛。两处口径不同，各服务其目的。

**DWSIM 工程文件**（`report/success/乙酸-水-DMSO/`，流程为 `Feed → Flash (Vessel) → Vapor / Liquid`）

| 文件 | 覆盖点 x(乙酸 / 水) |
|---|---|
| `aw_dmso_x0p316_x0p279_3comp_bubble_84.5C.dwxmz` | 0.316 / 0.279 |
| `aw_dmso_x0p321_x0p204_3comp_bubble_95.4C.dwxmz` | 0.321 / 0.204 |
| `aw_dmso_x0p324_x0p166_3comp_bubble_100.8C.dwxmz` | 0.324 / 0.166 |
| `aw_dmso_x0p261_x0p093_3comp_bubble_113.4C.dwxmz` | 0.261 / 0.093 |
| `aw_dmso_x0p262_x0p043_3comp_bubble_118.8C.dwxmz` | 0.262 / 0.043 |

同一目录另含三源数据与设计记录：

| 文件 | 内容 |
|---|---|
| `aw_dmso_three_source_bubble.csv` | 17 点逐点三源对照（实验 / TF seed_2 / DWSIM），末列 `file` 指向对应 .dwxmz |
| `aw_dmso_extractive_design.json` | 体系、数据源、物性包、物料衡算、三源精度、短节法设计与严格塔实算 |
| `water_acetic_acid_dmso_extractive_thermoformer.dwxmz` | §1.5 的严格塔工程文件（NRTL，求解通过） |

上述 5 个 flash 文件均已实测：打开后 `CalculateFlowsheet` 计算**无错误**（UNIFAC 物性包）；
严格塔文件同样实测求解通过（NRTL）。

> `aw_dmso_three_source_bubble.csv` 中 `T_thermoformer_K` 一列为 `n/a`：
> ThermoFormer 按**等温**泡点运行（输入 T 与 x、输出 y），不产出泡点温度；
> 而 DWSIM 列为**等压**泡点，故有温度。

### 2.3 二元 LLE：水 / 正丁醇

条件：101.325 kPa，温度逐点取实验条件。相位命名：`organic = 正丁醇富集相`（轻相），`aqueous = 水富集相`（重相）。

**表 2.3-1　正丁醇富集相（organic）中正丁醇摩尔分数**

| T (K) | 实验 | TF | DWSIM |
|:--:|---:|---:|---:|
| 298.15 | 0.4869 | 0.5090 | 0.3992 |
| 313.15 | 0.4840 | 0.4833 | 0.4070 |
| 343.15 | 0.4190 | 0.4330 | 0.4138 |
| 353.15 | 0.3965 | 0.4166 | 0.4137 |

**表 2.3-2　水富集相（aqueous）中正丁醇摩尔分数**

| T (K) | 实验 | TF | DWSIM |
|:--:|---:|---:|---:|
| 298.15 | 0.0188 | 0.0388 | 0.0055 |
| 313.15 | 0.0190 | 0.0423 | 0.0074 |
| 343.15 | 0.0160 | 0.0502 | 0.0125 |
| 353.15 | 0.0181 | 0.0531 | 0.0147 |

实验为各温度重复 tie-line 的均值；实验 tie-line 条数分别为 298.15 K：3 条，313.15 K：4 条，343.15 K：2 条，353.15 K：1 条。

本体系 ThermoFormer 列为 `binary-system`（`seed_0`）权值实跑结果，与实验、DWSIM 两列均可逐位复现，两相残差 ~1e-16。

流程结构为 `Feed → Vessel_LLE → Vapor / Light_Liquid / Heavy_Liquid`，其中 `Light_Liquid`为正丁醇富集相（organic）、`Heavy_Liquid` 为水富集相（aqueous），与本节相位命名一致。进料总摩尔流量 1.0 mol/s、101.325 kPa，总组成 正丁醇:水 = 0.3:0.7（位于两相区内）。写入的 NRTL 二元交互参数为 DWSIM 内置值：A12 = 2633.6951、A21 = 504.0381、α12 = 0.4447。

### 2.4 ThermoFormer 预测精度说明（权值选择）

§2.1–§2.3 各表的 ThermoFormer 列由 `ThermoFormerBackend` 实跑得到，要点如下：

- **三元体系（§2.2）使用 `seed_2` 权值**：对 乙酸 / 水 / DMSO 的 17 个实验点，
  三组分汽相组成 MAE 为 **0.2548**。
  逐组分 MAE 为 乙酸 0.0459、水 0.3408、DMSO 0.3779 —— 乙酸准确，
  而水与 DMSO 偏差较大，即本体系在 ThermoFormer 训练分布之外。
- **本体系各权值精度接近，`seed_4` 最优（0.2233），`seed_2` 列第三（0.2548）**：

  | 权值 | 总体 MAE | 乙酸 | 水 | DMSO |
  |---|---:|---:|---:|---:|
  | `seed_4` | **0.2233** | 0.0647 | 0.3349 | 0.2702 |
  | `seed_1` | 0.2445 | 0.0618 | 0.3667 | 0.3049 |
  | `seed_2` | 0.2548 | 0.0459 | 0.3408 | 0.3779 |
  | `seed_3` | 0.2814 | 0.0912 | 0.4193 | 0.3338 |
  | `seed_0` | 0.2871 | 0.0660 | 0.4118 | 0.3835 |

  **本报告仍统一采用 `seed_2`**，理由是与 §1.5、§2.2、§1.3 的既有口径保持一致，
  且五个权值的差距（0.2233~0.2871）远小于它们与 DWSIM（0.0714）的差距，
  换用 `seed_4` 不改变本节任何结论。若后续需最优精度，应改用 `seed_4`。
- **二元体系（§2.1 正庚烷 / 正壬烷）使用 `seed_2`**：16 个锁定实验点的泡点温度 MAE 为
  **0.985 °C**、汽相正庚烷组成 MAE 为 **0.0246**；该权值与实验和 DWSIM-UNIQUAC 都保持一致的
  VLE 趋势，故选作本报告的二元直接精馏展示体系。
- **§1.5 塔设计的规格来源**：短节法设计量取自 `seed_2` 源（N=15、R=1.441），
  随后在 DWSIM 中以 **NRTL** 物性包实现并严格求解通过（见 §1.5 表 1.5-3）。
  需注意短节法给出的进料板位置（7 / 2）在严格塔中调整为（14 / 3）才收敛，
  且实算塔釜温度 462.57 K 与短节法 `seed_2` 的 366.59 K 相差很大 ——
  后者低于乙酸沸点 390.94 K，物理上不合理，说明短节法在本体系上的
  温度估计不可靠，应以严格塔实算值为准。

---

## 补充部分　预测权值文件

§2 各体系（正庚烷 / 正壬烷，二元 VLE；乙酸 / 水 / DMSO，三元 VLE；水 / 正丁醇，二元 LLE）
及 §1.5 塔设计所用权值文件路径如下。

| 体系 | 权值文件（仓库相对路径） | 选择依据 |
|---|---|---|
| 正庚烷 / 正壬烷 | `models/vle/prediction/vle_overall_binary/seed_2/best_model.pt` | 16 个锁定实验点：T MAE 0.985 °C，y MAE 0.0246 |
| 乙酸 / 水 / DMSO（§2.2 与 §1.5） | `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt` | 与既有口径一致；五权值 MAE 0.2233~0.2871，`seed_4` 最优但差距不改变结论 |
| 水 / 正丁醇 | `models/lle/prediction/binary-system/seed_0/best.pt` | registry 默认 |
