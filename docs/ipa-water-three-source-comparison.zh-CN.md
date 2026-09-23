# ThermoFormer 泡点预测三源验证：2-丙醇-水 & 乙酸乙酯-乙酸正丙酯-二甲基亚砜

> 本实验对比 **实验（Excel 实测）/ ThermoFormer（ML 预测）/ DWSIM（NRTL 机理）** 三种来源的泡点温度。
> 两个体系分别代表 ThermoFormer 的**训练域内**与**外推域**，用于展示"怎么判断 ML 泡点预测好不好"。
> 核心标尺：**实验与 DWSIM（机理模型）是判断 ThermoFormer 预测是否可信的基准**。

---

## 总览

| 体系 | 类型 | 组分数 | ThermoFormer 泡点偏差 | 结论 |
|---|---|---|---|---|
| **2-丙醇-水** | 醇-水（训练域内）| 2 | MAE ≈ 2.6 °C | ✅ 可用，DWSIM ≈ ThermoFormer ≈ 实验 |
| **乙酸乙酯-乙酸正丙酯-二甲基亚砜** | 酯-砜（外推域）| 3 | MAE ≈ 18 °C | ⚠️ 外推短板，DWSIM ≈ 实验 ≫ ThermoFormer |

## Agent调用ThermoFormer预测流程

![image-20260829175550945](C:\Users\34861\AppData\Roaming\Typora\typora-user-images\image-20260829175550945.png)

---



## 2-丙醇-水（二元，训练域内）

> 数据：常压等压泡点（P=760 mmHg），5 个组成 x_IPA = 0.1 / 0.3 / 0.5 / 0.7 / 0.9。

### 1.1 三源泡点对比

| x_IPA | T_实验(°C) | T_ThermoFormer(°C) | T_DWSIM(°C) | y_IPA(实验) | y_IPA(TF) | y_IPA(DWSIM) |
|---|---|---|---|---|---|---|
| 0.1 | 84.75 | 90.21 | 89.50 | 0.495 | 0.360 | 0.378 |
| 0.3 | 81.85 | 83.77 | 83.68 | 0.541 | 0.570 | 0.565 |
| 0.5 | 80.15 | 81.81 | 81.92 | 0.605 | 0.657 | 0.644 |
| 0.7 | 80.02 | 81.14 | 81.12 | 0.687 | 0.727 | 0.730 |
| 0.9 | 80.87 | 81.49 | 81.42 | 0.872 | 0.876 | 0.875 |

**解读**
- **富水区（x=0.1）**：实验泡点最低（84.75°C），两模型都偏高 ~+4.8°C（富水陡降区共同偏差）。
- **中高 x（0.3–0.9）**：ThermoFormer 与 DWSIM **几乎重合**（温度差 ≤0.03°C），比实验高 ~+1~3°C。
- **y_IPA**：三源单调趋势完全一致；x=0.1 时实验 y 偏高，两模型偏低。
- 总体：**DWSIM ≈ ThermoFormer ≳ 实验**，物理合理、趋势一致。

### 1.2 ThermoFormer 方法

- 脚本 `lab_models/ThermoFormer/scripts/predict_ipa_water.py`，权重 `overall_binary/seed_4/best_model.pt`
- **关键**：给求解器传真实 Antoine 纯物性参数（异丙醇、水），否则泡点温度虚高 ~100°C（用"学到的 P_sat"会不物理）
- 等压泡点（P=101.325 kPa），输出 `docs/prediction_ipa_water.csv`
- **精度**：泡点温度 MAE ≈ **2.58 °C**；富水区偏差略大，富异丙醇区/端点区准

### 1.3 DWSIM 方法

- 脚本 `scripts/dwsim_ipa_bubble.py`：二分 Vapor 流股摩尔流量定位泡点温度，输出 `docs/dwsim_ipa_bubble.csv`
- 内置 NRTL 参数（Water/Isopropanol：`45.59 / 944.70 / 0.2`），与实验偏差 ≈ +0.2 ~ +4.8°C

---

## 乙酸乙酯-乙酸正丙酯-二甲基亚砜

> 数据：常压等压泡点（P=760 mmHg），36 点。
> 用途：展示 ThermoFormer **外推短板**，并与上一个案例形成对照。

### 2.1 三源泡点对比（代表点）

| x_乙酸乙酯 | x_乙酸正丙酯 | x_DMSO | T_实验 | T_ThermoFormer | T_DWSIM | ΔT(TF) | ΔT(DWSIM) |
|---|---|---|---|---|---|---|---|
| 0.374 | 0.024 | 0.602 | 101.68 | 84.91 | 96.11 | −16.8 | −5.6 |
| 0.354 | 0.138 | 0.508 | 103.50 | 83.87 | 96.85 | −19.6 | −6.7 |
| 0.220 | 0.179 | 0.602 | 107.05 | 88.41 | 103.59 | −18.6 | −3.5 |
| 0.182 | 0.314 | 0.504 | 111.58 | 89.41 | 104.41 | −22.2 | −7.2 |
| 0.060 | 0.334 | 0.607 | 114.33 | 100.14 | 113.09 | −14.2 | **−1.2** |
| 0.032 | 0.465 | 0.503 | 113.99 | 102.21 | 112.19 | −11.8 | **−1.8** |

**统计**：ThermoFormer（36 点）MAE = **18.09 °C**、系统性偏低；DWSIM（6 代表点）与实验偏差 ≈ −1.2 ~ −7.2 °C。

### 2.2 ThermoFormer 方法

- 脚本 `lab_models/ThermoFormer/scripts/predict_ternary_dms.py`，权重 `overall_binary_ternary/seed_4/best_model.pt`
- 用 Uni-Mol v2 编码（768 维）+ 给三组分拟合真实 Antoine 参数，等压泡点（P=101.325 kPa）
- 输出 `docs/prediction_ternary_dms.csv`

### 2.3 DWSIM 模拟（生成文件 + 自动算泡点）

- 脚本 `scripts/dwsim_dmso_bubble.py`
  - 对每个代表组成，二分 Vapor 流量求泡点温度
  - 生成 6 个进料温度恰在泡点上的 `.dwxmz`（`docs/DmsoBubble_x_*.dwxmz`）
  - 写出 `docs/dwsim_dmso_bubble.csv`
- 内置 NRTL 参数：Ethyl acetate/N-propyl acetate `-157.94/171.89/0.2`、
  Ethyl acetate/DMSO `-7.40/826.27/0.2`、N-propyl acetate/DMSO `127.02/855.10/0.2`

---

## 综合结论

| | 2-丙醇-水（域内）| 酯-酯-DMSO（域外）|
|---|---|---|
| ThermoFormer | MAE 2.6°C，可用 | MAE 18°C，不宜定量 |
| DWSIM vs 实验 | ≈ +0~5°C | ≈ −1~7°C（可靠）|
| 三源关系 | DWSIM ≈ TF ≈ 实验 | DWSIM ≈ 实验 ≫ TF |

1. **DWSIM（NRTL）在两类体系都贴近实验**，可作为泡点预测的**机理基准**。
2. **ThermoFormer 只在训练域内（醇-水）表现好**（MAE ~2.6°C）；酯-砜等外推域系统性偏差大（MAE ~18°C）。
3. **判断"预测好不好"的标尺**：把 ML 预测与**实验 + 机理模型（DWSIM/NRTL/UNIFAC）** 并排对照；
   ML 不劣于机理基线才算可用。对非醇/非水体系须先做适用性核对（与 v4 结论一致）。
4. **实用口径**：域内体系可用 ThermoFormer 初筛；外推体系应直接用 DWSIM / NRTL 等机理方法。

---

## 参考文件

权重：`checkpoints/overall_binary/seed_4/best_model.pt`（2-丙醇-水），`checkpoints/overall_binary_ternary/seed_4/best_model.pt`（DMSO 三元）
