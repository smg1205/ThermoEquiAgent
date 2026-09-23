# 乙醇-乙酸乙酯 泡点验证：实验 vs ThermoFormer vs DWSIM

> 案例：乙醇–乙酸乙酯（二元，含最低沸点共沸），数据来自 `docs/binary_vle_english.xlsx`（41 个等压点 @760 mmHg）。

---

## 1. 案例与任务

- **体系**：乙醇 `CCO` / 乙酸乙酯 `CCOC(=O)C`（常压 760 mmHg = 101.325 kPa 等压 T–x–y）
- **目标**：给定液相组成 x_乙醇 + 压力 760 mmHg → 求泡点温度 T 与汽相组成 y_乙醇
- **三源对照**：实验实测（Excel）↔ ThermoFormer 预测 ↔ DWSIM（NRTL）复现

---

## 2. ThermoFormer 预测

### 2.1 方法
用 `lab_models/ThermoFormer/scripts/predict_ea_thermoformer.py`：
- 加载 checkpoint `binary_to_ternary_scale_0.5/seed_2/best_model.pt`
- 分子特征：RDKit 描述子 + Uni-Mol v2(84M) + SMARTS 官能团
- 对每个实验点，用 `solve_isobaric`（给定 P=101.325 kPa + x）预测泡点温度与汽相组成

### 2.2 运行
```powershell
# 在仓库根目录、真实终端（非受限沙箱）运行：
conda run -n thermo python lab_models/ThermoFormer/scripts/predict_ea_thermoformer.py
```
输出写入 `docs/prediction_case1_ethanol_ethylacetate.csv`。

### 2.3 结果与实验对比（节选）

| x_乙醇 | T_实验 (°C) | T_ThermoFormer (°C) | ΔT (K) | y_乙醇(实验) | y_乙醇(TF) |
|---|---|---|---|---|---|
| 1.000 | 78.35 | 219.74 | **+141.4** | 1.000 | 1.000 |
| 0.841 | 74.27 | 223.31 | **+149.0** | 0.735 | 0.958 |
| 0.608 | 72.36 | 227.95 | +155.6 | 0.541 | 0.926 |
| 0.565 | 72.20 | 228.85 | +156.6 | 0.516 | 0.921 |
| 0.420 | 72.06 | 232.68 | +160.6 | 0.434 | 0.902 |
| 0.264 | 72.30 | 240.15 | +167.9 | 0.330 | 0.868 |
| 0.117 | 74.08 | 258.20 | +184.1 | 0.188 | 0.770 |
| 0.000 | 77.11 | 323.72 | **+246.6** | 0.000 | 0.000 |

> **如实标注（重要）**：当前预发布 checkpoint 对乙醇-乙酸乙酯（**醇–酯**体系）的
> **泡点温度系统性偏高约 +140 ~ +247 K**，汽相组成 y_乙醇也整体偏高。
> 这与 `docs/ThermoFormer萃取精馏过程v4.md` 已标注的局限一致：该 checkpoint 的**饱和蒸气压
> 分支校准不足**，且**酯/溶剂类体系**是其最弱端点。**此时 ThermoFormer 的泡点 T/y 不具备定量参考价值**，
> 不应直接用于工程设计。（完整 41 点见 `prediction_case1_ethanol_ethylacetate.csv`。）

---

## 3. DWSIM 复现

### 3.1 生成 DWSIM 文件
脚本：`scripts/generate_dwsim_ea_bubble.py`，复用项目 `thermo_engine.dwsim_export`
（pythonnet + DWSIM Automation3），生成含 Ethanol + Ethyl acetate + NRTL 包 + Flash 的
`.dwxmz` 文件。

```powershell
# 在仓库根目录、真实终端运行（需 DWSIM 已安装、.env 的 DWSIM_HOME 正确、装 pythonnet）：
conda run -n thermo python scripts/generate_dwsim_ea_bubble.py
# 生成 docs/BubbleData_case1_EA_bubble.dwxmz
```

### 3.2 在 DWSIM GUI 里做泡点
1. 双击打开 `docs/BubbleData_case1_EA_bubble.dwxmz`。
2. 设 Feed 物流：液相组成 x_乙醇 = 某实验值（如 0.44）、压力 **760 mmHg**。
3. 跑计算（Calculate）。Flash 出 Vapor/Liquid 两股；对泡点，接近 100% 液相、汽相分率≈0。
4. 读泡点温度与汽相组成：
   - 泡点温度 ≈ Flash 温度；
   - 汽相组成 = Vapor Product 流股组成 y_乙醇。
5. 对不同 x 重复，填入 §4 对比表。

> 说明：FP-flash 是给 T、P 算相平衡；要得泡点，把含 x 的物流压力设为 760 mmHg 且汽相分率→0
> 的温度即泡点温度。也可用 DWSIM 的 Phase Envelope / Bubble & Dew 工具直接生成 T–x–y。

---

## 4. 三源对比表（填 DWSIM 列后完成）

| x_乙醇 | T_实验 | T_ThermoFormer | T_DWSIM | y_乙醇(实验) | y_乙醇(TF) | y_乙醇(DWSIM) |
|---|---|---|---|---|---|---|
| 1.000 | 78.35 | 219.74 | | 1.000 | 1.000 | |
| 0.62（共沸） | ~71.8 | ~227 | | ~0.59 | ~0.93 | |
| 0.44 | ~72.0 | ~232 | | ~0.45 | ~0.90 | |
| 0.20 | ~73.2 | ~245 | | ~0.28 | ~0.85 | |
| 0.00 | 77.11 | 323.72 | | 0.000 | 0.000 | |

**判断标准**（参考 `docs/thermoformer-bubble-validation-plan.zh-CN.md`）：
- 泡点温度 ΔT：DWSIM 与实验应 ≤1–2 K；ThermoFormer 当前远超标（+140~247 K）。
- 汽相组成 Δy：DWSIM 应 ≤0.02–0.05；ThermoFormer 偏高。

---

## 5. 结论

- **实验数据**（`BubbleData_case1_ethanol_ethylacetate_760.csv`）本身物理合理（含共沸、y 归一）。
- **DWSIM（NRTL）** 应能很好地复现实验泡点（机理模型，参数来自数据库），作为可信基准。
- **ThermoFormer 当前 checkpoint** 在乙醇-乙酸乙酯上泡点 T/y 显著失准（饱和蒸气压分支校准不足 +
  酯类体系外推弱），仅作"模型局限"记录，不用于工程设计。

---

## 6. 参考文件

- `docs/BubbleData_case1_ethanol_ethylacetate_760.csv` —— 实验数据（41 行）
- `docs/prediction_case1_ethanol_ethylacetate.csv` —— ThermoFormer 预测（41 行）
- `docs/BubbleData_case1_EA_bubble.dwxmz` —— DWSIM 生成的 flowsheet（跑脚本后产生）
- `lab_models/ThermoFormer/scripts/predict_ea_thermoformer.py` —— ThermoFormer 预测脚本
- `scripts/generate_dwsim_ea_bubble.py` —— DWSIM 文件生成脚本
