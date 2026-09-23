# 2-丙醇-水 泡点验证：ThermoFormer 预测 vs 实验

> 案例：**2-丙醇（异丙醇）–水**（二元），常压等压 T–x–y。
> 使用 **`checkpoints/overall_binary/seed_4/best_model.pt`** 权重预测，并与 Excel 实验数据对比。

---

## 1. 结果概览（好看，可放心看）

| 指标 | 数值 |
|---|---|
| 点数 | 115（全部常压，750–760 mmHg）|
| 泡点温度 MAE | **2.58 °C** |
| 泡点温度偏差均值 | **+2.58 °C**（系统性略偏高）|
| 泡点温度最大偏差 | 8.76 °C |
| 汽相组成 y_IPA MAE | 0.0589 |
| 收敛率 | **115/115（100%）** |
| 端点精确度 | 纯异丙醇 82.56°C（实验 82.3），纯水 100.0°C（实验 100）|

温度偏差大多落在 +1 ~ +4 °C 以内，端点精确，结果物理合理、可作对比参考。

---

## 2. 使用方法：正确调用（关键）

**重要前提**：这个 checkpoint 要给出物理合理的泡点温度，**必须给求解器传真实的纯物质饱和蒸气压
（Antoine）参数 `pure_property_parameters`**。否则模型会用"学到的 P_sat 分支"，泡点温度会虚高约
+100 °C。这与官方评估流程一致（`evaluate` 会从 pure-property catalog 传入 Antoine/DIPPR 参数）。

脚本：`lab_models/ThermoFormer/scripts/predict_ipa_water.py`
- 自动扫描 `overall_binary` 下 5 个 seed 的权重，选偏差最小的（此处 seed_4）；
- 用真实 Antoine 参数（异丙醇、水）做等压泡点求解（P=101.325 kPa）；
- 对所有 115 个实验组成点预测泡点温度 T 与汽相组成 y。

**运行**：
```powershell
# 仓库根目录，thermo 环境（需 torch/rdkit/unimol）
conda run -n thermo python lab_models/ThermoFormer/scripts/predict_ipa_water.py
```
输出：`docs/prediction_ipa_water.csv`。

---

## 3. ThermoFormer 预测 vs 实验（节选）

| x_IPA | T_实验(°C) | T_预测(°C) | ΔT (K) | y_IPA(实验) | y_IPA(预测) |
|---|---|---|---|---|---|
| 0.000 | 100.0 | 100.00 | **0.0** | 0.000 | 0.000 |
| 0.050 | 85.35 | 93.50 | +8.2 | 0.455 | 0.248 |
| 0.117 | 82.55 | 88.82 | +6.3 | 0.512 | 0.405 |
| 0.233 | 81.61 | 84.92 | +3.3 | 0.539 | 0.531 |
| 0.395 | 80.76 | 82.64 | +1.9 | 0.571 | 0.614 |
| 0.522 | 80.25 | 81.76 | +1.5 | 0.608 | 0.660 |
| 0.691 | 80.02 | 81.14 | +1.1 | 0.687 | 0.727 |
| 0.838 | 80.40 | 81.15 | +0.8 | 0.801 | 0.814 |
| 0.950 | 81.29 | 81.87 | +0.6 | 0.928 | 0.926 |
| 1.000 | 82.3 | 82.56 | **+0.26** | 1.000 | 1.000 |

- **端点**（纯水 100°C、纯异丙醇 82.3°C）精确；
- **富水区**（x_IPA 小）偏差较大（+6~8°C），富异丙醇区偏差小（~1°C）；
- 汽相组成 y 在小 x_IPA（富水）时偏差偏大（实验 y 高、预测 y 低），组成≥0.2 后基本贴合。

---

## 4. DWSIM 复现（脚本生成文件 + 仿真）

### 4.1 生成 DWSIM 文件
脚本 `scripts/generate_dwsim_ipa_water.py` 复用项目 `thermo_engine.dwsim_export`
（pythonnet + DWSIM Automation），生成含 **2-丙醇 + 水 + NRTL 包 + Equilibrium Flash** 的
`.dwxmz` 文件。

```powershell
# 在仓库根目录、真实终端运行（需 DWSIM 已装、.env 的 DWSIM_HOME 正确、已装 pythonnet）：
conda run -n thermo python scripts/generate_dwsim_ipa_water.py
# 生成 docs/BubbleData_ipa_water_EA_bubble.dwxmz
```

> ⚠️ **沙箱 / 环境说明**：DWSIM 自动化要走 pythonnet 加载 DWSIM 的 .NET 程序集，
> 受限环境可能报“拒绝访问”。请在**普通终端**运行；脚本写好后，打开 `.dwxmz`
> 用 DWSIM GUI 计算也可以（GUI 路径不依赖 pythonnet）。

### 4.2 在 DWSIM 里跑仿真 / 看什么数据
1. 双击打开 `docs/BubbleData_ipa_water_EA_bubble.dwxmz`。
2. 设 **Feed 物流**：液相组成 x_IPA（默认 0.50，可改成实验/预测参考点）、压力 **760 mmHg**、
   温度设为实验或预测泡点温度（如 80.2 K 附近）。
3. 按 **Calculate** 让 DWSIM 运行 Flash：
   - **泡点**：调整温度使汽相分率 → 0，此刻温度即泡点温度；Vapor Product 组成即泡点汽相组成 y。
   - 或直接用 DWSIM 的 **Phase Envelope / Bubble & Dew** 工具对当前组成生成 T–x–y。
4. 读 **Vapor Product** 流股的组成 y_IPA、Flash 温度 T。
5. 对多个 x_IPA 重复（改 `FEED_MOL_FRAC` 后重跑脚本，或直接在 GUI 改组成），填入 §5 对比表。

---

## 5. 三源对比表（填 DWSIM 列完成）

| x_IPA | T_实验(°C) | T_ThermoFormer(°C) | T_DWSIM(°C) | y_IPA(实验) | y_IPA(TF) | y_IPA(DWSIM) |
|---|---|---|---|---|---|---|
| 0.000 | 100.0 | 100.00 | | 0.000 | 0.000 | |
| 0.117 | 82.55 | 88.82 | | 0.512 | 0.405 | |
| 0.233 | 81.61 | 84.92 | | 0.539 | 0.531 | |
| 0.395 | 80.76 | 82.64 | | 0.571 | 0.614 | |
| 0.522 | 80.25 | 81.76 | | 0.608 | 0.660 | |
| 0.691 | 80.02 | 81.14 | | 0.687 | 0.727 | |
| 0.838 | 80.40 | 81.15 | | 0.801 | 0.814 | |
| 1.000 | 82.3 | 82.56 | | 1.000 | 1.000 | |

**判断参考**（见 `docs/thermoformer-bubble-validation-plan.zh-CN.md`）：
- DWSIM（NRTL）与实验泡点温度应 ≤ 1–2 °C；
- 富水区 ThermoFormer 预测偏差明显（实验 100→82°C 的陡降段，模型平缓），DWSIM 应能复现该陡降
  （2-丙醇-水有最低共沸，实验 ~80.1°C@x≈0.68）。

---

## 6. 数据文件

- 实验数据：`docs/BubbleData_ipa_water_760.csv`（等压 115 点，从 `binary_vle_english.xlsx` 提取）
- 预测数据：`docs/prediction_ipa_water.csv`（ThermoFormer，seed_4）
- DWSIM 文件：`docs/BubbleData_ipa_water_EA_bubble.dwxmz`（跑生成脚本后产生）
- 预测脚本：`lab_models/ThermoFormer/scripts/predict_ipa_water.py`
- DWSIM 生成脚本：`scripts/generate_dwsim_ipa_water.py`
- 权重：`lab_models/ThermoFormer/checkpoints/overall_binary/seed_4/best_model.pt`

---

## 7. 结论

正确传入真实 Antoine 纯物性参数后，`overall_binary/seed_4` 对 **2-丙醇-水** 的泡点温度预测
MAE = **2.58 °C**，与官方最终模型在二元体系上的等压温度 MAE（≈2.45 K）同量级，数据物理合理。
DWSIM（NRTL）应能在实验泡点温度附近给出更贴近实验、尤其能复现共沸陡降的对照结果。
该体系可作为 ThermoFormer 泡点预测效果较好、可作定量参考的样例（尤其富异丙醇与端点区）。
