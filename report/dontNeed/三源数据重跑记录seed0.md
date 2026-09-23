# 三源数据重跑记录（ThermoFormer seed_0 权值 + DWSIM 9.x 实算）

> 本文档记录 2026-09-16 对 `report/` 下三个体系三源数据的**重新预测与重新实算**。
> 三源口径不变：**实验数据 / ThermoFormer 预测 / DWSIM 独立流程模拟**。
>
> 本文档为**新增**文档，与 `Agent整合ThermoFormer进度与三源验证报告v5.md` 并存：
> v5 已按本次结果刷新第二部分的 ThermoFormer 列与 §1.5 塔设计，
> 本文档记录完整重跑过程、双源塔设计刷新、以及关键差异与根因排查。

---

## 0. 权值与运行口径

| 项 | 取值 |
|---|---|
| ThermoFormer 权值 | `THERMOFORMER_CHECKPOINT` 留空 → 走 `_select_checkpoint` 从 `models/registry.json` 取 `(task, protocol, seed=0, status=provided)` |
| 体系① 二元 VLE | `models/vle/prediction/vle_overall_binary/seed_0/best_model.pt` |
| 体系② 三元 VLE | `models/vle/prediction/vle_overall_ternary/seed_0/best_model.pt` |
| 体系③ 二元 LLE | `models/lle/prediction/binary-system/seed_0/best.pt` |
| 预测入口 | `ThermoFormerBackend.bubble_point`（等压，760 mmHg）/ `ThermoFormerBackend.lle` |
| 计算设备 | CPU（本机 `thermo` 环境 torch 为 CPU-only 构建，`THERMOFORMER_USE_CUDA=0`） |
| DWSIM | DWSIM 9.x，`C:\Users\34861\AppData\Local\DWSIM`，经 Automation API 实算 |
| 实验数据 | `datasets/lle/binary_lle.csv`（LLE）；`docs/prediction_*.csv`（VLE） |

**数值来源约束**：所有平衡数值均来自 `thermo_engine` 后端或 DWSIM 求解器，脚本不夹带手算值。

---

## 1. 本次新增/修改的脚本

| 脚本 | 作用 |
|---|---|
| `scripts/rerun_three_source_tf.py` | 三体系 ThermoFormer 列重新预测（走 `ThermoFormerBackend`，seed_0） |
| `scripts/consolidate_three_source.py` | 汇总三源，输出 `report/dwsim/three_source_*.csv` 与 `three_source_summary.json` |
| `scripts/regenerate_column_design_dual_source.py` | 塔设计双源对照（UNIFAC / ThermoFormer），输出 `column_design_dual_source.csv/.json` |
| `scripts/export_dual_source_columns.py` | 由**真实**双源设计导出 4 个塔 `.dwxmz`（见 §7） |

DWSIM 列复用既有实算入口：
`scripts/generate_ternary_dwsim_demonstration.py`（体系①②）、`scripts/run_water_butanol_dwsim.py`（体系③），
本次均在同一环境下**重新实际运行**并重写 `.dwxmz` 与 DWSIM CSV。

---

## 2. 环境修复：`unimol_tools` 返回值兼容性

重跑首次失败于特征编码：

```text
TypeError: Uni-Mol v2 get_repr must return a dictionary containing 'cls_repr'
```

**原因**：本机安装的 `unimol_tools` 版本中 `UniMolRepr.get_repr()` 返回**每分子 `cls_repr` 向量组成的 list**，
而 `src/thermoformer/features/unimol_v2.py` 原先只接受 `{"cls_repr": ndarray}` 字典形式——属跨版本 API 差异，
并非 SMILES 或权重问题。

**修复**：`UniMolV2Encoder.encode` 同时接受 dict（含 `cls_repr`）与 list/tuple 两种返回形式。

**数值等价性验证**：修复后重算 `CC(C)O` / `O` / `CCOC(=O)C` 三个分子的 Uni-Mol 特征，
与既有缓存 `unimolv2_84m.npz` 逐元素比较，`max|diff| = 0.000e+00`（位对位一致）。
另核对编码输出维度为 **820**（= 24 RDKit + 768 Uni-Mol + 28 官能团），与 checkpoint 的
`feature_dim = 820` 配置完全一致，且全部数值有限。

---

## 3. 体系①：2-丙醇 / 水（二元 VLE，760 mmHg）

| x_IPA | T_实验 (°C) | T_TF (本次 seed_0) | T_TF (v5 旧值) | T_DWSIM (°C) | y_实验 | y_TF (本次) | y_TF (v5 旧值) | y_DWSIM |
|:--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 84.75 | **114.81** | 90.21 | 89.50 | 0.495 | **0.3480** | 0.360 | 0.3778 |
| 0.3 | 81.85 | **111.11** | 83.77 | 83.68 | 0.541 | **0.4501** | 0.570 | 0.5645 |
| 0.5 | 80.15 | **109.49** | 81.81 | 81.92 | 0.605 | **0.5504** | 0.657 | 0.6437 |
| 0.7 | 80.02 | **109.44** | 81.14 | 81.12 | 0.687 | **0.6657** | 0.727 | 0.7301 |
| 0.9 | 80.87 | **111.71** | 81.49 | 81.42 | 0.872 | **0.8520** | 0.876 | 0.8751 |

- DWSIM 列（本次实算）：89.50 / 83.68 / 81.92 / 81.12 / 81.42 °C，y = 0.3778 / 0.5645 / 0.6437 / 0.7301 / 0.8751，
  与旧表**完全一致**，DWSIM 侧可复现。
- 实验列与旧表一致。
- **ThermoFormer 列与旧值不一致**，根因见 §6。

---

## 4. 体系②：乙酸乙酯 / 乙酸正丙酯 + DMSO（三元 VLE，760 mmHg）

**泡点温度（°C）**

| x_乙酸乙酯 | x_乙酸正丙酯 | x_DMSO | T_实验 | T_TF (本次 seed_0) | T_TF (v5 旧值) | T_DWSIM |
|:--:|:--:|:--:|---:|---:|---:|---:|
| 0.3735 | 0.0244 | 0.6021 | 101.68 | **103.56** | 84.91 | 96.11 |
| 0.2195 | 0.1790 | 0.6015 | 107.05 | **102.30** | 88.41 | 103.59 |
| 0.0595 | 0.3336 | 0.6069 | 114.33 | **99.97** | 100.14 | 113.09 |
| 0.3540 | 0.1384 | 0.5076 | 103.50 | **104.08** | 83.87 | 96.85 |
| 0.1819 | 0.3137 | 0.5044 | 111.58 | **102.28** | 89.41 | 104.41 |
| 0.0323 | 0.4649 | 0.5028 | 113.99 | **100.05** | 102.21 | 112.19 |

**汽相组成 y（乙酸乙酯 / 乙酸正丙酯 / DMSO）**

| x_乙酸乙酯 | y_实验 | y_TF (本次 seed_0) | y_DWSIM |
|:--:|:--|:--|:--|
| 0.3735 | 0.914 / 0.056 / 0.030 | **0.3169 / 0.0215 / 0.6616** | 0.9358 / 0.0325 / 0.0317 |
| 0.2195 | 0.558 / 0.402 / 0.040 | **0.1643 / 0.1665 / 0.6692** | 0.6637 / 0.2922 / 0.0441 |
| 0.0595 | 0.166 / 0.777 / 0.057 | **0.0375 / 0.3146 / 0.6479** | 0.2284 / 0.7058 / 0.0658 |
| 0.3540 | 0.771 / 0.197 / 0.032 | **0.2686 / 0.1146 / 0.6168** | 0.8053 / 0.1642 / 0.0304 |
| 0.1819 | 0.390 / 0.562 / 0.049 | **0.1174 / 0.2704 / 0.6122** | 0.5004 / 0.4571 / 0.0425 |
| 0.0323 | 0.061 / 0.884 / 0.055 | **0.0174 / 0.3990 / 0.5836** | 0.1076 / 0.8336 / 0.0588 |

- DWSIM 列（本次实算）与旧表**完全一致**（T: 96.11 / 103.59 / 113.09 / 96.85 / 104.41 / 112.19 °C，y 亦一致）。
- 本次 TF 预测的关键特征是 **DMSO 在汽相中占主导（y_DMSO ≈ 0.58–0.67）**，而实验与 DWSIM 中
  DMSO 均为微量（0.03–0.07）。

---

## 5. 体系③：水 / 正丁醇（二元 LLE，101.325 kPa）

相位命名：`organic = 正丁醇富集相`，`aqueous = 水富集相`。

**表 5-1　正丁醇富集相（organic）中正丁醇摩尔分数**

| T (K) | 实验 | ThermoFormer (本次 seed_0) | DWSIM |
|:--:|---:|---:|---:|
| 298.15 | 0.4869 | **0.5090** | 0.3992 |
| 313.15 | 0.4840 | **0.4833** | 0.4070 |
| 343.15 | 0.4190 | **0.4330** | 0.4138 |
| 353.15 | 0.3965 | **0.4166** | 0.4137 |

**表 5-2　水富集相（aqueous）中正丁醇摩尔分数**

| T (K) | 实验 | ThermoFormer (本次 seed_0) | DWSIM |
|:--:|---:|---:|---:|
| 298.15 | 0.0188 | **0.0388** | 0.0055 |
| 313.15 | 0.0190 | **0.0423** | 0.0074 |
| 343.15 | 0.0160 | **0.0502** | 0.0125 |
| 353.15 | 0.0181 | **0.0531** | 0.0147 |

- 实验为各温度重复 tie-line 的均值，条数 298.15 K：3、313.15 K：4、343.15 K：2、353.15 K：1
  （由 `datasets/lle/binary_lle.csv` 现场聚合复算）。
- **体系③ 的 ThermoFormer 列与 DWSIM 列均与旧表完全一致**，两相残差 ~1e-16（两相等活度 RMS）。
  这说明本次链路对该体系是**逐位可复现**的。

---

## 6. 关键差异说明（务必随表引用）

### 6.1 体系①② 的 ThermoFormer 列与旧值不一致

本次以 registry 默认 **seed_0** 权值实跑，得到与旧表不同的 ThermoFormer 数值。
旧表中的 ThermoFormer 列（体系①：90.21 / 83.77 / 81.81 / 81.14 / 81.49 °C；
体系②：84.91 / 88.41 / 100.14 / 83.87 / 89.41 / 102.21 °C）对应
`docs/prediction_ipa_water.csv` 与 `docs/prediction_ternary_dms.csv` 里的 `T_tf_C` 列。

**这两列的生成脚本在当前仓库中不存在**（全仓检索 `prediction_ipa_water` 仅命中
`generate_ternary_dwsim_demonstration.py`，而该脚本只是**读取**该 CSV、并不产生 `T_tf_C`）。
即旧 ThermoFormer 数值来自**仓库外的外部进程**，当前代码 + 当前权值无法重现。

### 6.2 为什么本次 seed_0 数值偏差是「模型行为」而非「代码错误」

已逐项排除本次实现的正确性问题：

1. **特征维度匹配**：编码输出 820 维，与 checkpoint `feature_dim = 820` 一致，数值全部有限。
2. **权重选择正确**：`_select_checkpoint` 解析到 `vle_overall_binary/seed_0/best_model.pt`，
   路径与 registry 的 `seed=0, status=provided` 条目一致。
3. **求解器真收敛**：等压泡点走 `solve_isobaric` 括区间 Newton 法，残差 6e-06 ~ 3e-04 kPa，
   低于容差（1e-3 + 1e-5·P ≈ 1.42e-3 kPa），`converged=True`。
4. **特征数值位对位**：Uni-Mol 分支修复后与既有缓存 `max|diff| = 0`。
5. **链路自证**：同一套代码对**体系③ 完全复现**，说明管线本身没有问题。

进一步做纯组分外推检验（seed_0，760 mmHg），可见该权值对本体系的**纯组分饱和压力**即存在系统性偏差：

| 纯组分 | 本次预测沸点 | 真实常压沸点 | 偏差 |
|---|---:|---:|---:|
| 2-丙醇 (x_IPA=1) | 114.22 °C | 82.6 °C | **+31.6 °C** |
| 水 (x_IPA≈0) | 130.37 °C | 100.0 °C | **+30.4 °C** |

体系② 的 TF 汽相中 DMSO 占主导，同样属模型对该三元体系的预测行为。

**结论**：seed_0 权值对本报告三个体系中的 VLE 体系（①②）预测精度较差，属**权重/训练覆盖问题**，
不是本次重跑的实现缺陷。体系③（LLE，`binary-system` 权值）预测良好且完全可复现。

### 6.3 与报告末尾「权值选择」的关系

v5 正文 §1.3 与补充部分给出的权值为 **seed_0**；而
`report/三个体系三元三源数据总表.md` 指出原报告 §2 历史上曾手选 **seed_4**
（`checkpoints/overall_binary/seed_4/best_model.pt` 等），两者不是同一套权重。
本次按用户指定，**统一采用 registry 默认 seed_0**。

供参考，本次另测得 seed_4 的二元 VLE 结果（未写入正式表）：
x_IPA=0.3 → 98.18 °C（y=0.263）；x_IPA=0.5 → 99.48 °C（y=0.341）。仍显著高于实验 81.85 / 80.15 °C。

---

## 7. 塔设计（报告 §1.5）双源重新生成

报告 §1.5 的两个塔设计已按「UNIFAC / ThermoFormer 双源对照」重新生成。

**入口**：`scripts/regenerate_column_design_dual_source.py`（设计量）、
`scripts/export_dual_source_columns.py`（DWSIM `.dwxmz` 导出）。
数值全部来自 `thermo_engine.column_design`，ThermoFormer 源走 registry 默认 `seed_0`。

### 7.1 案例一：2-丙醇 / 水 普通精馏（x_IPA = 0.3 / 0.5）

| 项目 | x=0.3 UNIFAC | x=0.3 TF | x=0.5 UNIFAC | x=0.5 TF |
|---|---:|---:|---:|---:|
| alpha | 2.713 | 1.910 | 1.462 | 1.224 |
| N | 19 | 28 | 42 | 77 |
| N_min | 10.069 | 15.535 | 24.213 | 45.453 |
| R | 2.694 | 5.083 | 5.983 | 12.341 |
| R_min | 1.924 | 3.631 | 4.273 | 8.815 |
| 进料板 | 9 | 13 | 19 | 35 |
| 塔顶 / 塔釜 (K) | 355.26 / 367.46 | 387.22 / 400.22 | 355.26 / 362.99 | 387.22 / 396.95 |

### 7.2 案例二：乙酯 / 丙酯 + DMSO 萃取精馏

| 项目 | UNIFAC | ThermoFormer |
|---|---:|---:|
| alpha_base / alpha_ext | 2.187 / 1.761 | 1.345 / 0.822 |
| 选择性 | 0.805 | 0.612 |
| alpha_avg | 1.963 | 1.052 |
| N / N_min | 20 / 10.154 | 223 / 136.345 |
| R / R_min | 2.478 / 1.770 | 48.792 / 34.852 |
| 进料板 / 萃取剂板 | 9 / 2 | 100 / 2 |
| 塔顶 / 塔釜 (K) | 351.08 / 396.50 | 361.62 / 374.12 |
| 进料泡点 (K) | 360.13 | 366.49 |

**UNIFAC 两列与既有报告逐位一致**（案例一 x=0.5 的 alpha=1.462/N=42/R=5.983/T=355.26/362.99；
案例二的 2.187/1.761/0.805/1.963/N=20/N_min=10.154/R=2.478/351.08/396.50 全部吻合），
说明短节设计链路本身可复现。

**ThermoFormer 源的设计不具备工程可用性**：案例一 alpha 趋近 1，导致塔板数与回流比接近翻倍；
案例二 `alpha_ext = 0.822 < 1`（选择性 0.612 < 1），即该权值认为 DMSO 不但没有提高、
反而降低了乙酯/丙酯对的相对挥发度，短节设计退化为 N = 223 块板。
这与 §3/§4 中该权值对这两个 VLE 体系的预测偏差是同一根源。

### 7.3 关键发现：旧「TF 塔」文件的设计量是硬编码字面量

`scripts/generate_ipa_water_std_column_tf.py` 与 `scripts/generate_eac_npac_dmso_extractive_tf.py`
（分别生成 `ipa_water_binary_column_x0p3_tf.dwxmz` 与 `eac_npac_dmso_extractive_tf_column.dwxmz`）
中的 ThermoFormer 设计量**全部是源码里的硬编码常量**：

```python
# generate_ipa_water_std_column_tf.py
TF_STAGES = 17; TF_MIN_STAGES = 8.795; TF_REFLUX = 2.16
TF_MIN_REFLUX = 1.543; TF_FEED_STAGE = 8; TF_ALPHA = 3.135
# generate_eac_npac_dmso_extractive_tf.py
TF_STAGES = 18; TF_REFLUX = 2.179; TF_MIN_STAGES = 9.310
TF_ALPHA_BASE = 2.240; TF_ALPHA_EXT = 1.943; TF_SELECTIVITY = 0.931
```

这两个脚本**从未调用 `design_binary_distillation_column` / `design_ternary_extractive_column`**
（只调用 `bubble_temperature(..., "unifac")` 取进料泡点）。即 v4 §1.6 表中的 "TF 列"
（案例一 alpha=3.135 / N=17；案例二 2.240 / 1.943 / 0.931 / N=18）**并非 ThermoFormer 计算所得**，
而是人工预置的字面量，其数值在仓库中无任何可追溯的计算来源。

本次以真实 `alpha_source="thermoformer"` 重算，结果见 §7.1 / §7.2，与旧字面量差异显著。

**结论**：报告中「TF 源塔设计」应以本次双源重新生成的结果为准；旧 `*_tf*.dwxmz` 文件保留未删除，
但**不可作为 ThermoFormer 设计证据引用**。

### 7.4 双源塔文件

| 案例 | UNIFAC 源 | ThermoFormer 源 |
|---|---|---|
| 2-丙醇 / 水 (x=0.3) | `data/exports/flow_examples/ipa_water_binary_column_x0p3_unifac.dwxmz` | `..._x0p3_thermoformer.dwxmz` |
| 乙酯 / 丙酯 + DMSO | `data/exports/flow_examples/eac_npac_dmso_extractive_unifac.dwxmz` | `..._extractive_thermoformer.dwxmz` |

---

## 8. 产物清单（本次生成/更新）

| 文件 | 说明 |
|---|---|
| `report/dwsim/three_source_system1_ipa_vle.csv` | 体系① 三源总表（含本次 TF 与旧外部 TF 两列对照） |
| `report/dwsim/three_source_system2_dmso_vle.csv` | 体系② 三源总表（同上） |
| `report/dwsim/three_source_system3_water_butanol_lle.csv` | 体系③ 三源总表 |
| `report/dwsim/water_butanol_three_source.csv` | 体系③ 三源总表（沿用既有文件名，已按本次实算刷新） |
| `report/dwsim/three_source_summary.json` | 三体系合并 JSON 汇总 |
| `report/dwsim/tf_rerun_ipa_vle.csv` | 体系① TF 重跑原始输出（含残差） |
| `report/dwsim/tf_rerun_dmso_vle.csv` | 体系② TF 重跑原始输出（含残差、y 求和校验） |
| `report/dwsim/tf_rerun_water_butanol_lle.csv` | 体系③ TF 重跑原始输出（含残差） |
| `report/dwsim/water_butanol_dwsim.csv` | 体系③ DWSIM 实算原始输出（本次重写） |
| `docs/dwsim_ipa_bubble.csv` | 体系① DWSIM 实算（本次重写） |
| `docs/dwsim_dmso_bubble.csv` | 体系② DWSIM 实算（本次重写） |
| `report/dwsim/*.dwxmz` | 11 个泡点流程（5 二元 + 6 三元）+ 4 个 LLE 流程，均由本次实算重写 |
| `report/dwsim/column_design_dual_source.csv` | 塔设计双源对照总表（案例一 2 个 x × 案例二，逐量一行） |
| `report/dwsim/column_design_dual_source.json` | 同上，JSON 结构 |
| `data/exports/flow_examples/*_unifac.dwxmz`（2 个） | UNIFAC 源塔文件（本次生成） |
| `data/exports/flow_examples/*_thermoformer.dwxmz`（2 个） | **真实 ThermoFormer 源**塔文件（本次生成） |

---

## 9. 复现命令

```powershell
conda activate thermo
cd E:\codex\ThermoAgent\ThermoFormer\ThermoAgent

# 三体系 ThermoFormer 列（seed_0 走 registry）——纯 Python，无 DWSIM
python scripts\rerun_three_source_tf.py all

# DWSIM 列：体系①②（泡点 + .dwxmz + 补全 CSV）
#   注意：pythonnet 需完整进程权限，须在真实终端运行（受限沙盒下报
#   "Failed to initialize Python.Runtime.dll" / 拒绝访问）
python scripts\generate_ternary_dwsim_demonstration.py --outdir report\dwsim --write-files

# DWSIM 列：体系③（四个温度的 LLE 流程）
python scripts\run_water_butanol_dwsim.py

# 汇总三源总表
python scripts\consolidate_three_source.py

# 塔设计双源对照（UNIFAC vs ThermoFormer）——纯 Python，无 DWSIM
python scripts\regenerate_column_design_dual_source.py

# 双源塔 .dwxmz 导出（同样需要完整进程权限才能走通 DWSIM 导出）
python scripts\export_dual_source_columns.py

# 附：既有 UNIFAC 源塔导出
python scripts\export_flow_examples.py --outdir data\exports\flow_examples
```
