# 三元数据三源论证 DWSIM 处理与 .dwxmz 文件生成记录

> 本文档记录三个三元/二元体系的 **三源论证（实验 / ThermoFormer / DWSIM）** 的 DWSIM 处理结果，
> 以及为每个体系生成的 `.dwxmz` 工程文件。所有 DWSIM 数值均由本机 DWSIM（9.x,
> `C:\Users\34861\AppData\Local\DWSIM`）经 Automation API 实际计算得到，脚本不夹带任何手算平衡数。
>
> 生成日期：2026-09-15

---

## 0. 三个体系总览

| # | 体系 | 类型 | 物性包 | DWSIM 处理方式 | 对应报告章节 |
|---|---|---|---|---|---|
| ① | 2-丙醇 / 水 | 二元 VLE（等压泡点） | NRTL | TP-flash 二分法求泡点温度 | §2.1 |
| ② | 乙酸乙酯 / 乙酸正丙酯 + DMSO | 三元 VLE（萃取精馏溶剂） | NRTL | TP-flash 二分法求泡点温度 | §2.2 |
| ③ | 乙醇 / 乙酸乙酯 / 水 | 三元 LLE（液液萃取） | NRTL + 显式 BIP | 严格液液萃取塔（AbsorptionColumn=Extractor） | dwsim_export.md §10 |

每个体系均产出一套 `.dwxmz` 文件与补全 DWSIM 数据列的对比 CSV。产物统一归档在：

```text
lunwen/dwsim_demonstration/
```

---

## 1. 体系①：2-丙醇 / 水（二元 VLE 等压泡点）

- 输入：`docs/prediction_ipa_water.csv`（实验 + ThermoFormer 已备）
- 取样点（液体摩尔分数 x_IPA）：0.1 / 0.3 / 0.5 / 0.7 / 0.9
- 方法：对每个 x，在 760 mmHg 下二分 Vapor 摩尔流量，取「气相流量首次 > 0」的温度为泡点温度，并读取该温度下 Vapor 产物的汽相组成。
- 物性包：NRTL（Isopropanol/Water，DWSIM 内置 BIP）

**三源泡点温度与汽相组成（°C，y = y_2-丙醇）**

| x_IPA | T_实验 | T_TF | T_DWSIM | y_实验 | y_TF | y_DWSIM | .dwxmz |
|---|:--|:--|:--|:--|:--|:--|---|
| 0.1 | 84.75 | 90.21 | 89.50 | 0.495 | 0.360 | 0.378 | `ipa_x0.1_2comp_bubble_89.5C.dwxmz` |
| 0.3 | 81.85 | 83.77 | 83.68 | 0.541 | 0.570 | 0.565 | `ipa_x0.3_2comp_bubble_83.7C.dwxmz` |
| 0.5 | 80.15 | 81.81 | 81.92 | 0.605 | 0.657 | 0.644 | `ipa_x0.5_2comp_bubble_81.9C.dwxmz` |
| 0.7 | 80.02 | 81.14 | 81.12 | 0.687 | 0.727 | 0.730 | `ipa_x0.7_2comp_bubble_81.1C.dwxmz` |
| 0.9 | 80.87 | 81.49 | 81.42 | 0.872 | 0.876 | 0.875 | `ipa_x0.9_2comp_bubble_81.4C.dwxmz` |

- 补全产物：`docs/dwsim_ipa_bubble.csv`（含 `T_dwsim_C`、`y_ipa_dwsim`、`file` 列）。

---

## 2. 体系②：乙酸乙酯 / 乙酸正丙酯 + DMSO（三元 VLE 等压泡点）

- 输入：`docs/prediction_ternary_dms.csv`（实验 + ThermoFormer 已备，36 点）
- 取样点：6 个代表点（与报告 §2.2 表 2.2-1/2.2-2 对齐）
- 方法：同体系①，三分量 TP-flash 二分求泡点。
- 物性包：NRTL（Ethyl acetate / N-propyl acetate / Dimethyl sulfoxide，DWSIM 内置 BIP）

**三源泡点温度（°C）**

| x_乙酸乙酯 | x_乙酸正丙酯 | x_DMSO | T_实验 | T_TF | T_DWSIM | .dwxmz |
|:--|:--|:--|:--|:--|:--|---|
| 0.3735 | 0.0244 | 0.6021 | 101.68 | 84.91 | 96.11 | `dmso_etac0p373_3comp_bubble_96.1C.dwxmz` |
| 0.2195 | 0.1790 | 0.6015 | 107.05 | 88.41 | 103.59 | `dmso_etac0p220_3comp_bubble_103.6C.dwxmz` |
| 0.0595 | 0.3336 | 0.6069 | 114.33 | 100.14 | 113.09 | `dmso_etac0p059_3comp_bubble_113.1C.dwxmz` |
| 0.3540 | 0.1384 | 0.5076 | 103.50 | 83.87 | 96.85 | `dmso_etac0p354_3comp_bubble_96.9C.dwxmz` |
| 0.1819 | 0.3137 | 0.5044 | 111.58 | 89.41 | 104.41 | `dmso_etac0p182_3comp_bubble_104.4C.dwxmz` |
| 0.0323 | 0.4649 | 0.5028 | 113.99 | 102.21 | 112.19 | `dmso_etac0p032_3comp_bubble_112.2C.dwxmz` |

**三源汽相组成 y（乙酸乙酯 / 乙酸正丙酯 / DMSO）**

| x_乙酸乙酯 | y_实验 | y_TF | y_DWSIM |
|:--|:--|:--|:--|
| 0.3735 | 0.914 / 0.056 / 0.030 | 0.939 / 0.043 / 0.018 | 0.936 / 0.033 / 0.032 |
| 0.2195 | 0.558 / 0.402 / 0.040 | 0.691 / 0.288 / 0.021 | 0.664 / 0.292 / 0.044 |
| 0.0595 | 0.166 / 0.777 / 0.057 | 0.302 / 0.662 / 0.036 | 0.228 / 0.706 / 0.066 |
| 0.3540 | 0.771 / 0.197 / 0.032 | 0.796 / 0.188 / 0.016 | 0.805 / 0.164 / 0.030 |
| 0.1819 | 0.390 / 0.562 / 0.049 | 0.566 / 0.413 / 0.021 | 0.500 / 0.457 / 0.043 |
| 0.0323 | 0.061 / 0.884 / 0.055 | 0.167 / 0.795 / 0.038 | 0.108 / 0.834 / 0.059 |

- 补全产物：`docs/dwsim_dmso_bubble.csv`（含 `T_dwsim_C`、`y_*_dwsim`、`file` 列）。

> 口径说明（与报告一致）：ThermoFormer 在该体系泡点温度系统性偏低（MAE ≈ 18 °C，预测低约 18 °C）；
> DWSIM 泡点温度低于实验约 1.2–7.2 °C，汽相组成与实验/TF 趋势一致。

---

## 3. 体系③：乙醇 / 乙酸乙酯 / 水（三元 LLE 液液萃取）

- 类型：严格多级逆流液液萃取塔（DWSIM `AbsorptionColumn` + `OperationMode = Extractor`），
  非 `Vessel` / `Splitter` / `ComponentSeparator` 伪装。
- 物性包：NRTL，并**显式写入三对乙醇/乙酸乙酯/水经典 NRTL BIP**（A_ij 按 cal 尺度 = K 值 ÷ 4.184）。
- 闪蒸算法：`Nested Loops (VLLE)` / Gibbs 最小化，三液相稳定性测试 severity 3。
- 塔参数：8 级，101325 Pa，压降 1000 Pa；求解器 `Burningham-Otto (Sum Rates)`，初值 `Internal 2 (Experimental)`。
- 流程：`Feed（乙醇+乙酸乙酯，底部进） + Solvent（纯水，顶部进） → Raffinate（塔顶萃余相）+ Extract（塔底萃取相）`。

**写入的 NRTL BIP（K / K / α）**

| 组分 i | 组分 j | A_ij (K) | A_ji (K) | α_ij |
|---|---|---|:--:|:--:|
| Ethyl acetate | Ethanol | 1278.65 | 1382.86 | 0.2988 |
| Ethyl acetate | Water | 5380.57 | 6719.58 | 0.4393 |
| Ethanol | Water | -242.505 | 5195.44 | 0.2937 |

- 产物：`lunwen/dwsim_demonstration/ethanol_eac_water_lle_extractor_nrtl.dwxmz`（21,734 字节）。

**求解状态（如实记录）**：结构、BIP、VLLE 闪蒸、求解器、两相初始估计均已写入并可加载；
`--calc` 严格塔求解在初值内反馈 `PT Flash: Invalid solution`（Gibbs 最小化 Phase split 未收敛），
为 `lunwen/dwsim_export.md` §9.2/§9.3 已文档化的严格 LLE 塔典型难点（常压下强极性 LLE 体系
`Burningham-Otto` 冷启动）。文件可在 DWSIM GUI 打开，按 `dwsim_export.md` §6 的初值方法再收敛，
或改用该文档 §5 提到的单级 mixer-settler 简化路线。**LLE 平衡数值不从本脚本产出，只产出塔结构与初值。**

---

## 4. 运行方式（复现）

```powershell
# 必须在有 DWSIM + pythonnet 的环境运行（pythonnet 需要完整运行时权限初始化）
conda activate thermo

# 体系①②：DWSIM 泡点 + .dwxmz + 补全 CSV（一次性完成）
python scripts\generate_ternary_dwsim_demonstration.py `
    --outdir lunwen\dwsim_demonstration `
    --write-files

# 体系③：LLE 萃取塔 .dwxmz（可选 --calc 先求解）
python scripts\generate_lle_ethanol_eac_water_dwsim.py `
    --out lunwen\dwsim_demonstration\ethanol_eac_water_lle_extractor_nrtl.dwxmz `
    --stages 8
```

> 环境注意：`pythonnet`/`clr` 初始化需要进程内省（`OpenProcess`）能力，在受限沙盒
> （ConstrainedLanguage / 无进程句柄权限）下会报 `Failed to initialize Python.Runtime.dll`。
> 请在真实终端（完整权限）运行，如本机 `C:\Users\34861\miniconda3\envs\thermo`。

---

## 5. 关键实现要点（供后续维护）

1. **Vessel 必须显式设 `FlashTemperature` / `FlashPressure`**：DWSIM `Vessel` 默认
   `FlashTemperature = 298.15 K`，若不覆盖，闪蒸在错误温度进行，泡点二分会坍缩为 `nan`。
   正确做法是让 Vessel 在进料自身 T/P 下做等温等压闪蒸。
2. **每个新 flowsheet 都要重新 `AddCompound`**：化合物名不能跨 flowsheet 缓存
   （否则后续 flowsheet 无组分，闪蒸恒为单相、泡点为 `nan`）。
3. **NRTL BIP 写入用 cal 尺度**：`NRTL_IPData.A12 = A_ij(K) / 4.184`，`alpha12 = α`。
4. **LLE 走真实萃取塔**：`AbsorptionColumn` + `OperationMode = Extractor`，
   进料/溶剂经 `ConnectFeed` + `SetStreamFeedStage` 注册到塔内（溶剂 stage 0 = 顶，feed stage N-1 = 底）。
5. 平衡数值全部由 DWSIM 计算；脚本只做「搭结构 + 二分温度 + 读取 DWSIM 结果」。

---

## 6. 产物清单

| 文件 | 说明 |
|---|---|
| `lunwen/dwsim_demonstration/ipa_x*.dwxmz`（5 个） | 体系① 各代表点的泡点 TP-flash 流程 |
| `lunwen/dwsim_demonstration/dmso_etac*.dwxmz`（6 个） | 体系② 各代表点的泡点 TP-flash 流程 |
| `lunwen/dwsim_demonstration/ethanol_eac_water_lle_extractor_nrtl.dwxmz` | 体系③ 严格液液萃取塔 |
| `docs/dwsim_ipa_bubble.csv` | 体系① 补全后三源对比（含 DWSIM 列） |
| `docs/dwsim_dmso_bubble.csv` | 体系② 补全后三源对比（含 DWSIM 列） |
| `scripts/generate_ternary_dwsim_demonstration.py` | 统一生成脚本（体系①②） |
| `scripts/generate_lle_ethanol_eac_water_dwsim.py` | 体系③ LLE 萃取塔脚本（复用既有） |
