# DWSIM 数据 → ThermoFormer LLE 预测：可行路径

## 核心发现：参考数据里已经有 MIBK/水

`datasets/lle/binary_lle.csv` 中直接查到 6 行（去重后 3 条系线）：

| T (K) | P (kPa) | x_MIBK 水相 | x_MIBK 有机相 | DOI |
|---|---|---|---|---|
| 333.15 | 100 | 0.0023 | 0.8109 | 10.1016/j.fluid.2016.11.005 |
| 343.15 | 100 | 0.0022 | 0.7410 | 同上 |
| 353.15 | 100 | 0.0019 | 0.6924 | 同上 |

**这正是 `mibk_water_nrtl_fitted.json` 里参数的拟合来源 DOI。**
（文件每条系线存了两遍、相序互换，读取时需去重。）

数据集规模：

| 数据集 | 记录数 | DOI 数 | 文件 |
|---|---:|---:|---|
| Binary LLE | 5,983 | 254 | `datasets/lle/binary_lle.xlsx` |
| Ternary LLE | 13,861 | 432 | `datasets/lle/ternary_lle.xlsx` |

字段：`temperature_k, pressure_kpa, smiles1, x_alpha_1, x_beta_1, smiles2, x_alpha_2, x_beta_2, doi, source_url, …`
（CSV 是 UTF-8 **带 BOM**，读取要用 `utf-8-sig`。）

## 用参考数据反查：拟合参数质量不合格

我用独立等活度求解（scipy）验证 `A12=3601.07, A21=1747.85, α=0.3798`：

| T (K) | 参考 x_MIBK（水相/有机相） | 模型 x_MIBK | 误差 |
|---|---|---|---|
| 333.15 | 0.0023 / **0.8109** | 0.0018 / **0.6123** | 0.0005 / **0.1986** |
| 343.15 | 0.0022 / **0.7410** | 0.0021 / **0.6758** | 0.0001 / 0.0652 |
| 353.15 | 0.0019 / **0.6924** | 0.0025 / **0.7747** | 0.0006 / 0.0823 |

**水相（稀 MIBK）拟合很好（误差 ≤0.0006），但有机相系统性偏差，333.15 K 差 0.199 摩尔分数。**

这解释了 `mibk_water_nrtl_fitted.json` 里 `residual_norm = 0.305` 偏大 —— 参数确实没拟合好。
误差符号在 353.15 K 还翻转了（模型 0.7747 > 参考 0.6924），说明温度依赖也没抓住。

**所以：DWSIM 那边不分相，参数质量也是一个真实因素，不只是软件问题。**
（我此前说"参数完全正确"需要修正：参数能产生分相，但定量不准。）

## ThermoFormer 已有的 LLE 能力

不必从零开始，仓库里已有完整栈：

| 模块 | 作用 |
|---|---|
| `src/thermoformer/thermodynamics/lle_solver.py` | 可微 LLE 求解：`gmix_dimless`、`tpd`、`solve_lle`（多重初值 + TPD 稳定性筛） |
| `src/thermoformer/data/lle.py` | LLE 数据加载 |
| `src/thermoformer/lle_tp/` | 整套 TP 闪蒸：`binodal_model`、`hybrid_binodal_model`、`solver_certified`、`verified_campaign` |
| `src/thermoformer/protocols/lle_runner.py` | 实验协议入口 |
| `src/thermoformer/models/thermoformer.py` | 模型（需 `activity_mode=excess_gibbs`） |

实验配置：`configs/lle/` 下 12 个可运行协议（prediction 6 个 + generalization 5 个 + stability 1 个）。

**关键约束（`AGENTS.md`）**：
- 版本 0.1 **明确排除 flowsheet design 和 VLLE**
- "Every numerical result must come from `thermo_engine` and pass `validate_equilibrium_result`"
- **不得编造二元参数、实验数据或引用**

## 关于「用 DWSIM 数据做模拟实验」的重要警告

**不要把 DWSIM 的输出当作 LLE 真值标签。** 本次排查已证明：

| 路径 | 结果 |
|---|---|
| `Vessel` + `SimpleLLE` | `PT Flash: Invalid solution` |
| `Vessel` + 4 组不同参数 | 结果逐位相同，HEAVY = 0（对参数无响应）|
| `AbsorptionColumn` (Extractor) | `Error evaluating error functions`，0.5 s 崩溃 |

用这样的输出训练或验证模型，等于**用错误的标签污染数据**，直接违反仓库的科学规则。

## 建议方案（三条，按推荐度）

### 方案 A：用 ThermoFormer 参考数据做真值，DWSIM 只做独立复现（推荐）

```
datasets/lle/binary_lle.csv  →  ThermoFormer 训练/评估 真值
                                        ↓
                         预测 MIBK/水 tie-lines
                                        ↓
                    与参考数据比对（已有现成协议）
```
- 真值来自 NIST ThermoML，有 DOI 可溯源 —— 合规
- DWSIM 的角色改为**独立第三方复算**，用来交叉验证，而不是提供标签

### 方案 B：先修参数，再谈模拟

用 `datasets/lle` 里的 MIBK/水数据重新回归 NRTL（只有 3 个温度点，可加三元数据）。
目标是把有机相误差从 0.199 降到可接受范围。DWSIM 的
**Tools → Binary Data Regression → LLE** 就是干这个的。

### 方案 C：绕过 DWSIM 的单元操作

既然 `Vessel`/`Extractor` 都不执行液液闪蒸，就**不要用它们**。
直接用 `phasepy`（仓库已在 `src/thermoformer/baselines/phasepy_adapter.py` 引了它）
或 ThermoFormer 自己的 `solve_lle` 做 LLE 计算。

## 立即可做的事

1. **跑通现有协议**：`configs/lle/prediction/binary_system.json` —— 针对已知二元体系
2. **建 MIBK/水 专项测试**：只有 3 条系线，样本极少，适合做 qualitative 检查而非训练
3. **修正我之前的结论**：参数能分相但定量不准，`residual_norm=0.305` 是真实信号

## 产物

| 文件 | 说明 |
|---|---|
| `.tmp/dwsim/validate_against_dataset.py` | 参考数据 vs NRTL 比对（含去重/BOM 处理）|
| `.tmp/dwsim/compare_correctly.py` | 相序修正后的正确比对 |
| `docs/lle/water_mibk_*.md` | 前三轮的 DWSIM 排查记录 |
