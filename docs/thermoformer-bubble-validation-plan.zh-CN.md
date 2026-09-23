# 方案：ThermoFormer 泡点预测“好不好”的判断标准与落地流程

> 适用范围：本仓库 `ThermoFormer` 后端对**泡点 VLE**（等温泡点 `P–y`、等压泡点 `T–y`）预测的质量判定。
> 定位：把“怎样才算预测得好”从**单一误差数字**提升为**分层可验收的判定门**，与仓库数值纪律
> （所有数值来自确定性引擎、经过物理校验、缺失参数结构化失败）保持一致。
> 对齐的借鉴论文：`lunwen/` 下三篇（I&ECR Aspen Copilot、AIChE Chemasim 多智能体、JCTC OpenClaw）
> 的共同立场——“判别结果正确性必须由物理/实验把关；预测模型只负责协调与产生候选数”。

---

## 0. 一句话结论

**ThermoFormer 泡点“好”= 三层同时通过：**

1. **回归层**：对**实验实测**的泡点温度/压力/汽相组成，误差落在可接受阈值内，且**不比机理基线（UNIFAC）差**；
2. **物理层**：预测在**热力学上自洽**（拉乌尔/修正拉乌尔残差、吉布斯–杜亥姆、纯极限、收敛、相图光滑）；
3. **下游层**：由泡点推导的量（相对挥发度 α、塔板数 N、回流比 R、选择性）落在**物理可行域**内，能支撑合理的分离设计。

任何一层不过 → 该预测**不得直接进入工程设计**，应标记、回退机理模型或请求人工复核。

---

## 1. “真值”是什么：实验实测泡点

仓库中 `VLESample`（`lab_models/ThermoFormer/src/thermoformer/data/loading.py`）即实验真值：

| 字段 | 含义 | 单位来源 |
|---|---|---|
| `temperature_k` | 泡点实验温度 | Excel 摄氏 → K（+273.15）|
| `pressure_kpa` | 泡点实验压力 | Excel 毫米汞柱 → kPa（×0.133322368）|
| `liquid_composition` | 液相组成 x（输入/已知）| 归一化到 Σ=1 |
| `vapor_composition` | **汽相组成 y（真值）** | 归一化到 Σ=1 |
| `quality_weight` / `quality_status` | 数据质量（passed=1 / unverified=0.5 / failed）| 评分加权 |
| `experiment_mode` | isothermal / isobaric / full_state | **决定该样本按哪个方向评分** |
| `smiles` / `names` / `doi` / `source` | 体系标识与出处 | 溯源与防泄露分组 |

**因此判断“回归好坏”的标准答案 = 这些实验条目本身。** 评估时：

- **等温模式样本**（`experiment_mode=="isothermal"`）：给定实测 `temperature_k` 与实测液相
  `liquid_composition`，预测 `pressure_kpa` 和 `vapor_composition`，与实测对齐误差。
- **等压模式样本**（`experiment_mode=="isobaric"`）：给定实测 `pressure_kpa` 与实测液相
  `liquid_composition`，预测 `temperature_k` 和 `vapor_composition`，与实测对齐误差。
- **full_state**：两个方向都可评，也可只做物理层。

> 初始化提示：现有 `thermo_engine/thermoformer_backend.py` 的 `_warnings()` 写“结果未经实验验证”。
> 本方案落地后，应在具备实验真值样本的协议分区上执行下述回归层，把该警告替换为实测的判别结果。

---

## 2. 判断标准：三层判据详解

### 2.1 回归层（对实验实测）

对**每个有效预测**（`converged and not nonphysical`）计算，再跨样本聚合。已实现于
`evaluation/prediction.py::_scalar_metrics` / `_y_metrics`：

| 指标 | 定义 | 生效对象 |
|---|---|---|
| MAE, RMSE, R² | 点级聚合 | 泡点压力 `ΔP_kpa`、泡点温度 `ΔT_k` |
| `system_macro_mae/rmse` | 按体系先平均再聚合 | 防止大体系掩盖小体系 |
| `y_mae/y_rmse/y_r2` | 汽相组成 y 的误差 | `y`（摩尔分数）|
| `y_sample/system/component_macro` | 多种口径的 y 误差 | `y` |

**判定门槛（建议初值，可按数据集校准）：**

| 量 | 建议阈值 | 说明 |
|---|---|---|
| 泡点温度 `ΔT` | ≤ 1–2 K（均值），p95 ≤ 3 K | 多数工程短节计算可接受 |
| 泡点压力相对误差 | ≤ 1–3%（均值）| 等温模式 |
| 汽相组成 `y` | ≤ 0.01–0.05（摩尔分数）| 对 α/塔板数最敏感，宜从严 |
| 对比 UNIFAC | ThermoFormer MAE **不得显著劣于** UNIFAC | “不差于机理基线”才算可用 |

> 关键判据：**单独的 MAE 没有意义，必须与 UNIFAC 基线并排报告**。
> 参照 AIChE 论文：预测组与机理组在同一下游任务上对照，预测不劣出合理范围才算“好”。

### 2.2 物理层（热力学自洽）

已实现于 `evaluation/thermodynamic_consistency.py`。这些判据**不依赖实验数据**，属硬性物理门控：

| 判据 | 数学/含义 | 建议通过阈值 |
|---|---|---|
| 求解器收敛 | `converged==True` | 收敛率 ≥ 99%（P≤500 kPa、二/三元）|
| 平衡残差 | 修正拉乌尔 `P_calc − P_target` | `gross_equilibrium_residual_kpa ≤ 0.1`（现常量）|
| 物理边界 | P>0、y∈[0,1]、Σx=Σy=1、γ>0、P_sat>0、T∈[150,1500] K | 0 违约 |
| 吉布斯–杜亥姆 | `Σ xᵢ d ln γᵢ = 0` | 残差中位数 → 0，p95 尽量小 |
| 纯极限 | x→纯组分时 γ→1、P→P_sat、T 一致 | `pure_limit_vle_error ≤ 0.05`（现常量）|
| 置换不变性 | 换组分顺序结果不变 | `permutation_*` 近 0 |
| 相图光滑 | 泡点曲线一/二阶导跳变 | `phase_*` 无尖峰跳变 |

### 2.3 下游层（派生量物理合理）

泡点 y 会放大到分离设计。参照 `docs/paper-thermoformer-agent-experiment.zh-CN.md` 的对照实验
（UNIFAC α_avg≈2.82、N=19、R=2.536 vs ThermoFormer α_avg≈0.42、N=4、R=0.07），
**α<1 / R→0 / N 取整退化即是“物理上不合理”的信号**。

| 派生量 | 合理域 | 不合理信号 |
|---|---|---|
| 相对挥发度 α | α>0；萃取精馏希望有区分度 | α≈1（不可分）、α<1（反向异常）|
| 理论塔板数 N | 有限正整数、可被进料/目标解释 | N 取整下限退化、脱离塔顶塔釜泡点 |
| 回流比 R | 0<R 有限 | R 落到实现下限、无物理依据 |
| 塔顶/塔釜泡点温度 | 在组分泡点之间且趋势正确 | 超纯组分泡点区间 |

**下游层是三层里最容易被忽视、但三篇论文最强调的一层**——因为泡点误差会传播放大到塔设计，
预测即使“数值漂亮”也可能在设计端不可用。

---

## 3. 落地流程：判断树（建议固化进 `validate_equilibrium_result`）

```
输入：一个泡点预测（T/P/y/γ/P_sat + 状态位）
  │
  ├─1 求解器收敛？           否 → REJECT（不产出自洽平衡）
  ├─2 物理边界？            否 → NONPHYSICAL（标记、不进入指标池）
  ├─3 平衡残差 ≤ 阈值？      否 → REJECT
  ├─4 吉布斯–杜亥姆/纯极限/置换/光滑？  否 → REJECT
  ├─5 [有实验] MAE/RMSE ≤ 阈值 且 不劣于 UNIFAC？ 否 → FLAG（可报告但不可下发设计）
  ├─6 [有实验] 派生量(α,N,R,T) 物理合理？        否 → FLAG（回退机理模型）
  ├─ ✓ 全部通过 → VALID（可进入工程设计，带不确定性/置信度标注）
```

输出应为**结构化判定**，例如：

```json
{
  "verdict": "VALID | REJECT | NONPHYSICAL | FLAG",
  "reason": "一句话原因",
  "checks": {
    "converged": true,
    "physical_bounds": true,
    "equilibrium_residual_kpa": 0.02,
    "gibbs_duhem_pass": true,
    "pure_limit_pass": true,
    "regression_mae": {...},
    "unifac_comparison": {"thermoformer_mae": 0.9, "unifac_mae": 0.8, "better": false},
    "downstream": {"alpha": 2.5, "n_stages": 19, "reflux": 2.5, "reasonable": true}
  }
}
```

---

## 4. 推荐实现路径（分三步，先文档→再最小门控→后完整）

### 步骤 A（本文档）：定义标准与阈值 —— 已完成
见本文件；阈值 `2.1/2.2/2.3`。

### 步骤 B（最小改动）：新增 `assess_bubble_prediction` 门控
- 位置：`thermo_engine/`（后端 seam，对 Agent 可用）；或 `evaluation/score_card.py`（训练侧）。
- 输入：单个 `EquilibriumPoint` + `converged`/`nonphysical` + 有则实验真值 `VLESample`。
- 输出：上文 `verdict + checks` 结构化结果。
- 复用：`thermodynamic_consistency` 的物理判据、`prediction_metric_rows` 的回归聚合；**不新建数值**。

### 步骤 C（完整）：UNIFAC 基线对照 + 不确定性标注
- 在 `thermo_engine` 或独立对照脚本里，对同一批 `VLESample` 用 UNIFAC 算泡点，输出并排 MAE。
- 对每条预测给出置信度/适用区间（P≤500 kPa、二/三元、未见分子域），`FLAG` 时让 Agent 主动
  回退机理模型——对齐 AIChE 双 Agent“预测 Agent 决策、机理 Agent 把关”的思路。

---

## 5. 判断“好不好”的最终问答

| 问 | 答 |
|---|---|
| ThermoFormer 泡点“好”的标准答案是什么？ | **三层：实验回归误差 + 物理自洽 + 下游派生量合理** |
| 有真实实验数据吗？ | 有——`VLESample` 即实验实测（T/P/x/y，带 DOI 溯源）|
| 拿什么当真值打分？ | 回归层用实验实测；物理层用热力学恒等式；下游层用物理可行域 |
| 一定要逼近 UNIFAC 吗？ | 不是“逼近”，而是**不劣于 UNIFAC**（并列基准）|
| 没有实验数据怎么办？ | 先仅跑物理层 + 下游层，实验齐后叠加回归层并重评 |

---

## 附：与三篇论文判据的对照

| 本方案层 | 对应论文判据 | 借鉴点 |
|---|---|---|
| 回归层（实验） | I&ECR：文献规格/严格模拟对照 | “与已知正确值对齐、有余量” |
| 物理层（自洽） | I&ECR / JCTC：收敛、残差、有界校验 | “求解器状态≠物理有效，必须独立验证” |
| 下游层（派生量） | AIChE：分离序列物理合理性、严格模拟可否收敛 | “设计放大的自洽 + 掩码推理防作弊” |
| 机理基线对照 | AIChE：机理组 vs 预测组 | “不劣于基线才算可用” |
