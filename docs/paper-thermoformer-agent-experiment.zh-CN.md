# 实验：ThermoFormer 驱动的对话式萃取精馏 Agent

ThermoFormer 以分子视图融合 + 可微 VLE 求解器，提供两类泡点预测：**Isothermal P–x–y**（给定 $(\mathbf{x},T)$ 预测 $(\mathbf{y},P)$）与 **Isobaric T–x–y**（给定 $(\mathbf{x},P)$ 预测 $(\mathbf{y},T)$）。本实验将乙醇-水萃取精馏作为其下游应用，考察 ThermoFormer 预测的汽相组成 $\mathbf{y}$ 对塔设计的影响，并与机理模型 UNIFAC 对照。

---

## 1 方法

**Agent 流程。** 用户以自然语言提出萃取精馏任务，Agent 依次执行：

```
意图分流 → 萃取例行 → 提取参数 → 萃取剂候选 → 用户选定 → 塔设计 → 导出
```



其中萃取例行调用确定性短节法 `design_extractive_distillation_column`：用相对挥发度 α 做产品分割，经 Fenske–Underwood–Gilliland 求最板数与回流比，回归塔顶/塔釜泡点温度。

**两处可选用 ThermoFormer 相对挥发度 α 的来源。**

相对挥发度并非 ThermoFormer 的直接输出：模型输出泡点汽相组成 $\mathbf{y}$，α 由其按下式得到：

$$
\alpha_{E/W}=\frac{y_E/x_E}{y_W/x_W}
$$

萃取区（富萃取剂，$\mathbf{x}=[0.10,0.10,0.80]$）再算一次 $\alpha_{\mathrm{ext}}$，得平均相对挥发度与选择性：

$$
\alpha_{\mathrm{avg}}=\sqrt{\alpha_{\mathrm{base}}\,\alpha_{\mathrm{ext}}},\qquad \mathrm{selectivity}=\frac{\alpha_{\mathrm{ext}}}{\alpha_{\mathrm{base}}}
$$

T 指定走 `solve_isothermal` 出 $(P,\mathbf{y})$，P 指定走 `solve_isobaric` 出 $(T,\mathbf{y})$；分子特征来自 RDKit 描述子、Uni-Mol v2 嵌入与 SMARTS 官能团的融合。预测结果统一标记 `source_type="model_prediction"`。

---

## 2 对照设置

以 UNIFAC 活度系数为机理组、ThermoFormer 泡点预测为预测组，其余塔设计步骤相同。

| | 机理组（UNIFAC） | 预测组（ThermoFormer） |
|---|---|---|
| 汽相组成 $\mathbf{y}$ | $\gamma_i$（官能团法）+ 蒸气压 | VLE 求解器预测 $(\mathbf{y},P)$ 或 $(\mathbf{y},T)$ |
| 相对挥发度 α | $\alpha=\dfrac{\gamma_E P_E^{\mathrm{sat}}}{\gamma_W P_W^{\mathrm{sat}}}$ | $\alpha=\dfrac{y_E/x_E}{y_W/x_W}$ |

进料：乙醇/水 $=0.40/0.60$，$1\ \mathrm{mol/s}$，$25\,^\circ\mathrm{C}$，$101.325\ \mathrm{kPa}$，塔顶乙醇纯度 $0.995$。

---

## 3 结果

对照组样例（乙二醇为萃取剂）实测：

| 量 | 机理组（UNIFAC） | 预测组（ThermoFormer） |
|---|---|---|
| $\alpha_{\mathrm{base}}$ | — | ≈ 0.679 |
| $\alpha_{\mathrm{ext}}$ | — | ≈ 0.260 |
| $\alpha_{\mathrm{avg}}$ | ≈ 2.82 | ≈ 0.42 |
| 理论塔板数 $N$ | 19 | 4 |
| 回流比 $R$ | 2.536 | 0.07 |
| 塔顶 / 塔釜泡点温度 | 351.45 / 411.97 K | 351.45 / 415.07 K |

---

## 4 讨论

- ThermoFormer 通过预测 $\mathbf{y}$ 反推 α，因此 α 的误差直接传导至塔设计：本实验该 checkpoint 的 $\alpha_{\mathrm{avg}}<1$，使短节法输出退化的保守设计（$N$ 下取整、$R$ 取下限），与 UNIFAC 机理组的一致性结果形成对照。
- 结果强调：预测模型（尤其汽相组成 $\mathbf{y}$）的校准程度会放大到下游分离设计，提示需对模型预测进行基准校准与不确定性标注后再用于工程设计。
