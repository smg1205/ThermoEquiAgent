# 指定使用 ThermoFormer 的萃取精馏流程

> 触发示例：`乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，用乙二醇做萃取剂，塔顶乙醇纯度99.5%`

本文解释：当用户指定使用 ThermoFormer 时，系统如何用本地模型**筛选萃取剂**、再由用户**选定**萃取剂并完成塔设计。所有数值来自确定性引擎 `thermo_engine`；LLM/规则只识别意图、拆参数、编排，**从不直接算平衡数值**。

---

## 核心概念

**ThermoFormer 不直接输出相对挥发度 $\alpha$。** 它输出的是泡点 VLE 结果：$(x,T) \rightarrow (y,P)$（汽相组成与压力）。$\alpha$ 是在 `thermo_engine/column_design.py::_relative_volatility_from_model` 中，拿到模型输出的泡点汽相组成 $y$ **之后**，再按定义计算得到的：

$$
\alpha_{ij} = \frac{K_i}{K_j} = \frac{\,y_i/x_i\,}{\,y_j/x_j\,}
$$

$i$=乙醇、$j$=水。这里的 $y$ 是 ThermoFormer 模型**预测**输出（`thermoformer_backend.py` 中标注为 `source_type="model_prediction"`），公式 $(y_i/x_i)/(y_j/x_j)$ 本身是确定性的。

---

## 输入（本例）

进料乙醇 0.40 / 水 0.60，1 mol/s，25 °C=298.15 K，常压=101.325 kPa，塔顶乙醇 99.5%。乙二醇为萃取剂（如不指定，系统会先给你候选让你选）。回收率 0.98、萃取比 2.0 未给时用默认。

---

## 端到端流程（两个阶段）

### 阶段 A：本地模型筛选萃取剂 → 让用户选择

在 `agent/extractive_distillation.py`：

1. **识别意图** `is_extractive_distillation_request` → 走萃取精馏。
2. **是否指定 ThermoFormer** `requests_thermoformer`：命中 `thermoformer`/`用thermoformer` 等 → 本地筛选用 **ThermoFormer**；否则默认 **UNIFAC**（`alpha_source`）。
3. **给候选清单**：若消息**没指定萃取剂**，或写了"选萃取剂/看看候选"，则
   `offer_entrainer_choices(spec, alpha_source)` 调 `recommend_extraction_entrainer`：
   - 对每个候选萃取剂（乙二醇、甘油…），取萃取区组成 $[x_E,x_W,x_S]=[0.10,0.10,0.80]$，用所选模型算相对挥发度 $\alpha$ 与选择性；
   - 返回 `ExtractiveExportPayload(status="awaiting_entrainer", entrainer_candidates=[...])`，界面弹出候选按钮，等待用户选择。**此时不出塔设计。**

**筛选所用 $\alpha$（与塔设计同一套计算）**：

$$
\alpha_{\mathrm{ext}} = \frac{y_E / x_E}{y_W / x_W}
$$

### 阶段 B：用户选定萃取剂 → 出塔设计

用户在候选面板点选（前端会带上原请求整体重发，补齐进料上下文），消息含明确萃取剂（如 `用甘油`）：

1. `_find_entrainer` 命中 → 跳过候选环节，直接进 `design_extractive_distillation_column(spec, alpha_source=...)`。
2. 设计内部仍使用本地模型算相对挥发度：二元基础组 + 三元萃取区，几何平均得 $\alpha_{\mathrm{avg}}$：

$$
\alpha_{\mathrm{avg}} = \sqrt{\alpha_{\mathrm{base}} \cdot \alpha_{\mathrm{ext}}}, \qquad \text{selectivity} = \frac{\alpha_{\mathrm{ext}}}{\alpha_{\mathrm{base}}}
$$

3. 随后是确定性短节法设计：
   - **物料衡算**：按塔顶纯度与回收率分割乙醇/水/乙二醇三股流量；
   - **Fenske 最少塔板数**：$N_{\min} = \dfrac{\ln[(x_{D,E}/x_{D,W})/(x_{B,E}/x_{B,W})]}{\ln \alpha_{\mathrm{avg}}}$
   - **Underwood 最小回流比**（二元恒 $\alpha$）后取运行回流比 $R = 1.4\,R_{\min}$；
   - **Gilliland–Eduljee** 回流转板数权衡得到理论塔板数 $N$；定进料板与萃取剂板；
   - **泡点温度**回读塔顶/塔釜温度；
   - 返回 `status="ready"/"dwsim_unavailable"` 设计 + 可选 DWSIM 文件，`backend_version` 带 `+thermoformer`，并附 **ML 近似、未实验验证** 告警。

---

## 交互示例

```
你: 乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，塔顶纯度99.5%
AI: 已用本地模型（ThermoFormer）筛出候选萃取剂：glycerol、ethylene glycol，请选择。
    [候选面板：乙二醇 α=.. 选择性=.. | 甘油 α=.. 选择性=..]
你: 点选「甘油」
AI: 理论板数 .. 回流比 .. 塔顶/塔釜温度 .. (+ DWSIM 文件)
```

---

## 实测运行时数值（真实 checkpoint）

当前预发布 checkpoint 下完整链路实测：$\alpha_{\mathrm{base}} \approx 0.679$、$\alpha_{\mathrm{ext}} \approx 0.260$、$\alpha_{\mathrm{avg}} \approx 0.42$、理论塔板数 $N=4$、回流比 $0.07/0.05$、塔顶/塔釜 351.45 K / 415.07 K。

---

## 重要提醒

1. $\alpha$ 不是 ThermoFormer 的直接输出；它由模型**预测**的泡点汽相组成 $y$ 按 $(y_i/x_i)/(y_j/x_j)$ 计算得到。
2. 当前 checkpoint `production_ready: false`，实测 $\alpha_{\mathrm{avg}}<1$，使短节法给出退化的保守设计——这是模型预测偏弱的物理结果，非代码 bug。
3. 因此**默认仍用 UNIFAC**；只在指定 ThermoFormer 时切换。
4. 每次启用都强制带"ML 近似、未经实验验证"告警。

---

## 触发 / 切换对照

| 你的意图 | 请求里怎么写 |
|---|---|
| 用 ThermoFormer | 加上 `用ThermoFormer` / `thermoformer` |
| 用默认 UNIFAC | 不写 ThermoFormer 相关词 |
| 不指定萃取剂、先看候选 | 不写萃取剂，或写 `看看候选` / `选萃取剂` |
| 直接指定萃取剂出设计 | `用乙二醇` / `用甘油` |
