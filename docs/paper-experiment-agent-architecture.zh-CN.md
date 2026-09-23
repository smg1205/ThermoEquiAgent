# 论文实验章节：Agent 方法架构与 ThermoFormer 预测任务

> 状态：草稿，供撰写论文"方法/实验"章节使用
> 基础：`docs/Agent整合ThermoFormer-进度文档-v4.md`
> 与之对应的代码：`agent/orchestrator.py`、`agent/extractive_distillation.py`、`thermo_engine/thermoformer_backend.py`、`thermo_engine/column_design.py`

本附件把"乙醇-水萃取精馏"案例写成论文的一个实验章节。核心是**两张图**：一张画 Agent 的端到端方法架构，一张画 ThermoFormer 的预测任务边界；再用一张表把每个模型调用对到具体函数。

---

## 1 要讲清楚的两个概念

论文规定 ThermoFormer 的预测任务是**泡点两类描述**：

| 任务 | 输入 | 输出 | 对应代码函数 |
|---|---|---|---|
| 等温泡点 | 液相组成 $\mathbf{x}$、温度 $T$ | 汽相组成 $\mathbf{y}$、压力 $P$ | `predict_bubble_isothermal` |
| 等压泡点 | 液相组成 $\mathbf{x}$、压力 $P$ | 汽相组成 $\mathbf{y}$、温度 $T$ | `predict_bubble_isobaric` |

两个任务都由同一个入口 `ThermoFormerBackend.bubble_point` 分发（`thermo_engine/thermoformer_backend.py`）。当一个任务同时给了 $T$ 走**等温**分支；只给 $P$ 走**等压**分支。

预测结果的来源统一由代码标注为 `source_type="model_prediction"`（模型预测），并随结果附带"未实验验证"告警——这是论文"模型预测 vs 机理计算"对照的关键点。

---

## 2 Agent 方法架构（端到端）

下面这张图对应 `docs/Agent整合ThermoFormer-进度文档-v4.md` 的 2.2/2.3/2.4 节，并补上"萃取剂选择"这一步。

```
用户消息
  │
  ▼
[1] 意图判断  _classify_intent()
  │  (规则 + 大模型交叉确认；DeterministicProvider 兜底)
  ├────────────────────────────┬────────────────────────────┬───────────────
  问答类                       萃取                         计算/改条件
  │                            │                            │
  ▼                            ▼                            ▼
[2a] 记忆+知识回答           [2b] 萃取管线                [2c] 构建任务→有界执行图
                              │                            │
                              ▼                            ▼
                        run_extractive_export()          plan()→execute()→validate()→respond()
                              │
                              ▼
                    [3] 萃取剂候选排序
                        recommend_extraction_entrainer()
                        （本地模型：UNIFAC 默认 / ThermoFormer）
                              │
                              ▼
                    [4] 用户选定萃取剂（awaiting_entrainer）
                              │
                              ▼
                    [5] 塔设计 design_extractive_distillation_column()
                        ├─ 算相对挥发度 α、选择性（本地模型）
                        ├─ Fenske 最少塔板 / Underwood 回流 / Gilliland 塔板
                        ├─ 泡点温度（塔顶/塔釜）
                        ▼
                    [6] DWSIM 导出 export_dwsim_extractive_column()
                        ▼
                    [7] 返回设计 + .dwxmz（可选）
```

**说明（对着图上编号）：**

- **[1] 意图分流**：先规则、后大模型，大模型不可用则信任规则；专门意图（萃取、参数查询等）需关键词触发（`is_extractive_distillation_request`），大模型"脑补"不算。
- **[2b] 萃取管线入口** `run_extractive_export(message)`：参数由正则/规则抠取（`extract_extractive_params` → `build_extractive_spec`）。缺参数返回"缺哪几项"，不往下算。
- **[3] 萃取剂候选排序** `recommend_extraction_entrainer`：对每个候选萃取剂（乙二醇、甘油…），在萃取区组成 $[x_E,x_W,x_S]=[0.10,0.10,0.80]$ 用本地模型算相对挥发度与选择性，排序返回候选。**默认 UNIFAC；指定 ThermoFormer 时用其 $bubble\_point$ 结果。**
- **[4] 用户选定萃取剂**：消息含明确萃取剂则直接进塔设计（`_find_entrainer` 命中）；未指定或问"看看候选"则先返回候选面板（`status="awaiting_entrainer"`）。
- **[5] 塔设计** `design_extractive_distillation_column`：内部相对挥发度来源与[3]一致（同一 `alpha_source`）。
- **[6] DWSIM 导出**：建严格塔 + 4 条物流（进料/萃取剂/塔顶/塔釜），连线，存 `.dwxmz`；冷凝器/再沸器规格需手动补（见 v4 第 4 节）。
- **[7] 返回**：`ExtractiveExportPayload`，前端自动下载、可选记录。

> 与 v4 的关系：v4 的"主流程图"本质是 [1]→[2a/2b/2c] 分流；本图把**萃取支路**展开到塔设计与 DWSIM 导出。有界执行图四步（plan→execute→validate→respond）对应 [2c] 的普通计算，萃取支路则走 column_design 这一确定性链路。

---

## 3 ThermoFormer 预测任务边界（论文图）

展示论文两个任务在 Agent 萃取场景里如何被调用，以及它的输出如何进入塔设计。

```
           ┌─────────────── ThermoFormer 后端 ───────────────┐
           │            ThermoFormerBackend.bubble_point()   │
用户指定    │  range check  _check_system_scope               │
  (x,T) 或  │  ─（组分数≤3、压强≤500 kPa）─────────────────   │
  (x,P) ──►│  SMILES → 分子编码 encode_molecules             │
           │    （RDKit 描述子 + Uni-Mol v2 嵌入 + 官能团)    │
           │        │                                        │
           │        ▼                                        │
           │  给了 T ? ──是──► predict_bubble_isothermal     │
           │                 → 输出 y, P  （任务1）           │
           │        │                                        │
           │        └──否──► predict_bubble_isobaric         │
           │                 → 输出 y, T  （任务2）           │
           │                        │                        │
           │                        ▼                        │
           │  validate_equilibrium_result → build_result     │
           │  source_type="model_prediction" + 告警          │
           └───────────────────────┬─────────────────────────┘
                                   ▼
                      下游：column_design 用输出的 y
                      算相对挥发度 α=(y_E/x_E)/(y_W/x_W)
                      再走 Fenske/Underwood/Gilliland → 塔设计
```

**说明：**
- 两条预测任务共用同一编码与校验管线，区别只在"给了哪个热力学变量"。
- 模型的输出（$y$）进入 `column_design._relative_volatility_from_model` 反推 $\alpha$——**$\alpha$ 不是模型的直接输出**，是预测出 $y$ 后按定义计算得到。
- 输出一律 `source_type="model_prediction"`，保证论文可从结果里区分"模型预测"与"UNIFAC 机理计算"。

---

## 4 每一步的模型调用对照表

| 步 | 干什么 | 入口/代码 | 用的模型 |
|---|---|---|---|
| 意图判断 | 判断是否萃取/计算/问答 | `ConversationOrchestrator.chat()` / `_classify_intent()` | 规则 + 大模型交叉（DeterministicProvider 兜底） |
| 抠参数 | 进料组成/温度/压力/流量/纯度/回收率 | `extract_extractive_params()` + `build_extractive_spec()` | 正则 + 规则 |
| 萃取剂候选 | 候选排序 | `recommend_extraction_entrainer()` | 本地模型：**UNIFAC（默认）或 ThermoFormer** |
| 相对挥发度 | α、选择性 | `relative_volatility_eivw()` | 与上同源 |
| ThermoFormer 泡点 | 等温/等压预测 (x,T)→(y,P) 或 (x,P)→(y,T) | `ThermoFormerBackend.bubble_point()` → `predict_bubble_isothermal/isobaric` | ThermoFormer（checkpoint + Uni-Mol v2） |
| 塔设计 | Fenske/Underwood/Gilliland、塔板/回流/塔温 | `design_extractive_distillation_column()` | 确定性短节法（相对挥发度来源同[3]） |
| 泡点温度 | 塔顶/塔釜泡点 | `_bubble_temperature()` | 泡点方程 + 蒸气压（UNIFAC/NIST） |
| 导出 | 建塔建流存文件 | `export_dwsim_extractive_column()` | DWSIM 自动化（pythonnet） |

---

## 5 实验变量设计（供论文写"对照组"）

- **变量一：相对挥发度来源**。
  - 机理组：UNIFAC 活度系数 + 蒸气压；
  - 预测组：ThermoFormer `bubble_point` 输出的 $y$ 反推 $\alpha$（等温/等压两种子任务）。
- **变量二：候选萃取剂**。在本地模型打分下提供乙二醇/甘油等，由用户选定后再出塔设计。
- **因变量**：理论塔板数 $N$、最小回流比 $r_{\min}$、操作回流比 $R$、选择性、塔顶/塔釜泡点温度、DWSIM 收敛性。
- **输出标记**：所有用 ThermoFormer 的数值带 `source_type="model_prediction"` + "未实验验证"告警，便于论文明确标注哪些结果是 ML 预测、哪些是机理计算。

---

## 6 与代码的实际差距（如实写，论文别写过头）

- v4 第 7.7 已注明：萃取设计数值目前默认来自 `column_design`（UNIFAC + Fenske 系）。ThermoFormer 作为可选后端已能跑 `bubble_point`（本仓库 `thermo_engine/thermoformer_backend.py`），并在萃取剂候选/塔设计中可作为 `alpha_source="thermoformer"` 被请求启用。
- 若论文实验需要"整条链都用 ThermoFormer 出数值"，需确认部署机装了 torch/rdkit、有 checkpoint，且（本地跑 Uni-Mol v2 时）权重可访问；否则该组不可执行，属环境依赖，非代码问题。

---

## 7 建议的论文引用措辞（可直接抄）

> 我们以一个"对话式工程相平衡工作台"为例：用户以自然语言提出乙醇-水萃取精馏任务，Agent 先做意图分流，再通过本地模型筛选候选萃取剂并交由用户确认，最后以确定性短节法（Fenske–Underwood–Gilliland）生成塔设计，并可选导出 DWSIM 文件。其中，ThermoFormer 承担两类泡点预测任务——给定 $(\mathbf{x},T)$ 预测 $(\mathbf{y},P)$，或给定 $(\mathbf{x},P)$ 预测 $(\mathbf{y},T)$——其预测的汽相组成 $\mathbf{y}$ 用于反推相对挥发度，从而在"机理计算（UNIFAC）"与"模型预测（ThermoFormer）"两种模式下对相同的塔设计比较。
