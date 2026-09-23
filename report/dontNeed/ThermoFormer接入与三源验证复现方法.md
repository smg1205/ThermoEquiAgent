# ThermoFormer 接入与三源验证方法复现文档（LLEs / 二元&三元 VLE）

> 来源：以 `lunwen/Agent整合ThermoFormer进度与三源验证报告v4.md` 为方法蓝本，对照当前代码库
> （`thermo_engine/thermoformer_backend.py`、`thermo_engine/registry.py`、`thermo_engine/column_design.py`、
> `thermo_engine/dwsim_export.py`、`scripts/export_flow_examples.py`）逐条核实后整理。
> 目标是：任何人（包括 AI 编码代理）按本文顺序执行即可复现「ThermoFormer 接入 + 二元/三元 VLE
> 与 LLE 预测 + 三源（实验 / ThermoFormer / DWSIM，或 UNIFAC / ThermoFormer / DWSIM）比较」的完整链路。
> 文中所有模块名、函数名、字段名、数值口径均取自代码或原报告，不引入未在仓库中出现的命名。

---

## 0. 总览：一条主线 + 三类下游任务

报告的主线是「大模型只做意图理解与编排、数值一律来自 `thermo_engine` 并经独立校验」。围绕这条主线，
需要复现的工程能力分三类：

| 任务 | 求解入口 | 说明 |
|---|---|---|
| 二元 VLE（泡点/汽相组成） | `ThermoFormerBackend.bubble_point` / `isothermal_vle` / `isobaric_vle` | 由分子结构 + 状态预测平衡 |
| 三元 VLE（含萃取剂） | 同上（组分数 = 3） | 支撑萃取精馏的相对挥发度与剂选择性 |
| 二元/三元 LLE | `ThermoFormerBackend.lle` | 由 SMILES + T/P 预测两相共存端点 |

三条链路共享同一个接入骨架：**后端类 → 能力声明 → 注册 → 适用范围检查 → 结果校验 → 权重延迟加载**。

---

## 1. 接入骨架（后端如何接入）

报告 §1.3 给出的接入方式是五步，代码里逐条对应如下：

1. **编写后端类**：`thermo_engine/thermoformer_backend.py` 中的 `ThermoFormerBackend`，实现
   `bubble_point`、`isothermal_vle`、`isobaric_vle`、`lle` 等方法（见 §2、§3、§4）。
2. **声明能力**：在 `thermo_engine/registry.py` 的 `DEFAULT_BACKEND_REGISTRY`（文件底部）中，以
   `BackendRegistration` 声明：

   ```python
   BackendRegistration(
       canonical_name="ThermoFormer",
       aliases=frozenset({"thermoformer", "thermo former", "tf"}),
       supported_calculations=frozenset(
           {"bubble_point", "isothermal_vle", "isobaric_vle", "lle", "infinite_dilution_activity"}
       ),
       factory=ThermoFormerBackend,
   )
   ```

   其中 `factory=ThermoFormerBackend` 的导入在 `registry.py` 顶部
   `from thermo_engine.thermoformer_backend import ThermoFormerBackend`。

3. **加适用范围检查**：`_check_system_scope`（`thermoformer_backend.py`）限制
   —— 组分数 ≤ `_MAX_COMPONENTS = 3`；压力 ≤ `_MAX_PRESSURE_KPA = 500.0` kPa；
   并配合 `_smiles_from_components` 要求每个组分提供 SMILES。越界时抛结构化 `ThermoEquiError`
   （`UNSUPPORTED_MODEL` / `PARAMETER_OUT_OF_DOMAIN` / `MISSING_DATA`）。

4. **结果经 `validate_equilibrium_result` 校验**：后端返回的 `CalculationResult` 经
   `thermo_engine/validation.py` 的统一校验门（组成归一、物料衡算、平衡残差、收敛、参数适用性、相稳定性）。

5. **权重延迟加载**：真实实现里「权重延迟加载」对应
   - 环境解析 `resolve_settings()`（读 `THERMOFORMER_SRC`、`THERMOFORMER_CHECKPOINT`、
     `THERMOFORMER_FEATURE_CACHE`、`THERMOFORMER_USE_CUDA`）；
   - 模型类只在首次调用时通过 `_ensure_loaded()` 加载 torch/rdkit 依赖与 checkpoint，
     而不是 import 阶段加载。

> 说明：报告原文将第 5 步表述为「权重延迟加载」，代码中该语义由「可选依赖延迟 import
> （`_check_optional_dependencies` + `_ensure_loaded`）」与「checkpoint 按需选择 `_select_checkpoint`」共同承载，
> 二者在性质上一致：线程/进程启动时不加载模型，真正调用预测方法时才加载。

---

## 2. 二元 / 三元 VLE 复现方法

### 2.1 环境变量（与 `.env` 对齐）

当前 `.env` 已含：

```
THERMOFORMER_SRC=E:\codex\ThermoAgent\ThermoFormer\src
THERMOFORMER_CHECKPOINT=（空）
THERMOFORMER_FEATURE_CACHE=（空）
THERMOFORMER_USE_CUDA=1
```

- `THERMOFORMER_SRC` 指到 ThermoFormer 仓库的 `src` 目录；为空时回退到
  `Path(__file__).resolve().parents[2] / "src"`（相对 `thermoformer_backend.py` 上溯两级，即 ThermoFormer 根下的 `src`）。
- `THERMOFORMER_CHECKPOINT` 为空时，走 `_select_checkpoint`，从 `models/registry.json` 按
  `(task, protocol, seed=0, status="provided")` 选 checkpoint；
  显式设置则直接指向该 checkpoint 文件（仅用于覆盖，不参与 registry 选择）。

### 2.2 checkpoint 协议映射（代码 `_checkpoint_protocol`）

| 平衡类型 | 组分数 | registry 的 `task` | registry 的 `protocol` |
|---|---|---|---|
| VLE | 2 | `vle` | `vle_overall_binary` |
| VLE | 3 | `vle` | `vle_overall_ternary` |
| LLE | 2 | `lle` | `binary-system` |
| LLE | 3 | `lle` | `ternary-system` |

> 注意：这是 `models/registry.json` 里 `seed=0` 的约定。报告 §2 末尾「权值选择」写的两个权重
> `checkpoints/overall_binary/seed_4/best_model.pt`、`checkpoints/overall_binary_ternary/seed_4/best_model.pt`
> 是**该报告当时手选**的 `seed_4` 权重，与代码里 registry 默认 `seed_0` 不是同一套。复现报告 §2 数值
> 时应显式把 `THERMOFORMER_CHECKPOINT` 指到报告所述 `seed_4` 权重；复现「代码默认行为」则保持
> `THERMOFORMER_CHECKPOINT` 为空走 registry。

### 2.3 泡点（bubble_point）复现

`ThermoFormerBackend.bubble_point(request: TaskManifest) -> CalculationResult` 的行为：

- **等温**：`TaskManifest.conditions.temperature_K` 给定 → 输出 `pressure_kPa` + `y`；
- **等压**：`TaskManifest.conditions.pressure_kPa` 给定 → 输出 `temperature_K` + `y`；
- 两者都缺 → 抛 `MISSING_DATA`。

内部实现（可逐步核对）：
1. `_check_system_scope`（组分数、压力范围）；
2. `_smiles_from_components` 取各组分的 SMILES；
3. `_get_predictor` → `_ThermoFormerPredictor.encode_molecules(smiles)` 得到分子特征张量；
4. 等温走 `predict_bubble_isothermal`（内部 `_solver_isothermal`，`iterations=24`，`strict=False`）；
   等压走 `predict_bubble_isobaric`（内部 `_solver_isobaric`，`iterations=16`，`strict=False`）；
5. 用返回的 `residual`、`iterations`、`converged` 组装 `EquilibriumPoint`，
   经 `build_result(...)`（`thermo_engine/activity_coeff_utils.py`）产出 `CalculationResult`，
   `phase_state="two_phase"`。

预测返回的字段名（`predict_bubble_isothermal` / `predict_bubble_isobaric` 的 dict）：
`pressure_kpa` / `temperature_k`、`y`、`gamma`、`psat_kpa`、`residual`、`converged`、`iterations`。

### 2.4 全曲线 VLE（isothermal_vle / isobaric_vle）

- `isothermal_vle`：定温 `temperature_K`，用 `_build_composition_sweep(n_components, request.points)`
  生成组成扫描点，经 `_build_equilibrium_points` 逐点预测，返回整条 P-x-y 曲线；
- `isobaric_vle`：定压 `pressure_kPa`，其余同 `isothermal_vle`，返回 T-x-y 曲线。

两个方法都要求温度（等温）或压力（等压）之一给定，缺失抛 `MISSING_DATA`；组分数越界由
`_check_system_scope` 拒绝。

### 2.5 报告 §2.1 的二元复现要点（2-丙醇–水）

- 体系：`isopropanol`（轻组件，塔顶浓缩）+ `water`；等压 101.325 kPa。
- 报告取样：`x_IPA = 0.1, 0.3, 0.5, 0.7, 0.9`，逐点做等压泡点，比较
  `T_实验 / T_TF / T_DWSIM` 与 `y_实验 / y_TF / y_DWSIM`。
- 三源口径：实验值、ThermoFormer（`bubble_point` 等压预测）、DWSIM（加载 flowsheet 后判读
  「Vapor 摩尔分率 → 0 时的温度」为泡点温度）。
- 报告结论（防复述走样，直接引用）：在 `x_IPA=0.3–0.9` 段 ThermoFormer 与 DWSIM 温度预测差 ≤ 0.03 °C，
  二者均比该段实验泡点高约 1–3 °C；ThermoFormer 泡点温度 MAE ≈ 2.58 °C。

> 三元体系报告 §2.2（乙酸乙酯–乙酸正丙酯–DMSO）的三源比较表同理：等压泡点温度 + 汽相组成，
> ThermoFormer（36 点）泡点温度 MAE = 18.09 °C、平均误差约 −18.0 °C（预测系统低于实验），
> DWSIM（6 代表点）泡点温度低于实验约 1.2–7.2 °C。

---

## 3. LLE 复现方法

### 3.1 方法入口

`ThermoFormerBackend.lle(request: TaskManifest) -> CalculationResult`：

1. `_check_system_scope`（组分数 ≤ 3）；
2. 要求 `conditions.temperature_K` 与 `conditions.pressure_kPa` **同时给定**，缺一抛 `MISSING_DATA`；
3. `_get_lle_predictor(request)` → `_ThermoFormerLLEPredictor.predict_lle(smiles, T, P, device=...)`；
4. 取返回 `pairs[0]` 的 `endpoints`（应恰好 2 个端点）；
   - 无 `pairs` 或端点不满足 → `NUMERICAL_NONCONVERGENCE`；
5. 两相都以 `PhaseResult(phase="liquid", fraction=0.5, composition=endpoints[i])` 组装，
   `phase_state="two_phase"`；
6. `residual` 取 `pairs[0]["equilibrium_rms"]` 或回退 `max_equilibrium_rms`；
   `iterations` 取 `prediction["attempts"]`；若预测含 `stability_check` 则并入 `warnings`。

### 3.2 LLE 预测器的延迟导入

`_ThermoFormerLLEPredictor._ensure_loaded()` 在首次调用时：
- `_check_optional_dependencies()`（要求 `torch`、`rdkit` 已安装）；
- 把 `THERMOFORMER_SRC` 加进 `sys.path`；
- `from thermoformer.lle_tp.predict_verified import predict` 导入外置预测函数；
- 用 `settings.use_cuda and torch.cuda.is_available()` 决定 `cuda`/`cpu`。
- 导入失败抛 `MISSING_PARAMETERS`；预测本身异常抛 `NUMERICAL_NONCONVERGENCE`。

### 3.3 LLE 的明确边界（务必保留在复现里）

`lle` 返回的是**共存端点组成**，不含"由整体进料推出的相分率"；代码在 `warnings` 里显式写：

```
"ThermoFormer LLE returns coexistence endpoint compositions; phase fractions are not inferred from an overall feed."
```

即：本后端不承担「给定进料 → 两相各占多少」的相分率分配，只给两端组成。这一点与 PSMI 论文
「不判定是否分相、只重建已知分相体系的共存组成」是一致的（见仓库 `lunwen/PSMI_LLE_solve_notes.md`）。

---

## 4. 萃取/精馏塔设计（报告 §1.5–§1.6，VLE 侧的下游）

报告的塔设计链路是：**VLE 泡点/相对挥发度 ← ThermoFormer 或 UNIFAC，塔板/回流比 ← 确定性短节法
（Fenske–Underwood–Gilliland），工程文件 ← `dwsim_export.py`**。LLM 不参与任何平衡数值与塔参数数值计算。

### 4.1 确定性塔设计入口（`thermo_engine/column_design.py`）

| 函数 | 用途 |
|---|---|
| `bubble_temperature(components, xs, P_kPa, alpha_source=...)` | 等压泡点温度（`alpha_source` 可选 `unifac` 或模型源） |
| `relative_volatility_eivw(...)` / `relative_volatility_key(...)` | 相对挥发度计算 |
| `recommend_extraction_entrainer` / `recommend_entrainer_for` | 萃取剂推荐（规则/预测判定） |
| `design_binary_distillation_column(...)` | 二元普通精馏短节设计 |
| `design_ternary_extractive_column(...)` | 三元萃取精馏短节设计 |
| `design_extractive_distillation_column(...)` / `design_generic_extractive_column(...)` | 萃取/通用萃取塔设计 |

报告 §1.5.4 的确定性短节法约定：**操作回流比取 1.4 × 最小回流比**；由相对挥发度（含剂选择性）代入
Fenske 求最小塔板、Underwood 求最小回流比、Gilliland 关联得理论塔板与操作回流比；
塔顶/塔釜温度由（ThermoFormer 或 UNIFAC）等压泡点给出。

### 4.2 相对挥发度里的热力学源切换（`_alpha_source` / `_get_thermoformer_backend`）

`column_design.py` 顶部有 `_alpha_source()` 与 `_get_thermoformer_backend()`，`_relative_volatility_from_model`
通过它们切换「UNIFAC 源」或「ThermoFormer 源」。复现报告里的 UNIFAC / TF 双列对照，即分别以这两种
源调用 `design_binary_distillation_column` / `design_ternary_extractive_column`，得到 α、N、N_min、R、r_min、
塔顶/塔釜泡点两套数值（结果与报告 §1.6.1 / §1.6.2 的表对齐）。

### 4.3 报告下游两个案例（数据口径直接取自报告，用于核对复现）

**案例一：2-丙醇–水 普通精馏（`design_binary_distillation_column`，101.325 kPa，x_IPA=0.3）**

| 量 | UNIFAC | TF |
|---|---|---|
| 相对挥发度 α | 2.713 | 3.135 |
| 理论塔板 N | 19 | 17 |
| 最小塔板 N_min | 10.069 | 8.795 |
| 回流比 R | 2.694 | 2.16 |
| 最小回流比 r_min | 1.924 | 1.543 |
| 塔顶泡点 | 355.26 K | 355.25 K |
| 塔釜泡点 | 367.46 K | 371.72 K |

**案例二：乙酸乙酯/乙酸正丙酯 + DMSO 萃取精馏（`design_ternary_extractive_column`，乙酯:丙酯=0.5:0.5、1.0 mol/s、DMSO 比 2.0、常压）**

| 设计量 | UNIFAC | TF |
|---|---|---|
| α_base / α_ext（乙/丙酯对） | 2.187 / 1.761 | 2.240 / 1.943 |
| 选择性（α_ext/α_base） | 0.805 | 0.931 |
| α_avg | 1.963 | 2.086 |
| 理论塔板 N | 20 | 18 |
| N_min（Fenske） | 10.154 | 9.310 |
| 回流比 R | 2.478 | 2.179 |
| r_min | 1.770 | 1.557 |
| 塔顶/塔釜泡点 | 351.08 / 396.50 K | 349.73 / 391.20 K |

### 4.4 DWSIM 导出（`thermo_engine/dwsim_export.py`）

| 函数 | 对应报告 |
|---|---|
| `export_dwsim_binary_column(...)` | 案例一（二元普通精馏） |
| `export_generic_extractive_column(...)` | 案例二（带萃取剂三元萃取精馏） |
| `export_dwsim_extractive_column(...)` | 乙醇–水加萃取剂的历史路径（报告未用） |

`scripts/export_flow_examples.py` 是这两个案例的可运行入口：`case1_binary()`（2-丙醇/水）与
`case2_extractive()`（乙酸乙酯/n-丙酯 + DMSO），经确定性设计后分别调用上述两个导出函数，
`--dry-run` 只打印设计量不触达 DWSIM。运行方式见该脚本顶部 docstring：

```
python scripts/export_flow_examples.py --outdir data/exports/flow_examples
python scripts/export_flow_examples.py --dry-run
```

导出塔能否在 DWSIM 中算出结果，**以 DWSIM 图形界面加载计算后结果页的状态为准**（报告 §1.4、§1.5.4 原话）。

> 提醒：本仓库当前对**三元 LLE 的 DWSIM 导出**仍在严格萃取塔（`AbsorptionColumn + OperationMode=Extractor`）
> 上反复遇到收敛失败（`converged to the trivial solution`、`Error evaluating error functions`），
> 可用的替代是单级 mixer-settler / 平衡分离器路线（见 `scripts/generate_lle_mixer_settler_process.py`）。
> 该结论与 LLE 复现无关（LLE 侧走 `ThermoFormerBackend.lle`，不经 DWSIM），仅在需要把 LLE 结果
> 落到 DWSIM 流程文件时相关，特此注明以免混淆。

---

## 5. 端到端复现清单（按序执行）

1. **环境**：激活含 `torch`、`rdkit`（可选依赖组 `thermoformer`）的 Python 环境；
   确认 `.env` 里 `THERMOFORMER_SRC` 指向 ThermoFormer 的 `src`，并按需设置
   `THERMOFORMER_CHECKPOINT`（默认空 = registry 的 seed_0；复现报告数值则指到 seed_4 权重）。

2. **接入自检**：`registry.py` 中确有 `ThermoFormer` 的 `BackendRegistration`（§1 已列）。

3. **二元 VLE**：构造 `TaskManifest`（2 组分，给 SMILES + `temperature_K` 或 `pressure_kPa` +
   `liquid_composition`），调 `ThermoFormerBackend.bubble_point` / `isothermal_vle` / `isobaric_vle`，
   核对返回的温度/压力、`y`、`residual`，并过 `validate_equilibrium_result`。

4. **三元 VLE**：同上，组分数 = 3（含萃取剂），重点取等压泡点得到相对挥发度与剂选择性。

5. **LLEs**：构造 `TaskManifest`（2 或 3 组分，SMILES + `temperature_K` + `pressure_kPa`），
   调 `ThermoFormerBackend.lle`，核对返回的两个 `PhaseResult`（`fraction=0.5`、端点组成）与
   `equilibrium_rms` 残差；确认 `warnings` 里的「不分相、只给共存端点」说明。

6. **塔设计（可选下游）**：`design_binary_distillation_column` / `design_ternary_extractive_column`，
   分别以 `alpha_source` 的 UNIFAC 与 ThermoFormer 两种源跑，得到报告 §1.6 的双列设计量。

7. **DWSIM 导出（可选下游）**：`scripts/export_flow_examples.py`（或直接调 `dwsim_export` 的两个函数），
   导出 `.dwxmz` 后在 DWSIM 图形界面加载计算，以结果页状态为准判读可用性。

---

## 6. 数值口径与「不虚构」约束（复现时必须遵守）

- 大模型（Agent）只做意图分流、任务构建（`TaskManifest`）、工具路由与结果解释；**不直接计算任何
  平衡数值**（泡点、露点、y、LLE 端点、塔板、回流比均不由 LLM 给出）。
- 数值来源只有两类：`thermo_engine`（确定性引擎 + ThermoFormer 预测后端），以及三源比较里的
  DWSIM / 实验数据。
- 缺少 SMILES、缺少 T/P、缺少 checkpoint/无关 checkpoint 时，一律抛结构化 `ThermoEquiError`
  （`MISSING_PARAMETERS` / `MISSING_DATA` / `UNSUPPORTED_MODEL` / `PARAMETER_OUT_OF_DOMAIN` /
  `NUMERICAL_NONCONVERGENCE`），**不伪造**。
- 明确拒绝范围（报告 §1.1）：电解质、聚合物、固液平衡、汽液液平衡、完整精馏塔设计。
- 三源比较中的「实验」数据为外部输入，不属 `thermo_engine` 产出；表内数值应引用报告原文或
  重新采集的实验/DWSIM 读数，而非按趋势捏造。

---

## 附：关键文件定位速查

| 报告章节 | 代码/文档位置 |
|---|---|
| §1.2 编排（意图分流 + 有界执行） | `agent/orchestrator.py`、`agent/providers.py`、`agent/graph_workflow.py`、`agent/tools.py`、`agent/router.py` |
| §1.3 后端接入五步 | `thermo_engine/thermoformer_backend.py`、`thermo_engine/registry.py`（`BackendRegistration`） |
| §1.5.3 相平衡求解 | `ThermoFormerBackend.bubble_point`（等温/等压）、`predict_bubble_*`、`_get_predictor` |
| §1.5.4 塔设计短节法 | `thermo_engine/column_design.py`（`design_binary_distillation_column`、`design_ternary_extractive_column` 等） |
| §1.6 案例与导出 | `scripts/export_flow_examples.py`、`thermo_engine/dwsim_export.py` |
| §2 三源比较 | 报告原文 §2.1/§2.2 表（实验 / TF / DWSIM），对照 `bubble_point` 与 DWSIM 泡点判读 |
| LLE 预测后端 | `ThermoFormerBackend.lle`、`_ThermoFormerLLEPredictor`（`thermoformer.lle_tp.predict_verified.predict`） |
| LLE 方法论补充 | `lunwen/PSMI_LLE_solve_notes.md`（PSMI 论文的 LLE 处理方式） |
