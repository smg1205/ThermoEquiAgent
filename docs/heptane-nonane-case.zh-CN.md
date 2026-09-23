# 案例：正庚烷 / 正壬烷直接二元精馏

本案例是 ThermoAgent 推荐的首次运行对象。它是一次**不含溶剂的直接二元精馏**，因此可以在
没有萃取剂筛选步骤干扰的情况下，完整走通整条链路——实验数据、ThermoFormer 预测、DWSIM
计算、塔设计与文件导出。

- **体系**：正庚烷（轻组分，塔顶）/ 正壬烷（重组分，塔釜）
- **压力**：101.325 kPa，等压
- **实验来源**：NIST ThermoML，DOI `10.1016/j.fluid.2013.05.016`
- **ThermoFormer 权值**：`vle_overall_binary`、`seed_2`
- **DWSIM 物性包**：UNIQUAC

---

## 1. 三源沸点对比

101.325 kPa 下五个代表性液相组成。`x` 与 `y` 分别为液相与汽相中的正庚烷摩尔分数。

| x（正庚烷） | T 实验 (°C) | T ThermoFormer (°C) | T DWSIM (°C) | y 实验 | y ThermoFormer | y DWSIM |
|---|---|---|---|---|---|---|
| 0.117 | 140.75 | 138.93 | 139.97 | 0.3250 | 0.3830 | 0.3403 |
| 0.359 | 123.05 | 121.67 | 123.28 | 0.7260 | 0.7384 | 0.7006 |
| 0.466 | 117.15 | 116.28 | 117.66 | 0.8240 | 0.8163 | 0.7890 |
| 0.633 | 109.15 | 109.55 | 110.30 | 0.9160 | 0.8970 | 0.8844 |
| 0.837 | 102.55 | 103.10 | 103.01 | 0.9730 | 0.9613 | 0.9593 |

在全部 16 个锁定测试点上，ThermoFormer 的精度为：

| 指标 | 数值 |
|---|---|
| 沸点温度 MAE | **0.985 °C** |
| 汽相正庚烷 MAE | **0.0246** |

三个来源的温度一致性在约 2 °C 以内，两个模型也都一致地复现了汽相中正庚烷的富集。
该体系接近理想，这正说明它是很好的首个案例：能把"工程链路"与"热力学难度"分离开来考察。

数据来源：NIST ThermoML，DOI `10.1016/j.fluid.2013.05.016`。上表由
`scripts/generate_heptane_nonane_comparison.py` 重新生成。

---

## 2. 精馏塔设计

短节法 Fenske / Underwood / Gilliland 设计。相对挥发度**取自 DWSIM UNIQUAC 沸点计算，
而非假定值**——这是保证设计与模拟口径一致的关键。

| 规格 | 数值 |
|---|---|
| 进料组成 x（正庚烷） | 0.466 |
| 进料流量 | 1.0 mol/s |
| 压力 | 101.325 kPa |
| 塔顶纯度 x（正庚烷） | 0.995 |
| 正庚烷回收率 | 0.98 |
| 进料点相对挥发度 | 4.2859 |

塔计算结果：

| 项目 | 数值 |
|---|---|
| 理论塔板数 | 15 |
| 最小塔板数 | 6.42 |
| 进料板 | 7 |
| 最小回流比 | 0.638 |
| 操作回流比 | 0.893（1.4 × 最小） |
| 塔顶流量 | 0.459 mol/s |
| 塔釜流量 | 0.541 mol/s |
| 冷凝器类型 | 全凝器 |

DWSIM 计算得到的沸点温度：进料 390.81 K、塔顶 371.43 K、塔釜 422.40 K。

设计记录由 `scripts/generate_heptane_nonane_binary_column.py` 写出。

---

## 3. 从工作台运行

启动服务：

```powershell
python -m uvicorn apps.api.main:app --reload --port 8000   # 后端
pnpm --dir apps/web dev                                     # 前端
```

然后在输入框中粘贴以下任一语句。

**英文：**

```text
heptane 0.466, nonane 0.534, export the DWSIM distillation file
```

**中文：**

```text
正庚烷 0.466，正壬烷 0.534，导出 DWSIM 精馏塔文件
```

两条语句结果相同：返回短节法设计，并附带可下载的 `.dwxmz` 文件。

### 为什么两个要素都必需

路由需要**两个相互独立的条件**同时满足，才会触发 DWSIM 导出：

1. 已识别的组分对——`heptane`/`正庚烷` 与 `nonane`/`正壬烷` 必须同时出现
   （`_DISTILLATION_BINARY_ALIASES`）；并且
2. 明确的导出词——`dwsim`、`dwism`、`dwxmz`、`export`、`download`、`导出`、`下载`
   之一（`_DWSIM_FILE_MARKERS`）。

此外还需要进料组成。缺少要素时可观察到的行为：

| 语句 | 结果 |
|---|---|
| `heptane 0.466, nonane 0.534, export the DWSIM distillation file` | 完整设计 + 文件 |
| `正庚烷 0.466，正壬烷 0.534，导出 DWSIM 精馏塔文件` | 完整设计 + 文件 |
| `正庚烷-正壬烷精馏塔设计`（无导出词） | 仅返回设计数值，不生成文件 |
| 有导出词但无组成 | 结构化的 `missing_parameters` 失败 |

注意最后两行：系统**不会**猜测成 50/50 进料，也**不会**在用户只要数值时偷偷生成文件。

---

## 4. 复现案例文件

```powershell
python scripts\generate_heptane_nonane_comparison.py     # ThermoFormer vs 实验
python scripts\generate_heptane_nonane_dwsim.py          # 五个近沸点闪蒸
python scripts\generate_heptane_nonane_binary_column.py  # 短节法设计 + 严格塔
```

| 脚本 | 产出内容 |
|---|---|
| `generate_heptane_nonane_comparison.py` | 由锁定的 `seed_2` 测试预测生成 ThermoFormer 对实验对照表 |
| `generate_heptane_nonane_dwsim.py` | 五个近沸点 TP 闪蒸 `.dwxmz`；温度与汽相组成由 DWSIM 计算 |
| `generate_heptane_nonane_binary_column.py` | 短节法设计记录与严格二元精馏塔流程 |

### 生成的文件

脚本会写入 `report/success/正庚烷-正壬烷/`，该目录**不纳入 Git 跟踪**——请先运行上述命令生成。
文件及其内容如下：

| 文件 | 内容 |
|---|---|
| `heptane_nonane_three_source_bubble.csv` | 五个组成点的三源对照 |
| `heptane_nonane_binary_distillation_x0p466_design.json` | 设计记录与 DWSIM VLE 数值 |
| `heptane_nonane_binary_distillation_x0p466.dwxmz` | 严格二元精馏塔 |
| `heptane_x0p117_2comp_bubble_140.0C.dwxmz` | x = 0.117 的近沸点 TP 闪蒸 |
| `heptane_x0p359_2comp_bubble_123.3C.dwxmz` | x = 0.359 的近沸点 TP 闪蒸 |
| `heptane_x0p466_2comp_bubble_117.7C.dwxmz` | x = 0.466 的近沸点 TP 闪蒸 |
| `heptane_x0p633_2comp_bubble_110.3C.dwxmz` | x = 0.633 的近沸点 TP 闪蒸 |
| `heptane_x0p837_2comp_bubble_103.0C.dwxmz` | x = 0.837 的近沸点 TP 闪蒸 |

---

## 5. 在 DWSIM 中打开文件

打开 `.dwxmz` 文件需要 DWSIM 与 pythonnet，且**pythonnet 需要完整进程权限**——它在受限沙箱中
无法运行。请在 `.env` 中把 `DWSIM_HOME` 设为包含 `DWSIM.Automation.dll` 的目录。

打开前有一点需要了解：**冷凝器与再沸器规格无法通过 DWSIM Automation 接口可靠设置。** 导出的
塔在结构上是完整的（塔板数、进料、产品物流与回流比都已写入），但第一次严格计算需在 DWSIM
图形界面中完成——在界面上补上冷凝器与再沸器规格，然后重新计算。这是 DWSIM 自动化本身的能力
边界，而非导出功能的问题；本仓库所有精馏导出都适用同一说明。

五个 TP 闪蒸文件不需要这一步：它们是相平衡闪蒸，可直接计算。

---

## 6. 核查要点

- 随着正庚烷被蒸出，沸点温度应单调上升，从 x = 0.117 的约 140.75 °C 到 x = 0.837 的
  102.55 °C。
- 同一组成下，汽相正庚烷应始终高于液相正庚烷，因为正庚烷是更易挥发的组分。
- 塔顶与塔釜流量之和应等于进料流量（0.459 + 0.541 = 1.000 mol/s）。
- 塔釜温度（422.40 K）应接近正壬烷沸点，因为塔釜富含正壬烷。
