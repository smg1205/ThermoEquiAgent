# DWSIM LLE 萃取 — 乙醇 / 乙酸乙酯 / 水

## 交付物

| 文件 | 说明 |
|---|---|
| `experiments/lle/ethanol_eac_water/dwsim_extractor_lle.dwxmz` | DWSIM 流程文件（21,682 字节）|
| `experiments/lle/ethanol_eac_water/dwsim_lle_extractor.py` | 可复现构建脚本 |
| `experiments/lle/ethanol_eac_water/definitive_flash.py` | 判定性闪蒸测试 |
| `experiments/lle/ethanol_eac_water/test_point.json` | 严格构造的两相测试点 |
| `experiments/lle/ethanol_eac_water/pick_test_point.py` | 测试点构造依据 |

## 已建成的流程结构

```
Solvent_Water (1.5 mol/s 纯水, 298.15 K, 1 atm)
        |
        v   第 0 级（塔顶，重相）
   +-------------------+
   |  Liquid-Liquid    |   AbsorptionColumn
   |  Extractor        |   OperationMode = Extractor
   |  8 级             |   逆流
   +-------------------+
        ^   第 7 级（塔底，轻相）
        |
Feed_EtOH_EtAc (1.0 mol/s, x = [EtOH 0.40, EtAc 0.60, Water 0])

塔顶 -> Raffinate_Aqueous      塔底 -> Extract_Organic
```

配置（全部通过 Automation API 写入）：

- 物性包 **NRTL**，3 组分
- **NRTL 二元交互参数已注入**（文献值，Kelvin 除以 4.184 转 DWSIM 的 cal 标度）：

| 对 | A_ij (K) | A_ji (K) | α |
|---|---|---|---|
| EtOAc / Ethanol | 1278.65 | 1382.86 | 0.2988 |
| EtOAc / Water | 5380.57 | 6719.58 | 0.4393 |
| Ethanol / Water | −242.505 | 5195.44 | 0.2937 |

- `AutoEstimateMissingNRTLUNIQUACParameters = False`（**关键**，否则参数被静默覆盖）
- `ConfigParameters()` 重建参数表
- 闪蒸路线 `FlashCalculationApproach = NestedLoops`
- `ThreePhaseFlashStabTestSeverity = 3`
- `UsePhaseIdentificationAlgorithm = True`
- 两液相初值（水相富集 / 酯相富集，逐级插值）+ 流量/温度初值
- 溶剂进第 0 级（塔顶），进料进第 7 级（塔底），逆流

## 求解结果：失败

```
EXC: Liquid-Liquid Extractor: Error evaluating error functions.
Solved: False
```

## 关键排查：修好了一个真 bug，但没解决问题

**发现的真 bug**：既有脚本 `generate_lle_ethanol_eac_water_dwsim.py` 把
`PreferredFlashAlgorithmTag` 设在**塔对象**上，但那不是塔级平衡的入口 ——
塔级平衡由**物性包的** `DW_CalcEquilibrium` 计算，用的是物性包自己的
flash 路线 + `FlashSettings`。所以那个标签**完全无效**，traceback 里始终是
`GibbsMinimizationMulti.Flash_PT`。

我改为在**物性包**上设置，报错从
`GibbsMinimizationMulti.Flash_PT` 变成了 `Error evaluating error functions` ——
**说明这个修复是有效的**，失败点从闪蒸算法移回了塔求解器。

同时确认了 DWSIM 9.0.5 的枚举（反射得到）：

```
FlashCalculationApproachType : NestedLoops=0, InsideOut=1, GibbsMinimization=2
默认 ThreePhaseFlashStabTestSeverity = 0     <-- 不做三相稳定性检验
默认 FlashBase = UniversalFlash
```

参数标签必须是算法的 **Name** 而非类名，映射为：

| 标签字符串 | 实际类 |
|---|---|
| `Nested Loops (VLLE)` | `NestedLoops3PV3` |
| `Nested Loops (Immiscible VLLE)` | `NestedLoopsImmiscible` |
| `Simple LLE` | `SimpleLLE` |
| `Gibbs Minimization SVLLE` | `GibbsMinimizationMulti` |

## 判定性证据：闪蒸本身就不分相

为排除"进料组成选错"的可能，我**从实验系线中点**构造测试点 ——
中点按定义必在两相区内，任何正确的 LLE 闪蒸都必须把它劈回两端点：

```
z* = [EtOH 0.0000, EtOAc 0.4431, water 0.5569]
     （实验系线 alpha=[0, 0.8720, 0.1280] 与 beta=[0, 0.0142, 0.9858] 的中点）

必须劈成：
   有机相  [0.0000, 0.8720, 0.1280]
   水相    [0.0000, 0.0142, 0.9858]
```

6 种闪蒸配置的结果：

| 配置 | Solved | 液相槽位 | 结论 |
|---|---|---|---|
| NestedLoops severity 0 | True | **1** | 单液相 |
| NestedLoops severity 3 | True | **1** | 单液相 |
| GibbsMinimization severity 0 | **False** | 0 | `PT Flash: Invalid solution` |
| GibbsMinimization severity 3 | **False** | 0 | `PT Flash: Invalid solution` |
| InsideOut severity 0 | True | **1** | 单液相 |
| InsideOut severity 3 | True | **1** | 单液相 |

**0 / 6 产生两液相。**

成功求解时 `Liquid1` 占 100% 物料，`Liquid2` 为空。
Gibbs 路线直接抛 `Invalid solution`。

## 结论

**DWSIM 9.0.5 在本机对乙醇/乙酸乙酯/水不产生液液分相**，与之前 water/MIBK
的结论一致。这不是参数问题（参数已正确注入并验证），也不是设置问题
（6 种组合全试过），而是**内置单元操作/flash 的软件行为**。

因此：

1. `dwsim_extractor_lle.dwxmz` 是一个**结构完整、配置正确但未收敛**的流程，
   可在 DWSIM GUI 中打开检查。
2. 该流程**不能用于产出 LLE 数据**，也不应把它的输出当作标签 ——
   这违反 `AGENTS.md` 的 "Every numerical result must come from `thermo_engine`"。
3. 若要 ethanol/EtOAc/water 的萃取数据，应走 **ThermoFormer 的 `solve_lle`**
   或独立 NRTL 液液闪蒸求解（我已验证参数可用）。

## 未排查的剩余可能

- 换用 `UNIQUAC` / `UNIFAC-LL` 物性包（既有流程文件里有 `_uniquac_` 系列，
  暗示前人试过，但我未复测）
- 用自定义 Python UO 直接调用 `NestedLoopsImmiscible` 算法类，绕过 `Column`
- 升级 DWSIM 版本（9.0.5 的 whatsnew 显示三相不可混溶 flash 一直在修 bug）
