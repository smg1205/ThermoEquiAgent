# water–MIBK LLE：根因诊断记录

## 环境
- DWSIM 9.0.5（`C:\Users\34861\AppData\Local\DWSIM`）
- Python 3.13 + pythonnet 3.1.0，经 `DWSIM.Automation.dll` 驱动
- 物性包：**NRTL**（非 UNIQUAC）

## 被诊断文件
`ThermoFormer\ThermoAgent\lunwen\dwsim_demonstration\water_mibk_native_vessel_lle_333K_fitted.dwxmz`

结构：`Feed → Vessel_LL → {Vapor, Light_Liquid, Heavy_Liquid}`

## 磁盘中已确认的参数
`mibk_water_nrtl_fitted.json`（回归自 doi 10.1016/j.fluid.2016.11.005 系线，333.15/343.15/353.15 K）：

```
A12 = 7313.006843851335 J/mol   = 1747.8505840944874 cal/mol
A21 = 15066.862523158774 J/mol  = 3601.066568632594  cal/mol
alpha12 = 0.37982119574538054
residual_norm = 0.30522320737300107
```

该参数已写入 `.dwxmz`：

```xml
<InteractionParameter Compound1="Water" Compound2="Methyl isobutyl ketone"
  A12="3601.06656863259" A21="1747.85058409449"
  B12="0" B21="0" C12="0" C21="0"
  alpha12="0.379821195745381" />
```

## 复现到的真实异常

```
System.Exception: Feed: PT Flash: Invalid solution.
 ---> System.Exception: PT Flash: Invalid solution.

   at GibbsMinimizationMulti.Flash_PT(Double[] Vz, Double P, Double T,
        PropertyPackage PP, Boolean ReuseKI, Double[] PrevKi)
        GibbsMinimizationMulti.vb:line 546
   at PropertyPackage.DW_CalcEquilibrium(FlashSpec spec1, FlashSpec spec2)
        PropertyPackage.vb:line 2658
   at MaterialStream.Calculate(Boolean equilibrium, Boolean properties)
        MaterialStream.vb:line 691
   at BaseClass.Solve()  SimulationObjectBaseClasses.vb:line 495
```

- `fs.Solved = False`
- 报错对象：**`Feed`（进料流股）**，不是 Vessel
- 失败算法：**`GibbsMinimizationMulti.Flash_PT`**

## 根因链

### 1. 存储的进料条件在沸点以上
`.dwxmz` 里的进料规格：

| 项 | 值 |
|---|---|
| T | 333.15 K |
| P | **101325 Pa（1 atm）** |
| w | [0.847553, 0.152447] |
| F | 1.000 kg/s |

MIBK 常压沸点 389.15 K，水 373.15 K；但 15.2 wt% MIBK 的水溶液在 333.15 K、1 atm 已接近/超过泡点。于是 DWSIM 被要求做**汽液（VLE）闪蒸**，而不是液液闪蒸。

### 2. 走的是 Gibbs 最小化路径，而非 LLE 专用算法
`Vessel.PreferredFlashAlgorithmTag` 在文件中为**空字符串**：

```
CalculationMode            = Legacy
FlashTemperature           = 333.15
FlashPressure              = 101325
PreferredFlashAlgorithmTag = ""      <-- 空
PressureCalculation        = Minimum
OverrideP = false   OverrideT = false
```

空标签 → NRTL 物性包退回到默认的 `GibbsMinimizationMulti`，该算法在近泡点、强非理想体系上**无法收敛**，直接抛 `Invalid solution`。

### 3. 结果是三相全废
求解失败后，四条流股全部残留同一个组成：

```
Heavy_Liquid  F=1.00000  w=[0.847553, 0.152447]   <-- 等于进料
Light_Liquid  F=1.00000  w=[0.847553, 0.152447]   <-- 等于进料
Vapor         F=1.00000  w=[0.847553, 0.152447]   <-- 等于进料
Feed          F=0.05087  w=[0.787525, 0.212475]
```

即：**没有相平衡，没有分离，三条出口流股各自复制了进料组成**。

## 关键结论

1. **参数不是瓶颈。** 回归参数（A12/A21/α）已正确写入文件，`AutoEstimateMissingNRTLUNIQUACParameters = False`，DWSIM 不会覆盖它。
2. **失败发生在进料流股，不是澄清器。** 进料在 1 atm / 333.15 K 就崩了，还没走到 Vessel 的分离逻辑。
3. **`PreferredFlashAlgorithmTag` 为空是核心缺陷。** 未指定 LLE 算法，被送去 Gibbs 最小化。
4. **RAFFINATE/Heavy_Liquid 为空是这个失败的后果**，不是独立问题。

## 修复方向（按优先级）

1. **指定 LLE 闪蒸算法**：`Vessel_LL.PreferredFlashAlgorithmTag` 设为
   `SimpleLLE` 或 `NestedLoopsImmiscible`。
2. **提高压力至泡点以上**：1 atm → 3–5 bar，杜绝气相出现，强制液液两相。
   注意 333.15 K 的回归温度区间，与 298.15 K 外推需谨慎。
3. **改用 Absorption/Extraction Column（`OperationMode = Extractor`）**：
   按级求解，每级结构上即液液接触，且有 `SetInitialMolarCompositionEstimates`
   等初值接口——这是 Vessel 所没有的（Vessel 仅有
   `InitializationAction1/2/3`，无组成初值）。
4. 若仍不收敛，检查回归质量：`residual_norm = 0.305` 是否可接受，
   以及 α12 是否落在 0.2–0.47 的常规区间（0.3798 属正常）。

## 本次未能确证的部分

在 Automation API 中重建等价流程时，我无法把拟合参数**注入**新建的物性包
（`ParametersXMLString` 读写均为空，`InteractionParameters` 属性不存在），
因此重建流程用的仍是 DWSIM 自动估算的 `2586.46/7.38/0.2`，其结果为单液相
（`Heavy_Liquid = 0`）。**上述根因链来自对原文件的加载与复现，而非重建流程。**
要验证修复是否奏效，必须直接编辑原 `.dwxmz` 并重跑。
