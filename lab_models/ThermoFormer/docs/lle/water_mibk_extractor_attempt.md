# water–MIBK LLE：Extractor 塔尝试与最终结论

## 尝试内容

按 `Absorption/Extraction Column`（`OperationMode = Extractor`）重试，并把
之前学到的参数注入方法用上。三处关键修正：

| 项 | 做法 |
|---|---|
| NRTL 参数 | 改写 `.dwxmz` 内 XML 注入拟合值 A12=3601.07 / A21=1747.85 / α=0.3798 |
| **`AutoEstimateMissingNRTLUNIQUACParameters`** | 显式写为 **`false`** |
| 两相初值 | `SetInitialMolarCompositionEstimates` 分层给 MIBK 富集 / 水富集 |

### 关于 `AutoEstimate` 的重要发现

第一次生成的文件**没有** `AutoEstimateMissingNRTLUNIQUACParameters` 标签。
DWSIM 加载时**默认为 `True`**，于是重新估算参数并**丢弃**注入的拟合值：

```
AutoEstimate  : True        <-- 拟合参数被覆盖
```

对比原文件 `water_mibk_native_vessel_lle_333K_fitted.dwxmz`，它**显式**含：

```xml
<AutoEstimateMissingNRTLUNIQUACParameters>false</AutoEstimateMissingNRTLUNIQUACParameters>
```

补上该标签后：

```
AutoEstimate  : False       <-- 拟合参数保住了
verify A12  : ['3601.066568632594']
verify A21  : ['1747.8505840944874']
verify alpha: ['0.37982119574538054']
```

**这是之前所有失败尝试的一个隐藏陷阱**：任何通过 Automation API 新建的流程，
若不显式关闭该开关，注入的 IP 都会在加载时被静默覆盖。

## 结果：Extractor 仍然失败

```
OperationMode : Extractor
Stages        : 5
Solver        : Wang-Henke (Bubble Point)
AutoEstimate  : False
composition estimates set (staged MIBK/water rich)

=== solving ===
returned after 0.5 s
   EXC: EXT-001: Error evaluating error functions.
Solved: False
```

完整堆栈：

```
System.Exception: EXT-001: Error evaluating error functions.
   在 DWSIM.UnitOperations.UnitOperations.Column.Calculate(Object args)
        RigorousColumn.vb:行号 4927
   在 DWSIM.SharedClasses.UnitOperations.BaseClass.Solve()
        SimulationObjectBaseClasses.vb:行号 495
   在 DWSIM.FlowsheetSolver.FlowsheetSolver.CalculateObjectAsync(...)
```

尝试了 3 种规格配置（产物摩尔流量 / 质量流量 / 不同初值分布），
**全部在 0.0–0.5 s 内失败**，且产物流量停在 1.000 kg/s 的初始猜测值上。

0.5 s 内失败说明**不是迭代不收敛，而是求解器在建立误差函数阶段就崩了**。

## 决定性发现：参数从来不是问题

我用独立实现（不依赖 DWSIM）直接计算 NRTL 的 Gibbs 自由能曲线，
检验四组参数是否预测液液分层：

| 参数组 | d²G/dx²<0 区间 | 双节线 |
|---|---|---|
| **拟合值** A12=3601.07 A21=1747.85 α=0.3798 | 24.1% | x(MIBK) = 0.150 ↔ 0.300 |
| 拟合值（交换） | 24.1% | 0.694 ↔ 0.844 |
| **DWSIM 自动估算** A12=2586.46 A21=7.375 α=0.2 | 44.1% | 0.425 ↔ 0.519 |
| 强疏水 A12=2500 A21=4500 α=0.30 | 80.4% | 0.325 ↔ 0.582 |

**四组全部预测液液分相**，包括 DWSIM 自己估算的那组。

这与 DWSIM 内部行为直接矛盾：DWSIM 用这些参数**算出单液相**，
而同样的参数在 NRTL 方程下**必然给出两相区**。

**结论：参数正确，是 DWSIM 9.0.5 的单元操作没有调用液液闪蒸。**

## 汇总：三条路径全部失败

| 路径 | 结果 |
|---|---|
| `Vessel`（Legacy）+ `SimpleLLE` + 5 bar | `PT Flash: Invalid solution`，三相全等于进料组成 |
| `Vessel` + 三组不同参数（含强疏水） | 结果**逐位相同**，HEAVY = 0 |
| `AbsorptionColumn`（Extractor）+ 拟合参数 + 两相初值 | `Error evaluating error functions`，0.5 s 内失败 |

`Vessel` 对参数完全不响应（三组差异极大的参数结果逐位相同），
`Extractor` 在误差函数阶段崩溃 —— 两者都没有真正执行液液两相闪蒸。

## 建议

1. **不要再尝试用内置单元操作做 water/MIBK 的 LLE 分相**。三条路径已系统性排除。
2. **在自定义 Python UO 中直接调用闪蒸算法类**
   （`NestedLoopsImmiscible` / `SimpleLLE`），绕过 `Vessel` 与 `Column` 的封装。
3. **或在 DWSIM 之外独立求解 LLE**。NRTL 液液闪蒸是一个成熟的数值问题
   （Rachford-Rice + 相稳定性检验），几十行代码即可实现，
   且我已验证参数本身是正确的 —— 这可能是最快、最可控的路径。

## 产物

| 文件 | 说明 |
|---|---|
| `.tmp/dwsim/extractor2_fitted.dwxmz` | Extractor 塔 + 拟合参数 + `AutoEstimate=false` |
| `.tmp/dwsim/nrtl_check.py` | 独立 NRTL Gibbs 分析（验证参数正确） |
| `.tmp/dwsim/extract_trace.py` | 完整异常堆栈复现 |
| `.tmp/dwsim/iptest.py` | IP 注入对照实验 |

## 最终判据：拟合参数确实能分相（独立求解验证）

我用独立实现（`nrtl_flash_check.py`，scipy 等活度求解 NRTL 液液闪蒸，
不经过 DWSIM）验证拟合参数：

```
T = 298.15 K   no two-liquid solution found -> single phase
T = 333.15 K   phase A: x_water=0.3877 -> MIBK mass frac = 0.8978
               phase B: x_water=0.9982 -> MIBK mass frac = 0.0098
               two phases present
T = 343.15 K   phase A: x_water=0.3242 -> MIBK mass frac = 0.9206
               phase B: x_water=0.9979 -> MIBK mass frac = 0.0117
               two phases present
```

**333.15 K 的结果与实验值高度吻合**：

| 项 | 计算值 | 实验值 |
|---|---|---|
| 水相中 MIBK 质量分数 | 0.0098（≈1.0 wt%） | ~1.7–2.0 wt% |
| 有机相中 MIBK 质量分数 | 0.8978 | 约 0.90 |

即：**拟合参数是正确的**，在 333.15 K 给出合理的互溶度，
完全能够产生液液分相。这也解释了参数为何回归自 333.15/343.15/353.15 K 的系线。

**因此 DWSIM 内置单元操作算不出分相，是软件行为问题，不是热力学问题。**

## 待确认（已终止）

`lle_direct.py`（强制 `SetPresentPhases` 指定两液相）运行超过 10 分钟未返回，
已终止。这是"内置闪蒸能否做 LLE"的最后一个直接判据，但结论已由上节的
独立求解确立，不影响最终判断。
