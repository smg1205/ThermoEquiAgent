# water–MIBK LLE：修复验证结果

## 做了什么

对原文件
`water_mibk_native_vessel_lle_333K_fitted.dwxmz`
按你指定的方向打补丁，生成新文件（原文件已备份）：

`water_mibk_native_vessel_lle_333K_fitted_SIMPLELLE_5bar.dwxmz`

补丁内容（`.dwxmz` 是 ZIP，内含单个 XML，直接改写后重新打包）：

| 项 | 原值 | 新值 |
|---|---|---|
| `PreferredFlashAlgorithmTag` | `""`（空） | `SimpleLLE`（5 处，Vessel + 4 流股） |
| `Vessel.FlashPressure` | `101325` | `500000` |
| 缓存 `<pressure>` | `101325` ×4 | `500000` ×4 |
| `Minimum Pressure` 的 `Data` | `101325.0` | `500000.0` |

## 补丁生效了

重新加载后确认：

```
Vessel check after load:
   PreferredFlashAlgorithmTag = 'SimpleLLE'
   FlashPressure             = 500000.0
   FlashTemperature          = 333.15
   CalculationMode           = Legacy
```

DWSIM 正常读取，无 XML 报错。

## 报错位置前移了，但**仍然失败**

| | 补丁前 | 补丁后 |
|---|---|---|
| 报错对象 | **`Feed`** | **`Vessel_LL`** |
| 异常 | `PT Flash: Invalid solution` | `PT Flash: Invalid solution` |
| `Solved` | False | **False** |

```
exceptions returned: 1
    Vessel_LL: PT Flash: Invalid solution.
Solved: False

stream                 kg/s  mass frac [water, MIBK]
Heavy_Liquid       1.000000  [0.847553, 0.152447]   <-- 仍等于进料
Light_Liquid       1.000000  [0.847553, 0.152447]   <-- 仍等于进料
Vapor              1.000000  [0.847553, 0.152447]   <-- 仍等于进料
Feed               0.050873  [0.787525, 0.212475]
```

**分相没有发生，Heavy_Liquid 依然无效。** 指定 `SimpleLLE` + 升压到 5 bar，把失败点从进料推进到了澄清器，但没有解决它。

## 关键实验：拟合参数本身不产生分相

由于 Automation API 无法注入 NRTL 参数（`ParametersXMLString` 读写均空），我改用**改写 XML 的方式注入**，并验证注入确实到达了求解器：

```
=== ip_fitted.dwxmz ===
   A12="3601.066568632594"
   A21="1747.8505840944874"
   alpha12="0.37982119574538054"
```

（`base_min.dwxmz` 对照组为自动估算的 `A12=2586.46, A21=7.375, alpha=0.2`）

三组参数，同一流程，`SimpleLLE` + 5 bar + 333.15 K：

| 参数组 | A12 | A21 | α | Solved | HEAVY | VAP |
|---|---|---|---|---|---|---|
| **拟合值** | 3601.07 | 1747.85 | 0.3798 | True | **0.00000** | 0.00000 |
| **拟合值（交换）** | 1747.85 | 3601.07 | 0.3798 | True | **0.00000** | 0.00000 |
| 更强的假设值 | 2500 | 4500 | 0.30 | True | **0.00000** | 0.00000 |

**三组全部给出 `HEAVY = 0`，且 LIGHT 组成始终 = 进料组成 `[0.847558, 0.152442]`。**

注意第三行：我刻意给了一组 `A21` 远大于 `A12`（4500 vs 2500）的**强疏水**参数，理论上必然分层 —— **依然不分相**。

## 结论

**问题不在参数，在 DWSIM 9.0.5 的 `Vessel` 单元。**

理由是：

1. 换三组差异极大的 NRTL 参数，结果**逐位相同**（LIGHT 恒为 `[0.847558, 0.152442]`）—— 说明 `Vessel` 的闪蒸**根本没有对参数作出响应**。
2. `PreferredFlashAlgorithmTag = "SimpleLLE"` 已确认写入并读到，但行为与未设置时完全一致，说明该标签对 `Vessel` 无效。
3. 连刻意构造的强分层参数都不分层，排除"参数不够强"的可能。

即：**`Vessel` 在 Legacy 模式下只作单液相 + 可能的气相判断，不执行液液两相搜索。** 这解释了为什么仓库里 13 个 water–MIBK 流程**全部** `ErrorCalculating` 或未收敛 —— 它们都在试图用 `Vessel` 做液液分相，而这条路走不通。

## 建议

**放弃用 `Vessel` 做 LLE 分相**，改用：

1. **`Absorption/Extraction Column`**（`OperationMode = Extractor`）—— 专为萃取设计，
   `ProductDescription` 明确写 *"rigorous simulation of absorption/extraction columns"*，
   且有 `SetInitialMolarCompositionEstimates` / `SetInitialTemperatureEstimates` 等初值接口。
2. 或在**自定义 Python UO** 中直接调用物性包的液液闪蒸
   （`NestedLoopsImmiscible` / `SimpleLLE` 算法类），绕过 `Vessel`。

之前那次 `AbsorptionColumn`（Extractor 模式）的尝试超过 120 秒未收敛，
很可能同样是**初值**问题 —— 现在知道参数注入方法了（改写 XML），
可以把拟合参数和两相初值一起喂进去重试。

## 产物

| 文件 | 说明 |
|---|---|
| `..._SIMPLELLE_5bar.dwxmz` | 打过补丁的流程（已验证补丁生效） |
| `.tmp/dwsim/orig_backup.dwxmz` | 原文件备份 |
| `.tmp/dwsim/ip_fitted.dwxmz` | 注入拟合参数的最小流程 |
| `.tmp/dwsim/ip_swapped.dwxmz` | 交换 A12/A21 |
| `.tmp/dwsim/ip_strong.dwxmz` | 强疏水参数对照 |
