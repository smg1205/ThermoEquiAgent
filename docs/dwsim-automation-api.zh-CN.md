# DWSIM 自动化泡点求解 API（复用手册）

> 本文记录用 **DWSIM 8 的 .NET Automation 接口（pythonnet）** 自动创建流程、触发求解、读取泡点结果
> 的方法与踩坑经验。基于 `scripts/dwsim_ipa_bubble.py` 的实测（2-丙醇-水，760 mmHg，5 个组成跑通）。
> 目标：以后对任意体系复用，不必重新探查 API。

---

## 1. 前置条件

| 依赖 | 说明 |
|---|---|
| DWSIM | 本机装于 `C:\Users\34861\AppData\Local\DWSIM`（.env 的 `DWSIM_HOME`）|
| pythonnet | 需在运行环境可 `import clr`（受限沙箱会报"拒绝访问"，请在普通终端跑）|
| `thermo_engine.dwsim_export` | 项目已封装了创建 flowsheet、组件名映射、保存等基础函数 |

**自动化注册**（首次在机器上时）：
在 DWSIM 安装目录运行 `automation_reg.bat`（用 .NET Framework 4.x 的 regasm 注册 DWSIM 程序集）。

---

## 2. 核心 API（DWSIM 8 实测）

### 2.1 获取 Automation 实例（易错点！）
```python
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()   # 返回 (Automation3类, ObjectType类)
automation = factory()                              # ★ 必须先实例化！Factory 返回的是类
```
> **坑**：`_automation_factory()` 返回的是 **`Automation3` 类 / `ObjectType` 枚举类**，不是实例。
> 直接 `automation.CreateFlowsheet()`（在类上调用）会报 `not enough arguments`。
> **必须 `factory()` 得到实例后**才能调用。

### 2.2 创建 flowsheet 与对象
```python
fs = automation.CreateFlowsheet()          # 无参即可（Overloads 仅此一个无参重载）
fs.AddCompound("Isopropanol")              # 组分键名大小写敏感
fs.AddCompound(ded._dwsim_compound_name("water"))  # → "Water"
ded._add_property_package(fs, "NRTL")      # 物性包：NRTL / UNIQUAC / Wilson ...
feed   = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
sep    = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
vapor  = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
liquid = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
```
> **2-丙醇在 DWSIM 的键名是 `Isopropanol`**（不是 `2-Propanol`）。不同版本可能不同，
> 可用候选键回退：`["Isopropanol","2-Propanol","Propan-2-ol","Isopropyl alcohol"]`。

### 2.3 设置进料条件
```python
feed_stream = ded._simulation_object(feed)   # 包装 GetAsObject()
feed_stream.SetTemperature(t_c + 273.15)     # K
feed_stream.SetPressure(760.0 * 133.322)      # Pa（mmHg→Pa）
feed_stream.SetMolarFlow(1.0)
feed_stream.SetOverallComposition(ded._composition_argument([x_ipa, 1 - x_ipa]))
```

### 2.4 连线
```python
fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, liquid.GraphicObject, 1, 0)
```

### 2.5 ★ 触发计算（关键）
```python
automation.CalculateFlowsheet2(fs)   # DWSIM 8 正确入口
```
> **坑**：`CalculateFlowsheet`（无后缀）签名不同会报
> `TypeError: No method matches given arguments: (<class 'IFlowsheet'>)`。
> 用 **`CalculateFlowsheet2`**（或 3/4）正确。

### 2.6 读取泡点结果
```python
v = ded._simulation_object(vapor)
vapor_flow = float(v.GetMolarFlow())          # 泡点判定依据
comp = list(v.GetOverallComposition())         # 汽相摩尔组成 [comp0, comp1, ...]
T = float(v.GetTemperature())                  # K（这里 = 进料温度）
```

---

## 3. 泡点判定逻辑

用**汽相产品流股的摩尔流量**判定泡点（DWSIM Flash 是"给定 T、P → 汽相分率"）：

- `VaporFlow = 0` → 全液相（温度 **低于** 泡点）
- `VaporFlow > 0` → 开始出现汽相（温度 ≥ 泡点）
- **泡点温度 = VaporFlow 首次从 0 变 >0 的温度**

对固定组成 x，在温度区间二分扫描：
```python
lo, hi = 60.0, 105.0            # °C
while hi - lo > 1e-6:
    mid = 0.5 * (lo + hi)
    build_flowsheet(x, mid); CalculateFlowsheet2(fs)
    if vapor.GetMolarFlow() > 0:
        Tb, y = mid, vapor.GetOverallComposition()
        hi = mid                 # 泡点以上 → 下调上界
    else:
        lo = mid                 # 全液 → 上调下界
```
在泡点温度读 `vapor.GetOverallComposition()` 即泡点汽相组成。

> 实测（2-丙醇-水 760 mmHg）：x=0.5 → T_bubble ≈ 81.92 °C，y_IPA ≈ 0.644，与实验/ThermoFormer 吻合。

---

## 4. 打印噪音处理

每次 `CalculateFlowsheet2` 会打印 DWSIM 的 **Ipopt 公告**（大框 + `Estimated NRTL IP set for ...`）。
这是 DWSIM 库内部 `Console.WriteLine`，Python 端难完全抑制，**属正常噪音，不影响结果**。
如需干净的日志，运行时重定向：
```powershell
python scripts/dwsim_ipa_bubble.py > docs\_run_log.txt 2>&1
Get-Content docs\_run_log.txt | Select-String -Pattern "x_IPA|wrote|Error"
```

---

## 5. 关键 API 名速查表（DWSIM 8 实测）

| 用途 | 方法 | 备注 |
|---|---|---|
| Automation 实例 | `factory()` (`Automation3` 实例) | 类需实例化 |
| 建流程 | `CreateFlowsheet()` | 无参 |
| 加组分 | `AddCompound(canonical_name)` | 大小写敏感 |
| 加物性包 | `_add_property_package(fs, "NRTL")` | |
| 加单元 | `AddObject(ObjectType.Vessel/MaterialStream, x, y, name)` | |
| 设温度 | `SetTemperature(K)` | |
| 设压力 | `SetPressure(Pa)` | 注意单位 |
| 设组成 | `SetOverallComposition(double[])` | 用 `_composition_argument` |
| 连线 | `ConnectObjects(g1, g2, fromPort, toPort)` | |
| **计算** | **`CalculateFlowsheet2(fs)`** | 勿用无后缀版 |
| 读汽相流量 | `v.GetMolarFlow()` | 泡点判定 |
| 读汽相组成 | `v.GetOverallComposition()` | 返回 double[] |
| 读温度 | `v.GetTemperature()` | K |

---

## 6. 复用指引

- 直接改 `scripts/dwsim_ipa_bubble.py` 里的 `IPA_CANDIDATES`、`XS`、P、温度扫描范围即可跑新体系。
- 物性包按体系选：极性/非理想用 **NRTL / UNIQUAC**，非极性轻烃可用 **Peng-Robinson**。
- 组分键名用 `ded._dwsim_compound_name(name)` 规范化；未知名称用候选回退 + `AddCompound` 实测。
- **受限/沙箱环境报 `Python.Runtime ... 拒绝访问`** 是环境权限问题，换普通终端即可，不是代码问题。

---

## 7. 相关文件

- 参考脚本：`scripts/dwsim_ipa_bubble.py`（自动泡点）、`scripts/generate_dwsim_ipa_water.py`（生成单点 .dwxmz）
- 封装层：`thermo_engine/dwsim_export.py`（CreateFlowsheet / 组件映射 / 保存等）
- 结果示例：`docs/dwsim_ipa_bubble.csv`
