# DWSIM Automation Bubble Point Solving API (Reuse Manual)

> This document records the method and the pitfalls of using **DWSIM 8's .NET Automation interface (pythonnet)** to
> automatically create a flowsheet, trigger a solve, and read bubble point results. It is based on
> actual testing of `scripts/dwsim_ipa_bubble.py` (2-propanol–water, 760 mmHg, 5 compositions run successfully).
> Goal: reuse for any system later without re-exploring the API.

---

## 1. Prerequisites

| Dependency | Description |
|---|---|
| DWSIM | Installed locally at `C:\Users\34861\AppData\Local\DWSIM` (the `.env` `DWSIM_HOME`)|
| pythonnet | `import clr` must work in the runtime environment (a restricted sandbox reports "access denied"; run it in an ordinary terminal)|
| `thermo_engine.dwsim_export` | The project already wraps basic functions for creating a flowsheet, mapping component names, saving, etc. |

**Automation registration** (the first time on a machine):
Run `automation_reg.bat` in the DWSIM installation directory (this registers the DWSIM assemblies
with .NET Framework 4.x regasm).

---

## 2. Core API (verified on DWSIM 8)

### 2.1 Obtaining the Automation instance (common pitfall!)
```python
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()   # returns (Automation3 class, ObjectType class)
automation = factory()                              # ★ you must instantiate it first! Factory returns a class
```
> **Pitfall**: `_automation_factory()` returns the **`Automation3` class / `ObjectType` enum class**, not an instance.
> Calling `automation.CreateFlowsheet()` directly (on the class) reports `not enough arguments`.
> You **must call `factory()` to get an instance** before you can call it.

### 2.2 Creating a flowsheet and objects
```python
fs = automation.CreateFlowsheet()          # no arguments needed (Overloads has only this one no-arg overload)
fs.AddCompound("Isopropanol")              # component key names are case-sensitive
fs.AddCompound(ded._dwsim_compound_name("water"))  # → "Water"
ded._add_property_package(fs, "NRTL")      # property package: NRTL / UNIQUAC / Wilson ...
feed   = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
sep    = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
vapor  = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
liquid = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
```
> **The key name for 2-propanol in DWSIM is `Isopropanol`** (not `2-Propanol`). It may differ between versions;
> candidate key fallback is available: `["Isopropanol","2-Propanol","Propan-2-ol","Isopropyl alcohol"]`.

### 2.3 Setting feed conditions
```python
feed_stream = ded._simulation_object(feed)   # wraps GetAsObject()
feed_stream.SetTemperature(t_c + 273.15)     # K
feed_stream.SetPressure(760.0 * 133.322)      # Pa (mmHg→Pa)
feed_stream.SetMolarFlow(1.0)
feed_stream.SetOverallComposition(ded._composition_argument([x_ipa, 1 - x_ipa]))
```

### 2.4 Connecting
```python
fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, liquid.GraphicObject, 1, 0)
```

### 2.5 ★ Triggering calculation (the key step)
```python
automation.CalculateFlowsheet2(fs)   # the correct entry point for DWSIM 8
```
> **Pitfall**: `CalculateFlowsheet` (without the suffix) has a different signature and reports
> `TypeError: No method matches given arguments: (<class 'IFlowsheet'>)`.
> Use **`CalculateFlowsheet2`** (or 3/4) instead.

### 2.6 Reading bubble point results
```python
v = ded._simulation_object(vapor)
vapor_flow = float(v.GetMolarFlow())          # basis for the bubble point decision
comp = list(v.GetOverallComposition())         # vapor-phase molar composition [comp0, comp1, ...]
T = float(v.GetTemperature())                  # K (here = feed temperature)
```

---

## 3. Bubble point decision logic

Determine the bubble point from the **molar flow rate of the vapor product stream** (the DWSIM Flash is
"given T, P → vapor fraction"):

- `VaporFlow = 0` → all liquid (temperature **below** the bubble point)
- `VaporFlow > 0` → vapor phase has begun to appear (temperature ≥ bubble point)
- **Bubble point temperature = the temperature at which VaporFlow first changes from 0 to >0**

For a fixed composition x, bisection scan over the temperature interval:
```python
lo, hi = 60.0, 105.0            # °C
while hi - lo > 1e-6:
    mid = 0.5 * (lo + hi)
    build_flowsheet(x, mid); CalculateFlowsheet2(fs)
    if vapor.GetMolarFlow() > 0:
        Tb, y = mid, vapor.GetOverallComposition()
        hi = mid                 # above bubble point → lower the upper bound
    else:
        lo = mid                 # all liquid → raise the lower bound
```
Reading `vapor.GetOverallComposition()` at the bubble point temperature gives the bubble point vapor composition.

> Measured (2-propanol–water at 760 mmHg): x=0.5 → T_bubble ≈ 81.92 °C, y_IPA ≈ 0.644, consistent with experiment/ThermoFormer.

---

## 4. Handling print noise

Every `CalculateFlowsheet2` call prints DWSIM's **Ipopt notice** (a large box + `Estimated NRTL IP set for ...`).
This comes from `Console.WriteLine` inside the DWSIM library and is hard to suppress entirely from the
Python side; **it is normal noise and does not affect the results**.
If you need clean logs, redirect at run time:
```powershell
python scripts/dwsim_ipa_bubble.py > docs\_run_log.txt 2>&1
Get-Content docs\_run_log.txt | Select-String -Pattern "x_IPA|wrote|Error"
```

---

## 5. Quick reference table of key API names (verified on DWSIM 8)

| Purpose | Method | Notes |
|---|---|---|
| Automation instance | `factory()` (`Automation3` instance) | the class must be instantiated |
| Create flowsheet | `CreateFlowsheet()` | no arguments |
| Add compound | `AddCompound(canonical_name)` | case-sensitive |
| Add property package | `_add_property_package(fs, "NRTL")` | |
| Add unit | `AddObject(ObjectType.Vessel/MaterialStream, x, y, name)` | |
| Set temperature | `SetTemperature(K)` | |
| Set pressure | `SetPressure(Pa)` | mind the units |
| Set composition | `SetOverallComposition(double[])` | use `_composition_argument` |
| Connect | `ConnectObjects(g1, g2, fromPort, toPort)` | |
| **Calculate** | **`CalculateFlowsheet2(fs)`** | do not use the suffix-less version |
| Read vapor flow | `v.GetMolarFlow()` | bubble point decision |
| Read vapor composition | `v.GetOverallComposition()` | returns double[] |
| Read temperature | `v.GetTemperature()` | K |

---

## 6. Reuse guidance

- To run a new system, simply change `IPA_CANDIDATES`, `XS`, P, and the temperature scan range in `scripts/dwsim_ipa_bubble.py`.
- Choose the property package by system: use **NRTL / UNIQUAC** for polar/non-ideal systems, and **Peng-Robinson** for non-polar light hydrocarbons.
- Normalize component key names with `ded._dwsim_compound_name(name)`; for unknown names use candidate fallback + an actual `AddCompound` test.
- **A `Python.Runtime ... access denied` error in a restricted/sandboxed environment** is an environment permission issue; switch to an ordinary terminal. It is not a code problem.

---

## 7. Related files

- Reference scripts: `scripts/dwsim_ipa_bubble.py` (automatic bubble point), `scripts/generate_dwsim_ipa_water.py` (generate a single-point .dwxmz)
- Wrapper layer: `thermo_engine/dwsim_export.py` (CreateFlowsheet / component mapping / saving, etc.)
- Example results: `docs/dwsim_ipa_bubble.csv`
