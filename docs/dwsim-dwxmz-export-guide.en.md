# Generation, Format, and Export Data Requirements of DWSIM Project Files (`.dwxmz`)

## Abstract

This document explains the mechanism by which ThermoAgent generates DWSIM project files, dissects the `.dwxmz` format, and lists the data and validation constraints required by each export path. The system does not serialize the project file directly; instead it drives the locally installed DWSIM through the .NET Automation interface to build the flowsheet and write it to disk. The authority on the file format is DWSIM itself; this project's responsibility is limited to supplying components, states, and design parameters while working around version differences in the automation interface. The format description is based on inspection of actual exported files, and the interface and validation rules follow the current implementations of `thermo_engine/dwsim_export.py`, `agent/extractive_distillation.py`, and `schemas/column_design.py`.

> Colloquially, a "dwism file" means a DWSIM file; `dwism` is a spelling variant explicitly accepted in the code (`_DWSIM_FILE_MARKERS`).

---

## 1 Generation mechanism

### 1.1 Overall architecture

**This system does not generate the byte content of `.dwxmz`; it calls the .NET Automation interface of the locally installed DWSIM, and DWSIM itself performs the flowsheet construction and serialization.** The Python side is responsible only for intent recognition, parameter parsing, deterministic column design, and interface invocation. Its engineering implication is: for questions involving format details, the authoritative source is the DWSIM installation itself; what this project must handle is the differences between versions in the method names exposed by the automation interface, property writability, and silent-failure behavior.

```
Natural language request
  └─ agent/extractive_distillation.py      intent recognition, parameter parsing, column design orchestration
       └─ thermo_engine/column_design.py   deterministic shortcut method (Fenske–Underwood–Gilliland)
            └─ thermo_engine/dwsim_export.py    calls the DWSIM Automation API
                 └─ DWSIM .NET assemblies (pythonnet / clr) → .dwxmz
```

### 1.2 Assembly loading and flowsheet construction

Assemblies are loaded lazily (`_automation_factory`); `import clr` only happens when an export is initiated, so hosts without DWSIM installed can still run the calculation core. The loading sequence is: read `.env` to obtain `DWSIM_HOME`; verify that `<DWSIM_HOME>/DWSIM.Automation.dll` exists; point `TEMP`/`TMP` at `DWSIM_TEMP_DIR` (default `<cwd>/.tmp/dwsim`); `import clr` and append the installation directory to `sys.path`; load the automation assembly with `clr.AddReference`; import `Automation3` and `ObjectType`.

Note that this function returns **two classes** rather than instances, and the caller must then call `factory()` to complete instantiation. Taking a TP flash as an example, the construction process is:

```python
automation = factory()                     # instantiate Automation3
flowsheet  = automation.CreateFlowsheet()  # create an empty flowsheet
flowsheet.AddCompound("n-Heptane")         # add a component (key names are strictly case-sensitive)
_add_property_package(flowsheet, "NRTL")   # add a property package
feed  = flowsheet.AddObject(ObjectType.MaterialStream, 0,   0, "Feed")
sep   = flowsheet.AddObject(ObjectType.Vessel,        250,  0, "Equilibrium Flash")
vapor = flowsheet.AddObject(ObjectType.MaterialStream, 500, -80, "Vapor Product")
fs = _simulation_object(feed)              # unwrap to the concrete type via GetAsObject()
fs.SetTemperature(temperature_K)           # K
fs.SetPressure(pressure_kPa * 1000.0)      # kPa → Pa (DWSIM receives Pa internally)
fs.SetMolarFlow(1.0)                       # molar flow; TP flash is normalized to 1.0
fs.SetOverallComposition(Array[Double](composition))
flowsheet.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
_save_flowsheet_via_temp(automation, flowsheet, destination)
```

Three details require attention: `AddCompound` key names are strictly case-sensitive and must match the DWSIM compound dictionary; this project uses `_DWSIM_COMPOUND_MAP` to normalize model-side names (`heptane`, `2-propanol`, `DMSO`) into dictionary keys (`n-Heptane`, `Isopropanol`, `Dimethyl sulfoxide`), and a mismatched key name throws `KeyNotFoundException` and aborts the export; DWSIM 9 returns generic interface objects that must be unwrapped via `GetAsObject()` before state can be set; pressure must be converted to Pa, and the composition must be converted to `System.Array[Double]`.

Save methods differ across versions: `_save_flowsheet` tries `SaveFlowsheet2`, `SaveFlowsheet(_, _, True)`, `SaveFlowsheet(_, _)`, `SaveToXML`, `SaveToFile` in order; on encountering `UnauthorizedAccessException`, the outer `_save_flowsheet_via_temp` instead saves to a temporary directory first and copies back to the target path.

### 1.3 Timing of solving and writing to disk

DWSIM saves the **current state**, which gives rise to two kinds of artifacts. The first is a **file containing only the structural skeleton**: when no solve has been triggered before saving, every object's `<Calculated>` in the file is `false`, which shows in the interface as an empty phase fraction and zero product flow rates (the sample in this document is of this kind: all 8 occurrences of `<Calculated>` and 3 occurrences of `<AtEquilibrium>` are `false`). TP flash and column exports use this mode by default. The second is a **file carrying solve results**: LLE phase-splitting exports explicitly call `CalculateFlowsheet4` before saving and read back the two-phase flow rates and compositions, so the written file already contains real results.

When a solve fails, the file is still written out (the structure is correct and can be recomputed in the interface), but the system emits a warning and does not describe an unsplit result as converged. Note also that `CalculateFlowsheet2` **suppresses solve errors**, and only `CalculateFlowsheet4` reports them — deciding "whether the column solved successfully" must be based on the latter.

### 1.4 Additional assembly for rigorous distillation columns

Assembling a rigorous column is more complex than a flash. In addition to the general steps, the following items must be written, each of which corresponds to an interface defect that has been worked around.

| Assembly item | API used | Consequence if not performed |
|:--|:--|:--|
| Number of stages | `SetNumberOfStages(n)` | This method must be used; merely assigning `NumberOfStages` does not rebuild the `Stages` list, and DWSIM indexes the old list with the new count and throws `ArgumentOutOfRangeException: index out of range` |
| Feed stage binding | `ConnectFeed` first, then `SetStreamFeedStage` | The order is fixed and cannot be swapped; with only the former the stage number reads back as `-1`, with only the latter it throws `NullReferenceException` |
| Two column specifications | `SetCondenserSpec` / `SetReboilerSpec`, or write `Specs["C"]`/`Specs["R"]` | The rigorous column is under-specified; the reboiler specification defaults to 0 and reports `Failed to fulfill mass balance ... Relative Error = ~1.0` |
| Solver configuration | `SolvingMethodName` / `MaxIterations` | By default it cold-starts with the Wang–Henke method and 100 iterations, and a tall column will inevitably report an iteration limit |
| Initial profiles | `SetInitialTemperature/LiquidMolarFlow/VaporMolarFlowEstimates` | A missing initial profile is the primary cause of column non-convergence |
| Pressure drop | `ColumnPressureDrop` (Pa) | When not written, the DWSIM default value is used |
| Product connections | `ConnectDistillate` / `ConnectBottoms` | Using the generic `ConnectObjects` raises no error but the connection is not established, so the solve reports success while the product flow rates are zero |

The setter for `SolvingMethodName` performs no validation; an incorrect assignment is only exposed at solve time as `Unable to find column solver with name '...'`. Only two names were resolved in local testing: `Wang-Henke Bubble-Point (BP) Solver` and `Modified Wang-Henke Bubble-Point (MBP) Solver`.

### 1.5 Generation flow overview

```
① Intent recognition   contains dwsim/dwism/dwxmz/export/download, and a system can be identified
② Parameter parsing    components, composition, flow, pressure, purity, recovery, etc.; missing items take defaults
③ Component mapping    model name → DWSIM compound dictionary key; no mapping reports missing_parameters
④ Column design        the deterministic shortcut method gives stages, reflux ratio, feed stage, distillate and bottoms temperatures
⑤ Assembly loading     DWSIM.Automation.dll + Automation3 + ObjectType
⑥ Flowsheet construction CreateFlowsheet → AddCompound → property package → AddObject → set state → connect
⑦ Column parameter writing  number of stages, feed stage, the two specifications, solver, initial profiles, pressure drop
⑧ LLE solve first      CalculateFlowsheet4, read back the two-phase results
⑨ Saving               try five save methods in order, with a temporary-directory relay when necessary
⑩ Provide download     /api/export/extractive/<file-id>.dwxmz
```

---

## 2 File format

### 2.1 Container and top-level structure

`.dwxmz` is the native project file of DWSIM 9, and the container is a **ZIP archive** (magic number `50 4B 03 04`) containing exactly **one GUID-named XML** inside. An actual exported binary distillation column file was inspected as follows:

```
File        hep_nonane_web.dwxmz                          32,944 bytes
Archive content  4b95689b-7955-40a2-adf3-46bc8b5333c0.xml     265,082 bytes
```

That is, `.dwxmz` = ZIP(a single `{GUID}.xml`). The XML is uncompressed and can be unpacked and read directly with any ZIP tool. The older format `.dwrsd` is not used; all exports force the `.dwxmz` extension, otherwise a `ValueError` is thrown.

After unpacking, the root node is `<DWSIM_Simulation_Data>`, with child nodes grouped as follows. Among them `SimulationObjects` is the core, while the rest are mostly interface layout, result caches, or nodes left empty by this project (`ReactionSets`, `Reactions`, `DynamicsManager`, etc.).

| Top-level node | Role |
|---|---|
| `SimulationObjects` | property and state data of all unit operations and streams |
| `Compounds` | component list; the order is the global composition vector order |
| `PropertyPackages` | property packages and their internal options |
| `GraphicObjects` | canvas position, size, and connections |
| `GeneralInfo` / `Settings` / `Results` | meta information, global settings, cache of the last solve results |

### 2.2 Key nodes

Under `Compounds`, each component corresponds to an element of the same name; its name is the DWSIM compound dictionary key, and the order matches the composition vector. In `PropertyPackages`, `AutoEstimateMissingNRTLUNIQUACParameters` deserves particular attention: when the built-in library lacks a particular pair of binary interaction parameters, this option triggers automatic estimation, and the result should be reviewed. The sample records UNIQUAC:

```xml
<PropertyPackage>
  <Type>DWSIM.Thermodynamics.PropertyPackages.UNIQUACPropertyPackage</Type>
  <Tag>UNIQUAC</Tag>
  <AutoEstimateMissingNRTLUNIQUACParameters>true</AutoEstimateMissingNRTLUNIQUACParameters>
</PropertyPackage>
```

Under `SimulationObjects`, each object is a child node named "type prefix–GUID", such as `DC-3b76714b-...` (column) and `MAT-40aadbdb-...` (stream). The main fields of a material stream are `Type` (fully qualified class name), `SpecType` (e.g. `Temperature_and_Pressure`), `CompositionBasis` (e.g. `Molar_Fractions`), `DefinedFlow` (e.g. `Mole`), `ForcePhase`, `PreferredFlashAlgorithmTag`, `Calculated`, `AtEquilibrium`. The column object additionally contains `Specs` (key `C` for condenser, `R` for reboiler), `SType`, `SpecValue`, `SpecUnit`, as well as initial-value arrays `T0`/`Tf`, `V0`/`Vf`, `L0`/`Lf`, `P0`.

The values written to disk for the column configuration in the sample match the design values, which can serve as a criterion for assembly correctness and also confirms the two independent specifications required in section 3.2.2:

```
NumberOfStages        20
SType                 Stream_Ratio  |  Product_Molar_Flow_Rate
SpecValue             1.937         |  0.5
SpecUnit              (empty)       |  mol/s
SolvingMethodName     Naphtali-Sandholm      MaxIterations  500
ColumnPressureDrop    5000 (Pa)
```

### 2.3 Property package name mapping

| Model-side name | DWSIM property package name |
|---|---|
| `ideal/raoult` | `Raoult's Law` |
| `peng-robinson` / `phasepy/peng-robinson` / `clapeyron/peng-robinson` | `Peng-Robinson (PR)` |
| `nrtl` / `wilson` / `uniquac` | the same-named string |

Column exports are mapped through `_COLUMN_PROPERTY_PACKAGES`, supporting `NRTL`, `UNIQUAC`, `Wilson`, `Peng-Robinson (PR)`, `Ideal` (→ `Raoult's Law`).

---

## 3 Export data requirements

### 3.1 Environment prerequisites

| Condition | Description | Error when missing |
|---|---|---|
| DWSIM installed locally | the directory must contain `DWSIM.Automation.dll` | `DWSIM_HOME is not configured.` / `... dll was not found ...` |
| `pythonnet` | must be able to `import clr` | `pythonnet is not installed.` |
| Full process permissions | cannot run in a restricted sandbox; use an ordinary terminal | `Python.Runtime ... access denied` |

```dotenv
DWSIM_HOME=C:\Users\<user>\AppData\Local\DWSIM   # required
DWSIM_TEMP_DIR=E:\temp\dwsim                     # optional, default <cwd>/.tmp/dwsim
EXTRACTIVE_EXPORT_DIR=E:\exports\dwsim           # optional, default data/exports/extractive
```

The first time you use the automation interface, you must run `automation_reg.bat` in the DWSIM installation directory to register the assemblies.

### 3.2 Topology and data of each export path

| Export function | System | Flowsheet structure | Default property package |
|---|---|---|---|
| `export_dwsim_flowsheet` | any | `Feed → Vessel(TP flash) → Vapor / Liquid` | determined by the run's `model_name` |
| `export_dwsim_binary_column` | binary | rigorous distillation column + single feed + distillate/bottoms | UNIQUAC |
| `export_dwsim_extractive_column` | ethanol/water/entrainer | rigorous column + feed + entrainer + two products | specified by the design |
| `export_generic_extractive_column` | any ternary | rigorous column + feed + entrainer + two products | NRTL |
| `export_dwsim_binary_lle_flowsheet` | binary | `Feed → Vessel_LLE → Vapor / Light_Liquid / Heavy_Liquid` | NRTL |
| `export_dwsim_ternary_lle_flowsheet` | ternary | same as binary LLE | NRTL |
| `export_dwsim_lle_extraction` | ternary | `Feed + Solvent → Liquid-Liquid Extractor → Raffinate / Extract` | NRTL |

**TP flash** (`export_dwsim_flowsheet`) takes its data from `RunRecord` and accepts no extra parameters; the values follow this priority: components from `input_snapshot.components[].name`; feed composition from `conditions.feed_composition` → `conditions.liquid_composition` → `conditions.vapor_composition` → the liquid or vapor composition of `points[0]`; temperature from `result.temperature_K` → `conditions` → `points[0]`; pressure and model name likewise. The constraints are that the composition vector length equals the number of components, and the molar flow is normalized to 1.0.

**Binary distillation column** (`export_dwsim_binary_column`): all required parameters are supplied by the deterministic shortcut method: `components`(2), `feed_composition`(2), `feed_flow_mol_s`, `feed_temperature_K`, `operating_pressure_kPa`, `stages`, `minimum_stages`, `reflux_ratio`, `minimum_reflux_ratio`, `feed_stage`, `condenser_temperature_K`, `reboiler_temperature_K`, `destination`; optional `distillate_flow_mol_s`, `bottoms_flow_mol_s`, `pressure_drop_kPa`, `solving_method` (default `Naphtali-Sandholm`), `max_iterations` (default 500). Validation is `feed_stage ∈ [1, stages)`, and a rigorous column must have exactly two specifications. The chat entry point defaults to a flow of 1 mol/s, a pressure of 101.325 kPa, a distillate purity of 0.995, a recovery of 0.98, and a pressure drop of 5 kPa; the feed stage is taken as `max(2, theoretical stages − 1)`, and the bottoms as half the feed.

**Extractive distillation column** (`export_dwsim_extractive_column`) requires only one `ExtractiveColumnDesign`, with everything else read from that design: entrainer, property package, feed temperature/pressure/flow and composition, plus `theoretical_stages`, `reflux_ratio`, `feed_stage`, `entrainer_stage`. **Generic ternary extractive distillation** (`export_generic_extractive_column`) requires `light`, `heavy`, `entrainer`, `feed_composition`(2), and the various state and design parameters, with optional `property_package` (default NRTL) and others; validation is `1 ≤ feed_stage < stages`, `1 ≤ entrainer_stage < feed_stage` (the entrainer must be above the feed stage), and `D + B = feed + entrainer` (tolerance 1e-6) — the last is the most error-prone, because if D and B are taken from a different feed basis, DWSIM will report an iterative material balance failure that is hard to interpret directly.

**Binary and ternary LLE phase splitting** (`export_dwsim_*_lle_flowsheet`) require `components`(2 or 3), `feed_composition`, `temperature_K` (must be positive), and `destination`; optional `pressure_kPa` (default 101.325), `feed_flow_mol_s` (default 1.0), `property_package` (default NRTL), `split_out`. Validation is a strict match between the number of components and the composition vector length, fractions summing to 1 (tolerance 1e-6), no negative fractions for ternary, and pairwise distinct components. The temperature must be written **both** to the feed stream and to the vessel's `FlashTemperature`/`FlashPressure`; otherwise the `Vessel` will flash at its own default value of 298.15 K, computing the phase split at the wrong temperature.

**Liquid-liquid extractor** (`export_dwsim_lle_extraction`) requires `components` (exactly 3: two solutes plus one solvent), `feed_composition` (exactly 2), `solvent` (must belong to `components`), `solvent_ratio`, `feed_flow_mol_s`, `feed_temperature_K`, `feed_pressure_kPa`, `destination`; optional `property_package` (default NRTL) and preset product parameters `raffinate_*`/`extract_*`. Validation is that the number of components must be 3, the feed fractions must be 2, and the fractions sum to 1 (tolerance 1e-6).

### 3.3 Interface export path

The trigger condition is that "a system is identified" and "an explicit export intent" are satisfied at the same time; the intent keywords are `dwsim`, `dwism`, `dwxmz`, `导出`, `下载`, `export`, `download`. Examples:

```
2-propanol-water binary VLE distillation column, x_IPA=0.3, 1 atm, export DWSIM file
n-butanol-water binary liquid-liquid equilibrium, composition 0.3/0.7, 313.15 K, export DWSIM
ethanol, ethyl acetate, water ternary LLE, composition 0.129/0.188/0.683, 298.15 K, export DWSIM
n-propyl acetate 40%, ethyl acetate 60%, using dimethyl sulfoxide as solvent, solvent ratio 1.5, export DWSIM liquid-liquid extraction file
```

If a component in the system has no DWSIM name mapping, the system returns `status=missing_parameters` with `missing_parameters=["dwsim_compound_mapping"]`, rather than forcing a guessed name into the file. On successful export it returns `dwsim_file_uri` and `file_id`, downloaded via `GET /api/export/extractive/<file-id>.dwxmz`; a generic run export is `GET /api/runs/{run_id}/export?format=dwsim`.

### 3.4 Failure modes

| Symptom | Cause | Handling |
|---|---|---|
| `DWSIM_HOME is not configured.` | environment variable not configured | point it at the real installation directory in `.env` |
| `DWSIM.Automation.dll was not found` | wrong path | confirm that the DLL really is in that directory |
| `Python.Runtime ... access denied` | running in a restricted sandbox | switch to an ordinary terminal; not a code problem |
| `AddCompound` throws `KeyNotFoundException` | component name case or mapping mismatch | add the mapping in `_DWSIM_COMPOUND_MAP` |
| `Failed to fulfill mass balance ... Relative Error = ~1.0` | the rigorous column is missing its second specification, or the feed is not bound to a stage | write the bottoms product flow specification; bind the stage with `ConnectFeed` plus `SetStreamFeedStage` |
| `Solver reached the maximum number of iterations` | cold start with no initial values | write initial profiles and raise the iteration limit |
| `ArgumentOutOfRangeException: index out of range` | only `NumberOfStages` was changed; `Stages` was not rebuilt | `SetNumberOfStages(n)` must be used |
| Product flow rates are zero and results are default values | saved without solving, or the solve failed silently | use `CalculateFlowsheet4` |
| Ternary LLE heavy phase is empty | the automation path did not enable the immiscible flash kernel | in the interface switch to `Nested Loops (Immiscible)`, `(VLLE)`, or `Simple LLE` and recompute |

### 3.5 Acceptance points

Component names and order are correct and the fractions sum to 1; temperature is in K and pressure is converted by the interface from kPa to Pa; the property package matches the task type, and LLE preferentially uses NRTL or UNIQUAC parameters with supporting evidence; inlet and outlet connections are complete, and the feed enters the correct port or stage; the VLE column has two independent specifications and the convergence algorithm has been recorded; in LLE files the `Heavy_Liquid` or `Extract` flow rate should not be zero without explanation; the DWSIM status should show calculated or converged with the material balance within the allowed error. Before use in engineering design, it must be cross-validated against experimental data or ThermoFormer results.

---

## 4 Scientific boundaries

This project **does not compute LLE numbers, nor does it construct binary interaction parameters**. LLE-type exports are responsible only for the flowsheet structure, feed states, and property package; the phase split is solved by DWSIM based on its built-in binary parameters (or parameter estimation mechanism). All equilibrium data must originate from `thermo_engine` and pass `validate_equilibrium_result`; the language model is responsible only for classification, parsing, orchestration, and explanation.

## References

`thermo_engine/dwsim_export.py` (flowsheet construction, mapping, and saving), `agent/extractive_distillation.py` (request recognition and orchestration), `schemas/column_design.py` (input structures), `apps/api/main.py` (HTTP endpoints); companion documents `docs/dwsim-automation-api.zh-CN.md`, `docs/dwsim-extraction-export.zh-CN.md`; tests `tests/test_dwsim_export.py`, `test_binary_vle_dwsim.py`, `test_binary_lle_dwsim.py`, `test_lle_extraction_export.py`, `test_dwsim_extractive_export.py`.
