# DWSIM Phase-Equilibrium and Extraction Export Guide

This document summarizes the DWSIM `.dwxmz` export capabilities currently available in ThermoAgent. It covers binary and ternary VLE, binary and ternary LLE, liquid-liquid extraction, and extractive distillation. The interfaces and limitations described here follow the current implementations in `thermo_engine/dwsim_export.py` and `agent/extractive_distillation.py`.

## 1. Environment and Output Location

Exporting requires a local DWSIM installation and `pythonnet`. At minimum, configure the following value in the project `.env` file:

```dotenv
DWSIM_HOME=C:\path\to\DWSIM
```

`DWSIM_HOME` must point to the directory containing `DWSIM.Automation.dll`. Optional variables are:

```dotenv
DWSIM_TEMP_DIR=E:\temp\dwsim
EXTRACTIVE_EXPORT_DIR=E:\exports\dwsim
```

When `EXTRACTIVE_EXPORT_DIR` is not configured, files generated through the chat entry point are written to:

```text
data/exports/extractive/
```

The file extension must be `.dwxmz`. After a successful export, the API exposes the file at `/api/export/extractive/<file-id>.dwxmz`.

## 2. Supported Scope

| Type | Components | DWSIM topology | Default property package | Main entry points |
|---|---:|---|---|---|
| VLE distillation | Binary | Rigorous column + one feed + distillate/bottoms | UNIQUAC | `run_binary_distillation` / `export_dwsim_binary_column` |
| VLE extractive distillation | Ternary | Rigorous column + feed + entrainer + two products | Design-specific, usually NRTL/UNIQUAC | `run_extractive_export` / `export_dwsim_extractive_column` |
| LLE phase split | Binary | `Vessel_LLE` + light liquid/heavy liquid/vapor | NRTL | `run_binary_lle_export` / `export_dwsim_binary_lle_flowsheet` |
| LLE phase split | Ternary | `Vessel_LLE` + light liquid/heavy liquid/vapor | NRTL | `run_generic_ternary_lle_export` / `export_dwsim_ternary_lle_flowsheet` |
| Liquid-liquid extraction | Ternary | Feed + solvent + `Liquid-Liquid Extractor` + raffinate/extract | NRTL | `run_lle_extraction_export` / `export_dwsim_lle_extraction` |

VLE means vapor-liquid equilibrium, while LLE means liquid-liquid equilibrium. Extractive distillation is a VLE column process. Liquid-liquid extraction is an LLE process. The two export paths are not interchangeable.

## 3. Binary VLE Export

### 3.1 Systems Supported by the Chat Entry Point

The chat entry point currently recognizes:

- 2-propanol/water
- Methanol/water
- Ethanol/water
- Ethanol/toluene

The request must include the binary feed composition. If only one component mole fraction is supplied, the other fraction is calculated as `1 - x`. Parameters omitted from the request use the current design defaults: feed flow `1 mol/s`, pressure `101.325 kPa`, distillate purity `0.995`, recovery `0.98`, and total column pressure drop `5 kPa`.

Example:

```text
Export a DWSIM file for a 2-propanol/water binary VLE distillation column,
x_IPA=0.3, at 1 atm.
```

The output file contains a rigorous distillation column, feed stream, distillate stream, and bottoms stream. The condenser specification is the reflux ratio, while the reboiler specification is the bottoms molar flow. The feed stage, stage count, temperature estimates, and flow estimates are written into the file.

### 3.2 Selecting the Convergence Algorithm Manually

The binary VLE exporter does not lock the convergence algorithm. For a direct code call, select it with the `solving_method` argument:

```python
from pathlib import Path

from thermo_engine.dwsim_export import export_dwsim_binary_column

export_dwsim_binary_column(
    components=["isopropanol", "water"],
    feed_composition=[0.3, 0.7],
    feed_flow_mol_s=1.0,
    feed_temperature_K=354.60,
    operating_pressure_kPa=101.325,
    stages=19,
    minimum_stages=10.069,
    reflux_ratio=2.694,
    minimum_reflux_ratio=1.924,
    feed_stage=18,
    condenser_temperature_K=355.26,
    reboiler_temperature_K=367.46,
    distillate_flow_mol_s=0.5,
    bottoms_flow_mol_s=0.5,
    pressure_drop_kPa=5.0,
    solving_method="Naphtali-Sandholm",
    max_iterations=500,
    destination=Path("data/exports/extractive/ipa-water-binary-vle.dwxmz"),
)
```

Common choices are:

| Convergence algorithm | Recommended use |
|---|---|
| `Naphtali-Sandholm` | Current default; preferred first choice for a rigorous column with temperature and flow estimates |
| `Wang-Henke (Bubble Point)` | Bubble-point method; suitable for simpler systems but more sensitive to cold starts and initial estimates |
| `Burningham-Otto (Sum Rates)` | Sum-rates method; an alternative when the first two methods do not converge |

The algorithm name must exactly match a `SolvingMethodName` exposed by the installed DWSIM version. If the displayed name differs, use the value shown in the DWSIM GUI.

The algorithm can also be changed after export:

1. Open the `.dwxmz` file in DWSIM.
2. Double-click `Binary Distillation Column`.
3. Open the Solver or Convergence/Solver page.
4. Select the convergence algorithm and set a suitable maximum iteration count, such as `500`.
5. Keep the exported temperature, liquid-flow, and vapor-flow estimates, then run the calculation again.
6. Confirm that distillate and bottoms flows are nonzero, the component balance closes, and the solver status is converged.

The chat entry point currently uses the default `Naphtali-Sandholm` method. For precise algorithm control, call `export_dwsim_binary_column` directly or change the solver manually in the DWSIM GUI after export.

## 4. Ternary VLE and Extractive-Distillation Export

Ternary VLE export is represented as an extractive-distillation column in the current project. Two components form the process feed, while the third is a high-boiling entrainer. The file contains the feed, entrainer stream, rigorous column, distillate stream, and bottoms stream.

Fixed validation system:

```text
Export a DWSIM ternary VLE extractive-distillation file for
ethyl acetate / n-propyl acetate / DMSO.
```

Generic request example:

```text
Use tetrahydrofuran as the entrainer for acetonitrile/toluene extractive
distillation. Use the supplied feed composition, flow, pressure, and entrainer
ratio, and export a DWSIM file.
```

The ethanol/water workflow also supports ethylene glycol or glycerol as the entrainer. A request may explicitly select ThermoFormer as the relative-volatility source. Without that opt-in, the project uses its default deterministic design path.

A rigorous column requires two independent specifications. After opening the exported file, verify that:

- The feed and entrainer enter their intended stages.
- The condenser specification is a reflux ratio or distillate flow.
- The reboiler specification is a bottoms flow, boilup ratio, or heat duty.
- The property package and compound mappings are correct.
- The solver, maximum iteration count, and initial estimates are suitable for the system.

Some DWSIM versions do not expose a stable Automation API for every internal column specification. In that case, the file still retains its structure and feed connections, but the condenser and reboiler specifications must be completed in the GUI before calculation.

## 5. Binary LLE Export

Binary LLE uses the following topology:

```text
Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid
```

Example:

```text
Export a DWSIM file for the binary water/1-butanol LLE system,
composition 0.3/0.7, at 313.15 K.
```

The temperature is written to both the feed and the vessel flash conditions. Pressure defaults to `101.325 kPa`, and the property package defaults to NRTL. The exporter asks DWSIM to solve before saving and reads back the light-liquid and heavy-liquid flow rates and compositions. If DWSIM returns a single liquid phase, the file is still generated, but the result explicitly reports `separated=false`. Such a result must not be treated as a completed two-liquid-phase separation design.

Scientific boundary: the project supplies the flowsheet structure, feed state, and export operation. The LLE split is calculated from DWSIM's built-in parameters or parameter-estimation mechanism. Binary interaction parameters and experimental data must be reviewed before engineering use.

## 6. Ternary LLE Export

Ternary LLE uses the same `Vessel_LLE` topology as binary LLE, with three components in the feed.

Example:

```text
Export a DWSIM file for ethanol/ethyl acetate/water ternary LLE,
composition 0.129/0.188/0.683, at 298.15 K.
```

The current DWSIM 9.0.5 Automation path has a known limitation. Tested `NestedLoops`, `InsideOut`, and `GibbsMinimization` approaches, as well as several `PreferredFlashAlgorithmTag` values, may all return one liquid phase and leave `Heavy_Liquid` empty. The exporter preserves the structurally correct file and reports a warning.

When this occurs:

1. Open the file in the DWSIM GUI.
2. Select a flash algorithm that supports immiscible liquids or VLLE, such as `Nested Loops (Immiscible)`, `Nested Loops (VLLE)`, or `Simple LLE`, when available in the installed version.
3. Review the binary interaction parameters in the property package.
4. Recalculate and confirm that the `Heavy_Liquid` flow is greater than zero.

If the GUI calculation still returns one liquid phase, interpret the result as the selected model and parameters predicting no phase split at the specified temperature, pressure, and composition. Do not force an extraction-success interpretation.

## 7. Ternary Liquid-Liquid Extractor Export

When a request explicitly identifies two feed components and a separate solvent, the project can export a liquid-liquid extraction unit:

```text
Feed + Solvent -> Liquid-Liquid Extractor -> Raffinate + Extract
```

Validated example:

```text
The feed is 40% n-propyl acetate and 60% ethyl acetate. Use DMSO as the
solvent at a solvent ratio of 1.5 and export a DWSIM liquid-liquid extraction file.
```

For this example:

- The feed composition is written as `[0.4, 0.6, 0.0]`.
- The solvent stream is pure DMSO with composition `[0.0, 0.0, 1.0]`.
- The solvent flow equals `solvent_ratio × feed_flow`.
- The two product streams are `Raffinate` and `Extract`.

This is a liquid-liquid extraction unit, not a distillation column. If the intent is to use a high-boiling solvent to modify relative volatility, use the ternary VLE/extractive-distillation path described in Section 4.

## 8. Exported-File Acceptance Checks

Perform at least the following checks on every `.dwxmz` file:

1. Confirm that compound names and order are correct and mole fractions sum to 1.
2. Confirm that temperatures are in K. The interface converts pressure to Pa when writing to DWSIM.
3. Confirm that the property package matches the task. LLE work should use evidence-backed NRTL or UNIQUAC parameters.
4. Confirm that all inlet and outlet connections are complete and feeds enter the correct ports or stages.
5. Confirm that each VLE column has two independent specifications and records the selected convergence algorithm.
6. Confirm that an LLE file does not silently contain a zero `Heavy_Liquid` or `Extract` flow.
7. Confirm that DWSIM reports a calculated/converged state and the material balance is within the project's accepted tolerance.
8. Before engineering use, compare DWSIM results with experimental data, ThermoFormer, or another independent model.

## 9. Common Failures

| Symptom | Common cause | Resolution |
|---|---|---|
| `DWSIM.Automation.dll` cannot be found | Incorrect `DWSIM_HOME` | Point it to the actual DWSIM installation directory |
| `AddCompound` fails | Case-sensitive DWSIM name or missing mapping | Add an explicit mapping to `_DWSIM_COMPOUND_MAP` |
| Binary VLE reaches the iteration limit | Unsuitable solver or initial estimates | Switch manually among Naphtali-Sandholm, Wang-Henke, and Burningham-Otto; increase the iteration limit and retain the estimates |
| Rigorous-column product flows are zero | Missing second independent specification | Add a bottoms flow, boilup ratio, or heat-duty specification |
| Ternary LLE heavy phase is empty | The Automation flash kernel does not enable immiscible-liquid calculations, or the model predicts one phase | Select an LLE/VLLE kernel in the GUI and review binary parameters |
| The file is generated but not calculated | DWSIM solve failed or only the structure was saved | Open the GUI, inspect the error, correct the specifications, recalculate, and save |

## 10. Code Locations

- DWSIM flowsheet construction and saving: `thermo_engine/dwsim_export.py`
- Request recognition, parameter parsing, and download metadata: `agent/extractive_distillation.py`
- Binary VLE tests: `tests/test_binary_vle_dwsim.py`
- Binary LLE tests: `tests/test_binary_lle_dwsim.py`
- Ternary LLE tests: `tests/test_ternary_lle_dwsim.py`
- Liquid-liquid extractor tests: `tests/test_lle_extraction_export.py`
- Extractive-distillation column tests: `tests/test_dwsim_extractive_export.py`

