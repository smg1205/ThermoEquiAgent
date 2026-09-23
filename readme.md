# ThermoAgent

ThermoAgent is an intelligent thermodynamic assistant that connects natural-language problem statements, ThermoFormer phase-equilibrium prediction, classical activity-coefficient models, and DWSIM process simulation files.

The system is intended for researchers who need reproducible VLE and LLE calculations, transparent model selection, and traceable links from predicted phase-equilibrium quantities to downstream separation-process demonstrations.

## Scope

ThermoAgent currently focuses on:

- Binary VLE bubble-point prediction and DWSIM TP-flash demonstrations.
- Ternary VLE systems with an entrainer, including extractive-distillation solvent cases.
- Ternary LLE coexistence endpoint prediction.
- DWSIM flowsheet generation for selected distillation and liquid-liquid extraction demonstrations.
- Three-source validation against experimental data, ThermoFormer predictions, and DWSIM calculations.

Representative systems in the current validation set are:

| System | Equilibrium type | Demonstration role |
|---|---|---|
| 2-propanol / water | Binary VLE | Isobaric bubble point and conventional distillation |
| **n-heptane / n-nonane** | **Binary VLE** | **Reference alkane pair: bubble point and direct binary distillation** |
| Ethyl acetate / n-propyl acetate + DMSO | Ternary VLE | Entrainer-assisted extractive distillation |
| Ethanol / ethyl acetate / water | Ternary LLE | Coexistence endpoints and liquid-liquid extraction |
| Water / MIBK | Binary LLE | Additional DWSIM LLE stress-test example |

## ThermoFormer Integration

ThermoAgent routes phase-equilibrium tasks to ThermoFormer through the `thermo_engine` interface. The language model does not invent numerical equilibrium values. It identifies the task, validates the required information, and dispatches the calculation to the appropriate thermodynamic backend.

Default ThermoFormer checkpoints are selected from the local model registry:

| Task | Protocol | Checkpoint pattern |
|---|---|---|
| Binary VLE | `vle_overall_binary` | `models/vle/prediction/vle_overall_binary/seed_0/best_model.pt` |
| Ternary VLE | `vle_overall_ternary` | `models/vle/prediction/vle_overall_ternary/seed_0/best_model.pt` |
| Ternary LLE | `ternary-system` | `models/lle/prediction/ternary-system/seed_0/best.pt` |

## DWSIM Name Mapping

ThermoAgent automatically resolves common component names to DWSIM compound names before building flowsheets. Examples include:

| User-facing name | DWSIM candidate |
|---|---|
| 2-propanol | Isopropanol |
| n-propyl acetate | N-propyl acetate |
| DMSO | Dimethyl sulfoxide |
| MIBK | Methyl isobutyl ketone |
| n-heptane | n-Heptane |
| n-nonane | n-Nonane |

This mapping step is necessary because DWSIM compound names are not always identical to names used in manuscripts or experimental tables.

## Worked Example: n-Heptane / n-Nonane Direct Binary Distillation

This example is the simplest complete end-to-end reference case in the repository: a
**non-extractive** binary distillation with **no solvent**, so it isolates the three-source
chain (experiment, ThermoFormer, DWSIM) without entrainer-selection steps. It is a useful
template when adding a new binary system.

The experimental labels come from NIST ThermoML DOI `10.1016/j.fluid.2013.05.016`. All
equilibrium numbers are supplied by the experimental source, ThermoFormer, or DWSIM; none
are interpolated by hand.

### Three-source bubble-point comparison

At 101.325 kPa, five representative liquid compositions:

| x (n-heptane) | T exp (°C) | T ThermoFormer (°C) | T DWSIM (°C) | y exp | y ThermoFormer | y DWSIM |
|---|---|---|---|---|---|---|
| 0.117 | 140.75 | 138.93 | 139.97 | 0.325 | 0.383 | 0.340 |
| 0.359 | 123.05 | 121.67 | 123.28 | 0.726 | 0.738 | 0.701 |
| 0.466 | 117.15 | 116.28 | 117.66 | 0.824 | 0.816 | 0.789 |
| 0.633 | 109.15 | 109.55 | 110.30 | 0.916 | 0.897 | 0.884 |
| 0.837 | 102.55 | 103.10 | 103.01 | 0.973 | 0.961 | 0.959 |

All three sources agree closely: deviations stay within roughly ±2 °C for temperature, and
the vapor-phase enrichment of n-heptane is reproduced consistently by both models.

Source data: `report/success/正庚烷-正壬烷/heptane_nonane_three_source_bubble.csv`

### Column design and DWSIM flowsheet export

The design uses feed `x(heptane) = 0.466` at 101.325 kPa (an actual NIST ThermoML point),
1.0 mol/s total flow, targeting 0.995 distillate mole fraction with 98 % n-heptane recovery.
Relative volatility is taken from the DWSIM UNIQUAC bubble-point calculation rather than
assumed, and the short-cut Fenske–Underwood–Gilliland method then sizes the column:

| Quantity | Value |
|---|---|
| Theoretical stages | 15 |
| Minimum stages | 6.42 |
| Feed stage | 7 |
| Minimum reflux ratio | 0.638 |
| Operating reflux ratio | 0.893 (1.4 × minimum) |
| Distillate flow | 0.459 mol/s |
| Bottoms flow | 0.541 mol/s |
| Condenser type | Total |

The generated flowsheet contains one feed plus distillate and bottoms — no extraction
solvent. Property package: **DWSIM UNIQUAC**.

Design record: `report/success/正庚烷-正壬烷/heptane_nonane_binary_distillation_x0p466_design.json`

### Exported files

All artifacts live in `report/success/正庚烷-正壬烷/`:

| File | Contents |
|---|---|
| `heptane_nonane_three_source_bubble.csv` | Three-source comparison at five compositions |
| `heptane_nonane_binary_distillation_x0p466_design.json` | Design record and DWSIM VLE values |
| `heptane_nonane_binary_distillation_x0p466.dwxmz` | Rigorous binary distillation column |
| `heptane_x0p*_2comp_bubble_*.dwxmz` | Five near-bubble TP-flash files (one per point) |

### Reproducing this example

```powershell
# 1. ThermoFormer-vs-experiment table from the locked seed-2 test predictions
python scripts\generate_heptane_nonane_comparison.py

# 2. Five near-bubble DWSIM flashes (DWSIM calculates T and vapor composition)
python scripts\generate_heptane_nonane_dwsim.py

# 3. Short-cut design + rigorous binary column flowsheet
python scripts\generate_heptane_nonane_binary_column.py
```

> **Note on DWSIM automation.** The `.dwxmz` files open directly in DWSIM. As with the other
> distillation exports in this repository, condenser and reboiler specifications are not
> reliably settable through the DWSIM Automation interface, so they are completed in the
> DWSIM GUI and the column is recalculated there.

## Parameter Import

For systems requiring binary interaction parameters, ThermoAgent writes the parameters into the DWSIM property package or directly into the `.dwxmz` XML representation. Unit conventions are handled explicitly. For example, NRTL parameters reported in K or J/mol are converted to the internal DWSIM scale when required, and auto-estimation is disabled when explicit parameters must be preserved.

## Reproducible Demonstrations

The main DWSIM demonstration files are stored in:

```text
lunwen/dwsim_demonstration/
```

Flow-example column files are stored in:

```text
data/exports/flow_examples/
```

Useful entry points:

```powershell
# Refresh DWSIM VLE representative points and .dwxmz files
python scripts\generate_ternary_dwsim_demonstration.py --outdir lunwen\dwsim_demonstration --write-files --skip-lle

# Recompute deterministic column-design values without opening DWSIM
python scripts\export_flow_examples.py --dry-run

# Export DWSIM column flowsheets
python scripts\export_flow_examples.py --outdir data\exports\flow_examples
```

## Validation Reports

The latest integrated report is:

```text
report/Agent整合ThermoFormer进度与三源验证报告v5.md
```

It summarizes:

- ThermoAgent task orchestration.
- ThermoFormer model selection.
- DWSIM component-name mapping and parameter import.
- Three-source comparison for VLE and LLE systems.
- DWSIM supplementary-information assets for screenshots and video recording.

## Scientific Use

ThermoAgent is designed to make phase-equilibrium evidence traceable. A reported value should be attributable to one of three sources:

1. Experimental data.
2. ThermoFormer prediction.
3. DWSIM calculation.

When a value is not available from one of these sources, the report should state that limitation explicitly rather than filling the table by interpolation or assumption.
