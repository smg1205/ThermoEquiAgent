# ThermoAgent

ThermoAgent is a conversational thermodynamic engineering workbench. It turns a
natural-language problem statement into a reproducible VLE / LLE calculation and,
when asked, into a flowsheet file that opens in DWSIM.

The design goal is traceability: every number in the output is attributable to an
experimental source, a ThermoFormer prediction, or a DWSIM calculation. The language
model routes and explains — it never invents an equilibrium value.

## What it does

- **Binary and ternary VLE** — bubble point, dew point, isobaric and isothermal VLE,
  TP flash, and azeotrope search.
- **Binary and ternary LLE** — liquid-liquid coexistence endpoints and phase splits.
- **Distillation design** — short-cut Fenske / Underwood / Gilliland sizing (stages,
  reflux ratio, feed stage, product temperatures) for binary and extractive columns.
- **DWSIM export** — generates `.dwxmz` flowsheets for TP flash, binary distillation,
  extractive distillation, and liquid-liquid extraction.
- **Three-source validation** — experimental data, ThermoFormer prediction, and DWSIM
  are compared side by side so a model's reliability can be judged rather than assumed.

Out of scope in v0.1 (rejected explicitly): electrolytes, reactive equilibrium, SLE,
polymers, hydrates, petroleum pseudocomponents, and polymorphs.

## How it works

```
natural language
      │
      ▼
  intent routing ──► component resolution ──► model selection
      │                                            │
      │                        ┌───────────────────┴───────────────────┐
      ▼                        ▼                                       ▼
  missing-parameter      thermo_engine                        ThermoFormer
  report (no guessing)   (classical models)                   (neural VLE / LLE)
                               └───────────────┬───────────────────────┘
                                               ▼
                                    equilibrium validation
                                               ▼
                              design / report / DWSIM .dwxmz
```

Model selection is deterministic and happens outside the language model:

| Task | Protocol | Checkpoint |
|---|---|---|
| Binary VLE | `vle_overall_binary` | `models/vle/prediction/vle_overall_binary/seed_2/best_model.pt` |
| Ternary VLE | `vle_overall_ternary` | `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt` |
| Binary LLE | `binary-system` | `models/lle/prediction/binary-system/seed_0/best.pt` |

When required information is missing, the system returns a structured
`missing_parameters` failure instead of filling the gap with an assumption.

## Worked example: n-heptane / n-nonane

This is the recommended first case to run. It is a **direct binary distillation** with
no solvent, so it exercises the whole chain — experiment, ThermoFormer, DWSIM — without
any entrainer-selection steps. Experimental labels come from NIST ThermoML
(DOI `10.1016/j.fluid.2013.05.016`).

Three-source bubble-point comparison at 101.325 kPa:

| x (n-heptane) | T exp (°C) | T ThermoFormer (°C) | T DWSIM (°C) | y exp | y ThermoFormer | y DWSIM |
|---|---|---|---|---|---|---|
| 0.117 | 140.75 | 138.93 | 139.97 | 0.3250 | 0.3830 | 0.3403 |
| 0.359 | 123.05 | 121.67 | 123.28 | 0.7260 | 0.7384 | 0.7006 |
| 0.466 | 117.15 | 116.28 | 117.66 | 0.8240 | 0.8163 | 0.7890 |
| 0.633 | 109.15 | 109.55 | 110.30 | 0.9160 | 0.8970 | 0.8844 |
| 0.837 | 102.55 | 103.10 | 103.01 | 0.9730 | 0.9613 | 0.9593 |

Across all 16 locked test points the ThermoFormer temperature MAE is **0.985 °C** and
the vapour n-heptane MAE is **0.0246**. All three sources agree within roughly 2 °C.

The distillation design uses feed `x(heptane) = 0.466` at 101.325 kPa, 1.0 mol/s, with
0.995 distillate mole fraction and 0.98 recovery of n-heptane. Relative volatility is
taken from the DWSIM UNIQUAC bubble-point calculation rather than assumed:

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

### Run it from the workbench

Start the stack:

```powershell
python -m uvicorn apps.api.main:app --reload --port 8000   # backend
pnpm --dir apps/web dev                                     # frontend
```

Then paste either of these into the input box:

```text
正庚烷 0.466，正壬烷 0.534，导出 DWSIM 精馏塔文件
```

```text
heptane 0.466, nonane 0.534, export the DWSIM distillation file
```

Both the feed composition and an export word are required. Without a composition the
system reports a missing parameter; `正庚烷-正壬烷精馏塔设计` (no export word) returns
design numbers only, with no file.

### Reproducing the case files

```powershell
python scripts\generate_heptane_nonane_comparison.py     # ThermoFormer vs experiment
python scripts\generate_heptane_nonane_dwsim.py          # five near-bubble flashes
python scripts\generate_heptane_nonane_binary_column.py  # short-cut design + column
```

Artifacts land in `report/success/正庚烷-正壬烷/`:

| File | Contents |
|---|---|
| `heptane_nonane_three_source_bubble.csv` | Three-source comparison at five compositions |
| `heptane_nonane_binary_distillation_x0p466_design.json` | Design record and DWSIM VLE values |
| `heptane_nonane_binary_distillation_x0p466.dwxmz` | Rigorous binary distillation column |
| `heptane_x0p*_2comp_bubble_*.dwxmz` | Five near-bubble TP-flash files |

Opening them requires DWSIM plus pythonnet, and pythonnet needs a full process — it
will not run inside a restricted sandbox. Condenser and reboiler specifications are not
reliably settable through the DWSIM Automation API, so the exported columns are
completed and recalculated in the DWSIM GUI.

## Layout

| Path | Contents |
|---|---|
| `agent/` | Intent routing, orchestration, response assembly |
| `thermo_engine/` | Classical models, column design, DWSIM export |
| `apps/` | FastAPI backend and React workbench |
| `schemas/` | Pydantic domain models and API contracts |
| `lab_models/` | ThermoFormer source, configs, datasets, weights |
| `scripts/` | Reproduction entry points |
| `report/` | Validation reports and archived case artifacts |
| `docs/` | Architecture, DWSIM guides, methodology notes |

## Scientific rules

These are enforced in the code, not just documented:

1. The language model may classify, retrieve, orchestrate, and explain — never
   calculate or invent an equilibrium value.
2. Every numerical result comes from `thermo_engine` and passes validation.
3. A solver status is not physical validation: composition, material balance,
   equilibrium residuals, convergence, and parameter applicability are checked.
4. Missing parameters produce a structured `missing_parameters` failure. Binary
   parameters, experimental data, and citations are never fabricated.

## Development

```powershell
python -m pytest                      # backend tests
ruff check . && mypy .                # lint and types
pnpm --dir apps/web test              # frontend tests
docker compose up --build             # full stack
```

The full three-source validation report is `report/Agent整合ThermoFormer进度与三源验证报告v5.md`.
