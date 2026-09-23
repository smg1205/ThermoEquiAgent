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

## Try it

Start the stack:

```powershell
python -m uvicorn apps.api.main:app --reload --port 8000   # backend
pnpm --dir apps/web dev                                     # frontend
```

Then open the workbench and paste one of these into the input box.

**1. Direct binary distillation with a DWSIM file** — the n-heptane / n-nonane
reference case. Both the feed composition and an export word are required:

```text
正庚烷 0.466，正壬烷 0.534，导出 DWSIM 精馏塔文件
```

```text
heptane 0.466, nonane 0.534, export the DWSIM distillation file
```

The design uses feed `x(heptane) = 0.466` at 101.325 kPa with 0.995 distillate purity
and 0.98 recovery, giving 15 theoretical stages, feed stage 7, and reflux ratio 0.893.
Drop the composition or the export word and the request degrades predictably: without
a composition it reports a missing parameter, and `正庚烷-正壬烷精馏塔设计` (no export
word) returns design numbers only, with no file.

**2. Extractive distillation** — the entrainer is selected as the highest-boiling
component and the selectivity is checked to confirm the entrainer actually helps:

```text
乙酸 / 水 / DMSO 萃取精馏，导出 DWSIM 文件
```

**3. Liquid-liquid extraction**:

```text
水 / 正丁醇 液液萃取，导出 DWSIM 文件
```

## Validation

The current three-source comparison is documented in:

```text
report/Agent整合ThermoFormer进度与三源验证报告v5.md
```

Representative result — n-heptane / n-nonane at 101.325 kPa, experimental labels from
NIST ThermoML (DOI `10.1016/j.fluid.2013.05.016`):

| x (n-heptane) | T exp (°C) | T ThermoFormer (°C) | T DWSIM (°C) | y exp | y ThermoFormer | y DWSIM |
|---|---|---|---|---|---|---|
| 0.117 | 140.75 | 138.93 | 139.97 | 0.3250 | 0.3830 | 0.3403 |
| 0.466 | 117.15 | 116.28 | 117.66 | 0.8240 | 0.8163 | 0.7890 |
| 0.837 | 102.55 | 103.10 | 103.01 | 0.9730 | 0.9613 | 0.9593 |

Across all 16 locked test points the ThermoFormer temperature MAE is **0.985 °C** and
the vapour n-heptane MAE is **0.0246**. All three sources agree within roughly 2 °C,
which is why this pair is the recommended first case to run.

The report also documents where the models do *not* hold: for acetic acid / water /
DMSO, DMSO lies outside the ThermoFormer training distribution, and the resulting
vapour-composition MAE (0.22–0.29) is about three times worse than DWSIM. Both the
working and the failing cases are reported rather than only the favourable ones.

## Reproducing the case files

```powershell
python scripts\generate_heptane_nonane_comparison.py     # ThermoFormer vs experiment
python scripts\generate_heptane_nonane_dwsim.py          # five near-bubble flashes
python scripts\generate_heptane_nonane_binary_column.py  # short-cut design + column
```

Artifacts land in `report/success/正庚烷-正壬烷/`. Opening them requires DWSIM plus
pythonnet, and pythonnet needs a full process — it will not run inside a restricted
sandbox. Condenser and reboiler specifications are not reliably settable through the
DWSIM Automation API, so the exported columns are completed and recalculated in the
DWSIM GUI.

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
