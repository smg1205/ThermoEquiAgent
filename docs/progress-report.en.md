# ThermoEqui-Agent Progress Report

> Updated: 2026-08-27
> Scope: Model integration + DWSIM export

This document records the progress of ThermoEqui-Agent on two tracks — **model integration** and
**DWSIM extractive distillation export** — including whether each item is done, where it is stuck,
and what comes next.

---

## 1. Model integration progress

The project wires machine learning models in as **backends just like ordinary models**, registers them in
`thermo_engine/registry.py`, and routes everything through the same computation interface and validation
flow.

### 1.1 Which models exist now

| Model | Type | What it does | Status |
|------|------|---------|------|
| Ideal/Raoult | Ordinary computation | Bubble point, dew point, flash, etc. | Working |
| Peng-Robinson, SRK, RK, UNIFAC | Ordinary computation | Phase equilibrium | Working |
| Phasepy/PR, Clapeyron/PR | External libraries | Phase equilibrium | Working |
| NRTL, UNIQUAC, Wilson | Activity coefficient | Phase equilibrium, activity coefficients | Partially working |
| **ThermoFormer** | Deep learning | **Predicting VLE (bubble point, isobaric/isothermal), activity coefficients** | Wired up, not yet formally validated |
| GHGEAT | Deep learning | Predicting physical properties | Registered, pending verification |

### 1.2 Rules observed during integration

- **No arithmetic by the LLM**: the deep learning model only produces predicted values (for example VLE),
  and the final result still has to pass validation.
- **The model does not impersonate a phase equilibrium solver**: the model only performs the predictions it
  is meant to do; if you ask it for something it cannot do, it will state clearly
  "I don't do this / it is out of scope".
- **Applicability range limits exist**:
  - ThermoFormer: upper limits are set on the number of components, pressure, and molecular formula (SMILES);
    exceeding them results in rejection.
- **Lazy loading**: model weights and source code are read only at execution time, so missing heavy
  libraries such as torch and rdkit do not slow down the startup of the whole project.

### 1.3 Environment variables used (`.env`)

| Variable | Purpose |
|------|------|
| `THERMOFORMER_SRC` | ThermoFormer source directory |
| `THERMOFORMER_CHECKPOINT` | ThermoFormer weights file |
| `THERMOFORMER_USE_CUDA` | Whether to use the GPU |

### 1.4 How far model integration has come

**Done**
- The ThermoFormer backend is integrated, registered, and connected to validation.
- Applicability range limits have also been added.

**Not done yet / to confirm**
- The ThermoFormer backend is currently **experimental**; the code explicitly states that "benchmarking has
  not been done and the applicability range has not been formally reviewed".
- The deployment machine may not have libraries such as torch and rdkit installed; predictions can only
  really run once they are installed.

---

## 2. DWSIM export feature progress

### 2.1 What can be exported

`thermo_engine/dwsim_export.py` contains two export entry points:

| Function | What it does | Status |
|------|------|------|
| `export_dwsim_flowsheet` | Ordinary flash flowsheet → `.dwxmz` file | **Working** (validated) |
| `export_dwsim_extractive_column` | Extractive distillation column flowsheet → `.dwxmz` file | The structure is generated, but specifications must be added manually when running |

### 2.2 Several fixes made to the extractive column export

To make the extractive column file generate correctly, the following problems were fixed (each has
corresponding tests):

1. **Component name normalization**
   - DWSIM is **case-sensitive** when recognizing component names (writing `ethanol` errors out; writing
     `Ethanol` works).
   - Fix: automatically convert common lowercase names into the spelling DWSIM recognizes.

2. **Switching to a column type it supports**
   - The original shortcut column (ShortcutColumn) **can only accept one feed**, so it cannot take the two
     streams "feed + solvent", and the solvent could not be connected.
   - Fix: switch to the rigorous column (DistillationColumn), which can accept multiple feeds.

3. **Correcting the connection ports**
   - Testing confirmed the rigorous column's ports: feed `(0,0)`, solvent `(0,1)`, overhead `(0,0)`,
     bottoms `(1,0)`.
   - Fix: connect the solvent to the correct port; if DWSIM rejects a given connection, record a reminder
     rather than failing the whole export.

4. **Adapting to how column parameters are written**
   - The rigorous column's setter method names differ from what we originally used (such as
     `set_NumberOfStages`).
   - Fix: try several spellings and use whichever works.

### 2.3 Current situation (key points)

- The generated `.dwxmz` file **saves correctly**, with material connections, number of stages, and reflux
  ratio all written in.
- But before clicking **Calculate** on the rigorous column in DWSIM, the **condenser and reboiler
  specifications** and the energy streams still have to be added by hand. There is **no stable, reliable way
  to set these** through DWSIM's automation interface; they are configured in the UI after opening the file.
- **The design itself is fully calculated** (how many stages, the reflux ratio, and the overhead and bottoms
  temperatures all come from our own model); it is only after landing in DWSIM that **the first run requires
  manually completing the specifications**.

### 2.4 DWSIM behavior observed in testing (for the record)

Obtained by actually trying things with the diagnostic script (`scripts/diag_dwsim_extractive.py`):

| Observation | Conclusion |
|------|------|
| DWSIM recognizes component names case-sensitively | Normalization is needed (fixed) |
| The shortcut column cannot take a second feed | Switch to the rigorous column (fixed) |
| The rigorous column has 11 inlet and outlet ports each | Port numbering must be chosen according to the column type |
| The rigorous column reports "stream connections missing" | Everything external is connected, but internal specifications are still missing |
| The rigorous column with MWK (bubble-point method) never converges for ethanol-water-glycol | Switch to NS / Napthali (Newton-type) |
| The rigorous column with Newton reports "Error evaluating error functions" | Most likely **insufficient initial temperature** or **mutually contradictory specifications** |
| Some DWSIM versions lack Wilson in the property package dropdown | Must rely on initial values and self-consistent specifications to converge |

### 2.5 Unresolved difficulties and recommendations

- **Making the extractive column "compute right after download" is not achievable with DWSIM automation**:
  creating energy stream objects throws a null reference, and there is no stable interface for the column's
  internal specifications — this is the ceiling of DWSIM automation itself, not a mistake in our code.
- **Current approach**: accept the "file is usable + specifications are completed manually in DWSIM" pattern;
  see `docs/extractive-dwsim-usage.zh-CN.md` for the specific steps.
- **If less manual work is desired**, in the future we could:
  - keep looking for other ways to create energy streams in code;
  - or simply provide a "design report / JSON" that does not depend on DWSIM.

---

## 3. Code / documentation inventory

| File | What it is for |
|------|---------|
| `thermo_engine/dwsim_export.py` | DWSIM export (ordinary flash + extractive column) |
| `thermo_engine/column_design.py` | Extractive column design requirements (Fenske/Underwood/Gilliland) |
| `thermo_engine/thermoformer_backend.py` | ThermoFormer backend (predicting VLE) |
| `thermo_engine/registry.py` | Model registry |
| `agent/extractive_distillation.py` | Recognizes extractive requests + arranges the export |
| `docs/extractive-dwsim-usage.zh-CN.md` | Tutorial for manually configuring the extractive column in DWSIM |
| `scripts/diag_dwsim_extractive.py` | On-machine diagnostic script (for development) |

---

## 4. Suggested next steps (in order)

1. **Write the "extractive column won't converge" tuning steps into the tutorial**: that is, the operations
   "switch to the NS/Napthali solver, supply initial temperatures, trim the specifications", organized into
   `extractive-dwsim-usage.zh-CN.md`.
2. **Promote the deep learning backend from "experimental" to formal**: add benchmarks, complete the
   applicability range review, and clear the "not yet validated" warning.
3. **(Optional) Add a design report**: output the extractive design results as JSON or Markdown without
   depending on DWSIM, as a fallback deliverable.
