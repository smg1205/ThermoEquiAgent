# Paper experiment section: agent method architecture and ThermoFormer prediction tasks

> Status: draft, for writing the "methods / experiments" section of the paper
> Basis: `docs/Agent整合ThermoFormer-进度文档-v4.md`
> Corresponding code: `agent/orchestrator.py`, `agent/extractive_distillation.py`, `thermo_engine/thermoformer_backend.py`, `thermo_engine/column_design.py`

This appendix writes the "ethanol-water extractive distillation" case as one experimental section of
the paper. At its core are **two diagrams**: one drawing the agent's end-to-end method architecture,
one drawing the prediction-task boundary of ThermoFormer; a single table then maps every model call to
a concrete function.

---

## 1 Two concepts that must be stated clearly

The paper specifies ThermoFormer's prediction task as **two descriptions of the bubble point**:

| Task | Input | Output | Corresponding code function |
|---|---|---|---|
| Isothermal bubble point | Liquid composition $\mathbf{x}$, temperature $T$ | Vapour composition $\mathbf{y}$, pressure $P$ | `predict_bubble_isothermal` |
| Isobaric bubble point | Liquid composition $\mathbf{x}$, pressure $P$ | Vapour composition $\mathbf{y}$, temperature $T$ | `predict_bubble_isobaric` |

Both tasks are dispatched by the same entry point, `ThermoFormerBackend.bubble_point`
(`thermo_engine/thermoformer_backend.py`). When a task supplies $T$, it takes the **isothermal**
branch; when it supplies only $P$, it takes the **isobaric** branch.

The provenance of predictions is uniformly labelled by the code as `source_type="model_prediction"`,
and results carry a "not experimentally validated" warning — this is the key point for the paper's
"model prediction vs mechanistic calculation" comparison.

---

## 2 Agent method architecture (end to end)

The diagram below corresponds to sections 2.2/2.3/2.4 of
`docs/Agent整合ThermoFormer-进度文档-v4.md`, with the "entrainer selection" step added.

```
user message
  │
  ▼
[1] intent classification  _classify_intent()
  │  (rules + LLM cross-check; DeterministicProvider as fallback)
  ├────────────────────────────┬────────────────────────────┬───────────────
  question                     extraction                   calculation / edit conditions
  │                            │                            │
  ▼                            ▼                            ▼
[2a] memory + knowledge answer [2b] extraction pipeline      [2c] build task -> bounded graph
                               │                            │
                               ▼                            ▼
                        run_extractive_export()          plan()->execute()->validate()->respond()
                               │
                               ▼
                    [3] entrainer candidate ranking
                        recommend_extraction_entrainer()
                        (local model: UNIFAC default / ThermoFormer)
                               │
                               ▼
                    [4] user selects entrainer (awaiting_entrainer)
                               │
                               ▼
                    [5] column design design_extractive_distillation_column()
                        ├─ compute relative volatility alpha, selectivity (local model)
                        ├─ Fenske minimum stages / Underwood reflux / Gilliland stages
                        ├─ bubble temperatures (top / bottom)
                        ▼
                    [6] DWSIM export export_dwsim_extractive_column()
                        ▼
                    [7] return design + .dwxmz (optional)
```

**Notes (keyed to the diagram numbers):**

- **[1] Intent routing**: rules first, then the LLM; if the LLM is unavailable, trust the rules.
  Specialised intents (extraction, parameter lookup, and so on) require a keyword trigger
  (`is_extractive_distillation_request`); the LLM "imagining" one does not count.
- **[2b] Extraction pipeline entry** `run_extractive_export(message)`: parameters are pulled out by
  regex/rules (`extract_extractive_params` -> `build_extractive_spec`). If parameters are missing it
  returns "which items are missing" and does not compute further.
- **[3] Entrainer candidate ranking** `recommend_extraction_entrainer`: for each candidate entrainer
  (ethylene glycol, glycerol, ...) it computes relative volatility and selectivity with the local
  model at the extraction-region composition $[x_E,x_W,x_S]=[0.10,0.10,0.80]$ and returns a ranked
  candidate list. **UNIFAC by default; when ThermoFormer is requested, its $bubble\_point$ result is
  used.**
- **[4] User selects an entrainer**: if the message names an entrainer explicitly, the flow goes
  straight to column design (`_find_entrainer` matches); if none is named, or the user asks to see
  candidates, a candidate panel is returned first (`status="awaiting_entrainer"`).
- **[5] Column design** `design_extractive_distillation_column`: the internal source of relative
  volatility is the same as in [3] (the same `alpha_source`).
- **[6] DWSIM export**: build a rigorous column plus 4 streams (feed / entrainer / distillate /
  bottoms), connect them, and save a `.dwxmz`; condenser/reboiler specifications must be added
  manually (see section 4 of v4).
- **[7] Return**: an `ExtractiveExportPayload`, which the frontend downloads automatically and
  optionally records.

> Relationship to v4: v4's "main flow diagram" is essentially the [1] -> [2a/2b/2c] branching; this
> diagram expands the **extraction branch** down to column design and DWSIM export. The four bounded
> graph steps (plan -> execute -> validate -> respond) correspond to the ordinary calculation in [2c],
> while the extraction branch follows the deterministic column_design chain.

---

## 3 ThermoFormer prediction-task boundary (paper figure)

This shows how the paper's two tasks are invoked in the agent's extraction scenario, and how their
output enters column design.

```
           ┌─────────────── ThermoFormer backend ───────────────┐
           │            ThermoFormerBackend.bubble_point()       │
user gives │  range check  _check_system_scope                  │
  (x,T) or │  ─(component count <= 3, pressure <= 500 kPa)────  │
  (x,P) ──►│  SMILES -> molecular encoding encode_molecules     │
           │    (RDKit descriptors + Uni-Mol v2 embedding +     │
           │     functional groups)                             │
           │        │                                           │
           │        ▼                                           │
           │  T given? ──yes──► predict_bubble_isothermal       │
           │                 -> output y, P  (task 1)           │
           │        │                                           │
           │        └──no───► predict_bubble_isobaric           │
           │                 -> output y, T  (task 2)           │
           │                        │                           │
           │                        ▼                           │
           │  validate_equilibrium_result -> build_result       │
           │  source_type="model_prediction" + warning          │
           └───────────────────────┬────────────────────────────┘
                                   ▼
                       downstream: column_design uses the output y
                       to compute relative volatility
                       alpha=(y_E/x_E)/(y_W/x_W)
                       then Fenske/Underwood/Gilliland -> column design
```

**Notes:**
- Both prediction tasks share the same encoding and validation pipeline; they differ only in "which
  thermodynamic variable was supplied".
- The model's output ($y$) enters `column_design._relative_volatility_from_model` to back out
  $\alpha$ — **$\alpha$ is not a direct output of the model**; it is computed by definition after $y$
  is predicted.
- The output is always `source_type="model_prediction"`, which guarantees the paper can distinguish
  "model prediction" from "UNIFAC mechanistic calculation" in the results.

---

## 4 Correspondence table of model calls at each step

| Step | What it does | Entry point / code | Model used |
|---|---|---|---|
| Intent classification | Decide whether the request is extraction / calculation / question | `ConversationOrchestrator.chat()` / `_classify_intent()` | rules + LLM cross-check (DeterministicProvider fallback) |
| Parameter extraction | Feed composition / temperature / pressure / flow / purity / recovery | `extract_extractive_params()` + `build_extractive_spec()` | regex + rules |
| Entrainer candidates | Candidate ranking | `recommend_extraction_entrainer()` | local model: **UNIFAC (default) or ThermoFormer** |
| Relative volatility | alpha, selectivity | `relative_volatility_eivw()` | same source as above |
| ThermoFormer bubble point | Isothermal/isobaric prediction (x,T)->(y,P) or (x,P)->(y,T) | `ThermoFormerBackend.bubble_point()` -> `predict_bubble_isothermal/isobaric` | ThermoFormer (checkpoint + Uni-Mol v2) |
| Column design | Fenske/Underwood/Gilliland, stages/reflux/temperatures | `design_extractive_distillation_column()` | deterministic short-cut method (relative volatility source as in [3]) |
| Bubble temperatures | Top / bottom bubble point | `_bubble_temperature()` | bubble-point equation + vapour pressure (UNIFAC/NIST) |
| Export | Build column and streams, save file | `export_dwsim_extractive_column()` | DWSIM automation (pythonnet) |

---

## 5 Experimental variable design (for the paper's "control group")

- **Variable one: source of relative volatility.**
  - Mechanistic arm: UNIFAC activity coefficients + vapour pressure;
  - Prediction arm: the $y$ output by ThermoFormer `bubble_point`, used to back out $\alpha$
    (isothermal and isobaric sub-tasks).
- **Variable two: candidate entrainer.** Ethylene glycol, glycerol and others are offered under local
  model scoring, and the column design follows the user's selection.
- **Dependent variables**: theoretical stages $N$, minimum reflux ratio $r_{\min}$, operating reflux
  ratio $R$, selectivity, top/bottom bubble temperatures, DWSIM convergence.
- **Output labelling**: every value produced with ThermoFormer carries
  `source_type="model_prediction"` plus a "not experimentally validated" warning, so the paper can
  state clearly which results are ML predictions and which are mechanistic calculations.

---

## 6 Actual gap to the code (state honestly; do not overclaim in the paper)

- Section 7.7 of v4 already notes that the extractive design numbers currently come by default from
  `column_design` (UNIFAC + the Fenske family). ThermoFormer is an optional backend that can already
  run `bubble_point` (in this repository, `thermo_engine/thermoformer_backend.py`), and can be
  requested as `alpha_source="thermoformer"` in entrainer candidate ranking and column design.
- If a paper experiment needs "the whole chain producing numbers with ThermoFormer", confirm that the
  deployment machine has torch/rdkit installed, has a checkpoint, and (when running Uni-Mol v2
  locally) has accessible weights; otherwise that arm is not executable. This is an environment
  dependency, not a code problem.

---

## 7 Suggested paper wording (can be copied directly)

> We take a "conversational engineering phase-equilibrium workbench" as an example: the user poses an
> ethanol-water extractive distillation task in natural language; the agent first performs intent
> routing, then screens candidate entrainers with a local model and hands them to the user for
> confirmation, and finally generates a column design with a deterministic short-cut method
> (Fenske-Underwood-Gilliland), optionally exporting a DWSIM file. Within this pipeline, ThermoFormer
> handles two bubble-point prediction tasks — given $(\mathbf{x},T)$ predict $(\mathbf{y},P)$, or
> given $(\mathbf{x},P)$ predict $(\mathbf{y},T)$ — and its predicted vapour composition
> $\mathbf{y}$ is used to back out the relative volatility, so that the same column design can be
> compared under two modes: "mechanistic calculation (UNIFAC)" and "model prediction (ThermoFormer)".
