# Plan: Criteria and Implementation Process for Judging Whether ThermoFormer Bubble Point Predictions Are "Good"

> Scope: quality assessment of this repository's `ThermoFormer` backend predictions for **bubble point VLE**
> (isothermal bubble point `P–y`, isobaric bubble point `T–y`).
> Positioning: elevate "what counts as a good prediction" from **a single error number** to a
> **layered, acceptance-testable decision gate**, consistent with the repository's numerical discipline
> (all numbers come from the deterministic engine, pass physical validation, and missing parameters fail in a
> structured way).
> Aligned reference papers: the three papers under `lunwen/` (I&ECR Aspen Copilot, AIChE Chemasim
> multi-agent, JCTC OpenClaw) share the position that "judging the correctness of results must be guarded by
> physics/experiment; prediction models are only responsible for coordination and for producing candidate
> numbers".

---

## 0. One-sentence conclusion

**A ThermoFormer bubble point is "good" = all three layers pass at the same time:**

1. **Regression layer**: against **experimentally measured** bubble point temperature/pressure/vapor
   composition, the error falls within an acceptable threshold and is **no worse than the mechanistic
   baseline (UNIFAC)**;
2. **Physics layer**: the prediction is **thermodynamically self-consistent** (Raoult/modified-Raoult
   residual, Gibbs-Duhem, pure limits, convergence, smooth phase diagram);
3. **Downstream layer**: quantities derived from the bubble point (relative volatility α, number of stages
   N, reflux ratio R, selectivity) fall within the **physically feasible domain** and can support a
   reasonable separation design.

If any layer fails → the prediction **must not go directly into engineering design**; it should be flagged,
fall back to a mechanistic model, or request human review.

---

## 1. What "ground truth" is: experimentally measured bubble point

`VLESample` in the repository (`lab_models/ThermoFormer/src/thermoformer/data/loading.py`) is the
experimental ground truth:

| Field | Meaning | Unit source |
|---|---|---|
| `temperature_k` | Experimental bubble point temperature | Excel Celsius → K (+273.15)|
| `pressure_kpa` | Experimental bubble point pressure | Excel mmHg → kPa (×0.133322368)|
| `liquid_composition` | Liquid composition x (input/known) | Normalized to Σ=1 |
| `vapor_composition` | **Vapor composition y (ground truth)** | Normalized to Σ=1 |
| `quality_weight` / `quality_status` | Data quality (passed=1 / unverified=0.5 / failed) | Score weighting |
| `experiment_mode` | isothermal / isobaric / full_state | **Determines which direction the sample is scored in** |
| `smiles` / `names` / `doi` / `source` | System identity and provenance | Traceability and leakage-prevention grouping |

**Therefore the standard answer for judging "how good the regression is" = these experimental entries
themselves.** During evaluation:

- **Isothermal mode samples** (`experiment_mode=="isothermal"`): given the measured `temperature_k` and the
  measured liquid `liquid_composition`, predict `pressure_kpa` and `vapor_composition`, and compare the error
  against the measured values.
- **Isobaric mode samples** (`experiment_mode=="isobaric"`): given the measured `pressure_kpa` and the
  measured liquid `liquid_composition`, predict `temperature_k` and `vapor_composition`, and compare the
  error against the measured values.
- **full_state**: both directions can be evaluated, or only the physics layer can be run.

> Initialization note: the current `_warnings()` in `thermo_engine/thermoformer_backend.py` says "results are
> not experimentally validated".
> Once this plan is implemented, the regression layer described below should be executed on the protocol
> partitions that have experimental ground-truth samples, replacing that warning with the measured
> discriminative result.

---

## 2. Criteria: the three-layer decision rules in detail

### 2.1 Regression layer (against experimental measurements)

Computed for **each valid prediction** (`converged and not nonphysical`), then aggregated across samples.
Already implemented in `evaluation/prediction.py::_scalar_metrics` / `_y_metrics`:

| Metric | Definition | Applies to |
|---|---|---|
| MAE, RMSE, R² | Point-level aggregation | Bubble point pressure `ΔP_kpa`, bubble point temperature `ΔT_k` |
| `system_macro_mae/rmse` | Average per system first, then aggregate | Prevents large systems from masking small ones |
| `y_mae/y_rmse/y_r2` | Error in vapor composition y | `y` (mole fraction)|
| `y_sample/system/component_macro` | y error under several conventions | `y` |

**Decision thresholds (suggested initial values, calibratable per dataset):**

| Quantity | Suggested threshold | Note |
|---|---|---|
| Bubble point temperature `ΔT` | ≤ 1–2 K (mean), p95 ≤ 3 K | Acceptable for most engineering shortcut calculations |
| Bubble point pressure relative error | ≤ 1–3% (mean) | Isothermal mode |
| Vapor composition `y` | ≤ 0.01–0.05 (mole fraction) | Most sensitive to α/number of stages; keep it strict |
| Comparison with UNIFAC | ThermoFormer MAE **must not be significantly worse than** UNIFAC | "No worse than the mechanistic baseline" is what makes it usable |

> Key criterion: **a stand-alone MAE is meaningless; it must be reported side by side with the UNIFAC
> baseline**.
> Referring to the AIChE paper: the prediction group and the mechanistic group are compared on the same
> downstream task, and only predictions that are not worse beyond a reasonable range count as "good".

### 2.2 Physics layer (thermodynamic self-consistency)

Already implemented in `evaluation/thermodynamic_consistency.py`. These criteria **do not depend on
experimental data** and are hard physical gates:

| Criterion | Mathematics/meaning | Suggested passing threshold |
|---|---|---|
| Solver convergence | `converged==True` | Convergence rate ≥ 99% (P≤500 kPa, binary/ternary)|
| Equilibrium residual | Modified Raoult `P_calc − P_target` | `gross_equilibrium_residual_kpa ≤ 0.1` (current constant)|
| Physical bounds | P>0, y∈[0,1], Σx=Σy=1, γ>0, P_sat>0, T∈[150,1500] K | 0 violations |
| Gibbs-Duhem | `Σ xᵢ d ln γᵢ = 0` | Residual median → 0, p95 as small as possible |
| Pure limits | As x→pure component, γ→1, P→P_sat, T consistent | `pure_limit_vle_error ≤ 0.05` (current constant)|
| Permutation invariance | Results unchanged when component order is swapped | `permutation_*` near 0 |
| Smooth phase diagram | First/second derivative jumps of the bubble point curve | `phase_*` free of spike jumps |

### 2.3 Downstream layer (derived quantities physically reasonable)

Bubble point y is amplified into separation design. Taking one comparison experiment of ethanol-water
extractive distillation as an example
(UNIFAC α_avg≈2.82, N=19, R=2.536 vs ThermoFormer α_avg≈0.42, N=4, R=0.07),
**α<1 / R→0 / degenerate rounding of N is the signal of being "physically unreasonable"**.

| Derived quantity | Reasonable domain | Unreasonable signal |
|---|---|---|
| Relative volatility α | α>0; extractive distillation wants separation power | α≈1 (inseparable), α<1 (reverse anomaly)|
| Theoretical stages N | Finite positive integer, explainable by feed/target | Degenerate rounding lower bound of N, detached from overhead/bottoms bubble points |
| Reflux ratio R | 0<R finite | R falls to the implementation lower bound, with no physical basis |
| Overhead/bottoms bubble point temperature | Between the component bubble points with the correct trend | Outside the pure-component bubble point interval |

**The downstream layer is the most easily overlooked of the three, yet the one most emphasized by the three
papers** — because bubble point error propagates and is amplified into column design, so a prediction may be
unusable at the design end even if its "numbers look pretty".

---

## 3. Implementation process: the decision tree (recommended to be codified into `validate_equilibrium_result`)

```
Input: one bubble point prediction (T/P/y/γ/P_sat + status bits)
  │
  ├─1 Solver converged?           No → REJECT (does not produce a self-consistent equilibrium)
  ├─2 Physical bounds?            No → NONPHYSICAL (flag, does not enter the metric pool)
  ├─3 Equilibrium residual ≤ threshold?      No → REJECT
  ├─4 Gibbs-Duhem/pure limit/permutation/smoothness?  No → REJECT
  ├─5 [has experiment] MAE/RMSE ≤ threshold and no worse than UNIFAC?  No → FLAG (reportable but not releasable to design)
  ├─6 [has experiment] Derived quantities (α,N,R,T) physically reasonable?        No → FLAG (fall back to mechanistic model)
  ├─ ✓ All pass → VALID (may enter engineering design, annotated with uncertainty/confidence)
```

The output should be a **structured decision**, for example:

```json
{
  "verdict": "VALID | REJECT | NONPHYSICAL | FLAG",
  "reason": "one-sentence reason",
  "checks": {
    "converged": true,
    "physical_bounds": true,
    "equilibrium_residual_kpa": 0.02,
    "gibbs_duhem_pass": true,
    "pure_limit_pass": true,
    "regression_mae": {...},
    "unifac_comparison": {"thermoformer_mae": 0.9, "unifac_mae": 0.8, "better": false},
    "downstream": {"alpha": 2.5, "n_stages": 19, "reflux": 2.5, "reasonable": true}
  }
}
```

---

## 4. Recommended implementation path (three steps: document first → then minimal gate → then complete)

### Step A (this document): define criteria and thresholds — completed
See this file; thresholds in `2.1/2.2/2.3`.

### Step B (minimal change): add the `assess_bubble_prediction` gate
- Location: `thermo_engine/` (backend seam, usable by the Agent); or `evaluation/score_card.py` (training
  side).
- Input: a single `EquilibriumPoint` + `converged`/`nonphysical` + experimental ground truth `VLESample`
  when available.
- Output: the structured `verdict + checks` result described above.
- Reuse: the physical criteria of `thermodynamic_consistency`, the regression aggregation of
  `prediction_metric_rows`; **no new numerical values are introduced**.

### Step C (complete): UNIFAC baseline comparison + uncertainty annotation
- In `thermo_engine` or a standalone comparison script, compute bubble points with UNIFAC on the same batch
  of `VLESample` and output side-by-side MAE.
- Give each prediction a confidence level/applicability interval (P≤500 kPa, binary/ternary, unseen
  molecular domain), and on `FLAG` have the Agent actively fall back to the mechanistic model — aligning with
  the AIChE dual-Agent idea of "the prediction Agent decides, the mechanistic Agent guards".

---

## 5. Final questions and answers on judging "good or not"

| Question | Answer |
|---|---|
| What is the standard answer for a ThermoFormer bubble point being "good"? | **Three layers: experimental regression error + physical self-consistency + reasonable downstream derived quantities** |
| Is there real experimental data? | Yes — `VLESample` is experimentally measured (T/P/x/y, with DOI provenance)|
| What is used as ground truth for scoring? | The regression layer uses experimental measurements; the physics layer uses thermodynamic identities; the downstream layer uses the physically feasible domain |
| Must it approach UNIFAC? | Not "approach", but **no worse than UNIFAC** (side-by-side baseline)|
| What if there is no experimental data? | First run only the physics layer + downstream layer; once experiments are complete, layer the regression layer on top and re-evaluate |

---

## Appendix: comparison with the criteria of the three papers

| Layer in this plan | Corresponding paper criterion | Point borrowed |
|---|---|---|
| Regression layer (experimental) | I&ECR: comparison against literature specifications/rigorous simulation | "Align with known correct values and keep margin" |
| Physics layer (self-consistency) | I&ECR / JCTC: convergence, residuals, bounded validation | "Solver status ≠ physical validity; independent validation is required" |
| Downstream layer (derived quantities) | AIChE: physical reasonableness of the separation sequence, whether rigorous simulation converges | "Self-consistency of design scale-up + masked inference against cheating" |
| Mechanistic baseline comparison | AIChE: mechanistic group vs prediction group | "Usable only if not worse than the baseline" |
