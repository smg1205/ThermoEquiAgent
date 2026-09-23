# Extractive distillation with an explicit ThermoFormer request

> Example trigger: `乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，用乙二醇做萃取剂，塔顶乙醇纯度99.5%`

This document explains what happens when the user explicitly asks for ThermoFormer: how the system
uses the local model to **screen entrainers**, then lets the user **choose** one and completes the
column design. Every number comes from the deterministic `thermo_engine`; the LLM and routing rules
only recognise intent, split out parameters and orchestrate. They **never compute equilibrium values
directly**.

---

## Core concept

**ThermoFormer does not output relative volatility $\alpha$ directly.** It outputs a bubble-point VLE
result: $(x,T) \rightarrow (y,P)$, i.e. vapour composition and pressure. The value of $\alpha$ is
computed by definition in `thermo_engine/column_design.py::_relative_volatility_from_model`,
*after* the model's bubble-point vapour composition $y$ is obtained:

$$
\alpha_{ij} = \frac{K_i}{K_j} = \frac{\,y_i/x_i\,}{\,y_j/x_j\,}
$$

Here $i$ = ethanol and $j$ = water. The $y$ values are the ThermoFormer model's **predicted** output
(labelled `source_type="model_prediction"` in `thermoformer_backend.py`); the formula
$(y_i/x_i)/(y_j/x_j)$ itself is deterministic.

---

## Input (this example)

Feed ethanol 0.40 / water 0.60, 1 mol/s, 25 °C = 298.15 K, atmospheric = 101.325 kPa, distillate
ethanol 99.5 %. Ethylene glycol as entrainer (if unspecified, the system first offers candidates for
you to choose). Recovery 0.98 and extraction ratio 2.0 default when not given.

---

## End-to-end flow (two stages)

### Stage A: local model screens entrainers, then the user chooses

In `agent/extractive_distillation.py`:

1. **Intent recognition** `is_extractive_distillation_request` routes to extractive distillation.
2. **Whether ThermoFormer was requested** `requests_thermoformer`: matching `thermoformer` /
   `用thermoformer` and similar selects **ThermoFormer** for the local screening; otherwise
   **UNIFAC** is the default (`alpha_source`).
3. **Candidate list**: if the message **names no entrainer**, or asks to see candidates
   (`选萃取剂` / `看看候选`), then `offer_entrainer_choices(spec, alpha_source)` calls
   `recommend_extraction_entrainer`:
   - For each candidate entrainer (ethylene glycol, glycerol, ...) it takes the extraction-region
     composition $[x_E,x_W,x_S]=[0.10,0.10,0.80]$ and computes relative volatility $\alpha$ and
     selectivity with the selected model;
   - It returns `ExtractiveExportPayload(status="awaiting_entrainer", entrainer_candidates=[...])`
     and the UI shows candidate buttons, waiting for the user to choose. **No column design is
     produced at this point.**

**The $\alpha$ used for screening (the same calculation as the column design)**:

$$
\alpha_{\mathrm{ext}} = \frac{y_E / x_E}{y_W / x_W}
$$

### Stage B: the user picks an entrainer, and the design is produced

The user clicks a candidate in the panel (the frontend resends the whole original request, filling in
the feed context) and the message now contains an explicit entrainer (e.g. `用甘油`):

1. `_find_entrainer` matches, the candidate stage is skipped, and the flow goes straight to
   `design_extractive_distillation_column(spec, alpha_source=...)`.
2. The design still uses the local model to compute relative volatility: the binary base pair plus
   the ternary extraction region, combined geometrically into $\alpha_{\mathrm{avg}}$:

$$
\alpha_{\mathrm{avg}} = \sqrt{\alpha_{\mathrm{base}} \cdot \alpha_{\mathrm{ext}}}, \qquad \text{selectivity} = \frac{\alpha_{\mathrm{ext}}}{\alpha_{\mathrm{base}}}
$$

3. Then the deterministic short-cut design runs:
   - **Material balance**: split the ethanol / water / ethylene glycol flows according to distillate
     purity and recovery;
   - **Fenske minimum stages**: $N_{\min} = \dfrac{\ln[(x_{D,E}/x_{D,W})/(x_{B,E}/x_{B,W})]}{\ln \alpha_{\mathrm{avg}}}$
   - **Underwood minimum reflux** (constant $\alpha$ for a binary pair), then operating reflux
     $R = 1.4\,R_{\min}$;
   - **Gilliland-Eduljee** trade-off between reflux and stages gives the theoretical stage count $N$;
     feed stage and entrainer stage are then fixed;
   - **Bubble temperatures** are read back for the top and bottom;
   - Returns a `status="ready"` / `"dwsim_unavailable"` design plus an optional DWSIM file, with
     `backend_version` carrying `+thermoformer` and an explicit **ML approximation, not
     experimentally validated** warning.

---

## Interaction example

```
You: 乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，塔顶纯度99.5%
AI:  Candidate entrainers screened with the local model (ThermoFormer): glycerol, ethylene glycol.
     Please choose.
     [candidate panel: ethylene glycol alpha=.. selectivity=.. | glycerol alpha=.. selectivity=..]
You: click "glycerol"
AI:  stages .. reflux ratio .. top/bottom temperature .. (+ DWSIM file)
```

---

## Measured runtime values (real checkpoint)

With the current pre-release checkpoint the full chain measures:
$\alpha_{\mathrm{base}} \approx 0.679$, $\alpha_{\mathrm{ext}} \approx 0.260$,
$\alpha_{\mathrm{avg}} \approx 0.42$, theoretical stages $N=4$, reflux ratio $0.07/0.05$,
top/bottom temperatures 351.45 K / 415.07 K.

---

## Important notes

1. $\alpha$ is not a direct ThermoFormer output; it is computed from the model's **predicted**
   bubble-point vapour composition $y$ via $(y_i/x_i)/(y_j/x_j)$.
2. The current checkpoint is `production_ready: false`, and the measured
   $\alpha_{\mathrm{avg}}<1$ makes the short-cut method return a degenerate, conservative design.
   This is a physical consequence of weak model predictions, **not a code bug**.
3. The **default therefore remains UNIFAC**; ThermoFormer is used only when it is explicitly requested.
4. Every activation forces an "ML approximation, not experimentally validated" warning.

---

## Trigger / switch reference

| Your intent | What to write in the request |
|---|---|
| Use ThermoFormer | add `用ThermoFormer` / `thermoformer` |
| Use the default UNIFAC | do not mention ThermoFormer |
| Do not name an entrainer; see candidates first | omit the entrainer, or write `看看候选` / `选萃取剂` |
| Name an entrainer and get the design directly | `用乙二醇` / `用甘油` |
