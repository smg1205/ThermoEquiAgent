# Agent and thermodynamics integrations

## Decision

ThermoEqui-Agent keeps its FastAPI/Next.js product shell and selectively adopts the tool-oriented
agent boundary demonstrated by
[datamllab/CAi_copilot](https://github.com/datamllab/CAi_copilot). CAi_copilot's current agent core
uses LangGraph, an LLM, and a REPL, so it is part of the LangChain ecosystem; it is not simply a
classic LangChain `AgentExecutor`. No CAi_copilot source code is vendored in this repository.

This is an architectural integration rather than vendoring CAi_copilot source. Its unrestricted
REPL/Bash execution model conflicts with this repository's rule that an LLM may orchestrate but may
never calculate equilibrium values. The production calculation loop is now a bounded LangGraph
`StateGraph` with `plan -> execute -> validate -> respond` nodes, directly following CAi_copilot's
graph-oriented agent shell while exposing only the allowlisted `phase_equilibrium` tool. Numerical
work remains inside reviewed `ThermodynamicBackend` adapters; arbitrary Python, Bash, and notebook
execution is not available to the graph.

Deterministic Peng-Robinson calculations use the pinned
[`thermo==0.6.1`](https://github.com/CalebBell/thermo) package through a local adapter. Optional
adapters integrate Phasepy directly and Clapeyron.jl through the official `pyclapeyron` bridge.
Upstream objects never cross the application boundary: every adapter maps results to
`CalculationResult`, records parameter evidence, and sends them through
`validate_equilibrium_result`.

Install the optional engines with:

```bash
python -m pip install -e ".[phase-engines]"
```

Use Python 3.11 or 3.12. Phasepy does not currently publish a Windows wheel for Python 3.13. The
first `pyclapeyron` import may download Julia and precompile Clapeyron once.

## Think-execute contract

1. **Plan** — DeepSeek returns a `TaskManifest` matching the supplied JSON Schema. It may leave
   `model_name` null for deterministic routing and must not calculate equilibrium values.
2. **Execute** — the LangGraph workflow calls `EngineeringToolRegistry`, which permits only the
   `phase_equilibrium` tool. The backend registry resolves a reviewed model adapter.
3. **Validate** — compositions, phase fractions, material balance, equilibrium residual,
   convergence, applicability, and available phase-stability evidence are checked independently.
4. **Respond** — DeepSeek may explain only the tool result and validation report. Ungrounded
   numbers and citations are withheld.

The UI exposes these four high-level audit steps. It does not expose model chain-of-thought.

## Production implementation matrix

| Model | Backend | Current executable scope | Parameter policy | Status |
|---|---|---|---|---|
| Ideal/Raoult | Internal deterministic solver | Binary bubble/dew, isobaric/isothermal VLE, TP flash, azeotrope candidate search | Reviewed local pure-property correlations | Available, low-pressure baseline |
| Peng-Robinson | CalebBell/thermo adapter | Hydrocarbon/allowlisted light-gas bubble/dew, binary VLE curves, TP flash, phase-state classification, azeotrope candidate search | Pure properties from thermo; every binary pair must exist in ChemSep PR; exact matrix/order/form/units/version are snapshotted and hashed | Available |
| Phasepy/Peng-Robinson | Phasepy 0.0.56 optional adapter | Hydrocarbon/allowlisted light-gas bubble/dew, binary VLE curves, two-phase TP flash, azeotrope candidate search | Pure properties from thermo; every binary pair must exist in ChemSep PR; returned phases are checked again by fugacity residual | Available when `phase-engines` is installed |
| Clapeyron/Peng-Robinson | pyclapeyron 0.1.1 + Clapeyron.jl optional adapter | Bubble/dew, binary VLE curves, one/two-phase Gibbs TP flash, phase-state classification, azeotrope candidate search | CAS identities are grounded through thermo; Clapeyron packaged pure parameters and references plus the reviewed ChemSep PR `kij` matrix are snapshotted and hashed; absent pairs return `missing_parameters` | Available when Julia initialization succeeds |
| Wilson | Planned activity-coefficient adapter | VLE contract | Reviewed directional binary parameters required | Contract only |
| NRTL | Planned activity-coefficient adapter | VLE/LLE contract | Reviewed directional binary parameters required | Contract only |
| UNIQUAC | Planned activity-coefficient adapter | VLE/LLE contract | Reviewed binary parameters and structural constants required | Contract only |

"Contract only" models remain visible for recommendation and schema design but cannot emit
calculation results. They fail with structured `missing_parameters` instead of synthetic defaults.

## Extension seam

Add another model or engine by implementing `ThermodynamicBackend`, registering its aliases and
supported calculations in `ThermodynamicBackendRegistry`, providing evidence-bearing parameter
sources, and adding behavioral tests at `calculate_equilibrium` or the HTTP calculation endpoint.
Do not couple DeepSeek prompts or frontend components to an upstream engine object.

## Deliberate v0.1 boundary

Electrolytes, reactive equilibrium, SLE, polymers, hydrates, petroleum pseudocomponents,
polymorphs, VLLE, and flowsheet design are rejected explicitly. LLE currently has a typed contract
but no production numerical backend.
