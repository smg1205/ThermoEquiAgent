---
name: thermodynamic-calculation
description: Implement or execute deterministic bubble point, dew point, binary VLE, TP flash, azeotrope, and stability calculations through the ThermodynamicBackend protocol. Use whenever numerical phase-equilibrium results are requested or core algorithms change.
---

# Thermodynamic calculation

## Use cases

Use for solver implementation, CLI/API execution, numerical diagnostics, and backend adapters.

## Inputs

Require validated components, SI-normalized conditions/compositions, a model, and any evidence-
bearing parameters the model requires.

## Outputs

Return `CalculationResult` with input snapshot, points/phases, solver status, residuals, iterations,
warnings, backend version, and parameter-set identity.

## Procedure

1. Normalize units and compositions at the boundary.
2. Resolve pure properties and required binary parameters without guessing.
3. Execute through `ThermodynamicBackend`; bracket roots and report residuals.
4. Classify single/two-phase flash states and retain material-balance diagnostics.
5. Call thermodynamic validation before declaring success.

## Prohibitions

Do not calculate in the LLM, insert fake data, hide non-convergence, or label a test fixture as
engineering evidence.

## Failure handling

Map failures to the documented taxonomy and include a recovery action. Never emit a fabricated
partial curve after missing data or non-convergence.

## Related files

`thermo_engine/backend.py`, `thermo_engine/ideal.py`, `thermo_engine/service.py`, `schemas/domain.py`.

## Acceptance

The same manifest is runnable through Python and CLI without an LLM; results are deterministic and
validated by behavior tests using independent physical invariants.
