---
name: thermodynamic-validation
description: Audit phase-equilibrium outputs for composition and material balance, fugacity-equivalent residuals, convergence, parameter applicability, stability, and azeotrope criteria. Use after every calculation and when diagnosing physical validation failures.
---

# Thermodynamic validation

## Use cases

Use as the mandatory gate after a calculation or while investigating an invalid result.

## Inputs

Require a `CalculationResult`, its immutable input snapshot, tolerance configuration, model and
parameter applicability metadata.

## Outputs

Return `ValidationReport` with individual checks, overall status, warnings, and a recovery action.

## Procedure

1. Check positive T/P, bounded fractions, and normalized phases.
2. Check flash material balance and vapor-fraction bounds.
3. Evaluate maximum/mean equilibrium residuals rather than solver status alone.
4. Fail non-converged output; warn or fail out-of-domain parameters explicitly.
5. Identify only internal `x≈y` points as azeotrope candidates.

## Prohibitions

Do not let interpretation override validation. Do not infer success from a plotted crossing or HTTP
status. Do not silently relax centralized tolerances.

## Failure handling

Use `physical_validation_failure`, `numerical_nonconvergence`, `phase_instability`, or
`parameter_out_of_domain` and prescribe a deterministic next action.

## Related files

`thermo_engine/validation.py`, `docs/validation.md`, `tests/test_validation.py`.

## Acceptance

Every check contains a metric, tolerance, message, and pass state; overall status is derived from
checks; failures remain failures throughout API and agent responses.
