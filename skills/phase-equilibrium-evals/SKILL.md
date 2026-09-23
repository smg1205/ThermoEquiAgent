---
name: phase-equilibrium-evals
description: Create and run behavioral evaluations for ThermoEqui-Agent intent routing, task formulation, unsupported-system refusal, parameter integrity, multi-turn correction, solver failure preservation, and evidence labeling. Use when agent or thermodynamic behavior changes.
---

# Phase-equilibrium evals

## Use cases

Use for regression protection across natural-language, tool, numerical, and validation behavior.

## Inputs

Require a public seam, a scenario, independent expected behavior or invariant, and only explicitly
synthetic fixtures under `tests/fixtures`.

## Outputs

Produce deterministic pytest or frontend tests with a clear failure signal and no production data
contamination.

## Procedure

1. Test behavior through orchestrator, backend protocol, CLI, HTTP, or browser surface.
2. Separate knowledge questions from numerical tasks.
3. Cover missing conditions, standard-pressure assumptions, follow-up deltas, LLE/Wilson exclusion,
   electrolyte refusal, missing parameters, and failure preservation.
4. Verify physical invariants independently of the implementation formula where possible.
5. Run focused tests, then the full suite and static checks.

## Prohibitions

Do not test private helpers, recreate the implementation in assertions, expose synthetic parameters
in production, or accept non-converged output as successful.

## Failure handling

Classify whether the regression is semantic, contract, numerical, physical, evidence, or UI; fix the
smallest owning layer and retain the regression test.

## Related files

`tests`, `evals`, `AGENTS.md`, `docs/validation.md`.

## Acceptance

Tests fail when required behavior is removed, remain stable across internal refactors, and cover the
scientific and agent safety rules in the product specification.
