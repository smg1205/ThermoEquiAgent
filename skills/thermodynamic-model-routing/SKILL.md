---
name: thermodynamic-model-routing
description: Select and explain thermodynamic model candidates using hard exclusions, system rules, parameter evidence, applicability ranges, and post-calculation validation. Use for VLE, LLE, flash, and model-selection requests.
---

# Thermodynamic model routing

## Use cases

Use when recommending, ranking, excluding, or changing an equilibrium model.

## Inputs

Require a `TaskManifest`, program-derived `SystemProfile`, model cards, and available parameter sets.

## Outputs

Return ranked candidates with score components, exclusions, executability, risks, and reasons.

## Procedure

1. Reject unsupported systems before scoring.
2. Exclude models that do not support the phase task; exclude Wilson from LLE.
3. Require parameters declared by the model card.
4. Score phase, system, conditions, availability, evidence, extrapolation, and numerical risk.
5. Select only after solver convergence and validation are considered.

## Prohibitions

Do not return only a model name. Do not invent parameters. Do not treat an LLM profile as verified
facts when rules or structured component data can determine it.

## Failure handling

Return `unsupported_model`, `missing_parameters`, `parameter_out_of_domain`, or `model_conflict`
with a deterministic recovery action.

## Related files

`agent/router.py`, `knowledge/model_cards`, `docs/model_routing.md`.

## Acceptance

Every candidate has auditable score terms; hard exclusions cannot be overridden; unavailable
required parameters make the candidate non-executable.
