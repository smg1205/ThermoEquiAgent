---
name: phase-equilibrium-architecture
description: Design or review ThermoEqui-Agent service boundaries, deterministic solver isolation, adapters, schemas, persistence, and deployment. Use for architecture changes that cross the agent, API, thermodynamic core, database, or frontend.
---

# Phase-equilibrium architecture

## Use cases

Use for cross-layer design, new calculation families, backend integrations, or schema migrations.

## Inputs

Read the requested capability, `AGENTS.md`, `PLANS.md`, affected schemas, backend protocol, and API
types.

## Outputs

Produce a bounded design, updated contracts, implementation, tests, and architecture documentation.

## Procedure

1. Place every thermodynamic number behind `ThermodynamicBackend`.
2. Keep orchestration, calculation, validation, persistence, and presentation separate.
3. Define the Pydantic contract before wiring API and TypeScript types.
4. Store immutable input, model, parameter, version, result, and validation snapshots.
5. Add a public-boundary test and update architecture documentation.

## Prohibitions

Do not let LLM output become calculation data. Do not expose third-party internal objects. Do not
broaden phase scope implicitly or couple core calculations to HTTP/database state.

## Failure handling

Return a structured unsupported or configuration failure, preserve diagnostics, and leave existing
backends operational.

## Related files

`docs/architecture.md`, `schemas/domain.py`, `thermo_engine/backend.py`, `apps/api/main.py`.

## Acceptance

The core runs without LLM/API, contracts stay synchronized, snapshots are reproducible, and tests
cover the new public seam.
