# Repository instructions

## Goal and scope

Build a conversational engineering workbench for non-electrolyte molecular VLE, bubble/dew point,
TP flash, azeotrope search, validation, and an LLE contract. Version 0.1 excludes electrolytes,
reactive equilibrium, SLE, polymers, hydrates, petroleum pseudocomponents, polymorphs, VLLE, and
flowsheet design. Reject those tasks explicitly.

## Non-negotiable scientific rules

- The LLM may classify, formulate, retrieve, orchestrate, and explain. It must never invent or
  directly calculate equilibrium numbers.
- Never fabricate binary parameters, experimental data, or citations. Missing parameters must
  produce a structured `missing_parameters` failure.
- Every numerical result must come from `thermo_engine` and pass `validate_equilibrium_result`.
- A solver status alone is not physical validation. Check composition, material balance,
  equilibrium residuals, convergence, parameter applicability, and phase stability when relevant.
- Keep `test_fixture` and synthetic parameters under `tests/fixtures`; never expose them through the
  production parameter repository or an engineering report.

## Engineering rules

- Python: typed public interfaces, Ruff formatting/lint, strict mypy, Pydantic boundaries.
- TypeScript: strict mode, shared API types in `apps/web/src/lib/types.ts`.
- Schema changes require synchronized API models, frontend types, and contract tests.
- Thermodynamic core changes require behavioral tests at the public backend or CLI seam.
- Never commit secrets. Never log API keys. Production code must not import test fixtures.

## Commands

- Backend dev: `python -m uvicorn apps.api.main:app --reload --port 8000`
- Frontend dev: `pnpm --dir apps/web dev`
- Backend tests: `python -m pytest`
- Python checks: `ruff check . && ruff format --check . && mypy .`
- Frontend checks: `pnpm --dir apps/web test && pnpm --dir apps/web lint && pnpm --dir apps/web build`
- Full stack: `docker compose up --build`
