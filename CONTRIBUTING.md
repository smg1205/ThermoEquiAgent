# Contributing to ThermoEqui-Agent

This project is developed with GitHub Issues, short branches and Pull Requests. Two
boundaries must hold for every change:

1. The LLM only recognises intent, plans, calls tools and explains. It must not generate
   thermodynamic calculation results directly.
2. Numerical results must come from `thermo_engine` and pass
   `validate_equilibrium_result`.

## Development environment

Python 3.11 or 3.12, Node.js 22 and pnpm 11 are recommended.

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -e ".[dev,phase-engines]"
pnpm --dir apps/web install --frozen-lockfile
Copy-Item .env.example .env
```

`.env` is kept on your machine only. Never commit API keys, database files, logs, build
directories or test artifacts.

## Claiming a task

1. Create or claim an Issue on GitHub, stating the goal, scope and acceptance criteria.
2. One branch handles one Issue. Recommended naming:
   - `feat/123-phasepy-vle`
   - `fix/234-flash-composition`
   - `docs/345-model-guide`
3. Sync the main branch before starting:

```powershell
git switch main
git pull --ff-only
git switch -c feat/123-short-name
```

## Local verification

Backend and scientific core:

```powershell
python -m pytest
ruff check .
ruff format --check .
mypy .
```

Frontend:

```powershell
pnpm --dir apps/web test
pnpm --dir apps/web lint
pnpm --dir apps/web build
```

When changing the Phasepy or Clapeyron adapters, also run:

```powershell
python -m pytest tests/test_external_backends.py
```

The full Clapeyron integration test requires Julia and `RUN_CLAPEYRON_INTEGRATION=1`.

## Scientific change checklist

- The new model implements `ThermodynamicBackend` and does not expose third-party library
  objects to the API.
- Missing binary parameters return a structured `missing_parameters` failure; no default
  parameters are invented.
- Parameters carry their source, form, direction, units, applicability range and version.
- New numerical paths pass through `validate_equilibrium_result`.
- At least one behavioural test is added at a public backend or HTTP boundary.
- Test fixtures live only in `tests/fixtures`, and production code imports no test data.
- Tasks beyond the v0.1 scope are still rejected explicitly.

## Contract change checklist

When changing `schemas/domain.py`:

1. Synchronise `apps/web/src/lib/types.ts`.
2. Synchronise the API request/response and the OpenAPI tests.
3. Update `tests/test_frontend_contract.py`.
4. If persistence is affected, synchronise `database/models.py` and the repository
   conversion logic.

## Commits and Pull Requests

Use concise Conventional Commits:

```text
feat(engine): add reviewed Wilson parameter contract
fix(agent): preserve explicit flash feed composition
docs: clarify Phasepy applicability
```

After pushing, open a Pull Request and link the Issue:

```powershell
git push -u origin feat/123-short-name
```

A PR should stay reviewable and state the scientific assumptions, parameter sources, test
evidence and known limitations. Merge only after at least one other member has approved and
CI is green. Squash merge is preferred, to keep main branch history clean.

## Review priorities

Review in this order:

1. Scientific boundaries and parameter evidence;
2. Numerical validation and failure behaviour;
3. API and frontend contract consistency;
4. Maintainability, tests and documentation;
5. UI presentation and engineering ergonomics.

For the full directory guide see [docs/repository-guide.en.md](docs/repository-guide.en.md).
