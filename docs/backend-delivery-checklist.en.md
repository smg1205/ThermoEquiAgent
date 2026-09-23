# Backend phase-one delivery checklist

## 1. Environment

Python 3.12 is recommended. The current local environment lives at `E:\anaconda3\envs\thermoequi`.

```powershell
conda activate E:\anaconda3\envs\thermoequi
Copy-Item .env.example .env
python -m uvicorn apps.api.main:app --reload --port 8000
```

`LLM_PROVIDER=deterministic` by default, so the backend loop needs no external API key.

## 2. Demonstration order

1. Open `http://localhost:8000/health` and confirm service and provider status.
2. Open `http://localhost:8000/docs` and confirm the OpenAPI routes.
3. Call `/api/chat` with "compute the T-x-y curve of benzene-toluene at 101.325 kPa".
4. Take the `run_id` from the response and call `/api/runs` and `/api/runs/{run_id}`.
5. Call `/api/runs/{run_id}/export?format=json` and `format=csv`.
6. Show the validation report, parameter sources, backend version and request ID.
7. Submit requests with missing parameters, illegal ranges, and cases beyond the current scientific boundary to demonstrate structured failure.

## 3. Quality gates

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy agent apps database schemas thermo_engine
```

When a public schema changes, also run the frontend tests, lint and build.

## 4. Backend acceptance points

- All public inputs and outputs pass through Pydantic.
- Chat messages, tasks, runs, validations and evidence are written atomically.
- Direct calculations also persist their corresponding TaskRow.
- Run history supports reverse-chronological paging and status filtering.
- Parameter values are finite, units complete, applicability ranges legal, and sources traceable.
- Duplicate parameter sets return `409 duplicate_parameter_set`.
- Unexpected exceptions return a sanitized `500 internal_server_error`.
- Every normal and error response carries `X-Request-ID`.
- All numbers come from `thermo_engine` and pass independent validation.
- Test fixtures never enter the production parameter repository.

## 5. Current non-delivery scope

- Electrolytes, reactive equilibrium, SLE, polymers, hydrates, petroleum pseudocomponents, VLLE and flowsheet design.
- Production NRTL/UNIQUAC LLE, parameter regression, multi-model sensitivity analysis.
- User authentication and authorisation, formal Alembic migrations, and PostgreSQL driver deployment.
