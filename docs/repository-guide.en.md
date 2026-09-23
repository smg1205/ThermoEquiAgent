# Project Directory and File Guide

This document is the navigation map for the current repository. The principle is: source code, contracts,
tests, knowledge evidence, and collaboration configuration are tracked in Git; local environments, secrets,
databases, dependency directories, and build caches stay on the development machine.

## Root directory

| Path | Purpose |
|---|---|
| `.dockerignore` | Controls the Docker build context, excluding local dependencies, caches, and sensitive files. |
| `.env.example` | Environment variable template; contains only variable names and safe defaults, never real keys. |
| `.gitattributes` | Unifies Git text and line-ending behavior. |
| `.gitignore` | Excludes virtual environments, caches, databases, logs, frontend dependencies, and build artifacts. |
| `AGENTS.md` | Repository-level scientific boundaries and engineering rules for AI coding agents. |
| `CONTRIBUTING.md` | Team standards for branches, testing, commits, PRs, and scientific review. |
| `PLANS.md` | Architecture decisions, version phases, and delivery status. |
| `README.md` | Project entry point: capabilities, architecture, startup, testing, API, and known limitations. |
| `docker-compose.yml` | One-command container orchestration for the API, Web, and persistent volumes. |
| `package.json` | Root-level pnpm shortcut commands, forwarded to `apps/web`. |
| `pyproject.toml` | Python package metadata, dependencies, CLI, pytest, Ruff, and mypy configuration. |

## GitHub configuration

| Path | Purpose |
|---|---|
| `.github/workflows/ci.yml` | CI for the backend, frontend, and optional Phasepy/Clapeyron integrations. |
| `.github/pull_request_template.md` | PR checklist for scientific safety, contract synchronization, and validation evidence. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured issue form for reproducible bugs. |
| `.github/ISSUE_TEMPLATE/model_integration.yml` | Evidence and acceptance form for new models, parameter sources, or backend integrations. |
| `.github/ISSUE_TEMPLATE/config.yml` | Blocks unstructured blank issues and guides the team toward the templates. |

## Agent orchestration: `agent/`

| Path | Purpose |
|---|---|
| `agent/__init__.py` | Agent package boundary and public exports. |
| `agent/providers.py` | Deterministic, DeepSeek, and OpenAI providers; constrains structured output and filters unsupported content. |
| `agent/orchestrator.py` | Conversation state, component identification, intent classification, task list construction, and multi-turn context merging. |
| `agent/graph_workflow.py` | Bounded LangGraph: `plan → execute → validate → respond`. |
| `agent/tools.py` | Whitelist of tools the Agent may call; currently only exposes the controlled phase equilibrium tools. |
| `agent/router.py` | Reads model cards, builds a system profile, and recommends models by applicability. |
| `agent/executor.py` | Calls the deterministic service and enforces passage through the independent validation gate. |

This is the connection layer for "thinking + execution"; no LLM logic that computes phase equilibrium
numbers on its own is permitted here.

## API and frontend: `apps/`

| Path | Purpose |
|---|---|
| `apps/__init__.py` | Application package marker. |
| `apps/api/__init__.py` | FastAPI subpackage marker. |
| `apps/api/main.py` | Application entry point, provider configuration, middleware, exception mapping, and chat/computation/model/parameter/run-export endpoints. |
| `apps/api/Dockerfile` | Backend production image build and Uvicorn startup. |
| `apps/web/Dockerfile` | Next.js frontend production image build. |
| `apps/web/package.json` | Frontend dependencies and dev/test/lint/build commands. |
| `apps/web/pnpm-lock.yaml` | Pins the frontend dependency tree exactly, keeping CI and member environments consistent. |
| `apps/web/pnpm-workspace.yaml` | pnpm workspace configuration. |
| `apps/web/next.config.ts` | Next.js configuration. |
| `apps/web/next-env.d.ts` | Next.js automatic type declaration entry point. |
| `apps/web/tsconfig.json` | TypeScript strict mode, path aliases, and compiler settings. |
| `apps/web/eslint.config.mjs` | ESLint and Next.js rules. |
| `apps/web/vitest.config.ts` | Vitest, jsdom, and test setup configuration. |
| `apps/web/src/app/layout.tsx` | Global HTML layout and page metadata. |
| `apps/web/src/app/page.tsx` | Home page entry point, mounting the engineering workbench. |
| `apps/web/src/app/globals.css` | Global layout, color scheme, form, chart, and responsive styles. |
| `apps/web/src/components/Workbench.tsx` | Main interactive interface: conversation, task editing, runs, result tabs, and export. |
| `apps/web/src/components/VleChart.tsx` | Renders unified equilibrium point results as a Plotly phase diagram. |
| `apps/web/src/components/Workbench.test.tsx` | Workbench tests for chat, computation, failure messages, and recompute interactions. |
| `apps/web/src/components/VleChart.test.tsx` | Phase diagram component and data mapping tests. |
| `apps/web/src/lib/api.ts` | Typed HTTP client from the browser to FastAPI, with unified error handling. |
| `apps/web/src/lib/types.ts` | TypeScript types synchronized with the Pydantic API contract. |
| `apps/web/src/test/setup.ts` | Vitest DOM matchers and browser test setup. |
| `apps/web/src/types/react-plotly.d.ts` | Local type augmentation for `react-plotly.js`. |

## Deterministic thermodynamics core: `thermo_engine/`

| Path | Purpose |
|---|---|
| `thermo_engine/__init__.py` | Exposes the computation, routing, validation, and registry entry points. |
| `thermo_engine/backend.py` | The `ThermodynamicBackend` protocol that every computation backend must implement. |
| `thermo_engine/registry.py` | Central registry of backend aliases, supported tasks, and constructors. |
| `thermo_engine/service.py` | Public computation boundary: range rejection, identity verification, model routing, backend invocation, and validation entry point. |
| `thermo_engine/ideal.py` | Built-in Ideal/Raoult bubble point, dew point, VLE, Flash, and azeotrope candidate solving. |
| `thermo_engine/thermo_backend.py` | Peng–Robinson adapter for CalebBell/thermo. |
| `thermo_engine/phasepy_backend.py` | Optional Phasepy/Peng–Robinson adapter. |
| `thermo_engine/clapeyron_backend.py` | Calls Clapeyron.jl/Peng–Robinson through pyclapeyron. |
| `thermo_engine/properties.py` | Pure-component identity, Antoine correlations, and source records. |
| `thermo_engine/identity.py` | Component name/CAS resolution, alias verification, electrolyte and ambiguity detection. |
| `thermo_engine/parameters.py` | Explicit reverse conversion of directional binary parameters. |
| `thermo_engine/units.py` | Explicit normalization of pressure, temperature, and molar composition. |
| `thermo_engine/validation.py` | Composition, material balance, equilibrium residual, convergence, applicability, and stability checks. |
| `thermo_engine/errors.py` | Structured domain exceptions such as missing parameters and unsupported ranges. |
| `thermo_engine/cli.py` | Command-line entry point for running deterministic computations from a JSON task. |

## Shared contracts: `schemas/`

| Path | Purpose |
|---|---|
| `schemas/__init__.py` | Schema package exports. |
| `schemas/domain.py` | Pydantic source of truth for tasks, conditions, model cards, parameters, results, validation, chat, and errors. |

Changes here must be synchronized with `apps/web/src/lib/types.ts`, the API endpoints, and the contract tests.

## Data persistence: `database/`

| Path | Purpose |
|---|---|
| `database/__init__.py` | Database package exports. |
| `database/models.py` | SQLAlchemy tables for sessions, messages, tasks, parameters, runs, equilibrium points, validation, and exports. |
| `database/session.py` | Database engine, initialization, transaction context, and repository reads/writes. |

By default a local `thermoequi.db` is generated; it is runtime data and does not enter Git.

## Models and knowledge assets: `knowledge/`

| Path | Purpose |
|---|---|
| `knowledge/model_cards/ideal.yaml` | Applicability, risks, and executable status of Ideal/Raoult. |
| `knowledge/model_cards/peng-robinson.yaml` | Applicability and parameter requirements of Peng–Robinson. |
| `knowledge/model_cards/wilson.yaml` | Wilson model card; currently a contract/planning asset. |
| `knowledge/model_cards/nrtl.yaml` | NRTL model card; currently a contract/planning asset. |
| `knowledge/model_cards/uniquac.yaml` | UNIQUAC model card; currently a contract/planning asset. |
| `knowledge/model_selection_rules/core.yaml` | Rules for filtering in/excluding models by phase state, pressure, system, and task. |
| `knowledge/fundamentals/vle.md` | VLE fundamentals and terminology. |
| `knowledge/engineering_cases/benzene-toluene.md` | Benzene–toluene ideal system demonstration case and boundaries. |
| `knowledge/parameter_guides/import.md` | Parameter import, direction, sources, and review requirements. |
| `knowledge/validation_guides/vle.md` | Engineering validation checklist for VLE results. |
| `knowledge/software_mappings/README.md` | Mapping notes between various third-party software/libraries and the unified model contract. |

Even when a model card is not yet executable, it still participates in explicit recommendation, rejection,
and the roadmap; it is not a deprecated file.

## In-project AI work standards: `skills/`

| Path | Purpose |
|---|---|
| `skills/agent-tool-contract/SKILL.md` | Agent tool whitelist, inputs/outputs, and safety boundaries. |
| `skills/frontend-engineering-workbench/SKILL.md` | Frontend development constraints for the engineering workbench. |
| `skills/phase-equilibrium-architecture/SKILL.md` | Phase equilibrium system layering and deterministic boundaries. |
| `skills/phase-equilibrium-evals/SKILL.md` | Agent scientific evaluation and failure-case requirements. |
| `skills/thermodynamic-calculation/SKILL.md` | Deterministic computation implementation and testing rules. |
| `skills/thermodynamic-model-routing/SKILL.md` | Model applicability routing rules. |
| `skills/thermodynamic-validation/SKILL.md` | Requirements of the independent physical validation gate. |
| `skills/*/agents/openai.yaml` | Agent discovery and invocation metadata for the corresponding local Skill. |

These files are project knowledge for later collaboration with Codex/other agents; they do not participate in
production runs but should be kept in the repository.

## Documentation: `docs/`

Documents are written in English, using an `.en.md` suffix.

| Document | Purpose |
|---|---|
| `architecture` | System layering, data flow, and safety boundaries. |
| `agent-architecture` | Agent orchestration, intent routing, and the bounded execution graph. |
| `agent-thinking-execution` | Summary of the "thinking + execution" implementation status. |
| `api` | FastAPI routes, request/response, and error contracts. |
| `deployment` | Local and container deployment instructions. |
| `frontend` | Frontend layout, state, and interface behavior. |
| `integrations` | Integration matrix for the CAi_copilot approach, LangGraph, thermo, Phasepy, and Clapeyron. |
| `methodology` | Engineering method, assumptions, and computation flow. |
| `model_applicability` | Model scope, catalog status, and candidate filtering rules. |
| `model_routing` | Model filtering, scoring, and hard-exclusion logic. |
| `parameter_evidence` | Parameter evidence, traceability, and test fixture isolation. |
| `heptane-nonane-case` | Worked end-to-end example: three-source comparison, design, DWSIM export. |
| `repository-guide` | This file; repository map and retention policy. |
| `roadmap` | Upcoming phases and unfinished capabilities. |
| `thermodynamic_scope` | Scientific scope supported by v0.1 and explicitly rejected. |
| `validation` | Independent physical validation rules and status interpretation. |
| `pgssi_checkpoint` | Configuring private PGSSI weights without committing them. |
| `dwsim-automation-api` | DWSIM Automation bubble-point solving API. |
| `dwsim-dwxmz-export-guide` | Generation and format of `.dwxmz` project files. |
| `dwsim-extraction-export` | Phase-equilibrium and extraction export guide. |
| `extractive-dwsim-usage` | Using exported extractive-distillation files in the DWSIM GUI. |
| `ThermoFormer` | The ThermoFormer model, its formulation and its results. |

## Examples, tests, and evaluations

| Path | Purpose |
|---|---|
| `examples/benzene_toluene_isobaric.json` | Example isobaric benzene–toluene VLE input that the CLI can run directly. |
| `tests/fixtures/synthetic_nrtl.json` | Synthetic NRTL parameters for testing only; production code must never import them. |
| `tests/test_api.py` | HTTP, OpenAPI, chat, DeepSeek correction, run persistence, and export tests. |
| `tests/test_database.py` | Database and test fixture isolation tests. |
| `tests/test_deepseek_provider.py` | DeepSeek protocol, structured output, safety filtering, and orchestration boundary tests. |
| `tests/test_external_backends.py` | Phasepy/Clapeyron optional dependency, parameter, and validation gate tests. |
| `tests/test_frontend_contract.py` | Field synchronization tests between the Python API and TypeScript types. |
| `tests/test_schemas_validation.py` | Pydantic constraint and nonconforming result validation tests. |
| `tests/test_thermo_engine.py` | Units, bubble/dew point, VLE, Flash, routing, missing parameters, ranges, and physical invariant tests. |
| `evals/test_agent.py` | Agent behavior evaluations for natural-language tasks, including ambiguity, out-of-scope, and anti-fabrication. |

## Local content that does not enter Git

| Path | Handling |
|---|---|
| `.venv312/`, `.venv/` | Python virtual environments; kept on the local machine, rebuildable from the dependency list. |
| `apps/web/node_modules/` | Frontend dependencies; kept on the local machine, rebuildable from the lockfile. |
| `apps/web/.next/` | Next.js dev/build artifacts; can be deleted and rebuilt automatically. |
| `.mypy_cache/`, `.ruff_cache/`, `__pycache__/` | Static analysis and Python caches; can be deleted at any time. |
| `.coverage`, `htmlcov/`, `coverage/` | Test coverage artifacts; can be deleted at any time. |
| `thermoequi_agent.egg-info/` | Editable install metadata; regenerated on reinstall. |
| `thermoequi.db`, `*.db`, `*.sqlite3` | Local runtime data; not committed, and confirm whether history is needed before deleting. |
| `.env` | Local secrets and environment configuration; never committed. |

## Core call chain

```text
apps/web
  → apps/api/main.py
  → agent/orchestrator.py
  → agent/graph_workflow.py
  → agent/tools.py
  → agent/executor.py
  → thermo_engine/service.py
  → thermo_engine/registry.py
  → specific backend
  → thermo_engine/validation.py
  → API / database / frontend
```

Team members should first locate a problem along this call chain and then state the responsible layer in the
issue; this significantly reduces cross-layer edits and contract conflicts.
