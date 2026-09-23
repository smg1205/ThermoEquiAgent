# ThermoAgent

> **English** | [中文](readme.zh-CN.md)

ThermoAgent is a conversational thermodynamic engineering workbench. It turns a
natural-language problem statement into a reproducible VLE / LLE calculation and, when
asked, into a flowsheet file that opens in DWSIM.

The design goal is traceability: every number in the output is attributable to an
experimental source, a ThermoFormer prediction, or a DWSIM calculation. The language
model routes and explains — it never invents an equilibrium value.

## What it does

- **Binary and ternary VLE** — bubble point, dew point, isobaric and isothermal VLE,
  TP flash, and azeotrope search.
- **Binary and ternary LLE** — liquid-liquid coexistence endpoints and phase splits.
- **Distillation design** — short-cut Fenske / Underwood / Gilliland sizing (stages,
  reflux ratio, feed stage, product temperatures) for binary and extractive columns.
- **DWSIM export** — generates `.dwxmz` flowsheets for TP flash, binary distillation,
  extractive distillation, and liquid-liquid extraction.
- **Knowledge graph reasoning** — a thermodynamics knowledge graph over models, tasks,
  parameters and system types, supporting multi-hop queries with exclusion reasoning.
- **Retrieval-augmented answering** — concept and process questions are answered from
  versioned knowledge documents with source attribution.
- **Three-source validation** — experimental data, ThermoFormer prediction, and DWSIM
  are compared side by side so a model's reliability can be judged rather than assumed.

Out of scope in v0.1 (rejected explicitly): electrolytes, reactive equilibrium, SLE,
polymers, hydrates, petroleum pseudocomponents, and polymorphs.

## How it works

```
natural language
      │
      ▼
  intent routing ──► component resolution ──► model selection
      │                                            │
      │                        ┌───────────────────┴───────────────────┐
      ▼                        ▼                                       ▼
  missing-parameter      thermo_engine                        ThermoFormer
  report (no guessing)   (classical models)                   (neural VLE / LLE)
                               └───────────────┬───────────────────────┘
                                               ▼
                                    equilibrium validation
                                               ▼
                              design / report / DWSIM .dwxmz
```

Model selection is deterministic and happens outside the language model:

| Task | Protocol | Checkpoint |
|---|---|---|
| Binary VLE | `vle_overall_binary` | `models/vle/prediction/vle_overall_binary/seed_2/best_model.pt` |
| Ternary VLE | `vle_overall_ternary` | `models/vle/prediction/vle_overall_ternary/seed_2/best_model.pt` |
| Binary LLE | `binary-system` | `models/lle/prediction/binary-system/seed_0/best.pt` |

When required information is missing, the system returns a structured
`missing_parameters` failure instead of filling the gap with an assumption.

## Knowledge graph

`knowledge_graph/` holds a thermodynamics knowledge graph used for model-applicability
reasoning. It is built from the same reviewed model cards that drive routing, and it is
persisted to `data/knowledge_graph.json`.

**Entities** are extracted from text by `ThermoEntityExtractor`, which resolves
thermodynamic terms into typed nodes: `model`, `task`, `component`, `system_type`,
`parameter` and `property`. Entity resolution is exact, so `propane` never matches the
node for `propanol`.

**Relationships** connect those nodes with a fixed vocabulary
(`RelationshipType` in `knowledge_graph/graph.py`):

| Relationship | Meaning |
|---|---|
| `supports_task` | A model can perform a task (VLE, FLASH, LLE) |
| `excludes` | A model is explicitly not applicable to a system type |
| `requires_parameter` | A model needs a binary parameter set to execute |
| `uses_model` | A task or report is backed by a given model |
| `has_property` | A component or model carries a property |
| `has_relationship` | Two nodes are associated without a more specific type |
| `belongs_to` | A node belongs to a family or group |
| `describes` | A document or card describes a node |
| `applies_to` | A parameter or rule applies to a system type |

Two design points are worth noting. Node identifiers carry a type prefix
(`model:peng-robinson` versus `task:pr`), so identical short names in different
namespaces cannot collide. Exclusion is modelled as an explicit `excludes` edge rather
than the absence of a `belongs_to` edge, which is what makes negative questions
answerable.

**Querying** goes through `GraphQueryEngine`, which extracts entities from a question,
finds the matching nodes, and walks one hop to collect their relationships. This
supports multi-hop questions such as "which systems does NRTL not apply to?", where the
answer comes from following `excludes` edges rather than from pattern-matching text.
`GraphQuerySkill` wraps the engine for the agent, adding an optional LLM explanation
layer on top of the graph result — the graph supplies the facts, the LLM only phrases
them.

The graph can be exported to GraphML with `KnowledgeGraph.export_graphml()` for
inspection in standard graph viewers, and `build_graph_from_kb()` rebuilds it from the
YAML model cards under `knowledge/`.

## Retrieval and skills

- `rag/` — document loading, splitting, embedding and vector-store retrieval over the
  knowledge base.
- `skills/` — typed capabilities the agent may invoke: thermodynamic calculation, model
  routing, validation, knowledge-base Q&A, graph query, and phase-equilibrium
  architecture and evaluation skills. Each skill declares its inputs, outputs and
  failure behaviour.
- `evals/` — agent behaviour evaluations covering fabrication, ambiguity, prompt
  injection, omission and scope overreach.

## Worked example

A complete end-to-end case is documented separately:
**[docs/heptane-nonane-case.en.md](docs/heptane-nonane-case.en.md)**.

It covers the n-heptane / n-nonane direct binary distillation reference system, the
three-source bubble-point comparison against NIST ThermoML data, the column design, the
DWSIM export, and the exact prompts to reproduce it from the workbench. It is the
recommended first case to run, because it needs no entrainer selection.

A screen recording of the full run is in
[`demo_video/heptane-nonane-dwsim-demo.mp4`](demo_video/heptane-nonane-dwsim-demo.mp4)
(3.8 MB).

## Layout

| Path | Contents |
|---|---|
| `agent/` | Intent routing, orchestration, bounded execution graph |
| `thermo_engine/` | Classical models, column design, DWSIM export |
| `knowledge_graph/` | Thermodynamic knowledge graph and query engine |
| `knowledge/` | Model cards, parameter data, validation benchmarks |
| `rag/` | Retrieval-augmented generation pipeline |
| `skills/` | Agent skills and their contracts |
| `apps/` | FastAPI backend and React workbench |
| `schemas/` | Pydantic domain models and API contracts |
| `database/` | Persistence models and sessions |
| `lab_models/` | ThermoFormer source, configs, datasets, weights |
| `scripts/` | Reproduction entry points |
| `demo_video/` | Screen recordings of the workbench |
| `tests/`, `evals/` | Behavioural tests and agent evaluations |
| `docs/` | Architecture, DWSIM and methodology documentation |

## Scientific rules

These are enforced in the code, not just documented:

1. The language model may classify, retrieve, orchestrate, and explain — never
   calculate or invent an equilibrium value.
2. Every numerical result comes from `thermo_engine` and passes validation.
3. A solver status is not physical validation: composition, material balance,
   equilibrium residuals, convergence, and parameter applicability are checked.
4. Missing parameters produce a structured `missing_parameters` failure. Binary
   parameters, experimental data, and citations are never fabricated.

## Installation

### Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 or newer (3.12 recommended) | Phasepy has no Windows wheel for 3.13 yet |
| Node.js | 22 or newer | The frontend runs on Next.js 16 |
| pnpm | 11.9.0 | Declared by `packageManager`; activate it with Corepack |
| DWSIM | 9.x, Windows only | Needed only to open exported `.dwxmz` files |

### 1. Backend

Create a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1

# Option A — install the pinned dependency list
python -m pip install -r requirements.txt
```

Prefer installing the project itself, which also registers the `thermoequi` and
`thermoequi-seed` console scripts and lets you select dependency groups:

```powershell
# Option B — core runtime only
python -m pip install -e .

# ...plus every optional group
python -m pip install -e ".[dev,phase-engines,thermoformer,dwsim]"
```

The optional groups map to the sections of `requirements.txt`:

| Group | Adds | Needed for |
|---|---|---|
| `dev` | pytest, mypy, ruff | Running the test and lint suites |
| `phase-engines` | Phasepy, pyclapeyron | The Phasepy and Clapeyron backends |
| `thermoformer` | torch, rdkit, unimol-tools, scikit-learn | The ThermoFormer neural backend |
| `dwsim` | pythonnet | DWSIM automation (Windows, needs a .NET runtime) |
| `skills` | sentence-transformers | Retrieval over the knowledge base |

Then create your local environment file and seed the reviewed parameter store:

```powershell
Copy-Item .env.example .env
thermoequi-seed          # loads knowledge/parameters/*.yaml into the database
```

`.env` stays on your machine — it is git-ignored. Leave the checkpoint variables
blank if you do not have the private model weights; the corresponding backends
stay registered and return a structured `missing_parameters` failure instead of
fabricating numbers.

### 2. Frontend

The frontend is a pnpm workspace member under `apps/web`. Install pnpm through
Corepack if it is not already available, then install the dependencies:

```powershell
corepack enable          # makes the pinned pnpm 11.9.0 available
corepack prepare pnpm@11.9.0 --activate

pnpm --dir apps/web install --frozen-lockfile
```

`--frozen-lockfile` installs exactly the versions in `apps/web/pnpm-lock.yaml`,
which is what CI uses. Drop the flag only when you intend to change a dependency.

Alternatively, from the repository root:

```powershell
pnpm install --frozen-lockfile
```

### 3. Run it

Two processes, two terminals:

```powershell
# Terminal 1 — backend on http://localhost:8000
python -m uvicorn apps.api.main:app --reload --port 8000

# Terminal 2 — frontend on http://localhost:3000
pnpm --dir apps/web dev
```

Open `http://localhost:3000` for the workbench and `http://localhost:8000/docs`
for the OpenAPI browser. Confirm the service and provider status at
`http://localhost:8000/health`.

The default `LLM_PROVIDER=deterministic` needs no API key: intent routing, model
selection and calculation then run entirely on local rules. Set
`LLM_PROVIDER=deepseek` together with `DEEPSEEK_API_KEY` in `.env` to let the
model handle phrasing and explanation as well.

### 4. Containerised alternative

```powershell
docker compose up --build     # backend on 8000, frontend on 3000
```

SQLite is the default database. For a migration-ready deployment, set a
PostgreSQL SQLAlchemy URL in `.env`.

## Development

```powershell
python -m pytest                      # backend tests
ruff check . && ruff format --check . # lint and formatting
mypy .                                # strict type checking
pnpm --dir apps/web test              # frontend tests
pnpm --dir apps/web lint              # frontend lint
pnpm --dir apps/web build             # frontend production build
```

`python -m pytest` reads its configuration from `pyproject.toml` and runs both
`tests/` and `evals/`.

## Documentation

Technical documentation lives under `docs/` and is written in English, using an `.en.md`
suffix. A Chinese version of this readme is available at
[readme.zh-CN.md](readme.zh-CN.md).

Notable entry points:

| Document | Contents |
|---|---|
| `docs/heptane-nonane-case.en.md` | The worked end-to-end example |
| `docs/repository-guide.en.md` | Directory and file guide |
| `docs/agent-architecture.en.md` | Agent orchestration in detail |
| `docs/model_applicability.en.md` | Model scope and filtering rules |
| `docs/dwsim-dwxmz-export-guide.en.md` | DWSIM file generation and format |
| `docs/ThermoFormer.en.md` | The ThermoFormer model and its results |
