# ThermoAgent

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
**[docs/heptane-nonane-case.en.md](docs/heptane-nonane-case.en.md)** (Chinese:
[docs/heptane-nonane-case.zh-CN.md](docs/heptane-nonane-case.zh-CN.md)).

It covers the n-heptane / n-nonane direct binary distillation reference system, the
three-source bubble-point comparison against NIST ThermoML data, the column design, the
DWSIM export, and the exact prompts to reproduce it from the workbench. It is the
recommended first case to run, because it needs no entrainer selection.

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
| `tests/`, `evals/` | Behavioural tests and agent evaluations |
| `report/` | Validation reports and archived case artifacts |
| `docs/` | Bilingual architecture, DWSIM and methodology documentation |

## Scientific rules

These are enforced in the code, not just documented:

1. The language model may classify, retrieve, orchestrate, and explain — never
   calculate or invent an equilibrium value.
2. Every numerical result comes from `thermo_engine` and passes validation.
3. A solver status is not physical validation: composition, material balance,
   equilibrium residuals, convergence, and parameter applicability are checked.
4. Missing parameters produce a structured `missing_parameters` failure. Binary
   parameters, experimental data, and citations are never fabricated.

## Development

```powershell
python -m pytest                      # backend tests
ruff check . && mypy .                # lint and types
pnpm --dir apps/web test              # frontend tests
docker compose up --build             # full stack
```

## Documentation

All documentation is maintained in both Chinese and English under `docs/`. Files use a
`.zh-CN.md` or `.en.md` suffix, and every document has a counterpart in the other
language.

Notable entry points:

| Document | Contents |
|---|---|
| `docs/heptane-nonane-case.en.md` | The worked end-to-end example |
| `docs/repository-guide.en.md` | Directory and file guide |
| `docs/agent-architecture.en.md` | Agent orchestration in detail |
| `docs/model_applicability.en.md` | Model scope and filtering rules |
| `docs/dwsim-dwxmz-export-guide.en.md` | DWSIM file generation and format |
| `docs/ThermoFormer.en.md` | The ThermoFormer model and its results |

The full three-source validation report is archived under `report/`.
