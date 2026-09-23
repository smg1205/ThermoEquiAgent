# ThermoEqui-Agent team roles and delivery standard

Version: v0.1

Applies to: phase-one of the phase-equilibrium engineering workbench

Maintenance: before each Milestone begins, update the owner, delivery date and acceptance status

> "RAG" in this document means retrieval-augmented generation. If the team uses "RAD" internally for
> some other technique, add an explicit definition before development starts.

## 1. Shared phase-one goal

Phase one first completes an end-to-end loop in the phase-equilibrium domain:

```text
user question
-> DeepSeek + RAG + Skill constraints
-> structured task TaskManifest
-> model selection and parameter checks
-> deterministic phase-equilibrium calculation
-> independent physical validation
-> frontend display, traceability and export
```

Before phase one is complete, do not start production implementations of retrosynthesis, catalyst
screening, tray monitoring or impeller design in parallel. Domain plugin interfaces may be reserved,
but later domain requirements must not be allowed to break the current phase-equilibrium loop.

## 2. Shared rules that must not be violated

1. DeepSeek is responsible only for recognition, planning, retrieval, tool invocation and
   explanation; it must not generate phase-equilibrium values directly.
2. All engineering numbers must come from deterministic calculation tools or from explicitly labelled
   AI prediction models.
3. Deterministic phase-equilibrium results must pass `validate_equilibrium_result`.
4. Fabricating binary parameters, experimental data, physical properties, applicability ranges or
   literature citations is forbidden.
5. When parameters are missing, return a structured `missing_parameters` rather than filling in a
   guessed default.
6. AI predictions must be labelled with model version, training-data version, applicability domain,
   uncertainty and validation status.
7. Test fixtures may live only in `tests/fixtures`; production code must not import them.
8. API keys are injected only through environment variables or GitHub Secrets, and never enter Git,
   logs, database snapshots or the frontend.
9. Every piece of work is completed through a GitHub Issue, a dedicated branch and a Pull Request.

## 3. Member claim table

| Working group | Lead owner | Backup / reviewer | Current Milestone |
|---|---|---|---|
| Project architecture and product coordination | To be filled | To be filled | M1 |
| Frontend interface | To be filled | To be filled | M1 |
| Backend and data contracts | To be filled | To be filled | M1 |
| DeepSeek, RAG and Skills | To be filled | To be filled | M1-M2 |
| Phase-equilibrium models and calculation backends | To be filled | To be filled | M1-M3 |
| Scientific validation and quality assurance | To be filled | To be filled | M1-M4 |
| CI, deployment and release | To be filled | To be filled | M1 |

One person may take several roles, but the implementer of a scientific model cannot be the sole
approver of that model.

## 4. Project architecture and product coordination

### Responsibilities

- Maintain the phase-one scope, roadmap, Milestones and priorities.
- Define the boundaries between frontend, API, agent, calculation backends and database.
- Chair reviews of public schemas, plugin interfaces and major architectural changes.
- Break large tasks into GitHub Issues that can be accepted independently.
- Handle cross-group dependencies and interface conflicts.
- Ensure the current version does not silently expand into electrolytes, reactive equilibrium, SLE,
  polymers, VLLE or flowsheet simulation.

### Deliverables

- Architecture diagrams and interface descriptions;
- Milestone and Issue lists;
- Draft domain plugin interfaces;
- Release notes;
- Records of major decisions.

### Acceptance criteria

- Every Issue has an owner, scope, dependencies and acceptance criteria;
- Public interface changes are reviewed jointly by the relevant working groups;
- Every Milestone produces a runnable vertical slice.

## 5. Frontend group

Owns: `apps/web/`

### Responsibilities

- Build the engineering conversation interface and task input area.
- Display and allow editing of components, temperature, pressure, composition, task type and model.
- Display model recommendations, applicability scores, missing-parameter reasons and
  non-executable reasons.
- Display phase diagrams, data tables, phase compositions, phase fractions and calculation status.
- Display parameter sources, model versions, validation reports and run records.
- Handle loading, timeouts, API failures, missing inputs, missing parameters, solver failures and
  validation failures.
- Support JSON/CSV export.
- Keep fields synchronised with `schemas/domain.py`.

### Phase-one Issues

| ID | Task | Deliverable |
|---|---|---|
| FE-01 | Workbench page skeleton | Conversation, task, result areas and responsive layout |
| FE-02 | Typed API client | Unified requests, error parsing, request ID |
| FE-03 | Task manifest editing | Editing and recomputation of conditions, composition, model |
| FE-04 | Result visualisation | T-x-y, P-x-y, Flash, tables |
| FE-05 | Evidence and validation view | Parameter sources, model cards, validation checks |
| FE-06 | Failure states | Missing parameters, inapplicable, non-convergence, gateway errors |
| FE-07 | Frontend automated tests | Key user flows and error flows |

### Acceptance criteria

- A user can tell which stage a task is at without looking at a terminal;
- A successful LLM answer is never displayed as a successful thermodynamic calculation;
- Every numerical result exposes its model, backend, parameter source and validation status;
- `pnpm --dir apps/web test`, `lint` and `build` all pass.

## 6. Backend and data contracts group

Owns: `apps/api/`, `schemas/`, `database/`

### Responsibilities

- Maintain the FastAPI, OpenAPI and Pydantic public contracts.
- Maintain the chat, task parsing, model recommendation, calculation, validation, run query and
  export endpoints.
- Unify error responses and request IDs.
- Persist the original question, task manifest, parameter snapshots, results, validation reports and
  software versions.
- Isolate third-party calculation library objects; the API returns only unified schemas.
- Maintain database initialisation, transactions and repositories.
- Design unified task and run metadata for later multi-domain plugins.

### Phase-one Issues

| ID | Task | Deliverable |
|---|---|---|
| BE-01 | Public task and result schemas | Pydantic source of truth and frontend contracts |
| BE-02 | Chat and task endpoints | `/api/chat`, `/api/tasks/parse` |
| BE-03 | Calculation endpoints | bubble, dew, VLE, Flash, azeotrope |
| BE-04 | Error protocol | Unified error codes, request ID, sanitisation |
| BE-05 | Run persistence | Conversation, task, calculation, validation and export records |
| BE-06 | OpenAPI and contract tests | Route coverage, field synchronisation, error tests |
| BE-07 | Domain plugin boundary | `domain`, `tool`, `result_kind` metadata |

### Acceptance criteria

- All public inputs and outputs pass Pydantic validation;
- Python schemas and TypeScript types stay in sync;
- The API never returns third-party library internal objects;
- Run records can be queried, exported and traced;
- `python -m pytest` and `mypy .` pass.

## 7. DeepSeek, RAG and Skills group

Owns: `agent/`, `skills/`, and the later retrieval modules

### Responsibilities

- Maintain the DeepSeek provider, timeouts, retries and error sanitisation.
- Distinguish knowledge questions, calculations, parameter lookups, model comparisons and
  out-of-scope tasks.
- Extract components, CAS numbers, conditions, compositions and task type from user text.
- Constrain DeepSeek output with JSON Schema/Pydantic.
- Establish RAG document ingestion, retrieval, versioning and review status.
- Establish the Skill registry, inputs and outputs, tool allowlist and failure protocol.
- Prevent prompt injection, component omission, condition tampering, parameter fabrication and
  unauthorised tool calls.
- Maintain agent behaviour evaluations.

### DeepSeek structured output

The recommended unified form:

```json
{
  "intent": "calculation",
  "domain": "phase_equilibrium",
  "task_manifest": {},
  "selected_tool": "phase_equilibrium",
  "missing_inputs": [],
  "assumptions": [],
  "answer_mode": "execute_then_explain"
}
```

### Phase-one Skills

| Skill | Purpose |
|---|---|
| `task-classification` | Decide between question, calculation, parameter, comparison or out-of-scope request |
| `component-grounding` | Bind natural-language components to verified identities |
| `thermodynamic-model-routing` | Select candidate models from system, phase and conditions |
| `parameter-retrieval` | Find reviewed parameters and their sources |
| `phase-equilibrium-calculation` | Invoke the single permitted phase-equilibrium tool |
| `thermodynamic-validation` | Invoke the independent validator and explain the checks |
| `engineering-explanation` | Produce engineering explanation grounded in real tool results |
| `scope-rejection` | Explicitly reject tasks this version does not support |

### Phase-one Issues

| ID | Task | Deliverable |
|---|---|---|
| AI-01 | DeepSeek provider | Configuration, timeouts, retries, error sanitisation |
| AI-02 | Structured task output | JSON Schema, one repair attempt, failure protocol |
| AI-03 | RAG document standard | Source, version, review status, applicability range |
| AI-04 | Retrieval and answering | Retrieval of model cards, parameter specs, validation rules |
| AI-05 | Skill tool allowlist | No arbitrary shell, Python or database execution |
| AI-06 | Safety and behaviour evaluation | Fabrication, ambiguity, injection, omission, overreach tests |

### Acceptance criteria

- Invalid output cannot reach the calculation layer;
- Explicitly given components and compositions are never omitted or substituted;
- Knowledge questions do not accidentally trigger calculations;
- When required inputs are missing, the system asks or fails structurally;
- DeepSeek produces no final thermodynamic numbers;
- `evals/test_agent.py` and the provider tests pass.

## 8. Phase-equilibrium models and calculation backends group

Owns: `thermo_engine/`, `knowledge/model_cards/`, `knowledge/model_selection_rules/`

### Responsibilities

- Maintain the unified `ThermodynamicBackend` interface and backend registry.
- Integrate deterministic thermodynamic frameworks without leaking third-party objects upward.
- Establish the model capability matrix and applicability routing.
- Manage pure properties, binary parameters, sources, units, direction and applicability ranges.
- Implement bubble point, dew point, VLE, TP flash, azeotrope search and the planned LLE.
- Provide independent validation and benchmark tests for every numerical path.
- Establish isolated interfaces for AI physical-property/parameter/surrogate models.

### Model integration priority

| Priority | Model or framework | Goal |
|---|---|---|
| P0 | Ideal/Raoult | Low-pressure baseline and end-to-end loop |
| P0 | CalebBell/thermo + Peng-Robinson | Deterministic calculations for hydrocarbons and light gases |
| P0 | Phasepy/Peng-Robinson | Interchangeable Python backend |
| P0 | Clapeyron.jl/Peng-Robinson | Julia backend and cross-validation |
| P1 | SRK | A second cubic equation of state |
| P1 | Wilson, NRTL, UNIQUAC | Non-ideal liquid-phase VLE |
| P1 | UNIFAC/Modified UNIFAC | Group-contribution prediction and missing-parameter candidates |
| P2 | PC-SAFT, SAFT-VR Mie, CPA | Associating and complex molecular systems |
| P2 | LLE, VLLE, phase stability | Multi-liquid and multiphase extension |
| Separate later version | Electrolytes, reactions, SLE, polymers, hydrates | Not mixed into the current v0.1 |

"Integrating all models" is defined in engineering terms as: first register model capability and
applicability, then open execution only for models that have parameter evidence, validation cases
and an owner for maintenance.

### Standard delivery package for each model

1. Backend adapter;
2. Capability declaration;
3. Model card;
4. Parameter schema and sources;
5. Missing-parameter behaviour;
6. Inapplicable behaviour;
7. At least one public benchmark case;
8. Physical validation tests;
9. API behaviour tests;
10. Usage notes and known limitations.

### Requirements for AI phase-equilibrium models

AI may be used for:

- Physical-property prediction;
- Binary parameter candidates;
- Model selection;
- Flash or phase-diagram surrogates;
- Initial-value generation;
- Uncertainty estimation.

AI results must additionally carry:

```text
result_kind
model_version
training_data_version
applicability_domain
uncertainty
validated_against
review_status
```

An unreviewed AI prediction must not masquerade as a database parameter or a deterministic
engineering result.

### Phase-one Issues

| ID | Task | Deliverable |
|---|---|---|
| TH-01 | Backend protocol and registry | Unified interface, aliases and capability declarations |
| TH-02 | Parameter repository | Source, units, direction, applicability range and hash |
| TH-03 | Ideal/Raoult loop | bubble, dew, VLE, Flash |
| TH-04 | thermo/Peng-Robinson | PR adapter and ChemSep parameter evidence |
| TH-05 | Phasepy adapter | Unified results and validation gate |
| TH-06 | Clapeyron adapter | Julia bridge, parameter snapshots and validation |
| TH-07 | Model selection matrix | Applicable, exclusion, missing-parameter and risk rules |
| TH-08 | Independent physical validation | Balances, residuals, stability and applicability domain |

### Acceptance criteria

- All numbers come from registered backends;
- Missing parameters produce no result;
- Every executable model has benchmarks and behaviour tests;
- All results pass the independent validation gate;
- Upgrading a third-party framework does not change the public API.

## 9. Scientific validation and quality assurance group

Owns: `tests/`, `evals/`, `knowledge/validation_guides/`

### Responsibilities

- Establish public benchmark cases and allowed error tolerances.
- Compare against experimental data, papers, manuals or upstream software tests.
- Review model applicability ranges, parameter sources and licences.
- Maintain numerical regression, physical invariant and failure-behaviour tests.
- Independently review the applicability domain and uncertainty of AI models.
- Provide scientific review for model, parameter and validation PRs.

### Acceptance criteria

- The model implementer is not the sole reviewer;
- Benchmark data sources are traceable;
- Solver convergence is not treated as equivalent to passing validation;
- Results that cannot be validated must be marked warning or failed;
- Test parameters never enter the production parameter repository.

## 10. CI, deployment and release

Owns: `.github/`, Docker configuration and environment templates

### Responsibilities

- Maintain backend, frontend and optional calculation backend CI.
- Maintain Python, Node, pnpm, Julia and dependency versions.
- Maintain Dockerfiles, Compose and deployment documentation.
- Manage GitHub Secrets, branch protection and release tags.
- Ensure default CI does not call paid external LLM APIs.

### Acceptance criteria

- PRs must pass the backend, frontend and, where applicable, external-engines checks;
- Direct pushes and force pushes to `main` are forbidden;
- Release notes cover models, parameters, validation, limitations and migration information;
- No keys in the repository, images or logs.

## 11. Work dependencies and merge order

Recommended order:

```text
public schemas
-> backend protocol and error format
-> deterministic backends and validator
-> agent tool contracts
-> DeepSeek/RAG/Skills
-> frontend interaction and result display
-> end-to-end tests and release
```

Work that can proceed in parallel:

- Frontend development against fixed JSON mocks;
- The phase-equilibrium group working against the Python public interface;
- The RAG group preparing model cards and document metadata;
- The backend group maintaining the final contracts and owning integration;
- The validation group independently preparing benchmark data.

High-conflict files that must not be modified in parallel without coordination:

- `schemas/domain.py`
- `apps/api/main.py`
- `apps/web/src/lib/types.ts`
- `thermo_engine/service.py`
- `thermo_engine/registry.py`

## 12. GitHub workflow

1. Create an Issue for every task first.
2. State the scope, dependencies, inputs, outputs and acceptance criteria in the Issue.
3. Create a short branch from the latest `main`:

```text
feat/123-phasepy-flash
fix/234-component-grounding
docs/345-model-card
```

4. One PR solves one main problem.
5. A PR must link its Issue and follow the repository PR template.
6. A scientific model PR requires at least:
   - one software reviewer;
   - one thermodynamics/data reviewer.
7. Merge only after CI is green and discussions are resolved.
8. Prefer squash merge.

## 13. Milestone plan

### M1: phase-equilibrium platform loop

- Frontend workbench;
- FastAPI and public schemas;
- DeepSeek provider;
- LangGraph constrained tool chain;
- Ideal/Raoult;
- thermo/Peng-Robinson;
- Physical validation;
- Run records and export.

Definition of done: at least one knowledge question, one VLE and one TP Flash task can be executed
end to end from the web page and traced.

### M2: RAG and Skill constraints

- Knowledge document metadata;
- Retrieval and citation;
- Skill registry;
- JSON Schema output;
- Anti-fabrication and prompt-injection evaluation;
- Model selection explanation.

Definition of done: DeepSeek's tasks, tools and answers are all constrained by testable contracts.

### M3: phase-equilibrium model matrix

- Phasepy;
- Clapeyron;
- SRK;
- NRTL, Wilson, UNIQUAC;
- UNIFAC;
- Unified parameter repository;
- Unified validation benchmarks.

Definition of done: every model marked "executable" has parameter evidence, benchmark cases and
independent validation.

### M4: AI-assisted models

- Physical-property prediction;
- Parameter candidates;
- Phase-equilibrium surrogate models;
- Applicability domain;
- Uncertainty;
- Isolation of AI from deterministic results.

Definition of done: AI results cannot be mistaken for reviewed deterministic engineering results.

## 14. Phase-two domain expansion

After phase one completes, extend the system into a common platform plus domain plugins:

```text
platform/
  agent_runtime/
  rag/
  skills/
  api/
  database/
  validation/

domains/
  phase_equilibrium/
  retrosynthesis/
  catalyst_screening/
  tray_monitoring/
  impeller_design/
```

### Molecular retrosynthesis

- Molecular representation and identity;
- Reaction templates and route search;
- Synthesizability, cost, safety and green-chemistry scoring;
- Literature and supplier evidence;
- Route validation and expert review.

### Catalyst screening

- Catalyst and reaction data models;
- Descriptors and features;
- Machine-learning screening;
- Multi-objective optimisation;
- Uncertainty and experimental feedback loop.

### Tray monitoring

- Time-series data ingestion;
- Operating-condition labels and data quality;
- Anomaly detection;
- Soft sensing;
- Fault diagnosis;
- Alarm justification and human confirmation.

Tray monitoring is not flowsheet design. Full distillation-column design should be a separate later
scope.

### Impeller design

- Fluid properties and operating conditions;
- Impeller type selection;
- Power number and Reynolds number;
- Mixing time, mass transfer and suspension criteria;
- Empirical correlations and a CFD interface;
- Design safety margins and applicability ranges.

The five domains share identity, RAG, Skills, permissions, run records and the frontend framework,
but must each own their tools, schemas, knowledge base, calculation methods, validation standards and
responsible owner.

## 15. Pull Request definition of done

A task is complete only when all of the following hold:

- The code is implemented;
- Automated tests have been added;
- Scientific assumptions and applicability ranges are recorded;
- Parameter and data sources are recorded;
- Error and failure behaviour is covered;
- Public schemas are synchronised;
- Documentation is updated;
- CI is green;
- At least one other member has approved;
- No keys, databases, caches or test artifacts have entered Git.
