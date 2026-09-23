# Architecture

```mermaid
flowchart LR
  UI[Next.js workbench] --> API[FastAPI]
  API --> ORCH[Conversation orchestrator]
  ORCH --> LLM[DeepSeek / deterministic provider]
  ORCH --> TOOLS[Constrained engineering tool registry]
  TOOLS --> ROUTER[Applicability router]
  TOOLS --> EXEC[Calculation executor]
  EXEC --> BACKEND[Backend registry]
  BACKEND --> IDEAL[Ideal/Raoult adapter]
  BACKEND --> PR[CalebBell/thermo Peng-Robinson adapter]
  EXEC --> VALIDATE[Validation controller]
  API --> DB[(SQLite/PostgreSQL)]
  ROUTER --> CARDS[Model cards]
  EXEC --> PARAMS[Parameter repository]
```

Only the deterministic backend emits thermodynamic numbers. API, agent, and UI exchange Pydantic
contracts and immutable run snapshots. Third-party engines can be added behind the backend protocol
without leaking their internal object model into the service layer.

The tool registry follows the useful boundary in CAi_copilot—reasoning chooses a named tool, while
the tool performs the operation—but deliberately exposes no shell or notebook execution. The
public execution trace contains only auditable phase summaries, never private chain-of-thought.
See [integrations.md](integrations.md) for the implementation matrix and extension contract.
