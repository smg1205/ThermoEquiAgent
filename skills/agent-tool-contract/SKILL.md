---
name: agent-tool-contract
description: Design and enforce structured contracts between conversation orchestration, LLM providers, parameter lookup, deterministic calculation tools, validation, and interpretation. Use for agent workflows, tool schemas, multi-turn state, or provider integrations.
---

# Agent tool contract

## Use cases

Use when adding intents, LLM providers, calculation tools, conversation state, or result explanation.

## Inputs

Require the user utterance, prior structured task when present, provider capability, and canonical
Pydantic schemas.

## Outputs

Return an intent, task delta or manifest, invoked tool records, evidence labels, and an explanation
that preserves tool status.

## Procedure

1. Classify before formulating; knowledge questions must not invoke calculation.
2. Merge follow-up deltas into prior structured state, preserving unspecified fields.
3. Let programs generate verifiable profile fields.
4. Invoke calculation and validation tools for all numerical output.
5. Label statements as Knowledge, Database, Calculation, Inference, Estimate, or Warning.

## Prohibitions

Do not scatter provider SDK calls through business code. Do not let an LLM invent parameters,
numbers, sources, or override tool failures.

## Failure handling

Return structured semantic, missing-data, unsupported, or tool failures while keeping the
conversation recoverable.

## Related files

`agent/orchestrator.py`, `agent/providers.py`, `schemas/domain.py`, `evals/test_agent.py`.

## Acceptance

Deterministic mode works without a key; follow-ups inherit prior state; every number is traceable to
a tool/database field; failed validation stays visible.
