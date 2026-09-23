---
name: frontend-engineering-workbench
description: Build or modify the ThermoEqui-Agent Next.js engineering workbench, including chat, structured task editing, phase charts, tables, parameter evidence, validation diagnostics, run history, and exports. Use for frontend product work.
---

# Frontend engineering workbench

## Use cases

Use for workbench layout, interactions, visualizations, API type synchronization, and UI tests.

## Inputs

Read the relevant Pydantic response schema, `apps/web/src/lib/types.ts`, accessibility requirements,
and engineering state to display.

## Outputs

Produce strict TypeScript UI, persistent diagnostics, responsive styling, and behavior tests.

## Procedure

1. Keep system, task, model, parameter, calculation, validation, and risk status visible.
2. Render liquid/vapor lines with axes, units, model, condition, and azeotrope markers.
3. Let users edit conditions and rerun without mutating historical runs.
4. Expose raw table, evidence, validation, history, JSON, and CSV.
5. Test loading, input, updates, charts, errors, rerun, and downloads.

## Prohibitions

Do not calculate thermodynamics in React. Do not hide errors only in transient dialogs. Do not
invent absent fields or silently coerce invalid engineering inputs.

## Failure handling

Keep the last valid result, display structured API diagnostics in the task panel, and allow a
corrected rerun.

## Related files

`apps/web/src`, `docs/frontend.md`, `schemas/domain.py`.

## Acceptance

The UI is keyboard accessible, responsive, type-safe, explicit about provenance and validation, and
passes component tests plus production build.
