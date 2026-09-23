"""Deterministic multi-model comparison across executable backends."""

from __future__ import annotations

from agent.router import (
    available_parameter_models_for_task,
    rank_executable_models,
    recommend_models,
)
from schemas.domain import (
    CalculationResult,
    FailureDetail,
    FailureType,
    ModelComparisonEntry,
    ModelComparisonResponse,
    PhaseDiagramEntry,
    PhaseDiagramResponse,
    PhaseDiagramStatus,
    PhaseDiagramType,
    TaskManifest,
    ValidationReport,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.service import calculate_equilibrium, resolve_backend, validate_equilibrium_result


def compare_models(
    task: TaskManifest,
    available_parameter_models: set[str] | None = None,
) -> ModelComparisonResponse:
    """Run every executable model for one task and return per-model evidence.

    A failed execution is retained as a structured entry instead of aborting the
    whole comparison, so the response shows which models could and could not run.
    """
    if available_parameter_models is None:
        available_parameter_models = available_parameter_models_for_task(task)
    candidates = rank_executable_models(task, available_parameter_models)
    if not candidates:
        raise ThermoEquiError(
            FailureType.PARAMETER_OUT_OF_DOMAIN,
            "No executable model is available for multi-model comparison.",
            "Seed reviewed parameters with `thermoequi-seed` or import a parameter set via the parameter API.",
        )

    entries: list[ModelComparisonEntry] = []
    for candidate in candidates:
        attempt = task.model_copy(
            update={
                "model_name": candidate.model_name,
                "assumptions": [
                    *task.assumptions,
                    f"Compared {candidate.model_name} with score {candidate.score:.1f}.",
                ],
            }
        )
        try:
            backend = resolve_backend(attempt)
            result = calculate_equilibrium(attempt, backend=backend)
            validation = validate_equilibrium_result(result)
            entries.append(
                ModelComparisonEntry(
                    model_name=candidate.model_name,
                    score=candidate.score,
                    executable=True,
                    result=result,
                    validation=validation,
                    parameter_sources=backend.parameter_sources(attempt),
                    warnings=[*result.warnings, *validation.warnings],
                )
            )
        except ThermoEquiError as error:
            entries.append(
                ModelComparisonEntry(
                    model_name=candidate.model_name,
                    score=candidate.score,
                    executable=True,
                    failure=error.detail,
                    warnings=[error.detail.message],
                )
            )

    executed_count = sum(1 for entry in entries if entry.result is not None)
    passed_count = sum(
        1 for entry in entries if entry.validation is not None and entry.validation.overall_status == "passed"
    )
    warning_count = sum(
        1 for entry in entries if entry.validation is not None and entry.validation.overall_status == "warning"
    )
    failed_count = sum(
        1
        for entry in entries
        if entry.failure is not None or (entry.validation is not None and entry.validation.overall_status == "failed")
    )
    summary = f"对比 {len(entries)} 个可执行模型：{passed_count} 通过、{warning_count} 警告、{failed_count} 失败。"
    return ModelComparisonResponse(
        task=task,
        entries=entries,
        executed_count=executed_count,
        passed_count=passed_count,
        warning_count=warning_count,
        failed_count=failed_count,
        summary=summary,
    )


def _phase_diagram_type(task: TaskManifest) -> PhaseDiagramType:
    return "PXY" if task.calculation_type == "isothermal_vle" else "TXY"


def _entry_status(
    result: CalculationResult | None,
    validation: ValidationReport | None,
    failure: FailureDetail | None,
) -> PhaseDiagramStatus:
    """Derive the phase-diagram status for one model entry."""
    if failure is not None:
        return "failed"
    if validation is None or result is None:
        return "failed"
    return "passed" if validation.overall_status == "passed" else "warning"


def build_phase_diagram(
    task: TaskManifest,
    available_parameter_models: set[str] | None = None,
) -> PhaseDiagramResponse:
    """Build a multi-model phase diagram across every catalog model.

    Unlike :func:`compare_models`, which only runs executable backends, this
    walks the full model catalog so the response includes ``unsupported``
    entries for models that cannot execute for the requested diagram type.
    Each unsuccessful model surfaces as a structured entry (with its failure)
    instead of aborting the diagram.
    """
    if available_parameter_models is None:
        available_parameter_models = available_parameter_models_for_task(task)

    entries: list[PhaseDiagramEntry] = []
    for recommendation in recommend_models(task, available_parameter_models):
        if not recommendation.executable:
            exclusions = " ".join(recommendation.exclusions) or "Model is not executable for this task."
            entries.append(
                PhaseDiagramEntry(
                    model_name=recommendation.model_name,
                    status="unsupported",
                    executable=False,
                    score=recommendation.score,
                    result=None,
                    validation=None,
                    failure=FailureDetail(
                        failure_type=FailureType.UNSUPPORTED_MODEL,
                        message=exclusions,
                        recovery_action="Choose a supported model for this phase diagram.",
                        details={"score": recommendation.score},
                    ),
                    warnings=[exclusions],
                )
            )
            continue

        attempt = task.model_copy(
            update={
                "model_name": recommendation.model_name,
                "assumptions": [
                    *task.assumptions,
                    f"Compared {recommendation.model_name} with score {recommendation.score:.1f}.",
                ],
            }
        )
        try:
            backend = resolve_backend(attempt)
            result = calculate_equilibrium(attempt, backend=backend)
            validation = validate_equilibrium_result(result)
            entries.append(
                PhaseDiagramEntry(
                    model_name=recommendation.model_name,
                    status=_entry_status(result, validation, None),
                    executable=True,
                    score=recommendation.score,
                    result=result,
                    validation=validation,
                    parameter_sources=backend.parameter_sources(attempt),
                    warnings=[*result.warnings, *validation.warnings],
                )
            )
        except ThermoEquiError as error:
            entries.append(
                PhaseDiagramEntry(
                    model_name=recommendation.model_name,
                    status="failed",
                    executable=True,
                    score=recommendation.score,
                    result=None,
                    validation=None,
                    failure=error.detail,
                    warnings=[error.detail.message],
                )
            )

    passed_count = sum(1 for entry in entries if entry.status == "passed")
    warning_count = sum(1 for entry in entries if entry.status == "warning")
    failed_count = sum(1 for entry in entries if entry.status == "failed")
    unsupported_count = sum(1 for entry in entries if entry.status == "unsupported")
    diagram_type = _phase_diagram_type(task)
    summary = (
        f"Compared {len(entries)} models: {passed_count} passed, {warning_count} warnings, "
        f"{failed_count} failed, {unsupported_count} unsupported."
    )
    return PhaseDiagramResponse(
        task=task,
        diagram_type=diagram_type,
        entries=entries,
        total_models=len(entries),
        passed_count=passed_count,
        warning_count=warning_count,
        failed_count=failed_count,
        unsupported_count=unsupported_count,
        summary=summary,
    )
