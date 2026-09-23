"""Build the audited joint-protocol cumulative training-stage ablation report."""

from __future__ import annotations

import csv
import json
import math
from io import StringIO
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.reporting.artifacts import (
    artifact_sha256,
    atomic_write_json,
    atomic_write_text,
)
from src.thermoformer.reporting.training_stage_ablation import (
    STAGE_ORDER,
    build_stage_summary,
    paired_stage_deltas,
)


SPLIT_PROTOCOL = "overall_binary_ternary"
EXPERIMENT_NAMESPACE = Path(
    "configs/vle/ablation/studies/three_stage_training/overall_binary_ternary"
)
RESULT_NAMESPACE = Path(
    "experiments/vle/generalization/evaluations/ablations/three_stage_training/overall_binary_ternary"
)
RESULT_PROTOCOL = "c1_three_view_vanilla_three_stage.on.overall_binary_ternary"
SEEDS = (0, 1, 2, 3, 4)
STAGE_LABELS = {
    "stage0": "A0: supervised reference",
    "stage1": "A1: direct GE/RT and ln(gamma)",
    "stage2": "A2: joint VLE supervision",
    "stage3": "A3: fugacity-constrained fine-tuning",
    "selected": "Validation-selected deployment checkpoint",
}


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Malformed {name}: {path}")
    return value


def _project_path(project_root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} has no artifact path")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{name} escapes the project root") from error
    return resolved


def _artifact(project_root: Path, record: object, name: str, expected: Path | None = None) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{name} has no artifact record")
    path = _project_path(project_root, record.get("path"), name)
    digest = record.get("sha256")
    if expected is not None and path != expected.resolve():
        raise RuntimeError(f"{name} points to an unexpected location")
    if not isinstance(digest, str) or not path.is_file() or artifact_sha256(path) != digest:
        raise RuntimeError(f"{name} failed SHA-256 verification")
    return path


def _directions(rows: object, name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{name} metrics are not a list")
    indexed = {
        str(row.get("direction")): row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("scope") == "direction"
        and row.get("direction") in {"isothermal", "isobaric"}
    }
    if set(indexed) != {"isothermal", "isobaric"}:
        raise RuntimeError(f"{name} lacks directional metrics")
    return indexed


def _stage_payload(
    project_root: Path,
    *,
    seed: int,
    manifest: Mapping[str, Any],
    result_dir: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"seed {seed} lacks artifact provenance")
    comparison_path = _artifact(
        project_root,
        artifacts.get("stage_comparison"),
        f"seed {seed} stage comparison",
        result_dir / f"seed_{seed}" / "stage_comparison.json",
    )
    comparison = _read_json(comparison_path, f"seed {seed} stage comparison")
    if (
        comparison.get("selection_partition") != "validation"
        or comparison.get("evaluation_partition") != "test"
        or comparison.get("selected_stage") != manifest.get("selected_stage")
        or comparison.get("test_metrics_used_for_selection") is not False
    ):
        raise RuntimeError(f"seed {seed} violates validation-only selection")
    stages = comparison.get("stages")
    if not isinstance(stages, Mapping) or not set(STAGE_ORDER).issubset(stages):
        raise RuntimeError(f"seed {seed} lacks cumulative-stage entries")
    prepared: dict[str, Mapping[str, Any]] = {}
    for stage in STAGE_ORDER:
        record = stages[stage]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"seed {seed} {stage} is malformed")
        checkpoint = _artifact(project_root, artifacts.get(stage + "_checkpoint"), f"seed {seed} {stage} checkpoint")
        predictions = _artifact(
            project_root,
            artifacts.get(stage + "_predictions"),
            f"seed {seed} {stage} predictions",
            result_dir / f"seed_{seed}" / f"{stage}_predictions.csv",
        )
        if (
            _project_path(project_root, record.get("checkpoint"), f"seed {seed} {stage} checkpoint") != checkpoint
            or record.get("checkpoint_sha256") != artifact_sha256(checkpoint)
            or _project_path(project_root, record.get("predictions"), f"seed {seed} {stage} predictions") != predictions
            or record.get("predictions_sha256") != artifact_sha256(predictions)
        ):
            raise RuntimeError(f"seed {seed} {stage} artifact lineage is inconsistent")
        prepared[stage] = record
    selected = manifest.get("selected_stage")
    if selected not in STAGE_ORDER:
        raise RuntimeError(f"seed {seed} selected an invalid stage")
    return prepared, dict(comparison), {
        "seed": seed,
        "selected_stage": selected,
        "stage_comparison_sha256": artifact_sha256(comparison_path),
    }


def _csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write an empty report table: {path}")
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=sorted({key for row in rows for key in row}),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _fmt(value: object, digits: int) -> str:
    if value is None:
        return "not reported"
    return f"{float(value):.{digits}f}"


def _triple(row: Mapping[str, object], prefix: str, digits: int) -> str:
    return " / ".join(
        (
            _fmt(row[prefix + "_mae_mean"], digits)
            + " ± "
            + _fmt(row[prefix + "_mae_std"], digits),
            _fmt(row[prefix + "_rmse_mean"], digits)
            + " ± "
            + _fmt(row[prefix + "_rmse_std"], digits),
            _fmt(row[prefix + "_r2_mean"], 3)
            + " ± "
            + _fmt(row[prefix + "_r2_std"], 3),
        )
    )


def _table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    indexed = {(str(row["stage"]), str(row["direction"])): row for row in rows}
    lines = [
        "| Cumulative variant | P isothermal: MAE / RMSE / R2 (kPa) | y isothermal: MAE / RMSE / R2 | T isobaric: MAE / RMSE / R2 (K) | y isobaric: MAE / RMSE / R2 | Fugacity residual | Solver failure (iso / isob) | Nonphysical (iso / isob) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in (*STAGE_ORDER, "selected"):
        iso, isob = indexed[(stage, "isothermal")], indexed[(stage, "isobaric")]
        lines.append(
            "| "
            + STAGE_LABELS[stage]
            + " | "
            + _triple(iso, "state", 3)
            + " | "
            + _triple(iso, "y", 4)
            + " | "
            + _triple(isob, "state", 3)
            + " | "
            + _triple(isob, "y", 4)
            + " | "
            + _fmt(iso["teacher_forced_fugacity_mean"], 6)
            + " ± "
            + _fmt(iso["teacher_forced_fugacity_std"], 6)
            + " | "
            + _fmt(iso["solver_failure_rate_mean"], 4)
            + " / "
            + _fmt(isob["solver_failure_rate_mean"], 4)
            + " | "
            + _fmt(iso["nonphysical_rate_mean"], 4)
            + " / "
            + _fmt(isob["nonphysical_rate_mean"], 4)
            + " |"
        )
    return lines


def _delta_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Transition | Direction | Metric | Mean delta ± SD | Relative change (%) | Improved seeds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['from_stage']} → {row['to_stage']} | {row['direction']} | {row['metric']} | "
            f"{_fmt(row['mean_delta'], 5)} ± {_fmt(row['std_delta'], 5)} | "
            f"{_fmt(row['mean_relative_percent'], 2)} ± {_fmt(row['std_relative_percent'], 2)} | "
            f"{row['improved_seed_count']}/{row['available_seeds']} |"
        )
    return lines


def write_report(project_root: Path) -> tuple[Path, Path, Path]:
    """Validate all formal evidence and render the report after five seeds finish."""

    result_dir = project_root / RESULT_NAMESPACE / RESULT_PROTOCOL
    aggregate = _read_json(result_dir / "aggregate_manifest.json", "formal aggregate")
    if (
        aggregate.get("status") != "completed"
        or aggregate.get("aggregate_kind") != "formal"
        or aggregate.get("seeds") != list(SEEDS)
        or aggregate.get("protocol") != RESULT_PROTOCOL
    ):
        raise RuntimeError("The cumulative-stage report requires a complete five-seed formal aggregate")
    stage_metrics: dict[str, dict[int, dict[str, object]]] = {
        stage: {} for stage in STAGE_ORDER
    }
    selected_metrics: dict[int, dict[str, object]] = {}
    selected_rows: list[dict[str, object]] = []
    parameters: list[dict[str, object]] = []
    input_manifest_records: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_dir = result_dir / f"seed_{seed}"
        manifest_path = seed_dir / "manifest.json"
        manifest = _read_json(manifest_path, f"seed {seed} manifest")
        expected = {
            "status": "completed",
            "seed": seed,
            "protocol": RESULT_PROTOCOL,
            "split_protocol": SPLIT_PROTOCOL,
            "run_kind": "formal",
            "selection_partition": "validation",
            "evaluation_partition": "test",
            "test_metrics_used_for_selection": False,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"seed {seed} is not a valid formal joint-protocol run")
        stages, comparison, selected = _stage_payload(
            project_root, seed=seed, manifest=manifest, result_dir=result_dir
        )
        for stage, record in stages.items():
            stage_metrics[stage][seed] = {
                **_directions(record.get("metrics"), f"seed {seed} {stage}"),
                "physics_residuals": record.get("physics_residuals", {}),
            }
        selected_stage = str(selected["selected_stage"])
        selected_metrics[seed] = stage_metrics[selected_stage][seed]
        selected_rows.append(selected)
        summary = comparison.get("parameter_summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError(f"seed {seed} lacks parameter provenance")
        parameters.append({"seed": seed, **dict(summary)})
        input_manifest_records.append(
            {
                "seed": seed,
                "path": str(manifest_path.relative_to(project_root)),
                "sha256": artifact_sha256(manifest_path),
            }
        )
    summary_rows = build_stage_summary(stage_metrics)
    selected_summary = build_stage_summary({"selected": selected_metrics})
    all_summary = [*summary_rows, *selected_summary]
    delta_rows = paired_stage_deltas(stage_metrics)
    output_root = result_dir / "reporting"
    metrics_path = output_root / "cumulative_stage_metrics.csv"
    deltas_path = output_root / "cumulative_stage_deltas.csv"
    selected_path = output_root / "validation_selected_stages.csv"
    parameters_path = output_root / "parameter_provenance.json"
    _csv(metrics_path, all_summary)
    _csv(deltas_path, delta_rows)
    _csv(selected_path, selected_rows)
    atomic_write_json(parameters_path, parameters)
    report_path = project_root / "experiments/vle/ablation/study_records/three_stage_training/overall_binary_ternary/results.md"
    selected_counts = {
        stage: sum(row["selected_stage"] == stage for row in selected_rows)
        for stage in STAGE_ORDER
    }
    report = [
        "# C1 cumulative three-stage training ablation",
        "",
        "Protocol: `overall_binary_ternary` (binary + ternary training, validation, and joint test). All checkpoint decisions use validation data only. The test partition is evaluated only after each stage-local checkpoint is frozen.",
        "",
        "Each metric entry is MAE ± sample SD / RMSE ± sample SD / R2 ± sample SD over seeds 0--4.",
        "",
        "## Predictive and physical metrics",
        "",
        *_table(all_summary),
        "",
        "## Paired changes between consecutive cumulative stages",
        "",
        "Each comparison uses the same five seed pairs. Lower is better for MAE, RMSE, solver failure, and nonphysical rate; higher is better for R2 and coverage.",
        "",
        *_delta_table(delta_rows),
        "",
        "## Validation-selected deployment checkpoints",
        "",
        "| Stage 0 | Stage 1 | Stage 2 | Stage 3 |",
        "|---:|---:|---:|---:|",
        f"| {selected_counts['stage0']} | {selected_counts['stage1']} | {selected_counts['stage2']} | {selected_counts['stage3']} |",
        "",
        "The deployment row is auxiliary. The A0--A3 rows are the primary cumulative-stage ablation and do not replace an endpoint with an earlier stage merely because it has a lower global validation score.",
        "",
    ]
    atomic_write_text(report_path, "\n".join(report))
    report_manifest_path = output_root / "report_manifest.json"
    report_manifest = {
        "status": "completed",
        "protocol": RESULT_PROTOCOL,
        "split_protocol": SPLIT_PROTOCOL,
        "seeds": list(SEEDS),
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "test_metrics_used_for_selection": False,
        "stage_order": list(STAGE_ORDER),
        "input_seed_manifests": input_manifest_records,
        "outputs": {
            "metrics": {"path": str(metrics_path.relative_to(project_root)), "sha256": artifact_sha256(metrics_path)},
            "paired_deltas": {"path": str(deltas_path.relative_to(project_root)), "sha256": artifact_sha256(deltas_path)},
            "selected_stages": {"path": str(selected_path.relative_to(project_root)), "sha256": artifact_sha256(selected_path)},
            "parameters": {"path": str(parameters_path.relative_to(project_root)), "sha256": artifact_sha256(parameters_path)},
            "report": {"path": str(report_path.relative_to(project_root)), "sha256": artifact_sha256(report_path)},
        },
    }
    atomic_write_json(report_manifest_path, report_manifest)
    return report_path, metrics_path, report_manifest_path


def main(argv: list[str] | None = None) -> None:
    del argv
    report_path, metrics_path, manifest_path = write_report(PROJECT_ROOT)
    print(json.dumps({"report": str(report_path), "metrics": str(metrics_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
