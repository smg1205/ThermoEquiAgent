"""Publication-facing outputs for the final C1 fugacity campaign."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import (
    artifact_sha256,
    atomic_write_json,
    atomic_write_text,
    portable_artifact_path,
    resolve_artifact_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(
    "experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity"
)
RUN_PREFIX = "c1_three_view_vanilla_fugacity_finetune.on."


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    category: str
    label: str


PROTOCOLS = (
    ProtocolSpec("overall_binary", "overall", "Binary train -> binary test"),
    ProtocolSpec(
        "overall_binary_ternary",
        "overall",
        "Binary+ternary train -> joint test",
    ),
    ProtocolSpec("state_composition_interpolation", "state", "Composition interpolation"),
    ProtocolSpec("state_composition_edge_extrapolation", "state", "Composition edge extrapolation"),
    ProtocolSpec("state_temperature_low_extrapolation", "state", "Low-temperature extrapolation"),
    ProtocolSpec("state_temperature_high_extrapolation", "state", "High-temperature extrapolation"),
    ProtocolSpec("state_pressure_low_extrapolation", "state", "Low-pressure extrapolation"),
    ProtocolSpec("state_pressure_high_extrapolation", "state", "High-pressure extrapolation"),
    ProtocolSpec("unseen_component", "chemistry", "Unseen component"),
    ProtocolSpec("binary_to_ternary_zero_shot", "transfer", "0% ternary (zero-shot)"),
    ProtocolSpec("binary_to_ternary_scale_0.05", "transfer", "5.56% ternary"),
    ProtocolSpec("binary_to_ternary_scale_0.1", "transfer", "10% ternary"),
    ProtocolSpec("binary_to_ternary_scale_0.25", "transfer", "25% ternary"),
    ProtocolSpec("binary_to_ternary_scale_0.5", "transfer", "50% ternary"),
    ProtocolSpec("binary_to_ternary_scale_1", "transfer", "100% ternary"),
)


def _protocol_dir(project_root: Path, protocol: str) -> Path:
    return project_root / RESULTS_ROOT / f"{RUN_PREFIX}{protocol}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _verify_reference(project_root: Path, reference: dict[str, Any]) -> None:
    path = resolve_artifact_path(str(reference["path"]), project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing referenced artifact: {path}")
    actual = artifact_sha256(path)
    if actual != str(reference["sha256"]):
        raise RuntimeError(f"Artifact SHA mismatch: {path}")


def validate_protocol_bundle(project_root: Path, protocol: str) -> dict[str, Any]:
    """Validate the committed five-seed report bundle before consuming metrics."""

    directory = _protocol_dir(project_root, protocol)
    report_manifest = _load_json(directory / "report_manifest.json")
    expected = f"{RUN_PREFIX}{protocol}"
    invariants = {
        "status": "completed",
        "protocol": expected,
        "analysis_status": "confirmatory",
        "selection_partition": "validation",
        "evaluation_partition": "test",
    }
    for key, value in invariants.items():
        if report_manifest.get(key) != value:
            raise RuntimeError(
                f"Invalid {key} for {protocol}: {report_manifest.get(key)!r}"
            )
    if report_manifest.get("seeds") != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected seeds 0--4 for {protocol}")

    for key in ("aggregate_manifest", "stage_comparison_summary", "report"):
        _verify_reference(project_root, report_manifest[key])
    for key in ("run_manifests", "stage_comparisons"):
        references = report_manifest.get(key, [])
        if [int(item["seed"]) for item in references] != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"Invalid {key} seed set for {protocol}")
        for reference in references:
            _verify_reference(project_root, reference)

    aggregate_path = resolve_artifact_path(
        str(report_manifest["aggregate_manifest"]["path"]), project_root
    )
    aggregate = _load_json(aggregate_path)
    if aggregate.get("status") != "completed" or aggregate.get("seeds") != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Incomplete formal aggregate for {protocol}")
    for reference in aggregate.get("outputs", {}).values():
        _verify_reference(project_root, reference)
    input_hashes = aggregate.get("input_manifest_sha256", {})
    for reference in report_manifest["run_manifests"]:
        if input_hashes.get(str(reference["seed"])) != reference["sha256"]:
            raise RuntimeError(f"Aggregate input hash mismatch for {protocol}/seed_{reference['seed']}")
    return report_manifest


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _direction_rows(rows: Iterable[dict[str, str]], protocol: str) -> list[dict[str, str]]:
    rows = list(rows)
    if protocol == "overall_binary_ternary":
        return [
            row
            for row in rows
            if row["scope"] == "direction_cardinality"
            and row["direction"] in {"isothermal", "isobaric"}
            and row["component_count"] in {"2", "3", "2.0", "3.0"}
        ]
    return [
        row
        for row in rows
        if row["scope"] == "direction"
        and row["direction"] in {"isothermal", "isobaric"}
    ]


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value == "" else float(value)


def collect_generalization_rows(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Collect validation-selected final test metrics from all 15 protocols."""

    output: list[dict[str, Any]] = []
    for spec in PROTOCOLS:
        validate_protocol_bundle(project_root, spec.name)
        directory = _protocol_dir(project_root, spec.name)
        for row in _direction_rows(_read_rows(directory / "metrics_summary.csv"), spec.name):
            direction = row["direction"]
            cardinality = row.get("component_count", "")
            subset = (
                "binary"
                if cardinality in {"2", "2.0"}
                else "ternary"
                if cardinality in {"3", "3.0"}
                else "all"
            )
            if direction == "isothermal":
                state_name, unit = "P", "kPa"
                mae, rmse, r2 = "pressure_mae_kpa", "pressure_rmse_kpa", "pressure_r2"
            else:
                state_name, unit = "T", "K"
                mae, rmse, r2 = "temperature_mae_k", "temperature_rmse_k", "temperature_r2"
            output.append(
                {
                    "category": spec.category,
                    "protocol": spec.name,
                    "evaluation_setting": spec.label,
                    "test_subset": subset,
                    "direction": direction,
                    "known_inputs": "molecules,T,x" if direction == "isothermal" else "molecules,P,x",
                    "joint_outputs": "P,y" if direction == "isothermal" else "T,y",
                    "state_quantity": state_name,
                    "state_unit": unit,
                    "state_mae_mean": _float(row, f"{mae}_mean"),
                    "state_mae_std": _float(row, f"{mae}_std"),
                    "state_rmse_mean": _float(row, f"{rmse}_mean"),
                    "state_rmse_std": _float(row, f"{rmse}_std"),
                    "state_r2_mean": _float(row, f"{r2}_mean"),
                    "state_r2_std": _float(row, f"{r2}_std"),
                    "y_mae_mean": _float(row, "y_mae_mean"),
                    "y_mae_std": _float(row, "y_mae_std"),
                    "y_rmse_mean": _float(row, "y_rmse_mean"),
                    "y_rmse_std": _float(row, "y_rmse_std"),
                    "y_r2_mean": _float(row, "y_r2_mean"),
                    "y_r2_std": _float(row, "y_r2_std"),
                    "solver_failure_rate_mean": _float(row, "solver_failure_rate_mean"),
                    "nonphysical_rate_mean": _float(row, "nonphysical_rate_mean"),
                    "valid_coverage_mean": _float(row, "valid_coverage_mean"),
                    "available_seeds": int(row[f"{mae}_available_seeds"]),
                }
            )
    return output


def _fmt(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _task_table(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| Evaluation setting | subset | task | State MAE | State RMSE | State R² | y MAE | y RMSE | y R² | n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        state_digits = 2 if row["state_quantity"] in {"P", "T"} else 4
        unit = f" {row['state_unit']}" if row["state_unit"] else ""
        lines.append(
            f"| {row['evaluation_setting']} | {row['test_subset']} | "
            f"{row['direction']} ({row['known_inputs']} → {row['joint_outputs']}) | "
            f"{_fmt(row['state_mae_mean'], row['state_mae_std'], state_digits)}{unit} | "
            f"{_fmt(row['state_rmse_mean'], row['state_rmse_std'], state_digits)}{unit} | "
            f"{_fmt(row['state_r2_mean'], row['state_r2_std'], 3)} | "
            f"{_fmt(row['y_mae_mean'], row['y_mae_std'], 4)} | "
            f"{_fmt(row['y_rmse_mean'], row['y_rmse_std'], 4)} | "
            f"{_fmt(row['y_r2_mean'], row['y_r2_std'], 3)} | {row['available_seeds']} |"
        )
    return "\n".join(lines)


def _stage_summary(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    totals = {
        "stage1": 0,
        "stage2": 0,
        "residual_improved": 0,
        "predictive_improved": 0,
        "predictive_compared": 0,
    }
    for spec in PROTOCOLS:
        path = _protocol_dir(project_root, spec.name) / "stage_comparison_summary.json"
        payload = _load_json(path)
        counts = payload["selected_stage_counts"]
        stage1_residual = float(payload["stages"]["stage1"]["teacher_forced_fugacity"]["mean"])
        stage2_residual = float(payload["stages"]["stage2"]["teacher_forced_fugacity"]["mean"])
        totals["stage1"] += int(counts["stage1"])
        totals["stage2"] += int(counts["stage2"])
        totals["residual_improved"] += int(stage2_residual < stage1_residual)
        predictive_improved = 0
        predictive_compared = 0
        for direction, keys in (
            (
                "isothermal",
                ("pressure_mae_kpa", "pressure_rmse_kpa", "pressure_r2", "y_mae", "y_rmse", "y_r2"),
            ),
            (
                "isobaric",
                ("temperature_mae_k", "temperature_rmse_k", "temperature_r2", "y_mae", "y_rmse", "y_r2"),
            ),
        ):
            for key in keys:
                stage1_value = float(payload["stages"]["stage1"]["directions"][direction][key]["mean"])
                stage2_value = float(payload["stages"]["stage2"]["directions"][direction][key]["mean"])
                predictive_improved += int(
                    stage2_value > stage1_value if key.endswith("r2") else stage2_value < stage1_value
                )
                predictive_compared += 1
        totals["predictive_improved"] += predictive_improved
        totals["predictive_compared"] += predictive_compared
        rows.append(
            {
                "protocol": spec.name,
                "evaluation_setting": spec.label,
                "stage1_selected": int(counts["stage1"]),
                "stage2_selected": int(counts["stage2"]),
                "stage1_teacher_forced_fugacity": stage1_residual,
                "stage2_teacher_forced_fugacity": stage2_residual,
                "stage2_predictive_metrics_improved": predictive_improved,
                "predictive_metrics_compared": predictive_compared,
            }
        )
    return rows, totals


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().replace("\r\n", "\n")


def _write_generalization_leaf_records(
    project_root: Path,
    rows: list[dict[str, Any]],
    table_path: Path,
) -> None:
    """Write manuscript experiment leaves from the generated task table."""
    specifications = (
        (
            "experiments/vle/prediction/study_records/overall_binary/results.md",
            "Final C1 binary-only evaluation",
            [row for row in rows if row["protocol"] == "overall_binary"],
            "The binary-only C1 model provides the first overall-performance row of manuscript Table 1.",
        ),
        (
            "experiments/vle/prediction/study_records/overall_binary_ternary/results.md",
            "Final C1 overall binary/ternary evaluation",
            [row for row in rows if row["protocol"] == "overall_binary_ternary"],
            "The joint model retains strong binary performance and provides accurate ternary state predictions; ternary task availability is four seeds for several outputs.",
        ),
        (
            "experiments/vle/prediction/study_records/state_generalization/results.md",
            "Thermodynamic-state generalization",
            [row for row in rows if row["category"] == "state"],
            "Composition interpolation is the most accurate state task. High-temperature and high-pressure extrapolation are more difficult for bubble-pressure prediction.",
        ),
        (
            "experiments/vle/prediction/study_records/unseen_components/results.md",
            "Unseen-component generalization",
            [row for row in rows if row["category"] == "chemistry"],
            "Unseen-component prediction is the clearest limitation of the current model and is substantially harder than within-system state generalization.",
        ),
        (
            "experiments/vle/prediction/study_records/binary_to_ternary/results.md",
            "Binary-to-ternary transfer",
            [row for row in rows if row["category"] == "transfer"],
            "Zero-shot ternary state prediction is useful, while isobaric vapor-composition prediction remains the main transfer limitation; performance is not monotonic with ternary training fraction.",
        ),
    )
    source = portable_artifact_path(table_path, project_root)
    source_sha256 = artifact_sha256(table_path)
    for relative_path, title, selected, conclusion in specifications:
        if not selected:
            raise RuntimeError(f"No task rows available for {title}")
        text = "\n".join(
            [
                f"# {title}",
                "",
                "All values are mean ± standard deviation across seeds 0--4. "
                "Checkpoint selection uses validation only; test data are evaluated afterward.",
                "",
                _task_table(selected),
                "",
                conclusion,
                "",
                f"Machine-readable source: `{source}` (SHA-256 `{source_sha256}`).",
                "",
            ]
        )
        atomic_write_text(project_root / relative_path, text)


def write_c1_generalization_outputs(
    project_root: Path = PROJECT_ROOT,
    *,
    report_path: Path | None = None,
    table_path: Path | None = None,
) -> dict[str, str]:
    """Validate inputs and atomically publish the final C1 campaign summary."""

    project_root = project_root.resolve()
    publish_canonical_leaves = report_path is None and table_path is None
    report_path = report_path or project_root / "experiments/data_quality/reports/c1_fugacity_generalization_report.md"
    table_path = table_path or project_root / "experiments/vle/prediction/summary/c1_fugacity_generalization_by_task.csv"
    stage_table_path = table_path.with_name("c1_fugacity_stage_selection.csv")
    manifest_path = report_path.with_name("c1_fugacity_generalization_report_manifest.json")

    rows = collect_generalization_rows(project_root)
    stage_rows, totals = _stage_summary(project_root)
    sections = (
        ("Overall predictive performance", "overall"),
        ("State interpolation and extrapolation", "state"),
        ("Unseen-component generalization", "chemistry"),
        ("Binary-to-ternary transfer", "transfer"),
    )
    lines = [
        "# C1 Three-View Vanilla ThermoFormer: Predictive Performance and Generalization",
        "",
        "Final architecture: RDKit descriptors + Uni-Mol v2 + functional-group features, "
        "independent projections and fusion, vanilla Transformer, original pair potential. "
        "Stage 2 retains supervised loss and adds only the teacher-forced fugacity-equilibrium loss.",
        "",
        "All values are mean ± standard deviation across seeds 0--4. Checkpoint selection uses "
        "validation only; the test partition is evaluated after selection. Isothermal inference is "
        "`molecules,T,x -> P,y`; isobaric inference is `molecules,P,x -> T,y`.",
        "",
    ]
    for title, category in sections:
        lines.extend([f"## {title}", "", _task_table(row for row in rows if row["category"] == category), ""])
    lines.extend(
        [
            "## Fugacity fine-tuning selection",
            "",
            f"Across 15 protocols × 5 seeds, validation retained Stage 1 for "
            f"**{totals['stage1']}/75** runs and selected Stage 2 for **{totals['stage2']}/75** runs. "
            f"The teacher-forced fugacity residual decreased after Stage 2 in "
            f"**{totals['residual_improved']}/15** protocol means.",
            "",
            f"As a post-selection descriptive comparison, raw Stage 2 test means improved over "
            f"raw Stage 1 in **{totals['predictive_improved']}/{totals['predictive_compared']}** "
            "direction-resolved MAE/RMSE/R² cells. This comparison was not used to tune the loss "
            "or select checkpoints.",
            "",
            "This is evidence for using validation-gated fugacity fine-tuning, not a claim that "
            "Stage 2 uniformly improves every predictive metric or every random seed.",
            "",
            "## Numerical diagnostics",
            "",
            f"Maximum mean solver-failure rate: `{max(float(row['solver_failure_rate_mean']) for row in rows):.6f}`; "
            f"maximum mean nonphysical rate: `{max(float(row['nonphysical_rate_mean']) for row in rows):.6f}`; "
            f"minimum valid coverage: `{min(float(row['valid_coverage_mean']) for row in rows):.6f}`.",
            "",
            "Machine-readable task metrics and Stage-1/Stage-2 selection diagnostics are stored in "
            f"`{portable_artifact_path(table_path, project_root)}` and "
            f"`{portable_artifact_path(stage_table_path, project_root)}`.",
            "",
        ]
    )
    atomic_write_text(table_path, _csv_text(rows))
    atomic_write_text(stage_table_path, _csv_text(stage_rows))
    if publish_canonical_leaves:
        _write_generalization_leaf_records(project_root, rows, table_path)
    atomic_write_text(report_path, "\n".join(lines))

    inputs = []
    for spec in PROTOCOLS:
        path = _protocol_dir(project_root, spec.name) / "report_manifest.json"
        inputs.append(
            {
                "protocol": spec.name,
                "path": portable_artifact_path(path, project_root),
                "sha256": artifact_sha256(path),
            }
        )
    outputs = {
        "report": {
            "path": portable_artifact_path(report_path, project_root),
            "sha256": artifact_sha256(report_path),
        },
        "task_metrics": {
            "path": portable_artifact_path(table_path, project_root),
            "sha256": artifact_sha256(table_path),
        },
        "stage_selection": {
            "path": portable_artifact_path(stage_table_path, project_root),
            "sha256": artifact_sha256(stage_table_path),
        },
    }
    if publish_canonical_leaves:
        leaf_paths = [
            project_root / "experiments/vle/prediction/study_records/overall_binary/results.md",
            project_root / "experiments/vle/prediction/study_records/overall_binary_ternary/results.md",
            project_root / "experiments/vle/prediction/study_records/state_generalization/results.md",
            project_root / "experiments/vle/prediction/study_records/unseen_components/results.md",
            project_root / "experiments/vle/prediction/study_records/binary_to_ternary/results.md",
        ]
        outputs["canonical_experiment_leaves"] = [
            {
                "path": portable_artifact_path(path, project_root),
                "sha256": artifact_sha256(path),
            }
            for path in leaf_paths
        ]
    atomic_write_json(
        manifest_path,
        {
            "status": "completed",
            "analysis_status": "confirmatory",
            "protocols": [spec.name for spec in PROTOCOLS],
            "seeds": [0, 1, 2, 3, 4],
            "selection_partition": "validation",
            "evaluation_partition": "test",
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    return {
        key: value["path"]
        for key, value in outputs.items()
        if isinstance(value, dict) and "path" in value
    } | {
        "manifest": portable_artifact_path(manifest_path, project_root)
    }
