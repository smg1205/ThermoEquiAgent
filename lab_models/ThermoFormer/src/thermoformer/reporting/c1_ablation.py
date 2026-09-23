"""Build the focused C1 ablation report from frozen overall-test artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import artifact_sha256, atomic_write_text


@dataclass(frozen=True)
class AblationSource:
    label: str
    family: str
    result_dir: str


C1_ABLATION_SOURCES = {
    "c0_unimol": AblationSource(
        "C0 Uni-Mol v2 vanilla", "representation",
        "experiments/vle/ablation/multiview/chemical_attention/formal/runs/"
        "c0_current_vanilla.on.overall_binary_ternary",
    ),
    "v1_rdkit": AblationSource(
        "V1 RDKit descriptors only", "representation",
        "experiments/vle/ablation/multiview/formal/runs/v1_rdkit_only.on.overall_binary_ternary",
    ),
    "v3_fg": AblationSource(
        "V3 functional groups only", "representation",
        "experiments/vle/ablation/multiview/predictive/runs/"
        "v3_functional_group_only.on.overall_binary_ternary",
    ),
    "v4_rdkit_unimol": AblationSource(
        "V4 RDKit + Uni-Mol", "representation",
        "experiments/vle/ablation/multiview/predictive/runs/"
        "v4_rdkit_unimol_naive.on.overall_binary_ternary",
    ),
    "c1_final": AblationSource(
        "C1 RDKit + Uni-Mol + FG vanilla", "representation interaction",
        "experiments/vle/ablation/multiview/chemical_attention/formal/runs/"
        "c1_three_view_vanilla.on.overall_binary_ternary",
    ),
    "c2_chemical_bias": AblationSource(
        "C2 chemical-biased + context pair", "interaction",
        "experiments/vle/ablation/multiview/chemical_attention/formal/runs/"
        "c2_chemical_bias_full.on.overall_binary_ternary",
    ),
    "c3_context_pair": AblationSource(
        "C3 context pair without attention bias", "interaction",
        "experiments/vle/ablation/multiview/chemical_attention/formal/runs/"
        "c3_no_pair_bias.on.overall_binary_ternary",
    ),
}

PHYSICS_RESULT_DIR = (
    "experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity/"
    "c1_three_view_vanilla_fugacity_finetune.on.overall_binary_ternary"
)

ABLATION_LEAVES = {
    "c0_unimol": (
        "experiments/vle/ablation/study_records/molecular_representation/unimol_v2_only/results.md",
        "Uni-Mol vanilla baseline",
        "The three-view C1 model substantially reduces all four prediction errors relative to this Uni-Mol-only vanilla baseline.",
    ),
    "v1_rdkit": (
        "experiments/vle/ablation/study_records/molecular_representation/rdkit_only/results.md",
        "RDKit-only representation",
        "RDKit descriptors provide a strong low-dimensional baseline, but the complete three-view model gives lower pressure and isobaric-temperature errors.",
    ),
    "v3_fg": (
        "experiments/vle/ablation/study_records/molecular_representation/functional_groups_only/results.md",
        "Functional-group-only representation",
        "Functional-group counts alone are insufficient for VLE prediction. The error values must be interpreted with their reduced valid coverage.",
    ),
    "v4_rdkit_unimol": (
        "experiments/vle/ablation/study_records/molecular_representation/rdkit_unimol/results.md",
        "RDKit + Uni-Mol representation",
        "Adding the functional-group view to this two-view model improves pressure and isobaric-temperature MAE, while vapor-composition changes are small.",
    ),
    "c1_final": (
        "experiments/vle/ablation/study_records/molecular_representation/full_three_view/results.md",
        "Three-view vanilla representation",
        "The three complementary molecular views give the most balanced representation result and define the final C1 architecture.",
    ),
    "c1_final_interaction": (
        "experiments/vle/ablation/study_records/interaction_architecture/vanilla_transformer/results.md",
        "Three-view vanilla Transformer",
        "C1 is the selected interaction architecture because it balances pressure, temperature, vapor-composition accuracy, and model complexity.",
    ),
    "c2_chemical_bias": (
        "experiments/vle/ablation/study_records/interaction_architecture/chemical_interaction_bias/results.md",
        "Chemical-interaction-biased Transformer",
        "The chemical-attention bias does not consistently improve the three-view vanilla model and is not retained in the final architecture.",
    ),
    "c3_context_pair": (
        "experiments/vle/ablation/study_records/interaction_architecture/context_pair_without_attention_bias/results.md",
        "Context pair without attention bias",
        "The context-pair variant improves several vapor-composition metrics but has a larger pressure error than C1, so it is not selected as the balanced final model.",
    ),
}


def _write_ablation_leaf_records(
    project_root: Path,
    records: list[dict[str, object]],
) -> None:
    """Render each canonical ablation record from its validated aggregate row."""
    for variant_id, (relative_path, title, conclusion) in ABLATION_LEAVES.items():
        source_variant_id = "c1_final" if variant_id == "c1_final_interaction" else variant_id
        selected = sorted(
            (row for row in records if row["variant_id"] == source_variant_id),
            key=lambda row: str(row["direction"]),
        )
        if len(selected) != 2:
            raise RuntimeError(f"Expected two task directions for {variant_id}")
        lines = [
            f"# {title}",
            "",
            "Setting: `overall_binary_ternary`, seeds 0--4. Values are mean ± sample standard deviation.",
            "",
            "| Task | State MAE | State RMSE | State R² | y MAE | y RMSE | y R² | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in selected:
            task = "Isothermal" if row["direction"] == "isothermal" else "Isobaric"
            unit = str(row["state_unit"])
            lines.append(
                f"| {task} | {_mean_std(row, 'state_mae', 3)} {unit} | "
                f"{_mean_std(row, 'state_rmse', 3)} {unit} | {_mean_std(row, 'state_r2', 3)} | "
                f"{_mean_std(row, 'y_mae', 4)} | {_mean_std(row, 'y_rmse', 4)} | "
                f"{_mean_std(row, 'y_r2', 3)} | {100.0 * float(row['valid_coverage_mean']):.1f}% |"
            )
        source = selected[0]
        lines.extend(
            [
                "",
                conclusion,
                "",
                f"Machine-readable source: `{source['source']}` "
                f"(SHA-256 `{source['source_sha256']}`).",
            ]
        )
        atomic_write_text(project_root / relative_path, "\n".join(lines) + "\n")


def _validated_summary(
    project_root: Path,
    source: AblationSource,
) -> tuple[Path, Path, list[dict[str, str]]]:
    result_dir = project_root / source.result_dir
    manifest_path = result_dir / "aggregate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("seeds") != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Incomplete five-seed aggregate: {source.label}")
    if not str(manifest.get("protocol", "")).endswith(".on.overall_binary_ternary"):
        raise RuntimeError(f"Unexpected ablation protocol: {source.label}")
    summary_path = result_dir / "metrics_summary.csv"
    expected = manifest.get("outputs", {}).get("metrics_summary", {}).get("sha256")
    if artifact_sha256(summary_path) != expected:
        raise RuntimeError(f"Aggregate summary SHA mismatch: {source.label}")
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scope"] == "direction"]
    if {row["direction"] for row in rows} != {"isothermal", "isobaric"}:
        raise RuntimeError(f"Missing direction metrics: {source.label}")
    return manifest_path, summary_path, rows


def collect_c1_ablation_rows(project_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for variant_id, source in C1_ABLATION_SOURCES.items():
        manifest_path, summary_path, rows = _validated_summary(project_root, source)
        for row in rows:
            direction = row["direction"]
            prefix = "pressure" if direction == "isothermal" else "temperature"
            unit_suffix = "_kpa" if direction == "isothermal" else "_k"
            records.append(
                {
                    "variant_id": variant_id,
                    "variant": source.label,
                    "family": source.family,
                    "protocol": "overall_binary_ternary",
                    "direction": direction,
                    "state": "P" if direction == "isothermal" else "T",
                    "state_unit": "kPa" if direction == "isothermal" else "K",
                    "state_mae_mean": float(row[f"{prefix}_mae{unit_suffix}_mean"]),
                    "state_mae_std": float(row[f"{prefix}_mae{unit_suffix}_std"]),
                    "state_rmse_mean": float(row[f"{prefix}_rmse{unit_suffix}_mean"]),
                    "state_rmse_std": float(row[f"{prefix}_rmse{unit_suffix}_std"]),
                    "state_r2_mean": float(row[f"{prefix}_r2_mean"]),
                    "state_r2_std": float(row[f"{prefix}_r2_std"]),
                    "y_mae_mean": float(row["y_mae_mean"]),
                    "y_mae_std": float(row["y_mae_std"]),
                    "y_rmse_mean": float(row["y_rmse_mean"]),
                    "y_rmse_std": float(row["y_rmse_std"]),
                    "y_r2_mean": float(row["y_r2_mean"]),
                    "y_r2_std": float(row["y_r2_std"]),
                    "valid_coverage_mean": float(row["valid_coverage_mean"]),
                    "valid_coverage_std": float(row["valid_coverage_std"]),
                    "solver_failure_rate_mean": float(row["solver_failure_rate_mean"]),
                    "solver_failure_rate_std": float(row["solver_failure_rate_std"]),
                    "source": summary_path.relative_to(project_root).as_posix(),
                    "source_sha256": artifact_sha256(summary_path),
                    "aggregate_manifest": manifest_path.relative_to(project_root).as_posix(),
                    "aggregate_manifest_sha256": artifact_sha256(manifest_path),
                }
            )
    return records


def _mean_std(row: dict[str, object], name: str, digits: int) -> str:
    return f"{float(row[name + '_mean']):.{digits}f} ± {float(row[name + '_std']):.{digits}f}"


def _table(records: list[dict[str, object]], family: str, direction: str) -> list[str]:
    selected = [
        row for row in records
        if family in str(row["family"]).split() and row["direction"] == direction
    ]
    state = "P (kPa)" if direction == "isothermal" else "T (K)"
    lines = [
        f"| Variant | {state} MAE | {state} RMSE | {state} R² | y MAE | y RMSE | y R² | Valid coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['variant']} | {_mean_std(row, 'state_mae', 3)} | "
            f"{_mean_std(row, 'state_rmse', 3)} | {_mean_std(row, 'state_r2', 3)} | "
            f"{_mean_std(row, 'y_mae', 4)} | {_mean_std(row, 'y_rmse', 4)} | "
            f"{_mean_std(row, 'y_r2', 3)} | "
            f"{100.0 * float(row['valid_coverage_mean']):.1f}% |"
        )
    return lines


def _physics_rows(project_root: Path) -> tuple[Path, Path, dict[str, object]]:
    result_dir = project_root / PHYSICS_RESULT_DIR
    manifest_path = result_dir / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("seeds") != [0, 1, 2, 3, 4]
        or manifest.get("selection_partition") != "validation"
        or manifest.get("evaluation_partition") != "test"
    ):
        raise RuntimeError("Fugacity five-seed report is incomplete")
    summary_record = manifest.get("stage_comparison_summary", {})
    summary_path = project_root / str(summary_record.get("path", ""))
    if artifact_sha256(summary_path) != summary_record.get("sha256"):
        raise RuntimeError("Fugacity stage-comparison summary SHA mismatch")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("seeds") != [0, 1, 2, 3, 4]:
        raise RuntimeError("Fugacity stage summary has unexpected seeds")
    return manifest_path, summary_path, payload


def _write_fugacity_leaf_record(project_root: Path, manifest_path: Path) -> None:
    """Copy the validated five-seed stage report into its manuscript leaf."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_record = manifest.get("report", {})
    source_path = project_root / str(report_record.get("path", ""))
    if artifact_sha256(source_path) != report_record.get("sha256"):
        raise RuntimeError("Fugacity report SHA mismatch")
    source_text = source_path.read_text(encoding="utf-8").rstrip()
    canonical_path = (
        project_root / "experiments/vle/ablation/study_records/fugacity_finetuning/results.md"
    )
    footer = (
        "\n\nMachine-readable stage summary: `"
        + PHYSICS_RESULT_DIR
        + "/stage_comparison_summary.json` (SHA-256 `"
        + str(manifest["stage_comparison_summary"]["sha256"])
        + "`).\n"
    )
    atomic_write_text(canonical_path, source_text + footer)


def write_joint_c1_ablation_outputs(
    project_root: Path,
    *,
    output_root: Path | None = None,
    report_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    publish_canonical_leaves = output_root is None and report_path is None
    records = collect_c1_ablation_rows(project_root)
    if publish_canonical_leaves:
        _write_ablation_leaf_records(project_root, records)
    physics_manifest_path, physics_path, physics = _physics_rows(project_root)
    if publish_canonical_leaves:
        _write_fugacity_leaf_record(project_root, physics_manifest_path)
    output_root = output_root or project_root / "experiments/vle/ablation/binary_summary"
    report_path = report_path or project_root / "experiments/data_quality/reports/c1_ablation_overall_binary_ternary.md"
    metrics_path = output_root / "overall_binary_ternary_metrics.csv"
    manifest_path = output_root / "report_manifest.json"

    def recorded_path(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return str(path)

    fieldnames = list(records[0])
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    atomic_write_text(metrics_path, stream.getvalue())

    lines = [
        "# C1 three-view vanilla ablation results",
        "",
        "All representation and interaction ablations use the fixed `overall_binary_ternary` protocol: joint binary and ternary training, "
        "followed by evaluation on the joint binary and ternary test set. Values are the mean ± sample standard deviation over seeds 0--4.",
        "",
        "## Molecular-representation ablation",
        "",
        "### Isothermal task: molecules, T, x -> P, y",
        "",
        *_table(records, "representation", "isothermal"),
        "",
        "### Isobaric task: molecules, P, x -> T, y",
        "",
        *_table(records, "representation", "isobaric"),
        "",
        "The three-view C1 model reduces all four prediction errors relative to Uni-Mol only. Functional groups alone do not provide adequate VLE predictions; "
        "their isothermal/isobaric valid coverage is only 63.4%/96.7%, so errors must be interpreted together with coverage. "
        "Adding functional groups to RDKit and Uni-Mol improves P and isobaric T, with smaller changes in y.",
        "",
        "## Component-interaction ablation",
        "",
        "### Isothermal task",
        "",
        *_table(records, "interaction", "isothermal"),
        "",
        "### Isobaric task",
        "",
        *_table(records, "interaction", "isobaric"),
        "",
        "The C2 chemically biased Transformer does not consistently outperform C1. C3 performs better for T/y but substantially worse for pressure. "
        "Across P, T, y, and parameter complexity, C1 provides the most balanced architecture.",
        "",
        "## Fugacity-loss fine-tuning ablation (seeds 0--4)",
        "",
        "Stage 1 is the validation-best C1 checkpoint after supervised training; Stage 2 retains the supervised loss and adds only "
        "the teacher-forced fugacity-equilibrium loss for 10 fine-tuning epochs. Checkpoints are selected only on validation data.",
        "",
        "| Task output | Stage 1 MAE | Stage 1 RMSE | Stage 1 R² | Fugacity Stage 2 MAE | Fugacity Stage 2 RMSE | Fugacity Stage 2 R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stages = physics["stages"]

    def physics_value(stage: str, direction: str, key: str) -> str:
        record = stages[stage]["directions"][direction][key]
        return f"{float(record['mean']):.6f} ± {float(record['std']):.6f}"

    for direction, state, prefix, suffix in (
        ("isothermal", "P", "pressure", "_kpa"),
        ("isothermal", "y", "y", ""),
        ("isobaric", "T", "temperature", "_k"),
        ("isobaric", "y", "y", ""),
    ):
        lines.append(
            f"| {state}, {direction} | {physics_value('stage1', direction, prefix + '_mae' + suffix)} | "
            f"{physics_value('stage1', direction, prefix + '_rmse' + suffix)} | "
            f"{physics_value('stage1', direction, prefix + '_r2')} | "
            f"{physics_value('stage2', direction, prefix + '_mae' + suffix)} | "
            f"{physics_value('stage2', direction, prefix + '_rmse' + suffix)} | "
            f"{physics_value('stage2', direction, prefix + '_r2')} |"
        )
    counts = physics["selected_stage_counts"]
    fugacity1 = stages["stage1"]["teacher_forced_fugacity"]
    fugacity2 = stages["stage2"]["teacher_forced_fugacity"]
    lines.extend(
        [
            "",
            f"Validation selects Stage 2 for **{int(counts['stage2'])}/5** ** seeds and retains Stage 1 for "
            f"**{int(counts['stage1'])}/5**  seeds.",
            "The mean teacher-forced fugacity residual on the test set decreases from "
            f"`{float(fugacity1['mean']):.6g} ± {float(fugacity1['std']):.6g}` to "
            f"`{float(fugacity2['mean']):.6g} ± {float(fugacity2['std']):.6g}.",
            "Across five-seed means, 10 of 12 P/T/y metrics improve; T RMSE and T R² deteriorate slightly. "
            "Fugacity fine-tuning therefore provides an overall benefit but does not improve every seed; the final workflow retains the validation-selected Stage 1 fallback.",
            "",
            "## Final selection",
            "",
            "The selected model is **C1 RDKit descriptors + Uni-Mol v2 + functional groups + vanilla Transformer**. "
            "Stage 2 adds only the teacher-forced fugacity-equilibrium loss. The pure-endpoint Psat term in supervised training remains a data-supervision term and "
            "is not an additional physics fine-tuning loss.",
            "",
        ]
    )
    atomic_write_text(report_path, "\n".join(lines))
    manifest = {
        "status": "completed",
        "protocol": "overall_binary_ternary",
        "representation_and_interaction_seeds": [0, 1, 2, 3, 4],
        "physics_finetuning_seeds": [0, 1, 2, 3, 4],
        "inputs": {
            variant_id: {
                "metrics_summary": {
                    "path": next(row["source"] for row in records if row["variant_id"] == variant_id),
                    "sha256": next(row["source_sha256"] for row in records if row["variant_id"] == variant_id),
                },
                "aggregate_manifest": {
                    "path": next(row["aggregate_manifest"] for row in records if row["variant_id"] == variant_id),
                    "sha256": next(
                        row["aggregate_manifest_sha256"]
                        for row in records
                        if row["variant_id"] == variant_id
                    ),
                },
            }
            for variant_id in C1_ABLATION_SOURCES
        },
        "physics_stage_comparison": {
            "path": physics_path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(physics_path),
        },
        "physics_report_manifest": {
            "path": physics_manifest_path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(physics_manifest_path),
        },
        "outputs": {
            "metrics": {"path": recorded_path(metrics_path), "sha256": artifact_sha256(metrics_path)},
            "report": {"path": recorded_path(report_path), "sha256": artifact_sha256(report_path)},
        },
    }
    if publish_canonical_leaves:
        leaf_paths = [project_root / value[0] for value in ABLATION_LEAVES.values()]
        leaf_paths.append(project_root / "experiments/vle/ablation/study_records/fugacity_finetuning/results.md")
        manifest["outputs"]["canonical_experiment_leaves"] = [
            {
                "path": recorded_path(path),
                "sha256": artifact_sha256(path),
            }
            for path in leaf_paths
        ]
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report_path, metrics_path, manifest_path

from .c1_ablation_binary import (
    AblationSource as BinaryAblationSource,
    C1_ABLATION_SOURCES as BINARY_C1_ABLATION_SOURCES,
    collect_c1_ablation_rows as collect_binary_c1_ablation_rows,
    write_c1_ablation_outputs as write_binary_c1_ablation_outputs,
)

# Backward-compatible default for the registered binary three-stage report.
write_c1_ablation_outputs = write_binary_c1_ablation_outputs

