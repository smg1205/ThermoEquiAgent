"""Validated binary-only, three-stage Table 2 reporting.

Only the formal overall_binary_three_stage namespace is accepted. Historical
joint binary--ternary results are intentionally never read by this module.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from .artifacts import artifact_sha256, atomic_write_json, atomic_write_text

FORMAL_PROTOCOL = "overall_binary"
RESULT_ROOT = Path("experiments/vle/generalization/evaluations/ablations/overall_binary_three_stage")
SEEDS = (0, 1, 2, 3, 4)
STAGES = ("stage0", "stage1", "stage2", "stage3")
REPORT_STAGES = (*STAGES, "selected")
FIELDS = {
    "isothermal": ("pressure_mae_kpa", "pressure_rmse_kpa", "pressure_r2", "y_mae", "y_rmse", "y_r2", "valid_coverage", "solver_failure_rate", "nonphysical_rate"),
    "isobaric": ("temperature_mae_k", "temperature_rmse_k", "temperature_r2", "y_mae", "y_rmse", "y_r2", "valid_coverage", "solver_failure_rate", "nonphysical_rate"),
}

@dataclass(frozen=True)
class AblationSource:
    label: str
    family: str
    result_protocol: str

C1_ABLATION_SOURCES = {
    "c0_unimol": AblationSource("Uni-Mol v2 only", "representation", "c0_current_vanilla.on.overall_binary"),
    "v1_rdkit": AblationSource("RDKit descriptors only", "representation", "v1_rdkit_only.on.overall_binary"),
    "v3_fg": AblationSource("Functional-group features only", "representation", "v3_functional_group_only.on.overall_binary"),
    "v4_rdkit_unimol": AblationSource("RDKit descriptors + Uni-Mol v2", "representation", "v4_rdkit_unimol_naive.on.overall_binary"),
    "c1_final": AblationSource("Full three-view representation", "representation interaction", "c1_three_view_vanilla.on.overall_binary"),
    "c2_chemical_bias": AblationSource("Chemical-interaction-biased Transformer with context-conditioned pair potential", "interaction", "c2_chemical_bias_full.on.overall_binary"),
    "c3_context_pair": AblationSource("Context-conditioned pair potential without attention bias", "interaction", "c3_no_pair_bias.on.overall_binary"),
}
REPRESENTATION = tuple((key, C1_ABLATION_SOURCES[key].label) for key in ("c0_unimol", "v1_rdkit", "v3_fg", "v4_rdkit_unimol", "c1_final"))
INTERACTION = (("c1_final", "Vanilla multicomponent Transformer"), ("c2_chemical_bias", C1_ABLATION_SOURCES["c2_chemical_bias"].label), ("c3_context_pair", C1_ABLATION_SOURCES["c3_context_pair"].label))
STAGE_LABELS = {"stage0": "Stage 0 supervised reference", "stage1": "Stage 1 direct GE/RT and ln(gamma)", "stage2": "Stage 2 joint VLE supervision", "stage3": "Stage 3 fugacity-constrained fine-tuning", "selected": "Validation-selected checkpoint"}


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def _integer(value: object, name: str) -> int:
    result = _finite(value, name)
    if not result.is_integer():
        raise RuntimeError(f"{name} is not an integer")
    return int(result)


def _read(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Malformed {name}: {path}")
    return value


def _path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} has no artifact path")
    candidate = Path(value)
    result = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{name} escapes the project root") from error
    return result


def _artifact(root: Path, record: object, name: str, expected: Path | None = None) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{name} has no artifact record")
    path = _path(root, record.get("path"), name)
    if expected is not None and path != expected.resolve():
        raise RuntimeError(f"{name} points to an unexpected location")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not path.is_file() or artifact_sha256(path) != digest:
        raise RuntimeError(f"{name} SHA-256 verification failed")
    return path


def _stage_artifact_link(
    root: Path,
    payload: Mapping[str, Any],
    field: str,
    artifact_path: Path,
    artifact_record: object,
    name: str,
) -> None:
    """Require stage-comparison provenance to agree with its manifest artifact."""

    if not isinstance(artifact_record, Mapping):
        raise RuntimeError(f"{name} has no manifest artifact record")
    linked_path = _path(root, payload.get(field), f"{name} {field}")
    if linked_path != artifact_path:
        raise RuntimeError(f"{name} {field} path does not match the manifest")
    digest = payload.get(f"{field}_sha256")
    manifest_digest = artifact_record.get("sha256")
    if (
        not isinstance(digest, str)
        or not isinstance(manifest_digest, str)
        or digest != manifest_digest
        or not linked_path.is_file()
        or artifact_sha256(linked_path) != digest
    ):
        raise RuntimeError(f"{name} {field} SHA-256 verification failed")


def _directions(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["direction"]): row for row in rows if row.get("scope") == "direction" and row.get("direction") in FIELDS}
    if set(result) != set(FIELDS):
        raise RuntimeError(f"{name} must contain isothermal and isobaric metrics")
    return result


def _metrics(rows: Mapping[str, Mapping[str, Any]], name: str, summary: bool) -> None:
    for direction, row in rows.items():
        if summary and (_integer(row.get("seeds"), name + ".seeds") != 5 or _integer(row.get("scope_available_seeds"), name + ".scope") != 5):
            raise RuntimeError(f"{name}.{direction} is not five-seed formal output")
        for field in FIELDS[direction]:
            _finite(row.get(field + ("_mean" if summary else "")), name + "." + field)
            if summary:
                _finite(row.get(field + "_std"), name + "." + field + "_std")
                if _integer(row.get(field + "_available_seeds"), name + "." + field + " availability") != 5:
                    raise RuntimeError(f"{name}.{field} is incomplete")


def _load(root: Path, source_id: str, source: AblationSource) -> dict[str, Any]:
    if not source.result_protocol.endswith(".on.overall_binary") or "overall_binary_ternary" in source.result_protocol:
        raise RuntimeError(f"{source_id} is not binary-only")
    directory = root / RESULT_ROOT / source.result_protocol
    aggregate_path = directory / "aggregate_manifest.json"
    aggregate = _read(aggregate_path, source_id + " aggregate")
    if aggregate.get("status") != "completed" or aggregate.get("aggregate_kind") != "formal" or aggregate.get("seeds") != list(SEEDS) or aggregate.get("protocol") != source.result_protocol:
        raise RuntimeError(f"{source_id} aggregate is incomplete or wrong-protocol")
    provenance, outputs = aggregate.get("input_provenance"), aggregate.get("outputs")
    if not isinstance(provenance, Mapping) or provenance.get("protocol") != source.result_protocol or not isinstance(outputs, Mapping):
        raise RuntimeError(f"{source_id} aggregate provenance is invalid")
    summary_path = _artifact(root, outputs.get("metrics_summary"), source_id + " summary", directory / "metrics_summary.csv")
    _artifact(root, outputs.get("metrics_by_seed"), source_id + " metrics by seed", directory / "metrics_by_seed.csv")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary = _directions(list(csv.DictReader(handle)), source_id + " summary")
    _metrics(summary, source_id + " summary", True)
    manifests, comparisons = {}, {}
    for seed in SEEDS:
        seed_dir = directory / f"seed_{seed}"
        manifest = _read(seed_dir / "manifest.json", f"{source_id} seed {seed} manifest")
        expected = {"status": "completed", "seed": seed, "protocol": source.result_protocol, "split_protocol": FORMAL_PROTOCOL, "run_kind": "formal", "selection_partition": "validation", "evaluation_partition": "test"}
        if any(manifest.get(key) != value for key, value in expected.items()) or manifest.get("selected_stage") not in STAGES:
            raise RuntimeError(f"{source_id} seed {seed} is not formal binary validation-selected output")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"{source_id} seed {seed} lacks artifact provenance")
        comparison_path = _artifact(root, artifacts.get("stage_comparison"), f"{source_id} seed {seed} comparison", seed_dir / "stage_comparison.json")
        stage_artifacts: dict[str, tuple[Path, object, Path, object]] = {}
        for stage in STAGES:
            checkpoint_record = artifacts.get(stage + "_checkpoint")
            checkpoint_path = _artifact(
                root,
                checkpoint_record,
                f"{source_id} seed {seed} {stage} checkpoint",
            )
            prediction_record = artifacts.get(stage + "_predictions")
            prediction_path = _artifact(
                root,
                prediction_record,
                f"{source_id} seed {seed} {stage} predictions",
                seed_dir / f"{stage}_predictions.csv",
            )
            stage_artifacts[stage] = (
                checkpoint_path,
                checkpoint_record,
                prediction_path,
                prediction_record,
            )
        comparison = _read(comparison_path, f"{source_id} seed {seed} comparison")
        stages = comparison.get("stages")
        if comparison.get("selection_partition") != "validation" or comparison.get("evaluation_partition") != "test" or comparison.get("selected_stage") != manifest.get("selected_stage") or not isinstance(stages, Mapping) or not set(STAGES).issubset(stages):
            raise RuntimeError(f"{source_id} seed {seed} has invalid stage evidence")
        for stage in STAGES:
            payload = stages[stage]
            if not isinstance(payload, Mapping) or not isinstance(payload.get("metrics"), list):
                raise RuntimeError(f"{source_id} seed {seed} {stage} metrics are malformed")
            checkpoint_path, checkpoint_record, prediction_path, prediction_record = (
                stage_artifacts[stage]
            )
            _stage_artifact_link(
                root, payload, "checkpoint", checkpoint_path, checkpoint_record,
                f"{source_id} seed {seed} {stage}",
            )
            _stage_artifact_link(
                root, payload, "predictions", prediction_path, prediction_record,
                f"{source_id} seed {seed} {stage}",
            )
            _metrics(_directions(payload["metrics"], f"{source_id} seed {seed} {stage}"), f"{source_id} seed {seed} {stage}", False)
        manifests[seed], comparisons[seed] = manifest, comparison
    return {"id": source_id, "source": source, "directory": directory, "aggregate": aggregate_path, "summary_path": summary_path, "summary": summary, "manifests": manifests, "comparisons": comparisons}


def _stat(values: list[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise RuntimeError("Formal statistics require exactly five seed values")
    return fmean(values), stdev(values)


def _standard_row(item: Mapping[str, Any], direction: str) -> dict[str, object]:
    raw = item["summary"][direction]
    prefix, suffix, unit = ("pressure", "_kpa", "kPa") if direction == "isothermal" else ("temperature", "_k", "K")
    result = {"variant_id": item["id"], "variant": item["source"].label, "family": item["source"].family, "protocol": FORMAL_PROTOCOL, "result_protocol": item["source"].result_protocol, "direction": direction, "state": "P" if direction == "isothermal" else "T", "state_unit": unit}
    pairs = (("state_mae", prefix + "_mae" + suffix), ("state_rmse", prefix + "_rmse" + suffix), ("state_r2", prefix + "_r2"), ("y_mae", "y_mae"), ("y_rmse", "y_rmse"), ("y_r2", "y_r2"), ("valid_coverage", "valid_coverage"), ("solver_failure_rate", "solver_failure_rate"), ("nonphysical_rate", "nonphysical_rate"))
    for target, field in pairs:
        result[target + "_mean"], result[target + "_std"] = _finite(raw[field + "_mean"], field), _finite(raw[field + "_std"], field)
    result["source"] = str(item["summary_path"])
    result["source_sha256"] = artifact_sha256(item["summary_path"])
    return result


def _stage_rows(item: Mapping[str, Any], stage: str) -> list[dict[str, object]]:
    values = {direction: {field: [] for field in FIELDS[direction]} for direction in FIELDS}
    resolved = {}
    for seed in SEEDS:
        comparison = item["comparisons"][seed]
        actual = comparison["selected_stage"] if stage == "selected" else stage
        resolved[seed] = actual
        rows = _directions(comparison["stages"][actual]["metrics"], "stage metrics")
        for direction in FIELDS:
            for field in FIELDS[direction]:
                values[direction][field].append(_finite(rows[direction][field], field))
    answer = []
    for direction in FIELDS:
        prefix, suffix, unit = ("pressure", "_kpa", "kPa") if direction == "isothermal" else ("temperature", "_k", "K")
        summary = {field: _stat(data) for field, data in values[direction].items()}
        result = {"stage": stage, "stage_label": STAGE_LABELS[stage], "variant_id": item["id"], "direction": direction, "state": "P" if direction == "isothermal" else "T", "state_unit": unit, "resolved_stages_by_seed": ";".join(f"{seed}:{resolved[seed]}" for seed in SEEDS)}
        pairs = (("state_mae", prefix + "_mae" + suffix), ("state_rmse", prefix + "_rmse" + suffix), ("state_r2", prefix + "_r2"), ("y_mae", "y_mae"), ("y_rmse", "y_rmse"), ("y_r2", "y_r2"), ("valid_coverage", "valid_coverage"), ("solver_failure_rate", "solver_failure_rate"), ("nonphysical_rate", "nonphysical_rate"))
        for target, field in pairs:
            result[target + "_mean"], result[target + "_std"] = summary[field]
        answer.append(result)
    return answer


def _selected_stage_provenance(
    loaded: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, int]]]:
    """Return validation-selected stages by seed and complete per-stage counts."""

    by_seed: list[dict[str, object]] = []
    counts_by_variant: dict[str, dict[str, int]] = {}
    count_rows: list[dict[str, object]] = []
    for source_id, item in loaded.items():
        source = item["source"]
        counts = {stage: 0 for stage in STAGES}
        for seed in SEEDS:
            manifest = item["manifests"][seed]
            comparison = item["comparisons"][seed]
            selected = manifest.get("selected_stage")
            if selected not in STAGES or comparison.get("selected_stage") != selected:
                raise RuntimeError(
                    f"{source_id} seed {seed} has inconsistent selected-stage provenance"
                )
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise RuntimeError(f"{source_id} seed {seed} lacks selected-stage artifacts")
            comparison_record = artifacts.get("stage_comparison")
            if not isinstance(comparison_record, Mapping):
                raise RuntimeError(
                    f"{source_id} seed {seed} lacks stage-comparison provenance"
                )
            comparison_path = comparison_record.get("path")
            comparison_sha256 = comparison_record.get("sha256")
            if not isinstance(comparison_path, str) or not isinstance(comparison_sha256, str):
                raise RuntimeError(
                    f"{source_id} seed {seed} has malformed stage-comparison provenance"
                )
            manifest_path = item["directory"] / f"seed_{seed}" / "manifest.json"
            counts[selected] += 1
            by_seed.append(
                {
                    "variant_id": source_id,
                    "variant": source.label,
                    "family": source.family,
                    "protocol": FORMAL_PROTOCOL,
                    "result_protocol": source.result_protocol,
                    "seed": seed,
                    "selected_stage": selected,
                    "selection_partition": manifest["selection_partition"],
                    "evaluation_partition": manifest["evaluation_partition"],
                    "seed_manifest_path": str(manifest_path),
                    "seed_manifest_sha256": artifact_sha256(manifest_path),
                    "stage_comparison_path": comparison_path,
                    "stage_comparison_sha256": comparison_sha256,
                }
            )
        if sum(counts.values()) != len(SEEDS):
            raise RuntimeError(f"{source_id} selected-stage counts are incomplete")
        counts_by_variant[source_id] = counts
        for stage in STAGES:
            count_rows.append(
                {
                    "variant_id": source_id,
                    "variant": source.label,
                    "family": source.family,
                    "protocol": FORMAL_PROTOCOL,
                    "result_protocol": source.result_protocol,
                    "selected_stage": stage,
                    "selected_seed_count": counts[stage],
                    "total_seeds": len(SEEDS),
                }
            )
    return by_seed, count_rows, counts_by_variant


def _match_selected(item: Mapping[str, Any], selected: Sequence[Mapping[str, object]]) -> None:
    by_direction = {row["direction"]: row for row in selected}
    for direction, raw in item["summary"].items():
        prefix, suffix = ("pressure", "_kpa") if direction == "isothermal" else ("temperature", "_k")
        pairs = (("state_mae", prefix + "_mae" + suffix), ("state_rmse", prefix + "_rmse" + suffix), ("state_r2", prefix + "_r2"), ("y_mae", "y_mae"), ("y_rmse", "y_rmse"), ("y_r2", "y_r2"), ("valid_coverage", "valid_coverage"), ("solver_failure_rate", "solver_failure_rate"), ("nonphysical_rate", "nonphysical_rate"))
        for local, field in pairs:
            for suffix_name in ("mean", "std"):
                if not math.isclose(float(by_direction[direction][local + "_" + suffix_name]), _finite(raw[field + "_" + suffix_name], field), rel_tol=1e-10, abs_tol=1e-10):
                    raise RuntimeError("Validation-selected stage metrics do not match the formal aggregate")


def collect_c1_ablation_rows(project_root: Path) -> list[dict[str, object]]:
    loaded = [_load(project_root, key, source) for key, source in C1_ABLATION_SOURCES.items()]
    records = [_standard_row(item, direction) for item in loaded for direction in FIELDS]
    for item in loaded:
        _match_selected(item, _stage_rows(item, "selected"))
    return records


def _fmt(row: Mapping[str, object], name: str, digits: int) -> str:
    return f"{float(row[name + '_mean']):.{digits}f} ± {float(row[name + '_std']):.{digits}f}"


def _triple(row: Mapping[str, object], name: str, digits: int) -> str:
    return " / ".join((_fmt(row, name + "_mae", digits), _fmt(row, name + "_rmse", digits), _fmt(row, name + "_r2", 3)))


def _table(records: Sequence[Mapping[str, object]], variants: Sequence[tuple[str, str]], key: str = "variant_id") -> list[str]:
    indexed = {(str(row[key]), str(row["direction"])): row for row in records}
    lines = ["| Variant | P iso: MAE / RMSE / R2 (kPa) | y iso: MAE / RMSE / R2 | T isob: MAE / RMSE / R2 (K) | y isob: MAE / RMSE / R2 | Coverage iso / isob | Solver failure iso / isob | Nonphysical iso / isob |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for identifier, label in variants:
        iso, isob = indexed[(identifier, "isothermal")], indexed[(identifier, "isobaric")]
        lines.append(f"| {label} | {_triple(iso, 'state', 3)} | {_triple(iso, 'y', 4)} | {_triple(isob, 'state', 3)} | {_triple(isob, 'y', 4)} | {_fmt(iso, 'valid_coverage', 4)} / {_fmt(isob, 'valid_coverage', 4)} | {_fmt(iso, 'solver_failure_rate', 4)} / {_fmt(isob, 'solver_failure_rate', 4)} | {_fmt(iso, 'nonphysical_rate', 4)} / {_fmt(isob, 'nonphysical_rate', 4)} |")
    return lines


def _csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def write_c1_ablation_outputs(project_root: Path, *, output_root: Path | None = None, report_path: Path | None = None) -> tuple[Path, Path, Path]:
    """Write isolated binary-only reports; reject smoke, joint, and partial inputs."""
    canonical = output_root is None and report_path is None
    loaded = {key: _load(project_root, key, source) for key, source in C1_ABLATION_SOURCES.items()}
    records = [_standard_row(item, direction) for item in loaded.values() for direction in FIELDS]
    for item in loaded.values():
        _match_selected(item, _stage_rows(item, "selected"))
    stages = [row for stage in REPORT_STAGES for row in _stage_rows(loaded["c1_final"], stage)]
    _match_selected(loaded["c1_final"], [row for row in stages if row["stage"] == "selected"])
    selected_stages_by_seed, selected_stage_count_rows, selected_stage_counts = (
        _selected_stage_provenance(loaded)
    )
    output_root = output_root or project_root / RESULT_ROOT / "reporting"
    report_path = report_path or project_root / "experiments/data_quality/reports/c1_ablation_overall_binary_three_stage.md"
    metrics_path = output_root / "overall_binary_three_stage_metrics.csv"
    stages_path = output_root / "overall_binary_three_stage_stage_metrics.csv"
    parameters_path = output_root / "overall_binary_three_stage_parameters.csv"
    selected_stages_by_seed_path = (
        output_root / "overall_binary_three_stage_selected_stages_by_seed.csv"
    )
    selected_stage_counts_path = (
        output_root / "overall_binary_three_stage_selected_stage_counts.csv"
    )
    parameters = []
    for key, item in loaded.items():
        manifests = list(item["manifests"].values())
        totals = {_integer(value.get("total_parameters"), key + ".total") for value in manifests}
        initial = {_integer(value.get("initially_trainable_parameters"), key + ".stage0") for value in manifests}
        summary = item["comparisons"][0].get("parameter_summary")
        if len(totals) != 1 or len(initial) != 1 or not isinstance(summary, Mapping) or not isinstance(summary.get("stage3"), Mapping) or any(value.get("parameter_summary") != summary for value in item["comparisons"].values()):
            raise RuntimeError(f"{key} parameter provenance is incomplete")
        parameters.append({"variant_id": key, "variant": item["source"].label, "total_parameters": totals.pop(), "stage0_trainable_parameters": initial.pop(), "stage1_trainable_parameters": _integer(summary.get("stage1_trainable_parameters"), key + ".stage1"), "stage2_trainable_parameters": _integer(summary.get("stage2_trainable_parameters"), key + ".stage2"), "stage3_trainable_parameters": _integer(summary["stage3"].get("trainable_parameters"), key + ".stage3")})
    _csv(metrics_path, records)
    _csv(stages_path, stages)
    _csv(parameters_path, parameters)
    _csv(selected_stages_by_seed_path, selected_stages_by_seed)
    _csv(selected_stage_counts_path, selected_stage_count_rows)
    text = ["# ThermoFormer ablation analysis: binary-only three-stage training", "", "Protocol: overall_binary (binary train -> binary test), seeds 0--4. Values are mean ± sample standard deviation. Checkpoint selection uses validation only; test is evaluated after selection.", "", "Historical overall_binary_ternary artifacts are excluded.", "", "## Molecular representation", "", *_table(records, REPRESENTATION), "", "## Interaction architecture", "", "The vanilla row reuses the exact C1 source from the representation block, not a second training run.", "", *_table(records, INTERACTION), "", "## C1 training-stage ablation", "", *_table(stages, tuple((stage, STAGE_LABELS[stage]) for stage in REPORT_STAGES), key="stage"), "", "## Parameter counts", "", "| Variant | Total | Stage 0 trainable | Stage 1 trainable | Stage 2 trainable | Stage 3 trainable |", "|---|---:|---:|---:|---:|---:|"]
    text.extend(f"| {row['variant']} | {row['total_parameters']:,} | {row['stage0_trainable_parameters']:,} | {row['stage1_trainable_parameters']:,} | {row['stage2_trainable_parameters']:,} | {row['stage3_trainable_parameters']:,} |" for row in parameters)
    text.extend(
        (
            "",
            "## Validation-selected checkpoint counts",
            "",
            "| Variant | Stage 0 | Stage 1 | Stage 2 | Stage 3 |",
            "|---|---:|---:|---:|---:|",
        )
    )
    text.extend(
        f"| {item['source'].label} | {selected_stage_counts[key]['stage0']} | {selected_stage_counts[key]['stage1']} | {selected_stage_counts[key]['stage2']} | {selected_stage_counts[key]['stage3']} |"
        for key, item in loaded.items()
    )
    atomic_write_text(report_path, "\n".join(text) + "\n")
    manifest_path = output_root / "report_manifest.json"
    manifest = {"status": "completed", "protocol": FORMAL_PROTOCOL, "protocol_description": "binary train -> binary test", "seeds": list(SEEDS), "stage_order": list(REPORT_STAGES), "selection_partition": "validation", "evaluation_partition": "test", "shared_source_mappings": {"full_three_view_representation": "c1_final", "vanilla_multicomponent_transformer": "c1_final"}, "inputs": {key: {"result_protocol": item["source"].result_protocol, "aggregate_manifest": {"path": str(item["aggregate"]), "sha256": artifact_sha256(item["aggregate"])}, "metrics_summary": {"path": str(item["summary_path"]), "sha256": artifact_sha256(item["summary_path"])}} for key, item in loaded.items()}, "outputs": {"metrics": {"path": str(metrics_path), "sha256": artifact_sha256(metrics_path)}, "stage_metrics": {"path": str(stages_path), "sha256": artifact_sha256(stages_path)}, "parameters": {"path": str(parameters_path), "sha256": artifact_sha256(parameters_path)}, "report": {"path": str(report_path), "sha256": artifact_sha256(report_path)}}}
    manifest["selected_stage_counts"] = selected_stage_counts
    manifest["outputs"]["selected_stages_by_seed"] = {
        "path": str(selected_stages_by_seed_path),
        "sha256": artifact_sha256(selected_stages_by_seed_path),
    }
    manifest["outputs"]["selected_stage_counts"] = {
        "path": str(selected_stage_counts_path),
        "sha256": artifact_sha256(selected_stage_counts_path),
    }
    for key, item in loaded.items():
        input_record = manifest["inputs"][key]
        input_record["seed_manifests"] = [
            {
                "seed": seed,
                "path": str(item["directory"] / f"seed_{seed}" / "manifest.json"),
                "sha256": artifact_sha256(item["directory"] / f"seed_{seed}" / "manifest.json"),
            }
            for seed in SEEDS
        ]
        input_record["stage_artifacts"] = {
            str(seed): {
                stage: item["manifests"][seed]["artifacts"][stage + "_checkpoint"]
                for stage in STAGES
            }
            for seed in SEEDS
        }
        input_record["stage_prediction_artifacts"] = {
            str(seed): {
                stage: item["manifests"][seed]["artifacts"][stage + "_predictions"]
                for stage in STAGES
            }
            for seed in SEEDS
        }
    atomic_write_json(manifest_path, manifest)
    if canonical:
        leaf = project_root / "configs/vle/ablation/studies/overall_binary_three_stage/results.md"
        leaf_text = (
            "# Binary-only three-stage ablation results\n\n"
            "Historical overall_binary_ternary ablation results remain separate historical records and are not used here.\n\n"
            f"Canonical report: {report_path.relative_to(project_root)}\n\n"
            f"Validation-selected stages by seed: {selected_stages_by_seed_path.relative_to(project_root)}\n"
            f"Validation-selected stage counts: {selected_stage_counts_path.relative_to(project_root)}\n"
        )
        atomic_write_text(leaf, leaf_text)
    return report_path, metrics_path, manifest_path
