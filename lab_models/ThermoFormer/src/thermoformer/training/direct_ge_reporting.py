"""Five-seed aggregation and protocol-aware comparison for direct-GE training."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev

from ..reporting.artifacts import atomic_write_json, atomic_write_text


_DIRECTION_METRICS = {
    "isothermal": (
        "pressure_mae_kpa",
        "pressure_rmse_kpa",
        "pressure_r2",
        "y_mae",
        "y_rmse",
        "y_r2",
        "valid_coverage",
        "solver_failure_rate",
        "nonphysical_rate",
    ),
    "isobaric": (
        "temperature_mae_k",
        "temperature_rmse_k",
        "temperature_r2",
        "y_mae",
        "y_rmse",
        "y_r2",
        "valid_coverage",
        "solver_failure_rate",
        "nonphysical_rate",
    ),
}
_LABEL_METRICS = (
    "excess_gibbs_rt_mae",
    "log_gamma_mae",
    "ge_label_coverage",
    "gamma_label_coverage",
)
_SCOPES = {"joint": None, "binary": 2, "ternary": 3}
_WEIGHTS = {
    "excess_gibbs": 1.0,
    "activity_coefficient": 0.5,
    "vle": 1.0,
    "teacher_forced_fugacity": 0.01,
}


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Campaign metrics must be finite and non-empty")
    converted = [float(value) for value in values]
    return {
        "mean": fmean(converted),
        "sample_std": stdev(converted) if len(converted) > 1 else 0.0,
    }


def _optional_stats(values: Sequence[object]) -> dict[str, float | None]:
    available = [float(value) for value in values if value not in (None, "")]
    return _stats(available) if available else {"mean": None, "sample_std": None}


def _seed_from_path(path: Path) -> int:
    label = path.parent.name
    if not label.startswith("seed_"):
        raise ValueError(f"Cannot infer seed from {path}")
    return int(label.removeprefix("seed_"))


def _validate_seed_set(seeds: Sequence[int], expected_seeds: Sequence[int]) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError("Campaign inputs contain duplicate seeds")
    if tuple(sorted(seeds)) != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"Campaign requires exact seeds {tuple(expected_seeds)}, received {tuple(sorted(seeds))}"
        )


def _variant_stage(payload: Mapping[str, object], variant: str) -> str:
    if variant == "selected":
        return str(payload["selected_stage"])
    if variant == "stage3_candidate":
        return str(payload["trained_candidate_stage"])
    if variant == "stage0":
        return "stage0"
    raise KeyError(variant)


def _direction_row(
    payload: Mapping[str, object], variant: str, scope: str, direction: str
) -> Mapping[str, object] | None:
    stage = _variant_stage(payload, variant)
    component_count = _SCOPES[scope]
    wanted_scope = "direction" if component_count is None else "direction_cardinality"
    rows = payload["stages"][stage]["metrics"]
    return next(
        (
            row
            for row in rows
            if row.get("scope") == wanted_scope
            and row.get("direction") == direction
            and (
                component_count is None
                or int(row.get("component_count")) == component_count
            )
        ),
        None,
    )


def summarize_direct_ge_campaign(
    comparison_paths: Sequence[Path],
    *,
    expected_seeds: Sequence[int] = tuple(range(5)),
    expected_evaluation_partition: str = "test",
    manifest_paths: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Aggregate paired Stage-0/selected/Stage-3 evidence over registered seeds."""
    payloads: list[tuple[int, Path, dict[str, object]]] = []
    for path in comparison_paths:
        seed = _seed_from_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_partition") != "validation":
            raise RuntimeError(f"Seed {seed} did not use validation-only selection")
        if payload.get("evaluation_partition") != expected_evaluation_partition:
            raise RuntimeError(f"Seed {seed} has the wrong evaluation partition")
        weights = payload.get("thermodynamic_loss_weights")
        if not isinstance(weights, dict) or any(
            float(weights.get(name, float("nan"))) != value
            for name, value in _WEIGHTS.items()
        ):
            raise RuntimeError(f"Seed {seed} does not use the registered loss weights")
        selected = str(payload.get("selected_stage"))
        candidate = str(payload.get("trained_candidate_stage"))
        stages = payload.get("stages", {})
        if selected not in stages or candidate not in stages or "stage0" not in stages:
            raise RuntimeError(f"Seed {seed} lacks selected or candidate stage evidence")
        payloads.append((seed, path, payload))
    payloads.sort(key=lambda item: item[0])
    seeds = [seed for seed, _, _ in payloads]
    _validate_seed_set(seeds, expected_seeds)

    variants: dict[str, object] = {}
    seed_rows: list[dict[str, object]] = []
    for variant in ("stage0", "selected", "stage3_candidate"):
        scope_summaries: dict[str, object] = {}
        for scope in _SCOPES:
            directions: dict[str, object] = {}
            for direction, metric_names in _DIRECTION_METRICS.items():
                direction_rows = [
                    (seed, _direction_row(payload, variant, scope, direction))
                    for seed, _, payload in payloads
                ]
                metric_summaries: dict[str, object] = {}
                for metric in metric_names:
                    available = [
                        (seed, row.get(metric))
                        for seed, row in direction_rows
                        if row is not None and row.get(metric) not in (None, "")
                    ]
                    metric_summaries[metric] = {
                        **_optional_stats([value for _, value in available]),
                        "available_seeds": len(available),
                        "seed_ids": [seed for seed, _ in available],
                    }
                directions[direction] = metric_summaries
            scope_summaries[scope] = directions
            for seed, _, payload in payloads:
                row: dict[str, object] = {
                    "seed": seed,
                    "variant": variant,
                    "resolved_stage": _variant_stage(payload, variant),
                    "scope": scope,
                }
                for direction, metric_names in _DIRECTION_METRICS.items():
                    source = _direction_row(payload, variant, scope, direction)
                    row[f"{direction}_available"] = source is not None
                    for metric in metric_names:
                        row[f"{direction}_{metric}"] = (
                            source.get(metric) if source is not None else None
                        )
                seed_rows.append(row)
        label_metrics = {
            metric: _stats(
                [
                    float(
                        payload["stages"][_variant_stage(payload, variant)][
                            "evaluation_label_metrics"
                        ][metric]
                    )
                    for _, _, payload in payloads
                ]
            )
            for metric in _LABEL_METRICS
        }
        fugacity = _stats(
            [
                float(
                    payload["stages"][_variant_stage(payload, variant)][
                        "physics_residuals"
                    ]["teacher_forced_fugacity"]
                )
                for _, _, payload in payloads
            ]
        )
        variants[variant] = {
            **scope_summaries,
            "label_metrics": label_metrics,
            "teacher_forced_fugacity": fugacity,
        }

    selected_counts: dict[str, int] = {}
    for _, _, payload in payloads:
        stage = str(payload["selected_stage"])
        selected_counts[stage] = selected_counts.get(stage, 0) + 1
    parameter_summary = payloads[0][2]["parameter_summary"]
    if any(payload["parameter_summary"] != parameter_summary for _, _, payload in payloads[1:]):
        raise RuntimeError("Direct-GE parameter summaries differ across seeds")
    resources = None
    if manifest_paths is not None:
        manifests: list[tuple[int, dict[str, object]]] = []
        for path in manifest_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed = int(payload.get("seed", -1))
            if payload.get("status") != "completed":
                raise RuntimeError(f"Seed {seed} manifest is not completed")
            if payload.get("evaluation_partition") != expected_evaluation_partition:
                raise RuntimeError(f"Seed {seed} manifest has the wrong partition")
            manifests.append((seed, payload))
        manifests.sort(key=lambda item: item[0])
        _validate_seed_set([seed for seed, _ in manifests], expected_seeds)
        resources = {
            "training_seconds": _stats(
                [float(payload["training_seconds"]) for _, payload in manifests]
            ),
            "peak_gpu_memory_mb": _stats(
                [float(payload["peak_gpu_memory_mb"]) for _, payload in manifests]
            ),
            "total_parameters": int(manifests[0][1]["total_parameters"]),
            "stage3_trainable_parameters": int(
                manifests[0][1]["trainable_parameters"]
            ),
        }
    return {
        "seeds": seeds,
        "selected_stage_counts": selected_counts,
        "parameter_summary": parameter_summary,
        "resources": resources,
        "variants": variants,
        "seed_rows": seed_rows,
        "inputs": [
            {"seed": seed, "path": path.as_posix()}
            for seed, path, _ in payloads
        ],
    }


def load_joint_baseline_metrics(
    result_dir: Path,
    *,
    expected_provenance: Mapping[int, Mapping[str, object]],
    expected_seeds: Sequence[int] = tuple(range(5)),
    allow_registered_id_fallback: bool = False,
    allow_partial: bool = False,
) -> dict[str, object]:
    """Load joint 2+3 rows while enforcing the frozen test protocol."""
    seeds = sorted(int(seed) for seed in expected_provenance)
    _validate_seed_set(seeds, expected_seeds)
    rows_by_direction: dict[str, list[dict[str, str]]] = {
        "isothermal": [],
        "isobaric": [],
    }
    provenance_status = "hash_verified"
    for seed in seeds:
        seed_dir = result_dir / f"seed_{seed}"
        manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
        invariants = {
            "status": "completed",
            "seed": seed,
            "evaluation_partition": "test",
            "test_labels_used_for_selection": False,
        }
        for name, expected in invariants.items():
            if manifest.get(name) != expected:
                raise RuntimeError(f"Baseline seed {seed} has invalid {name}")
        if manifest.get("benchmark") not in {
            "joint_train_joint_test",
            "official_pretrained_to_joint_test",
            "external_fixed_to_joint_test",
        }:
            raise RuntimeError(f"Baseline seed {seed} is not a joint-test benchmark")
        expected = expected_provenance[seed]
        for name in ("dataset_sha256", "split_sha256"):
            observed = manifest.get(name)
            if observed is None and allow_registered_id_fallback:
                provenance_status = "registered_id_audited"
                continue
            if observed != expected.get(name):
                raise RuntimeError(f"Baseline seed {seed} has mismatched {name}")
        if provenance_status == "registered_id_audited":
            predictions = seed_dir / "predictions.csv"
            allowed_ids = set(expected.get("test_sample_ids", ()))
            if not predictions.is_file() or not allowed_ids:
                raise RuntimeError("Hash-less baseline requires registered test-ID audit")
            with predictions.open("r", encoding="utf-8", newline="") as handle:
                observed_ids = {row["sample_id"] for row in csv.DictReader(handle)}
            if not observed_ids or not observed_ids.issubset(allowed_ids):
                raise RuntimeError(f"Baseline seed {seed} predictions do not match test IDs")
        with (seed_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            metric_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("component_count") == "2+3"
            ]
        for direction in rows_by_direction:
            matches = [row for row in metric_rows if row.get("direction") == direction]
            if len(matches) != 1:
                raise RuntimeError(f"Baseline seed {seed} lacks evaluated {direction} 2+3 row")
            status = str(matches[0].get("status", ""))
            if status.startswith("evaluated"):
                rows_by_direction[direction].append(matches[0])
            elif not allow_partial:
                raise RuntimeError(
                    f"Baseline seed {seed} has non-evaluated {direction} 2+3 row"
                )
    directions = {
        direction: {
            metric: _optional_stats([row.get(metric) for row in rows])
            for metric in _DIRECTION_METRICS[direction]
        }
        for direction, rows in rows_by_direction.items()
    }
    return {
        "seeds": seeds,
        "provenance_status": provenance_status,
        "directions": directions,
        "evaluated_seed_counts": {
            direction: len(rows) for direction, rows in rows_by_direction.items()
        },
    }


def write_seed_metrics_csv(summary: Mapping[str, object], path: Path) -> Path:
    rows = list(summary["seed_rows"])
    if not rows:
        raise ValueError("No seed rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _formatted(record: Mapping[str, object]) -> str:
    if record["mean"] is None:
        return "N/A"
    return f"{float(record['mean']):.4f} +/- {float(record['sample_std']):.4f}"


def write_direct_ge_campaign_report(
    summary: Mapping[str, object],
    baseline_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Write the pre-registered five-seed result and scope-aware baseline comparison."""
    variants = summary["variants"]
    lines = [
        "# Direct-GE three-stage ThermoFormer: five-seed formal comparison",
        "",
        "Protocol: `overall_binary_ternary`; seeds 0--4; validation-only checkpoint selection; test used once after checkpoint locking.",
        "",
        "## Validation-selected and diagnostic results",
        "",
        "| Variant | Scope | P MAE / RMSE / R2 | y isothermal MAE / RMSE / R2 | T MAE / RMSE / R2 | y isobaric MAE / RMSE / R2 | Coverage iso / isob |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {
        "stage0": "Original C1",
        "selected": "Validation-selected final",
        "stage3_candidate": "Stage 3 diagnostic",
    }
    for variant in ("stage0", "selected", "stage3_candidate"):
        for scope in ("joint", "binary", "ternary"):
            iso = variants[variant][scope]["isothermal"]
            isob = variants[variant][scope]["isobaric"]
            lines.append(
                f"| {labels[variant]} | {scope} | "
                f"{_formatted(iso['pressure_mae_kpa'])} / {_formatted(iso['pressure_rmse_kpa'])} / {_formatted(iso['pressure_r2'])} | "
                f"{_formatted(iso['y_mae'])} / {_formatted(iso['y_rmse'])} / {_formatted(iso['y_r2'])} | "
                f"{_formatted(isob['temperature_mae_k'])} / {_formatted(isob['temperature_rmse_k'])} / {_formatted(isob['temperature_r2'])} | "
                f"{_formatted(isob['y_mae'])} / {_formatted(isob['y_rmse'])} / {_formatted(isob['y_r2'])} | "
                f"{_formatted(iso['valid_coverage'])} / {_formatted(isob['valid_coverage'])} |"
            )
    counts = ", ".join(
        f"{name}: {count}" for name, count in sorted(summary["selected_stage_counts"].items())
    )
    lines.extend(
        [
            "",
            f"Validation-selected stage counts: {counts}.",
            "The Stage 3 row is pre-registered diagnostic evidence and is never selected using test metrics.",
            "",
            "## Same-protocol baseline comparison",
            "",
            "Only joint 2+3 test rows are shown. External and partial-coverage references are labelled and do not enter a same-data overall-winner claim.",
            "",
            "| Model | Group | P MAE | y isothermal MAE | T MAE | y isobaric MAE | Coverage iso / isob | Provenance |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in baseline_rows:
        iso = row["summary"]["directions"]["isothermal"]
        isob = row["summary"]["directions"]["isobaric"]
        lines.append(
            f"| {row['label']} | {row['group']} | {_formatted(iso['pressure_mae_kpa'])} | "
            f"{_formatted(iso['y_mae'])} | {_formatted(isob['temperature_mae_k'])} | "
            f"{_formatted(isob['y_mae'])} | {_formatted(iso['valid_coverage'])} / "
            f"{_formatted(isob['valid_coverage'])} | {row['summary']['provenance_status']} |"
        )
    lines.extend(
        [
            "",
            "SolvGNN remains a 298.15 +/- 0.5 K partial-coverage reference and is excluded from full-test ranking. Binary-only baselines remain in their registered binary table and are not ranked against this joint protocol.",
            "",
        ]
    )
    resources = summary.get("resources")
    if resources is not None:
        lines.extend(
            [
                "## Resources",
                "",
                f"Training time per seed: {_formatted(resources['training_seconds'])} s; peak allocated GPU memory: {_formatted(resources['peak_gpu_memory_mb'])} MB.",
                f"Total parameters: {int(resources['total_parameters']):,}; Stage-3 trainable parameters: {int(resources['stage3_trainable_parameters']):,}.",
                "",
            ]
        )
    atomic_write_text(path, "\n".join(lines))
    return path


def write_campaign_summary(summary: Mapping[str, object], path: Path) -> Path:
    atomic_write_json(path, dict(summary))
    return path
