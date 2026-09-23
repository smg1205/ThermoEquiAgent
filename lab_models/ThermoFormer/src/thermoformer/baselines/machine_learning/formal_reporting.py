"""Paper-facing report generation for formal ML baseline aggregates."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from ...reporting.artifacts import artifact_sha256, atomic_write_json, atomic_write_text
from .protocols import benchmark_for_baseline
from .schema import BASELINE_CAPABILITIES


def _mean_and_sample_std(row: dict[str, str], metric: str) -> str:
    mean = row.get(f"{metric}_mean", "")
    deviation = row.get(f"{metric}_sample_std", "")
    if mean in ("", None):
        return "N/A"
    if deviation in ("", None):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(deviation):.4f}"


def write_formal_report(result_root: Path, report_path: Path) -> None:
    """Generate one unambiguous mean +/- sample-std row per task."""
    lines = [
        "# Machine-learning VLE baseline comparison",
        "",
        "Status: five-seed formal execution completed for executable models; unavailable models remain not evaluated.",
        "",
        "Trainable baselines used validation-only checkpoint selection. HANNA uses fixed official weights and performs no selection on these data; test labels were used only for evaluation.",
        "",
        "HANNA uses the unchanged official ten-model ensemble. It was trained on the authors' binary corpus and applies the official Muggianu projection for ternary inference; its training-system overlap with this dataset is unknown because the official inventory is not published.",
        "",
        "HANNA was executed in the project's ggnn39 compatibility environment, not the upstream version-pinned environment. Source, weights, scalers and architecture are unchanged, but exact upstream-environment numerical parity has not been established.",
        "",
        "| Baseline | Benchmark | Direction | Components | Status | Valid seeds | State MAE | State RMSE | State R2 | y MAE | y RMSE | y R2 | Coverage | Nonphysical | Solver failure |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline in BASELINE_CAPABILITIES:
        benchmark = benchmark_for_baseline(baseline)
        summary = result_root / f"{baseline}.on.{benchmark.key}" / "metrics_summary.csv"
        with summary.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            direction = row["direction"]
            prefix = "pressure" if direction == "isothermal" else "temperature"
            suffix = "kpa" if direction == "isothermal" else "k"
            metrics = (
                f"{prefix}_mae_{suffix}", f"{prefix}_rmse_{suffix}", f"{prefix}_r2",
                "y_mae", "y_rmse", "y_r2", "valid_coverage", "nonphysical_rate",
                "solver_failure_rate",
            )
            formatted = [_mean_and_sample_std(row, metric) for metric in metrics]
            lines.append(
                f"| {baseline} | {benchmark.key} | {direction} | {row['component_count']} | {row['status']} | "
                f"{row.get('evaluated_seed_count', 0)}/{row.get('seed_count', 5)} | "
                + " | ".join(formatted) + " |"
            )
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    project_root = Path(__file__).resolve().parents[4]
    aggregate_inputs = {}
    for baseline in BASELINE_CAPABILITIES:
        benchmark = benchmark_for_baseline(baseline)
        path = result_root / f"{baseline}.on.{benchmark.key}" / "aggregate_manifest.json"
        aggregate_inputs[baseline] = {
            "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
            "sha256": artifact_sha256(path),
        }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    atomic_write_json(
        result_root / "report_manifest.json",
        {
            "schema_version": 1,
            "status": "completed_with_blocked_models",
            "source_commit": source_commit,
            "aggregate_inputs": aggregate_inputs,
            "output": {
                "path": report_path.resolve().relative_to(project_root.resolve()).as_posix(),
                "sha256": artifact_sha256(report_path),
            },
        },
    )
