"""Human-readable experiment result reporting."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

from ..configuration import ExperimentConfig


def _format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        numeric = float(value)
        return f"{numeric:.8g}" if math.isfinite(numeric) else str(numeric)
    return str(value)


def write_experiment_results(
    path: Path,
    experiment: ExperimentConfig,
    manifest: dict[str, object],
    out_dir: Path,
) -> None:
    """Write a compact Markdown index for one completed experiment."""
    test_metrics = manifest.get("final_test_metrics", {})
    cv_summary = manifest.get("cross_validation_summary", {})
    run_status = str(manifest.get("run_status", "completed"))
    data_audit = manifest.get("data_loading_audit", {})
    lines = [
        f"# {experiment.name} results",
        "",
        f"- Status: {run_status}",
        f"- Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Evaluation: {manifest.get('evaluation_mode', 'unknown')}",
        f"- Retained samples: {manifest.get('n_samples', 'unknown')}",
        f"- Output directory: `{out_dir}`",
        f"- Final checkpoint: `{out_dir / 'final_model.pt'}`",
    ]
    if isinstance(data_audit, dict) and data_audit:
        lines.extend(
            [
                "",
                "## Data loading audit",
                "",
                "| Item | Count |",
                "|---|---:|",
                f"| Raw workbook rows | {_format_metric(data_audit.get('raw_rows', ''))} |",
                "| Accepted before deduplication | "
                f"{_format_metric(data_audit.get('accepted_before_deduplication', ''))} |",
                f"| Duplicates removed | {_format_metric(data_audit.get('duplicates_removed', ''))} |",
                f"| Loaded samples | {_format_metric(data_audit.get('loaded_samples', ''))} |",
                "| Pure-anchor filter removals | "
                f"{_format_metric(manifest.get('n_samples_removed_by_pure_anchor_filter', ''))} |",
            ]
        )
        rejected = data_audit.get("rejected", {})
        if isinstance(rejected, dict):
            lines.extend(
                f"| Rejected: {reason} | {_format_metric(count)} |"
                for reason, count in sorted(rejected.items())
            )
    lines.extend(
        [
            "",
            "## Final test metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    if isinstance(test_metrics, dict) and test_metrics:
        lines.extend(
            f"| {name} | {_format_metric(value)} |"
            for name, value in sorted(test_metrics.items())
        )
    else:
        lines.append("| _No test metrics recorded_ | — |")
    lines.extend(["", "## Cross-validation summary", ""])
    if isinstance(cv_summary, dict) and cv_summary:
        lines.extend(["| Metric | Mean | Std | Folds |", "|---|---:|---:|---:|"])
        for name, values in sorted(cv_summary.items()):
            if isinstance(values, dict):
                lines.append(
                    f"| {name} | {_format_metric(values.get('mean', ''))} | "
                    f"{_format_metric(values.get('std', ''))} | "
                    f"{_format_metric(values.get('folds', ''))} |"
                )
    else:
        lines.append("Not applicable or validation was skipped.")
    lines.extend(
        [
            "",
            "Full machine-readable records are stored in "
            f"`{out_dir / 'dataset_manifest.json'}` and `{out_dir / 'history.json'}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
