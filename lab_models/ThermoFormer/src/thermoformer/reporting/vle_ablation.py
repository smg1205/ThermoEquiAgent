"""Aggregate the release-data joint VLE structural ablation benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import artifact_sha256, atomic_write_text


PROTOCOL = "vle_overall_binary_ternary"
SEEDS = tuple(range(5))
OUTPUT = Path("experiments/vle/ablation/joint")
TRAINED_ROOT = Path("experiments/vle/ablation/joint/formal/three_stage")
REUSED_ROOT = Path("experiments/vle/training_records/v1/three/results")


@dataclass(frozen=True)
class Source:
    identifier: str
    label: str
    category: str
    result_protocol: str
    reused: bool = False
    split_root: Path | None = None


SOURCES = (
    Source("c0_current_vanilla", "Uni-Mol v2 only", "Molecular representation", f"c0_current_vanilla.on.{PROTOCOL}"),
    Source("v1_rdkit_only", "RDKit descriptors only", "Molecular representation", f"v1_rdkit_only.on.{PROTOCOL}"),
    Source("v4_rdkit_unimol_naive", "RDKit descriptors + Uni-Mol v2", "Molecular representation", f"v4_rdkit_unimol_naive.on.{PROTOCOL}"),
    Source("c1_three_view_vanilla", "Full three-view representation", "Molecular representation", f"thermoformer_vle_three_stage.on.{PROTOCOL}", True, Path("experiments/vle/ablation/staged_validation/frozen_splits")),
    Source("c1_three_view_vanilla", "Vanilla multicomponent Transformer", "Interaction architecture", f"thermoformer_vle_three_stage.on.{PROTOCOL}", True, Path("experiments/vle/ablation/staged_validation/frozen_splits")),
    Source("c2_chemical_bias_full", "Chemical-interaction-biased Transformer with context-conditioned pair potential", "Interaction architecture", f"c2_chemical_bias_full.on.{PROTOCOL}"),
    Source("c3_no_pair_bias", "Context-conditioned pair potential without attention bias", "Interaction architecture", f"c3_no_pair_bias.on.{PROTOCOL}"),
)

METRICS = (
    ("pressure_mae_kpa", "P isothermal MAE"),
    ("pressure_rmse_kpa", "P isothermal RMSE"),
    ("pressure_r2", "P isothermal R2"),
    ("y_mae", "y MAE"),
    ("y_rmse", "y RMSE"),
    ("y_r2", "y R2"),
)


def _result_dir(root: Path, source: Source) -> Path:
    base = REUSED_ROOT if source.reused else TRAINED_ROOT
    return root / base / source.result_protocol


def _load_direction(root: Path, source: Source, seed: int, direction: str) -> dict[str, Any]:
    directory = _result_dir(root, source) / f"seed_{seed}"
    manifest_path = directory / "manifest.json"
    metrics_path = directory / "metrics.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("seed") != seed:
        raise RuntimeError(f"Incomplete result: {source.identifier}, seed {seed}")
    if manifest.get("protocol") != source.result_protocol:
        raise RuntimeError(f"Protocol mismatch: {manifest_path}")
    split_path = (root / source.split_root / f"seed_{seed}.json") if source.split_root else (root / 'datasets/splits/vle' / PROTOCOL / f"seed_{seed}.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != split.get("dataset_sha256"):
        raise RuntimeError(f"Dataset mismatch: {manifest_path}")
    if manifest.get("split_sha256") != artifact_sha256(split_path):
        raise RuntimeError(f"Split mismatch: {manifest_path}")
    record = manifest.get("artifacts", {}).get("metrics", {})
    if record.get("sha256") != artifact_sha256(metrics_path):
        raise RuntimeError(f"Metrics checksum mismatch: {metrics_path}")
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = [row for row in rows if row.get("scope") == "direction" and row.get("direction") == direction]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {direction} row: {metrics_path}")
    row = dict(selected[0])
    row.update(
        variant_id=source.identifier,
        variant=source.label,
        category=source.category,
        seed=seed,
        reused_from_main_benchmark=source.reused,
        manifest=str(manifest_path.resolve()),
    )
    return row


def collect(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in SOURCES:
        for seed in SEEDS:
            for direction in ("isothermal", "isobaric"):
                records.append(_load_direction(root, source, seed, direction))
    return records


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for source in SOURCES:
        row: dict[str, Any] = {
            "category": source.category,
            "variant_id": source.identifier,
            "variant": source.label,
            "reused_from_main_benchmark": source.reused,
            "seeds": "0,1,2,3,4",
        }
        for direction in ("isothermal", "isobaric"):
            selected = [r for r in records if r["variant"] == source.label and r["direction"] == direction]
            prefix = "pressure" if direction == "isothermal" else "temperature"
            names = (f"{prefix}_mae_kpa" if direction == "isothermal" else f"{prefix}_mae_k",
                     f"{prefix}_rmse_kpa" if direction == "isothermal" else f"{prefix}_rmse_k",
                     f"{prefix}_r2", "y_mae", "y_rmse", "y_r2", "valid_coverage")
            short = ("state_mae", "state_rmse", "state_r2", "y_mae", "y_rmse", "y_r2", "coverage")
            for name, output_name in zip(names, short):
                values = [float(r[name]) for r in selected]
                mean, std = _mean_std(values)
                row[f"{direction}_{output_name}_mean"] = mean
                row[f"{direction}_{output_name}_std"] = std
        summary.append(row)
    return summary


def _format(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _cell(row: dict[str, Any], direction: str, output: str, digits: int) -> str:
    return _format(row[f"{direction}_{output}_mean"], row[f"{direction}_{output}_std"], digits)


def _markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Joint binary--ternary VLE ablation results",
        "",
        "All variants use the release VLE dataset, the registered `vle_overall_binary_ternary` assignments, and seeds 0--4. Values are mean ± sample standard deviation. The C1 row references the completed main VLE benchmark and is shared between the two ablation categories.",
        "",
    ]
    for category in ("Molecular representation", "Interaction architecture"):
        lines += [f"## {category}", "", "| Variant | P MAE | P RMSE | P R² | y MAE (isothermal) | y RMSE | y R² | T MAE | T RMSE | T R² | y MAE (isobaric) | y RMSE | y R² | Coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary:
            if row["category"] != category:
                continue
            coverage = 100.0 * min(row["isothermal_coverage_mean"], row["isobaric_coverage_mean"])
            lines.append("| " + " | ".join((
                str(row["variant"]),
                _cell(row, "isothermal", "state_mae", 3), _cell(row, "isothermal", "state_rmse", 3), _cell(row, "isothermal", "state_r2", 3),
                _cell(row, "isothermal", "y_mae", 4), _cell(row, "isothermal", "y_rmse", 4), _cell(row, "isothermal", "y_r2", 3),
                _cell(row, "isobaric", "state_mae", 3), _cell(row, "isobaric", "state_rmse", 3), _cell(row, "isobaric", "state_r2", 3),
                _cell(row, "isobaric", "y_mae", 4), _cell(row, "isobaric", "y_rmse", 4), _cell(row, "isobaric", "y_r2", 3), f"{coverage:.1f}%",
            )) + " |")
        lines.append("")
    return "\n".join(lines)


def _latex(summary: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Category & Variant & $P$ (kPa), isothermal & $y$, isothermal & $T$ (K), isobaric & $y$, isobaric \\",
        r"\midrule",
    ]
    for row in summary:
        def triple(direction: str, state: bool) -> str:
            base = "state" if state else "y"
            digits = 3 if state else 4
            return " / ".join(_cell(row, direction, f"{base}_{metric}", digits) for metric in ("mae", "rmse", "r2"))
        lines.append(" & ".join((str(row["category"]), str(row["variant"]), triple("isothermal", True), triple("isothermal", False), triple("isobaric", True), triple("isobaric", False))) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(root: Path) -> dict[str, Any]:
    records = collect(root)
    summary = summarize(records)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    per_seed = output / "per_seed_metrics.csv"
    summary_path = output / "summary.csv"
    markdown = output / "results.md"
    latex = output / "results.tex"
    _write_csv(per_seed, records)
    _write_csv(summary_path, summary)
    atomic_write_text(markdown, _markdown(summary))
    atomic_write_text(latex, _latex(summary))
    manifest = {
        "status": "completed",
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "dataset_sha256": json.loads((root / 'datasets/splits/vle' / PROTOCOL / "seed_0.json").read_text(encoding="utf-8"))["dataset_sha256"],
        "reused_variant": "c1_three_view_vanilla",
        "trained_variants": [s.identifier for s in SOURCES if not s.reused],
        "outputs": {path.name: artifact_sha256(path) for path in (per_seed, summary_path, markdown, latex)},
    }
    manifest_path = output / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(markdown)
    return manifest


