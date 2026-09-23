"""Human-readable smoke report generated from machine-readable artifacts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .protocols import native_evaluation_cells


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_smoke_report(smoke_dir: Path, report_path: Path) -> None:
    manifest_path = smoke_dir / "manifest.json"
    summary_path = smoke_dir / "smoke_summary.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# Machine-learning VLE baseline smoke audit",
        "",
        "Status: diagnostic smoke completed; no formal test metrics have been produced.",
        "",
        "The smoke used seed 0 of the frozen `overall_binary_ternary` split, performed one "
        "finite forward/backward update on train-only rows, and used validation only as the "
        "diagnostic partition. It also validated the exact seed-0 IDs for all three requested "
        "overall benchmark settings. Registered test IDs were audited for isolation, but test "
        "labels were not consumed.",
        "",
        "| Baseline | Source | Native mixtures | Native tasks | Local parameters exercised | Author/reference parameters | Smoke status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        reference = (
            f"{int(row['reference_trainable_parameters']):,}"
            if row["reference_trainable_parameters"]
            else "N/A"
        )
        lines.append(
            f"| {row['display_name']} | {row['implementation_source']} | "
            f"{row['native_component_counts']} | {row['native_directions']} | "
            f"{int(row['trainable_parameters']):,} | {reference} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`passed` means that the audited architecture completed a finite train-only "
            "forward/backward update. `passed_architecture_only` additionally means that "
            "the native data/temperature boundary prevents a dataset-level smoke metric. "
            "Neither status is a predictive-performance result. "
            "`blocked_external_assets` means that the official checkpoint and its training-"
            "system inventory are required before an overlap-safe evaluation can run.",
            "",
            "SolvGNN, GDI-GNN and GE-GNN retain their native 298.15 K restriction; non-298 K "
            "and isobaric cells remain N/A. Descriptor ANN and UALF-GNN retain only their "
            "native binary isobaric task. No baseline is silently expanded to unsupported tasks.",
            "",
            "## Outstanding fairness and dependency gates",
            "",
            "- Descriptor ANN still needs the author-defined 21-property input table; its "
            "repository and the SMILES-RNN/SolvGNN repositories have no explicit license at "
            "the audited revisions.",
            "- SMILES-RNN uses a leakage-free task adapter because the distributed example "
            "contains an input-output leak and the author training table is absent.",
            "- SolvGNN/GDI-GNN/GE-GNN require VLE-derived gamma labels, shared Psat coverage and "
            "exact 298.15 K rows. The derivation and resulting coverage must be reported.",
            "- HANNA's 651,900 reference parameters describe the ten-head ensemble; TeNNet-SAC's "
            "2,743,206 describe its three learned heads. Frozen language encoders are excluded.",
            "- HANNA and TeNNet-SAC cannot enter a confirmatory table until official assets are "
            "checksummed and their training systems are intersected with every test split.",
            "",
            "## Reproduction boundary",
            "",
            f"- Dataset digest: `{manifest['dataset_sha256']}`",
            f"- Split digest: `{manifest['split_sha256']}`",
            "- Formal seeds 0--4: not run",
            "- State extrapolation, unseen-component and scaling protocols: not run",
            "",
        ]
    )
    _atomic_text(report_path, "\n".join(lines))
