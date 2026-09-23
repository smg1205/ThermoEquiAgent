"""Seed-level metric and aggregate artifacts for formal baseline campaigns."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .protocols import UNIFIED_METRIC_KEYS


IDENTITY_FIELDS = ("baseline", "benchmark", "direction", "component_count", "status")
SEED_FIELDS = ("seed", *IDENTITY_FIELDS, *UNIFIED_METRIC_KEYS)


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_seed_metrics(path: Path, rows: Sequence[dict[str, object]], seed: int) -> None:
    if seed not in (0, 1, 2, 3, 4):
        raise ValueError("Formal seed metrics are restricted to seeds 0--4")
    normalized: list[dict[str, object]] = []
    identities: set[tuple[object, ...]] = set()
    for row in rows:
        payload = {field: row.get(field) for field in SEED_FIELDS}
        payload["seed"] = seed
        identity = tuple(payload[field] for field in IDENTITY_FIELDS[:-1])
        if identity in identities:
            raise ValueError(f"Duplicate seed metric identity: {identity}")
        identities.add(identity)
        normalized.append(payload)
    if not normalized:
        raise ValueError("At least one seed metric row is required")
    _atomic_csv(path, normalized, SEED_FIELDS)


def aggregate_seed_metrics(
    seed_files: Sequence[Path],
    output_path: Path,
    required_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> list[dict[str, object]]:
    comparison_fields = IDENTITY_FIELDS[:-1]
    by_identity: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    seen_seeds: set[int] = set()
    for path in seed_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty seed metric file: {path}")
        file_seeds = {int(row["seed"]) for row in rows}
        if len(file_seeds) != 1:
            raise ValueError(f"Seed metric file mixes seeds: {path}")
        seed = next(iter(file_seeds))
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed metric file for seed {seed}")
        seen_seeds.add(seed)
        for row in rows:
            identity = tuple(row[field] for field in comparison_fields)
            by_identity[identity].append(row)
    if seen_seeds != set(required_seeds):
        raise ValueError(f"Expected seeds {required_seeds}, found {tuple(sorted(seen_seeds))}")

    output: list[dict[str, object]] = []
    fields = [*IDENTITY_FIELDS, "seed_count", "evaluated_seed_count"]
    for metric in UNIFIED_METRIC_KEYS:
        fields.extend((f"{metric}_mean", f"{metric}_sample_std"))
    for identity, rows in sorted(by_identity.items()):
        result: dict[str, object] = dict(zip(comparison_fields, identity))
        statuses = {row["status"] for row in rows}
        evaluated = sum(row["status"] == "evaluated" for row in rows)
        if statuses == {"evaluated"}:
            result["status"] = "evaluated"
        elif "evaluated" in statuses:
            result["status"] = "evaluated_partial_seeds"
        elif len(statuses) == 1:
            result["status"] = next(iter(statuses))
        else:
            result["status"] = "not_evaluated_mixed_reasons"
        result["seed_count"] = len(rows)
        result["evaluated_seed_count"] = evaluated
        for metric in UNIFIED_METRIC_KEYS:
            values = []
            for row in rows:
                raw = row.get(metric, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            result[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            result[f"{metric}_sample_std"] = (
                float(np.std(values, ddof=1)) if len(values) >= 2 else ""
            )
        output.append(result)
    _atomic_csv(output_path, output, fields)
    return output
