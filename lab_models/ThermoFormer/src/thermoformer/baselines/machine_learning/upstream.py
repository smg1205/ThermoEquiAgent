"""Strict preflight and test-overlap auditing for official pretrained baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PretrainedOverlapAudit:
    training_systems: int
    benchmark_systems: int
    overlapping_systems: tuple[str, ...]

    @property
    def overlap_rate(self) -> float:
        return len(self.overlapping_systems) / self.benchmark_systems if self.benchmark_systems else 0.0


def audit_pretrained_overlap(
    training_inventory: Path,
    benchmark_system_ids: Sequence[str],
) -> PretrainedOverlapAudit:
    """Require a normalized system-ID inventory before official pretrained evaluation."""
    with training_inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "system_id" not in rows[0]:
        raise ValueError("Pretrained training inventory must contain a system_id column")
    training = {str(row["system_id"]).strip() for row in rows if str(row["system_id"]).strip()}
    if not training:
        raise ValueError("Pretrained training inventory contains no system IDs")
    benchmark = set(benchmark_system_ids)
    overlap = tuple(sorted(training & benchmark))
    return PretrainedOverlapAudit(len(training), len(benchmark), overlap)
