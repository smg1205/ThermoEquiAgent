"""Validate every committed split artifact and report leakage/anchor coverage."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.data import load_vle_samples, pure_anchor_temperatures, retain_pure_anchored_systems
from src.auditing import ternary_subsystem_rows
from src.thermoformer.protocols.registry import PAPER_SEEDS, PROTOCOL_CONFIGS
from src.thermoformer.data.splitting import canonical_smiles, load_split_assignment, sample_id, system_id


def _composition_edge_score(row) -> float:
    order = sorted(
        range(row.component_count),
        key=lambda index: canonical_smiles(row.smiles[index]),
    )
    return float(np.min([row.liquid_composition[index] for index in order]))


def _strict_boundary_violations(split) -> int:
    definitions = {
        "state_temperature_low_extrapolation": (lambda row: row.temperature_k, "low"),
        "state_temperature_high_extrapolation": (lambda row: row.temperature_k, "high"),
        "state_pressure_low_extrapolation": (lambda row: row.pressure_kpa, "low"),
        "state_pressure_high_extrapolation": (lambda row: row.pressure_kpa, "high"),
        "state_composition_edge_extrapolation": (_composition_edge_score, "low"),
    }
    if split.protocol not in definitions:
        return 0
    score, tail = definitions[split.protocol]
    training: dict[str, list[float]] = defaultdict(list)
    testing: dict[str, list[float]] = defaultdict(list)
    for row in split.train:
        training[system_id(row)].append(score(row))
    for row in split.test:
        testing[system_id(row)].append(score(row))
    violations = 0
    for key, values in testing.items():
        reference = training.get(key, [])
        if not reference:
            violations += 1
        elif tail == "low" and not max(values) < min(reference):
            violations += 1
        elif tail == "high" and not min(values) > max(reference):
            violations += 1
    return violations


def main() -> None:
    samples = retain_pure_anchored_systems(
        load_vle_samples(PROJECT_ROOT / 'datasets/vle_reference', max_pressure_kpa=500.0),
        minimum_temperatures=2,
    )
    rows = []
    paths = sorted((PROJECT_ROOT / 'datasets/splits/vle').glob("*/seed_*.json"))
    found = {
        (path.parent.name, int(path.stem.removeprefix("seed_"))) for path in paths
    }
    expected = {
        (protocol, seed) for protocol in PROTOCOL_CONFIGS for seed in PAPER_SEEDS
    }
    missing_configs = [
        path for path in PROTOCOL_CONFIGS.values() if not (PROJECT_ROOT / path).is_file()
    ]
    if missing_configs:
        raise RuntimeError(f"Registered experiment configs are missing: {missing_configs}")
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"Split registry mismatch; missing={missing}, extra={extra}")
    for path in paths:
        split = load_split_assignment(path, samples)
        partitions = (split.train, split.validation, split.test)
        id_sets = [{sample_id(row) for row in values} for values in partitions]
        row_overlap = sum(
            len(id_sets[first] & id_sets[second])
            for first, second in ((0, 1), (0, 2), (1, 2))
        )
        train_components = {value for row in split.train for value in row.smiles}
        test_components = {value for row in split.test for value in row.smiles}
        anchors = pure_anchor_temperatures(split.train)
        unanchored = sum(len(anchors.get(value, set())) < 2 for value in train_components)
        train_systems = {system_id(row) for row in split.train}
        validation_systems = {system_id(row) for row in split.validation}
        test_systems = {system_id(row) for row in split.test}
        boundary_violations = _strict_boundary_violations(split)
        coverage_mismatch = False
        if split.protocol.startswith("binary_to_ternary"):
            actual = Counter(
                int(row["covered_binary_subsystems"])
                for row in ternary_subsystem_rows(
                    split.test, binary_reference_samples=split.train
                )
            )
            declared = {
                int(key): int(value)
                for key, value in split.metadata.get(
                    "test_binary_subsystem_coverage", {}
                ).items()
            }
            coverage_mismatch = dict(actual) != declared
        rows.append(
            {
                "protocol": split.protocol,
                "seed": split.seed,
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(split.test),
                "row_overlap": row_overlap,
                "train_validation_system_overlap": len(train_systems & validation_systems),
                "train_test_system_overlap": len(train_systems & test_systems),
                "validation_test_system_overlap": len(validation_systems & test_systems),
                "train_components": len(train_components),
                "test_components": len(test_components),
                "train_test_component_overlap": len(train_components & test_components),
                "training_components_without_two_anchors": unanchored,
                "strict_state_boundary_violations": boundary_violations,
                "binary_subsystem_coverage_mismatch": coverage_mismatch,
            }
        )
    if any(row["row_overlap"] for row in rows):
        raise RuntimeError("At least one split artifact has row leakage")
    if any(row["strict_state_boundary_violations"] for row in rows):
        raise RuntimeError("At least one state split violates its strict train/test boundary")
    if any(row["binary_subsystem_coverage_mismatch"] for row in rows):
        raise RuntimeError("At least one binary-to-ternary split has stale coverage labels")
    system_disjoint_prefixes = (
        "overall_",
        "unseen_component",
        "binary_to_ternary",
    )
    for row in rows:
        if row["protocol"].startswith(system_disjoint_prefixes) and (
            row["train_validation_system_overlap"]
            or row["train_test_system_overlap"]
            or row["validation_test_system_overlap"]
        ):
            raise RuntimeError(
                f"System leakage in {row['protocol']}/seed_{row['seed']}"
            )
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "split_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Split artifact audit",
        "",
        f"All {len(expected)} registered protocol/seed files were found and reloaded against the exact modeling-dataset digest. `system overlap` is expected for within-system state protocols and forbidden across all three partition pairs for system/generalization protocols; row overlap must always be zero.",
        "",
        "| Protocol/seed | Train/validation/test rows | Row overlap | Train-val / train-test / val-test system overlap | Train-test component overlap | Unanchored train components | Boundary violations | Coverage mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['protocol']}/seed_{row['seed']} | {row['train_rows']}/{row['validation_rows']}/{row['test_rows']} | "
        f"{row['row_overlap']} | {row['train_validation_system_overlap']} / {row['train_test_system_overlap']} / {row['validation_test_system_overlap']} | {row['train_test_component_overlap']} | "
        f"{row['training_components_without_two_anchors']} | {row['strict_state_boundary_violations']} | "
        f"{row['binary_subsystem_coverage_mismatch']} |"
        for row in rows
    )
    (reports / "split_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(rows), "row_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
