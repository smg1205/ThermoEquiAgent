"""Generate digest-checked splits for the registered NIST VLE dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.data import load_vle_dataset, retain_pure_anchored_systems
from src.thermoformer.data.splitting import (
    DatasetPartitions,
    load_split_assignment,
    sample_id,
    save_split_assignment,
)
from src.thermoformer.protocols.generalization import (
    composition_edge_split,
    composition_interpolation_split,
    overall_system_split,
    protect_pure_reference_systems,
    state_extreme_split,
    unseen_component_split,
)

PREFIX = "vle_"
SEEDS = tuple(range(5))
PROTOCOLS = (
    "overall_binary",
    "overall_binary_ternary",
    "overall_ternary",
    "binary_state_composition_interpolation",
    "binary_state_composition_edge_extrapolation",
    "binary_state_temperature_low_extrapolation",
    "binary_state_temperature_high_extrapolation",
    "binary_state_pressure_low_extrapolation",
    "binary_state_pressure_high_extrapolation",
    "binary_unseen_component",
)


def renamed(split: DatasetPartitions, name: str, **metadata: object) -> DatasetPartitions:
    return DatasetPartitions(
        train=split.train,
        validation=split.validation,
        test=split.test,
        protocol=PREFIX + name,
        seed=split.seed,
        metadata={**split.metadata, **metadata},
    )


def ternary_projection(split: DatasetPartitions) -> DatasetPartitions:
    parts = {
        name: tuple(row for row in values if row.component_count == 3)
        for name, values in {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }.items()
    }
    if any(not values for values in parts.values()):
        raise ValueError("Ternary projection produced an empty partition")
    return DatasetPartitions(
        train=parts["train"],
        validation=parts["validation"],
        test=parts["test"],
        protocol=PREFIX + "overall_ternary",
        seed=split.seed,
        metadata={
            "source_protocol": split.protocol,
            "split_rule": "component_count=3 projection of the joint system split",
            "rows": {name: len(values) for name, values in parts.items()},
        },
    )


def build_protocols(samples, seed: int) -> tuple[DatasetPartitions, ...]:
    binary_samples = tuple(row for row in samples if row.component_count == 2)
    binary = overall_system_split(
        binary_samples, seed, component_counts=(2,),
        minimum_anchor_temperatures=2, protocol=PREFIX + "overall_binary",
    )
    joint = overall_system_split(
        samples, seed, component_counts=(2, 3),
        minimum_anchor_temperatures=2, protocol=PREFIX + "overall_binary_ternary",
    )
    raw_state = (
        ("binary_state_composition_interpolation", composition_interpolation_split(binary_samples, seed)),
        ("binary_state_composition_edge_extrapolation", composition_edge_split(binary_samples, seed)),
        ("binary_state_temperature_low_extrapolation", state_extreme_split(binary_samples, "temperature", "low", seed)),
        ("binary_state_temperature_high_extrapolation", state_extreme_split(binary_samples, "temperature", "high", seed)),
        ("binary_state_pressure_low_extrapolation", state_extreme_split(binary_samples, "pressure", "low", seed)),
        ("binary_state_pressure_high_extrapolation", state_extreme_split(binary_samples, "pressure", "high", seed)),
    )
    binary_components = {smiles for row in binary_samples for smiles in row.smiles}
    states = tuple(
        renamed(
            protect_pure_reference_systems(
                binary_samples, split, minimum_temperatures=2,
                allowed_components=binary_components,
                required_components=binary_components,
            ),
            name,
            component_scope="binary",
        )
        for name, split in raw_state
    )
    unseen = renamed(
        unseen_component_split(
            binary_samples, seed, minimum_anchor_temperatures=2
        ),
        "binary_unseen_component",
        component_scope="binary",
    )
    return (binary, joint, ternary_projection(joint), *states, unseen)


def audit_split(split: DatasetPartitions) -> dict[str, object]:
    ids = {
        name: {sample_id(row) for row in values}
        for name, values in {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }.items()
    }
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Row leakage in {split.protocol}: {overlaps}")
    return {
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "row_overlap": overlaps,
        "metadata": split.metadata,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/vle")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / 'datasets/splits/vle')
    args = parser.parse_args(argv)
    loaded = load_vle_dataset(args.dataset, failed_weight=0.0, max_pressure_kpa=500.0)
    samples = retain_pure_anchored_systems(loaded.samples, minimum_temperatures=2)
    dataset_audit = dataclasses.asdict(loaded.audit) if dataclasses.is_dataclass(loaded.audit) else loaded.audit
    summary = {"dataset_audit": dataset_audit, "modeling_rows": len(samples), "splits": {}}
    for seed in SEEDS:
        for split in build_protocols(samples, seed):
            path = args.output / split.protocol / f"seed_{seed}.json"
            save_split_assignment(path, samples, split)
            if load_split_assignment(path, samples) != split:
                raise RuntimeError(f"Split round-trip mismatch: {path}")
            summary["splits"][f"{split.protocol}/seed_{seed}"] = audit_split(split)
    index = args.output / "vle_index.json"
    index.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"modeling_rows": len(samples), "artifacts": len(summary["splits"]), "index": str(index)}, indent=2))


if __name__ == "__main__":
    main()
