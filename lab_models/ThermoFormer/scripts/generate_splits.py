"""Generate every fixed, digest-checked split used by paper experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.data import load_vle_dataset, retain_pure_anchored_systems
from src.thermoformer.protocols.generalization import (
    binary_generalization_splits,
    binary_to_ternary_split,
    composition_edge_split,
    composition_interpolation_split,
    overall_system_split,
    protect_pure_reference_systems,
    state_extreme_split,
    unseen_component_split,
)
from src.thermoformer.protocols.registry import PAPER_SEEDS
from src.thermoformer.data.splitting import DatasetPartitions, load_split_assignment, save_split_assignment


SEEDS = PAPER_SEEDS
TERNARY_FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)


def _ternary_only_projection(split: DatasetPartitions) -> DatasetPartitions:
    """Preserve the registered joint system assignment while removing binary rows."""
    if split.protocol != "overall_binary_ternary":
        raise ValueError("Ternary-only projection requires the registered joint split")
    partitions = {
        name: tuple(row for row in rows if row.component_count == 3)
        for name, rows in {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }.items()
    }
    if any(not rows for rows in partitions.values()):
        raise ValueError("Ternary-only projection produced an empty partition")
    metadata = {
        "source_protocol": split.protocol,
        "source_seed": split.seed,
        "split_rule": (
            "component_count=3 projection of the registered joint split; "
            "system assignments are unchanged"
        ),
        "rows": {name: len(rows) for name, rows in partitions.items()},
        "component_count_rows": {
            name: {"3": len(rows)} for name, rows in partitions.items()
        },
    }
    return DatasetPartitions(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        protocol="overall_ternary",
        seed=split.seed,
        metadata=metadata,
    )


def _protocols(samples, seed: int) -> list[DatasetPartitions]:
    binary = overall_system_split(
        samples,
        seed,
        component_counts=(2,),
        minimum_anchor_temperatures=2,
        protocol="overall_binary",
    )
    joint = overall_system_split(
        samples,
        seed,
        component_counts=(2, 3),
        minimum_anchor_temperatures=2,
        protocol="overall_binary_ternary",
    )
    protocols = [
        binary,
        joint,
        _ternary_only_projection(joint),
        unseen_component_split(samples, seed, minimum_anchor_temperatures=2),
    ]
    state_protocols = [
        composition_interpolation_split(samples, seed),
        composition_edge_split(samples, seed),
        state_extreme_split(samples, "temperature", "low", seed),
        state_extreme_split(samples, "temperature", "high", seed),
        state_extreme_split(samples, "pressure", "low", seed),
        state_extreme_split(samples, "pressure", "high", seed),
    ]
    all_components = {smiles for row in samples for smiles in row.smiles}
    protocols.extend(
        protect_pure_reference_systems(
            samples,
            split,
            minimum_temperatures=2,
            allowed_components=all_components,
            required_components=all_components,
        )
        for split in state_protocols
    )
    protocols.extend(
        binary_to_ternary_split(
            samples,
            seed,
            fraction,
            minimum_anchor_temperatures=2,
        )
        for fraction in TERNARY_FRACTIONS
    )
    protocols.extend(
        binary_generalization_splits(
            samples,
            seed,
            minimum_anchor_temperatures=2,
        )
    )
    return protocols


def main() -> None:
    loaded = load_vle_dataset(
        PROJECT_ROOT / 'datasets/vle_reference',
        failed_weight=0.0,
        max_pressure_kpa=500.0,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=2,
    )
    summary: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        for split in _protocols(samples, seed):
            path = PROJECT_ROOT / 'datasets/splits/vle' / split.protocol / f"seed_{seed}.json"
            save_split_assignment(path, samples, split)
            restored = load_split_assignment(path, samples)
            if restored != split:
                raise RuntimeError(f"Split round-trip mismatch: {path}")
            summary[f"{split.protocol}/seed_{seed}"] = split.metadata
    index = PROJECT_ROOT / 'datasets/splits/vle/index.json'
    index.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_rows": len(samples),
                "seeds": list(SEEDS),
                "artifacts": len(summary),
                "index": str(index),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
