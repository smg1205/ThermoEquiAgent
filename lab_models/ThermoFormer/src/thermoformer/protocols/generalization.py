"""Deterministic split protocols for ThermoFormer generalization studies."""

from __future__ import annotations

import math
import random
import itertools
from collections import Counter, defaultdict
from typing import Callable, Literal, Sequence

import numpy as np

from ..data.auditing import ternary_subsystem_rows
from ..data import VLESample, build_split_plan, retain_pure_anchored_systems
from ..data.splitting import DatasetPartitions, canonical_smiles, component_id, sample_id, system_id


def _ordered(rows: Sequence[VLESample]) -> tuple[VLESample, ...]:
    return tuple(sorted(rows, key=sample_id))


def _metadata(
    train: Sequence[VLESample],
    validation: Sequence[VLESample],
    test: Sequence[VLESample],
    **extra: object,
) -> dict[str, object]:
    partitions = {"train": train, "validation": validation, "test": test}
    return {
        "rows": {name: len(values) for name, values in partitions.items()},
        "systems": {
            name: len({system_id(value) for value in values})
            for name, values in partitions.items()
        },
        "component_count_rows": {
            name: {
                str(key): count
                for key, count in sorted(
                    Counter(value.component_count for value in values).items()
                )
            }
            for name, values in partitions.items()
        },
        **extra,
    }


def _select_pure_reference_rows(
    dataset: Sequence[VLESample],
    minimum_temperatures: int,
    allowed_components: set[str],
    required_components: set[str],
) -> dict[str, VLESample]:
    candidates: dict[str, dict[float, VLESample]] = defaultdict(dict)
    for sample in dataset:
        if not set(sample.smiles).issubset(allowed_components):
            continue
        for smiles, fraction in zip(sample.smiles, sample.liquid_composition):
            if smiles not in required_components or fraction < 0.999:
                continue
            temperature = round(sample.temperature_k, 6)
            previous = candidates[smiles].get(temperature)
            if previous is None or (
                sample.quality_weight,
                sample_id(sample),
            ) > (
                previous.quality_weight,
                sample_id(previous),
            ):
                candidates[smiles][temperature] = sample
    reference_rows: dict[str, VLESample] = {}
    for smiles in sorted(required_components):
        temperatures = sorted(candidates.get(smiles, {}))
        if len(temperatures) < minimum_temperatures:
            raise ValueError(
                f"Insufficient leakage-safe pure references for component {smiles}"
            )
        if minimum_temperatures == 1:
            selected_temperatures = [temperatures[len(temperatures) // 2]]
        else:
            positions = np.linspace(
                0, len(temperatures) - 1, minimum_temperatures
            ).round().astype(int)
            selected_temperatures = [temperatures[index] for index in positions]
        for temperature in selected_temperatures:
            row = candidates[smiles][temperature]
            reference_rows[sample_id(row)] = row
    return reference_rows


def protect_pure_reference_rows(
    dataset: Sequence[VLESample],
    split: DatasetPartitions,
    minimum_temperatures: int = 2,
    allowed_components: set[str] | None = None,
    required_components: set[str] | None = None,
) -> DatasetPartitions:
    """Move a compact pure-endpoint reference set to the training partition.

    Reference rows calibrate the separate Psat branch and are excluded from
    validation/test metrics.  ``allowed_components`` prevents a reference row
    from exposing a chemically held-out component through a zero-composition
    slot.
    """
    if minimum_temperatures < 0:
        raise ValueError("minimum_temperatures cannot be negative")
    if minimum_temperatures == 0:
        return split
    required = required_components if required_components is not None else {
        smiles
        for partition in (split.train, split.validation, split.test)
        for sample in partition
        for smiles in sample.smiles
    }
    allowed = allowed_components if allowed_components is not None else required
    reference_rows = _select_pure_reference_rows(
        dataset, minimum_temperatures, allowed, required
    )
    reference_ids = set(reference_rows)

    def without_references(rows: Sequence[VLESample]) -> list[VLESample]:
        return [row for row in rows if sample_id(row) not in reference_ids]

    train = without_references(split.train) + list(reference_rows.values())
    validation = without_references(split.validation)
    test = without_references(split.test)
    if not validation or not test:
        raise ValueError("Pure references exhausted a validation or test partition")
    metadata = {
        **split.metadata,
        "pure_reference_rows": len(reference_rows),
        "pure_reference_systems": len(
            {system_id(row) for row in reference_rows.values()}
        ),
        "pure_reference_rule": (
            f"{minimum_temperatures} range-spanning endpoint temperatures per component; "
            "reference rows excluded from validation/test metrics"
        ),
    }
    metadata.update(_metadata(train, validation, test))
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol=split.protocol,
        seed=split.seed,
        metadata=metadata,
    )


def protect_pure_reference_systems(
    dataset: Sequence[VLESample],
    split: DatasetPartitions,
    minimum_temperatures: int = 2,
    allowed_components: set[str] | None = None,
    required_components: set[str] | None = None,
) -> DatasetPartitions:
    """Reserve complete endpoint-bearing systems for training calibration.

    Complete-system reservation cannot place one mixture in both training and
    validation, and cannot inject a held-out extreme state into the training
    side of a within-system extrapolation.
    """
    if minimum_temperatures < 0:
        raise ValueError("minimum_temperatures cannot be negative")
    if minimum_temperatures == 0:
        return split
    required = required_components if required_components is not None else {
        smiles
        for partition in (split.train, split.validation, split.test)
        for sample in partition
        for smiles in sample.smiles
    }
    allowed = allowed_components if allowed_components is not None else required
    endpoints = _select_pure_reference_rows(
        dataset, minimum_temperatures, allowed, required
    )
    reference_system_ids = {system_id(row) for row in endpoints.values()}
    reference_rows = [
        row
        for row in dataset
        if system_id(row) in reference_system_ids
        and set(row.smiles).issubset(allowed)
    ]

    def without_reference_systems(rows: Sequence[VLESample]) -> list[VLESample]:
        return [row for row in rows if system_id(row) not in reference_system_ids]

    train = without_reference_systems(split.train) + reference_rows
    validation = without_reference_systems(split.validation)
    test = without_reference_systems(split.test)
    if not validation or not test:
        raise ValueError("Pure reference systems exhausted a validation or test partition")
    metadata = {
        **split.metadata,
        "pure_reference_rows": len(reference_rows),
        "pure_reference_endpoint_rows": len(endpoints),
        "pure_reference_systems": len(reference_system_ids),
        "pure_reference_system_ids": sorted(reference_system_ids),
        "pure_reference_rule": (
            f"complete systems containing {minimum_temperatures} range-spanning "
            "endpoint temperatures per required component; reference systems "
            "excluded from validation/test"
        ),
    }
    metadata.update(_metadata(train, validation, test))
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol=split.protocol,
        seed=split.seed,
        metadata=metadata,
    )


def overall_system_split(
    samples: Sequence[VLESample],
    seed: int,
    component_counts: tuple[int, ...] = (2, 3),
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    minimum_anchor_temperatures: int = 2,
    protocol: str = "overall_binary_ternary",
) -> DatasetPartitions:
    selected = [sample for sample in samples if sample.component_count in component_counts]
    plan = build_split_plan(
        selected,
        mode="holdout",
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        folds=5,
        seed=seed,
        minimum_anchor_temperatures=minimum_anchor_temperatures,
    )
    split = plan.folds[0]
    metadata = _metadata(
        split.train,
        split.validation,
        plan.test,
        split_rule="unordered canonical system disjoint, row-balanced by cardinality",
        requested_fraction={
            "train": 1.0 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        anchor_reference_systems=len(plan.anchor_reference_systems),
    )
    return DatasetPartitions(
        train=_ordered(split.train),
        validation=_ordered(split.validation),
        test=_ordered(plan.test),
        protocol=protocol,
        seed=seed,
        metadata=metadata,
    )


def _canonical_x(sample: VLESample) -> np.ndarray:
    order = sorted(
        range(sample.component_count),
        key=lambda index: canonical_smiles(sample.smiles[index]),
    )
    return np.asarray([sample.liquid_composition[index] for index in order], dtype=float)


def composition_interpolation_split(
    samples: Sequence[VLESample],
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetPartitions:
    """Hold out interior composition states while keeping bracketing states per system."""
    grouped: dict[str, list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[system_id(sample)].append(sample)
    train: list[VLESample] = []
    validation: list[VLESample] = []
    test: list[VLESample] = []
    skipped = 0
    rng = random.Random(seed)
    for key in sorted(grouped):
        rows = grouped[key]
        if len(rows) < 5:
            skipped += len(rows)
            continue
        matrix = np.stack([_canonical_x(row) for row in rows])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        axis = vectors[0]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis = -axis
        scores = centered @ axis
        ordered = [
            row
            for _, _, row in sorted(
                zip(scores.tolist(), [sample_id(row) for row in rows], rows),
                key=lambda item: (item[0], item[1]),
            )
        ]
        interior = list(range(1, len(ordered) - 1))
        n_test = min(len(interior), max(1, round(test_fraction * len(ordered))))
        evenly_spaced = np.linspace(0, len(interior) - 1, n_test + 2)[1:-1]
        test_indices = {interior[int(round(value))] for value in evenly_spaced}
        validation_candidates = [index for index in interior if index not in test_indices]
        rng.shuffle(validation_candidates)
        n_validation = min(
            len(validation_candidates),
            max(1, round(validation_fraction * len(ordered))),
        )
        validation_indices = set(validation_candidates[:n_validation])
        for index, row in enumerate(ordered):
            if index in test_indices:
                test.append(row)
            elif index in validation_indices:
                validation.append(row)
            else:
                train.append(row)
    if not train or not validation or not test:
        raise ValueError("Composition interpolation protocol has an empty partition")
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol="state_composition_interpolation",
        seed=seed,
        metadata=_metadata(
            train,
            validation,
            test,
            split_rule="per-system PCA composition coordinate; interior test states bracketed by train states",
            skipped_rows=skipped,
        ),
    )


def _tail_partition(
    rows: Sequence[VLESample],
    score: Callable[[VLESample], float],
    tail: Literal["low", "high"],
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[VLESample], list[VLESample], list[VLESample]] | None:
    by_value: dict[float, list[VLESample]] = defaultdict(list)
    for row in rows:
        by_value[round(score(row), 10)].append(row)
    values = sorted(by_value, reverse=tail == "high")
    if len(values) < 3:
        return None
    target_test = max(1, round(test_fraction * len(rows)))
    target_validation = max(1, round(validation_fraction * len(rows)))
    test_values: list[float] = []
    validation_values: list[float] = []
    count = 0
    while values and count < target_test and len(values) > 2:
        value = values.pop(0)
        test_values.append(value)
        count += len(by_value[value])
    count = 0
    while values and count < target_validation and len(values) > 1:
        value = values.pop(0)
        validation_values.append(value)
        count += len(by_value[value])
    if not test_values or not validation_values or not values:
        return None
    return (
        [row for value in values for row in by_value[value]],
        [row for value in validation_values for row in by_value[value]],
        [row for value in test_values for row in by_value[value]],
    )


def state_extreme_split(
    samples: Sequence[VLESample],
    variable: Literal["temperature", "pressure"],
    tail: Literal["low", "high"],
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetPartitions:
    """Evaluate a strict low/high state tail within every eligible system."""
    score = (
        (lambda sample: sample.temperature_k)
        if variable == "temperature"
        else (lambda sample: sample.pressure_kpa)
    )
    grouped: dict[str, list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[system_id(sample)].append(sample)
    train: list[VLESample] = []
    validation: list[VLESample] = []
    test: list[VLESample] = []
    skipped = 0
    for key in sorted(grouped):
        partition = _tail_partition(
            grouped[key], score, tail, validation_fraction, test_fraction
        )
        if partition is None:
            skipped += len(grouped[key])
            continue
        group_train, group_validation, group_test = partition
        train.extend(group_train)
        validation.extend(group_validation)
        test.extend(group_test)
    if not train or not validation or not test:
        raise ValueError(f"No systems support {variable}-{tail} extrapolation")
    protocol = f"state_{variable}_{tail}_extrapolation"
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol=protocol,
        seed=seed,
        metadata=_metadata(
            train,
            validation,
            test,
            split_rule=f"strict per-system {tail} tail of unique {variable} values",
            skipped_rows=skipped,
        ),
    )


def composition_edge_split(
    samples: Sequence[VLESample],
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetPartitions:
    """Hold out compositions closest to a simplex edge within each system."""
    grouped: dict[str, list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[system_id(sample)].append(sample)
    train: list[VLESample] = []
    validation: list[VLESample] = []
    test: list[VLESample] = []
    skipped = 0
    for key in sorted(grouped):
        partition = _tail_partition(
            grouped[key],
            lambda sample: float(np.min(_canonical_x(sample))),
            "low",
            validation_fraction,
            test_fraction,
        )
        if partition is None:
            skipped += len(grouped[key])
            continue
        group_train, group_validation, group_test = partition
        train.extend(group_train)
        validation.extend(group_validation)
        test.extend(group_test)
    if not train or not validation or not test:
        raise ValueError("No systems support composition-edge extrapolation")
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol="state_composition_edge_extrapolation",
        seed=seed,
        metadata=_metadata(
            train,
            validation,
            test,
            split_rule="strict per-system low tail of min(x_i), closest to simplex edges",
            skipped_rows=skipped,
        ),
    )


def _system_validation_split(
    rows: Sequence[VLESample], seed: int, fraction: float = 0.15
) -> tuple[list[VLESample], list[VLESample]]:
    grouped: dict[str, list[VLESample]] = defaultdict(list)
    for row in rows:
        grouped[system_id(row)].append(row)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    target = max(1, round(fraction * len(rows)))
    validation_keys: list[str] = []
    count = 0
    while len(keys) > 1 and count < target:
        key = keys.pop(0)
        validation_keys.append(key)
        count += len(grouped[key])
    validation_set = set(validation_keys)
    train = [row for key, values in grouped.items() if key not in validation_set for row in values]
    validation = [row for key in validation_keys for row in grouped[key]]
    return train, validation


def unseen_component_split(
    samples: Sequence[VLESample],
    seed: int,
    target_component_fraction: float = 0.20,
    minimum_anchor_temperatures: int = 0,
) -> DatasetPartitions:
    """Hold out components, then test every system containing at least one of them."""
    if not 0.0 < target_component_fraction < 1.0:
        raise ValueError("target_component_fraction must be between zero and one")
    grouped: dict[str, list[VLESample]] = defaultdict(list)
    components_by_system: dict[str, frozenset[str]] = {}
    component_system_count: Counter[str] = Counter()
    for sample in samples:
        key = system_id(sample)
        grouped[key].append(sample)
        components = frozenset(canonical_smiles(value) for value in sample.smiles)
        components_by_system[key] = components
    for components in components_by_system.values():
        component_system_count.update(components)
    all_components = sorted(component_system_count)
    target_count = max(1, math.ceil(target_component_fraction * len(all_components)))
    rng = random.Random(seed)
    component_tie = {value: rng.random() for value in all_components}
    candidate_sets: list[tuple[int, int, float, str, set[str]]] = []
    strict_seed_keys = [
        key for key in sorted(grouped) if len(components_by_system[key]) == 3
    ] or sorted(grouped)
    for seed_key in strict_seed_keys:
        candidate = set(components_by_system[seed_key])
        remaining = sorted(
            (value for value in all_components if value not in candidate),
            key=lambda value: (component_system_count[value], component_tie[value], value),
        )
        candidate.update(remaining[: max(0, target_count - len(candidate))])
        test_key_count = sum(
            bool(components & candidate) for components in components_by_system.values()
        )
        training_key_count = len(grouped) - test_key_count
        # Prefer a viable training side, then the smallest held-out chemical
        # footprint and the least destructive system cut.  Starting from one
        # complete system guarantees that the strict-unseen subset is non-empty.
        candidate_sets.append(
            (
                0 if training_key_count >= 2 else 1,
                test_key_count,
                rng.random(),
                seed_key,
                candidate,
            )
        )
    viable, _, _, _, held_out = min(candidate_sets, key=lambda item: item[:4])
    if viable:
        raise ValueError("Cannot hold out components while retaining two training systems")
    test_keys = {
        key for key, components in components_by_system.items() if components & held_out
    }
    strict_keys = {
        key for key in test_keys if components_by_system[key].issubset(held_out)
    }
    train_candidates = [
        row for key, values in grouped.items() if key not in test_keys for row in values
    ]
    if minimum_anchor_temperatures:
        train_candidates = retain_pure_anchored_systems(
            train_candidates,
            minimum_temperatures=minimum_anchor_temperatures,
        )
    train, validation = _system_validation_split(train_candidates, seed + 1)
    test = [row for key in sorted(test_keys) for row in grouped[key]]
    if not train or not validation or not test:
        raise ValueError("Unseen-component selection produced an empty partition")
    strict_ids = [sample_id(row) for key in sorted(strict_keys) for row in grouped[key]]
    initial = DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol="unseen_component",
        seed=seed,
        metadata=_metadata(
            train,
            validation,
            test,
            split_rule="test systems contain at least one component absent from train",
            held_out_components=sorted(held_out),
            held_out_component_ids=[component_id(value) for value in sorted(held_out)],
            strict_unseen_test_rows=len(strict_ids),
            strict_unseen_sample_ids=strict_ids,
        ),
    )
    training_components = {smiles for row in train_candidates for smiles in row.smiles}
    return (
        protect_pure_reference_systems(
            train_candidates,
            initial,
            minimum_temperatures=minimum_anchor_temperatures,
            allowed_components=training_components,
            required_components=training_components,
        )
        if minimum_anchor_temperatures
        else initial
    )


def _rename_binary_protocol(
    split: DatasetPartitions,
    protocol: str,
) -> DatasetPartitions:
    return DatasetPartitions(
        train=split.train,
        validation=split.validation,
        test=split.test,
        protocol=protocol,
        seed=split.seed,
        metadata={
            **split.metadata,
            "source_protocol": split.protocol,
            "component_counts": [2],
            "data_scope": "binary train, binary validation, binary test",
        },
    )


def binary_generalization_splits(
    samples: Sequence[VLESample],
    seed: int,
    minimum_anchor_temperatures: int = 2,
) -> tuple[DatasetPartitions, ...]:
    """Build binary-only state-space and unseen-component protocols."""
    binary = tuple(sample for sample in samples if sample.component_count == 2)
    if not binary:
        raise ValueError("Binary generalization protocols require binary samples")
    state_splits = (
        composition_interpolation_split(binary, seed),
        composition_edge_split(binary, seed),
        state_extreme_split(binary, "temperature", "low", seed),
        state_extreme_split(binary, "temperature", "high", seed),
        state_extreme_split(binary, "pressure", "low", seed),
        state_extreme_split(binary, "pressure", "high", seed),
    )
    components = {smiles for sample in binary for smiles in sample.smiles}
    protected = (
        protect_pure_reference_systems(
            binary,
            split,
            minimum_temperatures=minimum_anchor_temperatures,
            allowed_components=components,
            required_components=components,
        )
        for split in state_splits
    )
    unseen = unseen_component_split(
        binary,
        seed,
        minimum_anchor_temperatures=minimum_anchor_temperatures,
    )
    return (
        *(
            _rename_binary_protocol(split, f"binary_{split.protocol}")
            for split in protected
        ),
        _rename_binary_protocol(unseen, "binary_unseen_component"),
    )


def binary_to_ternary_split(
    samples: Sequence[VLESample],
    seed: int,
    ternary_training_fraction: float,
    minimum_anchor_temperatures: int = 0,
) -> DatasetPartitions:
    """Create zero-shot or fixed-test ternary scaling experiments."""
    if not 0.0 <= ternary_training_fraction <= 1.0:
        raise ValueError("ternary_training_fraction must be in [0, 1]")
    binary = [sample for sample in samples if sample.component_count == 2]
    ternary = [sample for sample in samples if sample.component_count == 3]
    binary_plan = overall_system_split(
        binary,
        seed=seed + 11,
        component_counts=(2,),
        test_fraction=0.15,
        validation_fraction=0.15,
        minimum_anchor_temperatures=minimum_anchor_temperatures,
        protocol="binary_reference_split",
    )
    binary_train = list(binary_plan.train) + list(binary_plan.test)
    binary_validation = list(binary_plan.validation)
    ternary_grouped: dict[str, list[VLESample]] = defaultdict(list)
    for sample in ternary:
        ternary_grouped[system_id(sample)].append(sample)
    ternary_keys = sorted(ternary_grouped)
    coverage_by_system = {
        str(row["ternary_system_id"]): int(row["covered_binary_subsystems"])
        for row in ternary_subsystem_rows(
            ternary, binary_reference_samples=binary_train
        )
    }
    rng = random.Random(seed)

    def take_stratified(keys: list[str], count: int) -> tuple[list[str], list[str]]:
        strata: dict[int, list[str]] = defaultdict(list)
        for key in keys:
            strata[coverage_by_system.get(key, -1)].append(key)
        for values in strata.values():
            rng.shuffle(values)
        selected: list[str] = []
        nonempty_coverages = [coverage for coverage in sorted(strata) if strata[coverage]]
        if count >= len(nonempty_coverages) and len(nonempty_coverages) > 1:
            combinations = itertools.product(
                *(strata[coverage] for coverage in nonempty_coverages)
            )
            target_rows = 0.15 * sum(len(ternary_grouped[key]) for key in keys)
            ranked = []
            for combination in combinations:
                modes = {
                    row.experiment_mode
                    for key in combination
                    for row in ternary_grouped[key]
                }
                direction_penalty = int("isothermal" not in modes) + int(
                    "isobaric" not in modes
                )
                row_count = sum(len(ternary_grouped[key]) for key in combination)
                ranked.append(
                    (
                        direction_penalty,
                        abs(row_count - target_rows),
                        rng.random(),
                        combination,
                    )
                )
            _, _, _, best = min(ranked, key=lambda item: item[:3])
            selected.extend(best)
            for coverage, key in zip(nonempty_coverages, best):
                strata[coverage].remove(key)
        while len(selected) < count and any(strata.values()):
            for coverage in sorted(strata):
                if strata[coverage] and len(selected) < count:
                    selected.append(strata[coverage].pop())
        remaining = [key for values in strata.values() for key in values]
        rng.shuffle(remaining)
        return selected, remaining

    n_test = max(1, round(0.15 * len(ternary_keys)))
    test_ternary_keys, remaining = take_stratified(ternary_keys, n_test)
    if ternary_training_fraction == 0.0:
        train = binary_train
        validation = binary_validation
        test = [row for key in test_ternary_keys for row in ternary_grouped[key]]
        training_ternary_keys: list[str] = []
        validation_ternary_keys: list[str] = []
        selection = (
            "binary validation only; zero-shot test uses the same fixed, "
            "coverage-stratified held-out ternary systems as positive scaling fractions"
        )
    else:
        n_validation = max(1, round(0.15 * len(ternary_keys)))
        validation_ternary_keys, pool = take_stratified(remaining, n_validation)
        n_training = max(1, math.ceil(ternary_training_fraction * len(pool)))
        training_ternary_keys = pool[:n_training]
        train = binary_train + [
            row for key in training_ternary_keys for row in ternary_grouped[key]
        ]
        validation = binary_validation + [
            row for key in validation_ternary_keys for row in ternary_grouped[key]
        ]
        test = [row for key in test_ternary_keys for row in ternary_grouped[key]]
        selection = "fixed ternary system test/validation; fraction samples the remaining ternary training pool"
    test_coverage = Counter(coverage_by_system.get(key, -1) for key in test_ternary_keys)
    protocol = (
        "binary_to_ternary_zero_shot"
        if ternary_training_fraction == 0.0
        else f"binary_to_ternary_scale_{ternary_training_fraction:g}"
    )
    return DatasetPartitions(
        train=_ordered(train),
        validation=_ordered(validation),
        test=_ordered(test),
        protocol=protocol,
        seed=seed,
        metadata=_metadata(
            train,
            validation,
            test,
            split_rule=selection,
            ternary_training_fraction=ternary_training_fraction,
            ternary_training_fraction_actual=(
                len(training_ternary_keys) / len(pool)
                if ternary_training_fraction > 0.0 and pool
                else 0.0
            ),
            ternary_training_systems=len(training_ternary_keys),
            ternary_validation_systems=len(validation_ternary_keys),
            ternary_test_systems=len(test_ternary_keys),
            test_binary_subsystem_coverage={str(key): value for key, value in sorted(test_coverage.items())},
        ),
    )
