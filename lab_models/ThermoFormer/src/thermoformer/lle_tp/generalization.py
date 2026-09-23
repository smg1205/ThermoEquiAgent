"""Leakage-safe LLE state and chemical-space generalization splits."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Literal, Sequence

from .data import Condition


def _ordered(values: Sequence[Condition]) -> list[Condition]:
    return sorted(values, key=lambda value: value.key)


def _tail_partition(
    values: Sequence[Condition],
    score: Callable[[Condition], float],
    tail: Literal["low", "high"],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[Condition], list[Condition], list[Condition]] | None:
    by_value: dict[float, list[Condition]] = defaultdict(list)
    for value in values:
        by_value[round(score(value), 8)].append(value)
    levels = sorted(by_value, reverse=tail == "high")
    if len(levels) < 3:
        return None
    targets = (
        max(1, round(test_fraction * len(values))),
        max(1, round(validation_fraction * len(values))),
    )
    selected: list[list[float]] = [[], []]
    for destination, target in zip(selected, targets):
        count = 0
        reserve = 2 if destination is selected[0] else 1
        while levels and count < target and len(levels) > reserve:
            level = levels.pop(0)
            destination.append(level)
            count += len(by_value[level])
    if not all(selected) or not levels:
        return None
    return (
        [row for level in levels for row in by_value[level]],
        [row for level in selected[1] for row in by_value[level]],
        [row for level in selected[0] for row in by_value[level]],
    )


def state_extreme_partition(
    conditions: Sequence[Condition],
    variable: Literal["temperature", "pressure"],
    tail: Literal["low", "high"],
) -> tuple[dict[str, list[Condition]], dict[str, object]]:
    binary = [value for value in conditions if value.n == 2]
    grouped: dict[tuple[str, ...], list[Condition]] = defaultdict(list)
    for value in binary:
        grouped[value.key[0]].append(value)
    index = 1 if variable == "temperature" else 2
    parts = {name: [] for name in ("train", "validation", "test")}
    skipped = 0
    eligible = 0
    for system in sorted(grouped):
        result = _tail_partition(grouped[system], lambda value: value.key[index], tail)
        if result is None:
            skipped += len(grouped[system])
            continue
        eligible += 1
        for name, values in zip(("train", "validation", "test"), result):
            parts[name].extend(values)
    if any(not values for values in parts.values()):
        raise ValueError(f"No usable LLE {variable}-{tail} split")
    return (
        {name: _ordered(values) for name, values in parts.items()},
        {
            "split_rule": f"strict per-system {tail} tail of unique {variable} values",
            "eligible_systems": eligible,
            "skipped_conditions": skipped,
        },
    )


def unseen_component_partition(
    conditions: Sequence[Condition], seed: int, target_fraction: float = 0.20
) -> tuple[dict[str, list[Condition]], dict[str, object]]:
    binary = [value for value in conditions if value.n == 2]
    grouped: dict[tuple[str, ...], list[Condition]] = defaultdict(list)
    for value in binary:
        grouped[value.key[0]].append(value)
    all_components = sorted({component for system in grouped for component in system})
    target = max(1, math.ceil(target_fraction * len(all_components)))
    rng = random.Random(seed)
    tie = {component: rng.random() for component in all_components}
    candidates = []
    for seed_system in sorted(grouped):
        held = set(seed_system)
        remaining = sorted(
            (component for component in all_components if component not in held),
            key=lambda component: (sum(component in system for system in grouped), tie[component], component),
        )
        held.update(remaining[: max(0, target - len(held))])
        test_systems = {system for system in grouped if set(system) & held}
        candidates.append((len(test_systems), rng.random(), held, test_systems))
    _, _, held, test_systems = min(
        (value for value in candidates if len(grouped) - len(value[3]) >= 2),
        key=lambda value: value[:2],
    )
    train_systems = [system for system in grouped if system not in test_systems]
    rng.shuffle(train_systems)
    target_validation = max(1, round(0.15 * sum(len(grouped[s]) for s in train_systems)))
    validation_systems: set[tuple[str, ...]] = set()
    count = 0
    while len(train_systems) > 1 and count < target_validation:
        system = train_systems.pop(0)
        validation_systems.add(system)
        count += len(grouped[system])
    parts = {
        "train": [value for system in train_systems for value in grouped[system]],
        "validation": [value for system in validation_systems for value in grouped[system]],
        "test": [value for system in test_systems for value in grouped[system]],
    }
    if any(not values for values in parts.values()):
        raise ValueError("LLE unseen-component split produced an empty partition")
    train_components = {component for value in parts["train"] for component in value.key[0]}
    assert not (train_components & held)
    return (
        {name: _ordered(values) for name, values in parts.items()},
        {
            "split_rule": "test systems contain at least one component absent from train",
            "held_out_components": sorted(held),
            "held_out_component_count": len(held),
        },
    )


def partition_generalization(
    conditions: Sequence[Condition], protocol: str, seed: int
) -> tuple[dict[str, list[Condition]], dict[str, object]]:
    mapping = {
        "binary-temperature-low": ("temperature", "low"),
        "binary-temperature-high": ("temperature", "high"),
        "binary-pressure-low": ("pressure", "low"),
        "binary-pressure-high": ("pressure", "high"),
    }
    if protocol == "binary-unseen-component":
        return unseen_component_partition(conditions, seed)
    if protocol not in mapping:
        raise ValueError(protocol)
    variable, tail = mapping[protocol]
    return state_extreme_partition(conditions, variable, tail)
