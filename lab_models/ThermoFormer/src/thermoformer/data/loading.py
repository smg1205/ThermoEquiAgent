"""Excel ingestion and leakage-safe system-grouped dataset splitting."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import openpyxl
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..thermodynamics.vapor_pressure import (
    CORRELATION_PARAMETER_COUNT,
    PurePropertyCatalog,
    empty_pure_property_catalog,
)

MMHG_TO_KPA = 0.133322368
QualityStatus = Literal["passed", "unverified", "failed"]
ExperimentMode = Literal["isothermal", "isobaric", "full_state"]
EXPERIMENT_MODE_INDEX: dict[ExperimentMode, int] = {
    "isothermal": 0,
    "isobaric": 1,
    "full_state": 2,
}


@dataclass(frozen=True)
class VLESample:
    smiles: tuple[str, ...]
    names: tuple[str, ...]
    temperature_k: float
    pressure_kpa: float
    liquid_composition: tuple[float, ...]
    vapor_composition: tuple[float, ...]
    quality_weight: float
    quality_status: QualityStatus
    source: str
    doi: str
    experiment_mode: ExperimentMode = "full_state"
    experiment_mode_confidence: float = 0.0

    @property
    def component_count(self) -> int:
        return len(self.smiles)

    @property
    def system_key(self) -> tuple[str, ...]:
        return tuple(sorted(self.smiles))


@dataclass(frozen=True)
class FoldSplit:
    train: tuple[VLESample, ...]
    validation: tuple[VLESample, ...]


@dataclass(frozen=True)
class SplitPlan:
    cv: tuple[VLESample, ...]
    test: tuple[VLESample, ...]
    folds: tuple[FoldSplit, ...]
    anchor_reference_systems: tuple[tuple[str, ...], ...] = ()
    mode: Literal["kfold", "holdout"] = "kfold"


@dataclass(frozen=True)
class DatasetAudit:
    raw_rows: int
    accepted_before_deduplication: int
    duplicates_removed: int
    loaded_samples: int
    rejected: dict[str, int]


@dataclass(frozen=True)
class DatasetLoadResult:
    samples: tuple[VLESample, ...]
    audit: DatasetAudit


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def classify_quality_flags(
    values: Sequence[object], failed_weight: float
) -> tuple[QualityStatus, float]:
    """Map workbook codes: 1=passed, 0=failed, -1=undetermined."""
    flags = {_number(value) for value in values}
    if 0.0 in flags:
        return "failed", failed_weight
    if 1.0 in flags:
        return "passed", 1.0
    return "unverified", 0.5


def _closed_composition(values: Sequence[float], tolerance: float = 1e-6) -> tuple[float, ...] | None:
    if any(not math.isfinite(value) for value in values):
        return None
    final = 1.0 - sum(values)
    composition = tuple(values) + (final,)
    if any(value < -tolerance or value > 1.0 + tolerance for value in composition):
        return None
    clipped = tuple(min(1.0, max(0.0, value)) for value in composition)
    total = sum(clipped)
    if total <= 0.0:
        return None
    return tuple(value / total for value in clipped)


def _row_to_sample(
    row: Sequence[object],
    component_count: int,
    source: Path,
    failed_weight: float,
    explicit_experiment_mode: object = None,
) -> tuple[VLESample | None, str | None]:
    offset = 3 * component_count
    names = tuple(str(row[3 * index] or "").strip() for index in range(component_count))
    smiles = tuple(str(row[3 * index + 2] or "").strip() for index in range(component_count))
    if not all(smiles):
        return None, "missing_smiles"

    pressure_mmhg = _number(row[offset + 2])
    temperature_c = _number(row[offset + 3])
    if not math.isfinite(pressure_mmhg) or pressure_mmhg <= 0.0 or not math.isfinite(temperature_c):
        return None, "invalid_temperature_or_pressure"

    independent = component_count - 1
    liquid = _closed_composition([_number(row[offset + 4 + index]) for index in range(independent)])
    vapor = _closed_composition([_number(row[offset + 4 + independent + index]) for index in range(independent)])
    if liquid is None or vapor is None:
        return None, "invalid_composition"

    quality_status, quality_weight = classify_quality_flags(
        row[offset : offset + 2], failed_weight
    )
    if quality_weight <= 0.0:
        return None, "failed_quality"
    doi_index = offset + 4 + 2 * independent
    doi = str(row[doi_index] or "").strip() if doi_index < len(row) else ""
    explicit_mode_text = str(explicit_experiment_mode or "").strip()
    explicit_mode = _parse_experiment_mode(explicit_experiment_mode)
    if explicit_mode_text and explicit_mode is None:
        return None, "invalid_experiment_mode"
    return (
        VLESample(
            smiles=smiles,
            names=names,
            temperature_k=temperature_c + 273.15,
            pressure_kpa=pressure_mmhg * MMHG_TO_KPA,
            liquid_composition=liquid,
            vapor_composition=vapor,
            quality_weight=quality_weight,
            quality_status=quality_status,
            source=str(source),
            doi=doi,
            experiment_mode=explicit_mode or "full_state",
            experiment_mode_confidence=1.0 if explicit_mode is not None else 0.0,
        ),
        None,
    )


def _parse_experiment_mode(value: object) -> ExperimentMode | None:
    normalized = str(value or "").strip().casefold().replace("_", "-").replace(" ", "")
    aliases: dict[str, ExperimentMode] = {
        "isothermal": "isothermal",
        "constant-temperature": "isothermal",
        "constanttemperature": "isothermal",
        "p-x-y": "isothermal",
        "pxy": "isothermal",
        "isobaric": "isobaric",
        "constant-pressure": "isobaric",
        "constantpressure": "isobaric",
        "t-x-y": "isobaric",
        "txy": "isobaric",
        "full-state": "full_state",
        "fullstate": "full_state",
        "tp-x-y": "full_state",
        "tpxy": "full_state",
    }
    return aliases.get(normalized)


def infer_experiment_modes(
    samples: Sequence[VLESample],
    minimum_series_points: int = 3,
    minimum_confidence: float = 2.0 / 3.0,
) -> list[VLESample]:
    """Infer the controlled variable from repeated T/P conditions within a source series.

    A DOI can contain several experimental series. Classification is therefore
    performed per row: a repeated pressure identifies an isobaric series, while
    a repeated temperature identifies an isothermal series. Ambiguous points
    remain full-state observations and can be evaluated in both directions.
    """
    if minimum_series_points < 2:
        raise ValueError("minimum_series_points must be at least two")
    if not 0.5 < minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in (0.5, 1.0]")
    grouped: dict[tuple[str, str, tuple[str, ...]], list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.source, sample.doi, sample.system_key)].append(sample)
    inferred: list[VLESample] = []
    for rows in grouped.values():
        temperature_counts = Counter(round(row.temperature_k, 4) for row in rows)
        pressure_counts = Counter(round(row.pressure_kpa, 4) for row in rows)
        for row in rows:
            if row.experiment_mode_confidence >= 1.0:
                inferred.append(row)
                continue
            if not row.doi:
                inferred.append(
                    replace(
                        row,
                        experiment_mode="full_state",
                        experiment_mode_confidence=0.0,
                    )
                )
                continue
            repeated_temperature = temperature_counts[round(row.temperature_k, 4)]
            repeated_pressure = pressure_counts[round(row.pressure_kpa, 4)]
            if (
                repeated_pressure >= minimum_series_points
                and repeated_pressure > repeated_temperature
            ):
                mode: ExperimentMode = "isobaric"
                confidence = repeated_pressure / (
                    repeated_pressure + repeated_temperature
                )
            elif (
                repeated_temperature >= minimum_series_points
                and repeated_temperature > repeated_pressure
            ):
                mode = "isothermal"
                confidence = repeated_temperature / (
                    repeated_temperature + repeated_pressure
                )
            else:
                mode = "full_state"
                confidence = 0.0
            if mode != "full_state" and confidence < minimum_confidence:
                mode = "full_state"
            inferred.append(
                replace(
                    row,
                    experiment_mode=mode,
                    experiment_mode_confidence=confidence,
                )
            )
    return inferred


def discover_vle_workbooks(root: Path, source_filter: str = "") -> tuple[Path, ...]:
    """Return the exact recursive workbook set consumed by the VLE loader."""
    filter_text = source_filter.casefold()
    return tuple(
        path
        for path in sorted(root.rglob("*.xlsx"))
        if not filter_text or filter_text in str(path).casefold()
    )


def load_vle_dataset(
    root: Path,
    source_filter: str = "",
    failed_weight: float = 0.0,
    max_pressure_kpa: float | None = 500.0,
) -> DatasetLoadResult:
    """Load VLE samples together with an auditable row-rejection summary."""
    if failed_weight < 0.0 or failed_weight > 1.0:
        raise ValueError("failed_weight must be between zero and one")

    samples: list[VLESample] = []
    rejected: Counter[str] = Counter()
    raw_rows = 0
    for path in discover_vle_workbooks(root, source_filter):
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for worksheet in workbook.worksheets:
                rows = worksheet.iter_rows(values_only=True)
                header = next(rows, None)
                if not header:
                    continue
                normalized = [str(value).strip().casefold() for value in header]
                if "smiles1" not in normalized:
                    continue
                if "smiles4" in normalized:
                    raise ValueError(f"Only binary and ternary data are supported: {path}::{worksheet.title}")
                component_count = 3 if "smiles3" in normalized else 2
                mode_index = next(
                    (
                        normalized.index(field)
                        for field in ("experiment_mode", "experimental_mode", "mode")
                        if field in normalized
                    ),
                    None,
                )
                for row in rows:
                    raw_rows += 1
                    sample, reason = _row_to_sample(
                        row,
                        component_count,
                        path,
                        failed_weight,
                        row[mode_index]
                        if mode_index is not None and mode_index < len(row)
                        else None,
                    )
                    if reason is not None:
                        rejected[reason] += 1
                    elif sample is not None and (
                        max_pressure_kpa is not None
                        and sample.pressure_kpa > max_pressure_kpa
                    ):
                        rejected["pressure_above_limit"] += 1
                    elif sample is not None:
                        samples.append(sample)
        finally:
            workbook.close()
    if not samples:
        raise RuntimeError(f"No supported VLE rows found under {root}")
    accepted_before_deduplication = len(samples)
    unique = deduplicate_samples(samples)
    unique = infer_experiment_modes(unique)
    audit = DatasetAudit(
        raw_rows=raw_rows,
        accepted_before_deduplication=accepted_before_deduplication,
        duplicates_removed=accepted_before_deduplication - len(unique),
        loaded_samples=len(unique),
        rejected=dict(sorted(rejected.items())),
    )
    return DatasetLoadResult(samples=tuple(unique), audit=audit)


def load_vle_samples(
    root: Path,
    source_filter: str = "",
    failed_weight: float = 0.0,
    max_pressure_kpa: float | None = 500.0,
) -> list[VLESample]:
    """Load binary/ternary rows, converting °C/mmHg to K/kPa.

    A positive ``failed_weight`` retains consistency-test failures as a
    low-confidence robustness set. Set it to zero for strict main-set training.
    """
    return list(
        load_vle_dataset(
            root,
            source_filter=source_filter,
            failed_weight=failed_weight,
            max_pressure_kpa=max_pressure_kpa,
        ).samples
    )


def _canonical_state_key(sample: VLESample) -> tuple[object, ...]:
    order = sorted(range(sample.component_count), key=lambda index: sample.smiles[index])
    return (
        tuple(sample.smiles[index] for index in order),
        round(sample.temperature_k, 7),
        round(sample.pressure_kpa, 7),
        tuple(round(sample.liquid_composition[index], 9) for index in order),
        tuple(round(sample.vapor_composition[index], 9) for index in order),
    )


def deduplicate_samples(samples: Sequence[VLESample]) -> list[VLESample]:
    """Remove exact thermodynamic states, including reversed component order."""
    selected: dict[tuple[object, ...], VLESample] = {}
    order: list[tuple[object, ...]] = []
    for sample in samples:
        key = _canonical_state_key(sample)
        previous = selected.get(key)
        if previous is None:
            selected[key] = sample
            order.append(key)
        elif (
            sample.experiment_mode_confidence,
            sample.quality_weight,
        ) > (
            previous.experiment_mode_confidence,
            previous.quality_weight,
        ):
            selected[key] = sample
    return [selected[key] for key in order]


def pure_anchor_temperatures(
    samples: Sequence[VLESample],
    threshold: float = 0.999,
) -> dict[str, set[float]]:
    anchors: dict[str, set[float]] = defaultdict(set)
    for sample in samples:
        for smile, fraction in zip(sample.smiles, sample.liquid_composition):
            if fraction >= threshold:
                anchors[smile].add(round(sample.temperature_k, 6))
    return anchors


def retain_pure_anchored_systems(
    samples: Sequence[VLESample],
    minimum_temperatures: int = 2,
) -> list[VLESample]:
    """Retain systems whose every Psat branch has pure-state temperature anchors."""
    if minimum_temperatures < 0:
        raise ValueError("minimum_temperatures cannot be negative")
    retained = list(samples)
    if minimum_temperatures == 0:
        return retained
    while True:
        anchor_map = pure_anchor_temperatures(retained)
        anchored = {
            smile
            for smile, temperatures in anchor_map.items()
            if len(temperatures) >= minimum_temperatures
        }
        filtered = [
            sample for sample in retained if all(smile in anchored for smile in sample.smiles)
        ]
        if len(filtered) == len(retained):
            return filtered
        retained = filtered


@dataclass
class VLEBatch:
    molecules: Tensor
    temperature_k: Tensor
    pressure_kpa: Tensor
    x: Tensor
    y: Tensor
    mask: Tensor
    quality_weight: Tensor
    experiment_mode: Tensor
    pure_property_parameters: Tensor

    def to(self, device: torch.device) -> "VLEBatch":
        return VLEBatch(**{name: value.to(device) for name, value in vars(self).items()})


@dataclass(frozen=True)
class _TensorRow:
    molecules: Tensor
    temperature_k: float
    pressure_kpa: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    quality_weight: float
    experiment_mode: ExperimentMode
    pure_property_parameters: Tensor


class VLETensorDataset(Dataset[_TensorRow]):
    def __init__(
        self,
        samples: Sequence[VLESample],
        feature_map: dict[str, np.ndarray],
        pure_property_catalog: PurePropertyCatalog | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.feature_map = feature_map
        self.pure_property_catalog = (
            pure_property_catalog or empty_pure_property_catalog()
        )
        missing = sorted({smile for sample in samples for smile in sample.smiles if smile not in feature_map})
        if missing:
            raise KeyError(f"Missing molecular features for {len(missing)} SMILES; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _TensorRow:
        sample = self.samples[index]
        features = np.stack([self.feature_map[smile] for smile in sample.smiles]).astype(np.float32)
        return _TensorRow(
            molecules=torch.from_numpy(features),
            temperature_k=sample.temperature_k,
            pressure_kpa=sample.pressure_kpa,
            x=sample.liquid_composition,
            y=sample.vapor_composition,
            quality_weight=sample.quality_weight,
            experiment_mode=sample.experiment_mode,
            pure_property_parameters=torch.from_numpy(
                self.pure_property_catalog.parameters_for(sample.smiles)
            ),
        )


def collate_vle(rows: Sequence[_TensorRow]) -> VLEBatch:
    """Pad every batch to the supported maximum of three components."""
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(rows)
    feature_dim = rows[0].molecules.shape[-1]
    molecules = torch.zeros(batch_size, 3, feature_dim, dtype=torch.float32)
    x = torch.zeros(batch_size, 3, dtype=torch.float32)
    y = torch.zeros_like(x)
    mask = torch.zeros_like(x)
    pure_property_parameters = torch.zeros(
        batch_size,
        3,
        CORRELATION_PARAMETER_COUNT,
        dtype=torch.float32,
    )
    for index, row in enumerate(rows):
        component_count = row.molecules.shape[0]
        if component_count not in (2, 3):
            raise ValueError("Only binary and ternary rows can be collated")
        molecules[index, :component_count] = row.molecules
        x[index, :component_count] = torch.tensor(row.x, dtype=torch.float32)
        y[index, :component_count] = torch.tensor(row.y, dtype=torch.float32)
        mask[index, :component_count] = 1.0
        pure_property_parameters[index, :component_count] = row.pure_property_parameters
    return VLEBatch(
        molecules=molecules,
        temperature_k=torch.tensor([[row.temperature_k] for row in rows], dtype=torch.float32),
        pressure_kpa=torch.tensor([[row.pressure_kpa] for row in rows], dtype=torch.float32),
        x=x,
        y=y,
        mask=mask,
        quality_weight=torch.tensor([[row.quality_weight] for row in rows], dtype=torch.float32),
        experiment_mode=torch.tensor(
            [EXPERIMENT_MODE_INDEX[row.experiment_mode] for row in rows],
            dtype=torch.long,
        ),
        pure_property_parameters=pure_property_parameters,
    )


def _stratified_group_holdout(
    grouped: dict[tuple[str, ...], list[VLESample]],
    candidate_keys: Sequence[tuple[str, ...]],
    fraction: float,
    reserve_groups_per_stratum: int,
    total_rows_by_component_count: dict[int, int],
    rng: random.Random,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Select a row-balanced holdout while preserving whole system groups."""
    strata: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for key in candidate_keys:
        strata[len(key)].append(key)
    selected: list[tuple[str, ...]] = []
    retained: list[tuple[str, ...]] = []
    for component_count in sorted(strata):
        keys = strata[component_count]
        rng.shuffle(keys)
        target_rows = max(
            1,
            round(fraction * total_rows_by_component_count[component_count]),
        )
        maximum_selected_groups = max(0, len(keys) - reserve_groups_per_stratum)
        selected_rows = 0
        selected_count = 0
        while selected_count < maximum_selected_groups and selected_rows < target_rows:
            key = keys[selected_count]
            selected.append(key)
            selected_rows += len(grouped[key])
            selected_count += 1
        retained.extend(keys[selected_count:])
    return selected, retained


def grouped_holdout_and_folds(
    samples: Sequence[VLESample],
    test_fraction: float,
    folds: int,
    seed: int,
    minimum_anchor_temperatures: int = 0,
    pure_threshold: float = 0.999,
) -> SplitPlan:
    """Split whole systems while protecting a compact pure-property reference pool."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if minimum_anchor_temperatures < 0:
        raise ValueError("minimum_anchor_temperatures cannot be negative")

    grouped: dict[tuple[str, ...], list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.system_key].append(sample)
    if len(grouped) <= folds:
        raise ValueError(f"Need more than {folds} distinct systems, found {len(grouped)}")

    anchor_pairs: dict[tuple[str, ...], set[tuple[str, float]]] = defaultdict(set)
    for sample in samples:
        for smile, fraction in zip(sample.smiles, sample.liquid_composition):
            if fraction >= pure_threshold:
                anchor_pairs[sample.system_key].add((smile, round(sample.temperature_k, 6)))
    molecules = {smile for sample in samples for smile in sample.smiles}
    covered_temperatures: dict[str, set[float]] = defaultdict(set)
    available_anchor_keys = set(anchor_pairs)
    reference_keys: list[tuple[str, ...]] = []
    while minimum_anchor_temperatures and any(
        len(covered_temperatures[smile]) < minimum_anchor_temperatures
        for smile in molecules
    ):
        def coverage_score(key: tuple[str, ...]) -> tuple[float, int, int, tuple[str, ...]]:
            gain = sum(
                temperature not in covered_temperatures[smile]
                and len(covered_temperatures[smile]) < minimum_anchor_temperatures
                for smile, temperature in anchor_pairs[key]
            )
            return gain / len(grouped[key]), gain, -len(grouped[key]), key

        if not available_anchor_keys:
            missing = sorted(
                smile
                for smile in molecules
                if len(covered_temperatures[smile]) < minimum_anchor_temperatures
            )
            raise ValueError(f"Pure-property anchors are insufficient for {len(missing)} molecules")
        selected = max(available_anchor_keys, key=coverage_score)
        if coverage_score(selected)[1] == 0:
            missing = sorted(
                smile
                for smile in molecules
                if len(covered_temperatures[smile]) < minimum_anchor_temperatures
            )
            raise ValueError(f"Pure-property anchors are insufficient for {len(missing)} molecules")
        available_anchor_keys.remove(selected)
        reference_keys.append(selected)
        for smile, temperature in anchor_pairs[selected]:
            covered_temperatures[smile].add(temperature)

    reference_set = set(reference_keys)
    candidate_keys = [key for key in grouped if key not in reference_set]
    rng = random.Random(seed)
    total_rows_by_component_count = {
        component_count: sum(
            len(rows) for key, rows in grouped.items() if len(key) == component_count
        )
        for component_count in {len(key) for key in grouped}
    }
    test_keys, cv_evaluation_keys = _stratified_group_holdout(
        grouped,
        candidate_keys,
        fraction=test_fraction,
        reserve_groups_per_stratum=folds,
        total_rows_by_component_count=total_rows_by_component_count,
        rng=rng,
    )
    if len(cv_evaluation_keys) < folds:
        raise ValueError("Not enough non-reference systems remain for grouped cross-validation")

    fold_keys: list[list[tuple[str, ...]]] = [[] for _ in range(folds)]
    for component_count in sorted({len(key) for key in cv_evaluation_keys}):
        stratum_keys = [key for key in cv_evaluation_keys if len(key) == component_count]
        tie_break = {key: rng.random() for key in stratum_keys}
        ordered_keys = sorted(
            stratum_keys,
            key=lambda key: (-len(grouped[key]), tie_break[key]),
        )
        fold_sizes = [0] * folds
        for key in ordered_keys:
            target_fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
            fold_keys[target_fold].append(key)
            fold_sizes[target_fold] += len(grouped[key])

    def rows_for(selected: Sequence[tuple[str, ...]]) -> tuple[VLESample, ...]:
        return tuple(row for key in selected for row in grouped[key])

    ordered_reference_keys = sorted(reference_set)
    cv_keys = ordered_reference_keys + cv_evaluation_keys
    cv = rows_for(cv_keys)
    test = rows_for(test_keys)
    fold_splits = []
    for validation_keys in fold_keys:
        validation_set = set(validation_keys)
        training_keys = ordered_reference_keys + [
            key for key in cv_evaluation_keys if key not in validation_set
        ]
        fold_splits.append(FoldSplit(train=rows_for(training_keys), validation=rows_for(validation_keys)))
    return SplitPlan(
        cv=cv,
        test=test,
        folds=tuple(fold_splits),
        anchor_reference_systems=tuple(ordered_reference_keys),
        mode="kfold",
    )


def build_split_plan(
    samples: Sequence[VLESample],
    mode: Literal["kfold", "holdout"] = "kfold",
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    folds: int = 5,
    seed: int = 42,
    minimum_anchor_temperatures: int = 0,
    pure_threshold: float = 0.999,
) -> SplitPlan:
    """Build either grouped K-fold CV or one grouped train/validation/test split."""
    if mode == "kfold":
        return grouped_holdout_and_folds(
            samples,
            test_fraction=test_fraction,
            folds=folds,
            seed=seed,
            minimum_anchor_temperatures=minimum_anchor_temperatures,
            pure_threshold=pure_threshold,
        )
    if mode != "holdout":
        raise ValueError("mode must be 'kfold' or 'holdout'")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test_fraction + validation_fraction must be below one")

    base = grouped_holdout_and_folds(
        samples,
        test_fraction=test_fraction,
        folds=2,
        seed=seed,
        minimum_anchor_temperatures=minimum_anchor_temperatures,
        pure_threshold=pure_threshold,
    )
    grouped_cv: dict[tuple[str, ...], list[VLESample]] = defaultdict(list)
    for sample in base.cv:
        grouped_cv[sample.system_key].append(sample)
    reference_set = set(base.anchor_reference_systems)
    candidates = [key for key in grouped_cv if key not in reference_set]
    total_rows_by_component_count = dict(
        Counter(sample.component_count for sample in samples)
    )
    validation_keys, training_evaluation_keys = _stratified_group_holdout(
        grouped_cv,
        candidates,
        fraction=validation_fraction,
        reserve_groups_per_stratum=1,
        total_rows_by_component_count=total_rows_by_component_count,
        rng=random.Random(seed + 1),
    )

    def rows_for(keys: Sequence[tuple[str, ...]]) -> tuple[VLESample, ...]:
        return tuple(row for key in keys for row in grouped_cv[key])

    reference_keys = list(base.anchor_reference_systems)
    training = rows_for(reference_keys + training_evaluation_keys)
    validation = rows_for(validation_keys)
    if not training or not validation or not base.test:
        raise ValueError("Grouped holdout produced an empty train, validation, or test split")
    return SplitPlan(
        cv=base.cv,
        test=base.test,
        folds=(FoldSplit(train=training, validation=validation),),
        anchor_reference_systems=base.anchor_reference_systems,
        mode="holdout",
    )
