"""LLE data loading, strict grouped splits, and directed phase-pair batching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import openpyxl
import torch
from rdkit import Chem
from torch import Tensor
from torch.utils.data import Dataset

from .loading import MMHG_TO_KPA, classify_quality_flags


@dataclass(frozen=True)
class LLESample:
    record_id: str
    smiles: tuple[str, ...]
    names: tuple[str, ...]
    temperature_k: float
    pressure_kpa: float
    phase_alpha: tuple[float, ...]
    phase_beta: tuple[float, ...]
    quality_weight: float
    source: str
    doi: str

    @property
    def component_count(self) -> int:
        return len(self.smiles)

    @property
    def system_key(self) -> tuple[str, ...]:
        return self.smiles


@dataclass(frozen=True)
class LLEDatasetLoadResult:
    samples: tuple[LLESample, ...]
    audit: dict[str, object]


@dataclass(frozen=True)
class LLESplit:
    train: tuple[LLESample, ...]
    validation: tuple[LLESample, ...]
    test: tuple[LLESample, ...]
    protocol: str
    seed: int


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _canonical_smiles(value: object) -> str:
    text = str(value or "").strip()
    molecule = Chem.MolFromSmiles(text) if text else None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else ""


def _phase(values: Sequence[object], count: int) -> tuple[float, ...] | None:
    independent = tuple(_number(value) for value in values)
    if len(independent) != count - 1 or any(not math.isfinite(value) for value in independent):
        return None
    phase = (*independent, 1.0 - sum(independent))
    if any(value < -1e-6 or value > 1.0 + 1e-6 for value in phase):
        return None
    clipped = tuple(max(0.0, min(1.0, value)) for value in phase)
    total = sum(clipped)
    return tuple(value / total for value in clipped) if total > 0.0 else None


def _canonicalize(
    names: tuple[str, ...],
    smiles: tuple[str, ...],
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    order = tuple(sorted(range(len(smiles)), key=lambda index: smiles[index]))
    ordered_smiles = tuple(smiles[index] for index in order)
    if len(set(ordered_smiles)) != len(ordered_smiles):
        raise ValueError("duplicate_component")
    ordered_names = tuple(names[index] for index in order)
    alpha = tuple(first[index] for index in order)
    beta = tuple(second[index] for index in order)
    return ordered_names, ordered_smiles, *( (alpha, beta) if alpha <= beta else (beta, alpha) )


def _identifier(sample: LLESample) -> str:
    payload = {
        "components": sample.smiles,
        "temperature_k": round(sample.temperature_k, 7),
        "pressure_kpa": round(sample.pressure_kpa, 7),
        "alpha": [round(value, 9) for value in sample.phase_alpha],
        "beta": [round(value, 9) for value in sample.phase_beta],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lle_dataset_digest(samples: Sequence[LLESample]) -> str:
    identities = sorted(_identifier(sample) for sample in samples)
    if len(identities) != len(set(identities)):
        raise ValueError("LLE dataset contains duplicate tie-line identities")
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def load_lle_dataset(
    root: Path,
    *,
    binary_workbook: str = "binary_lle_english.xlsx",
    ternary_workbook: str = "ternary_lle_english.xlsx",
    component_count: int | None = None,
    failed_weight: float = 0.0,
) -> LLEDatasetLoadResult:
    """Read actual LLE workbook headers and canonicalize both liquid phases."""
    if not 0.0 <= failed_weight <= 1.0:
        raise ValueError("failed_weight must lie in [0, 1]")
    paths = (root / binary_workbook, root / ternary_workbook)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing LLE workbook(s): " + ", ".join(missing))
    accepted: list[LLESample] = []
    rejected: dict[str, int] = {}
    raw_rows = 0
    for path in paths:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                rows = sheet.iter_rows(values_only=True)
                header_row = next(rows, None)
                if header_row is None:
                    continue
                header = {str(value or "").strip().casefold(): index for index, value in enumerate(header_row)}
                count = 3 if "smiles3" in header else 2
                if component_count is not None and count != component_count:
                    continue
                required = {
                    "pressure_mmhg", "temperature_c", "quality_check_1", "quality_check_2",
                    *(f"smiles{index}" for index in range(1, count + 1)),
                    *(f"component_{index}_original_name" for index in range(1, count + 1)),
                    *(f"x{index}" for index in range(1, count)),
                    *(f"y{index}" for index in range(1, count)),
                }
                absent = sorted(required - set(header))
                if absent:
                    raise ValueError(f"{path.name} is missing columns: {absent}")
                for row_number, row in enumerate(rows, start=2):
                    raw_rows += 1
                    names = tuple(str(row[header[f"component_{index}_original_name"]] or "").strip() for index in range(1, count + 1))
                    smiles = tuple(_canonical_smiles(row[header[f"smiles{index}"]]) for index in range(1, count + 1))
                    pressure = _number(row[header["pressure_mmhg"]])
                    temperature = _number(row[header["temperature_c"]])
                    first = _phase([row[header[f"x{index}"]] for index in range(1, count)], count)
                    second = _phase([row[header[f"y{index}"]] for index in range(1, count)], count)
                    status, weight = classify_quality_flags(
                        [row[header["quality_check_1"]], row[header["quality_check_2"]]], failed_weight
                    )
                    reason = ""
                    if not all(smiles):
                        reason = "invalid_smiles"
                    elif not math.isfinite(pressure) or pressure <= 0.0 or not math.isfinite(temperature):
                        reason = "invalid_state"
                    elif first is None or second is None:
                        reason = "invalid_composition"
                    elif weight <= 0.0:
                        reason = "failed_quality"
                    if reason:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue
                    try:
                        names, smiles, alpha, beta = _canonicalize(names, smiles, first, second)
                    except ValueError as error:
                        rejected[str(error)] = rejected.get(str(error), 0) + 1
                        continue
                    doi = str(row[header["doi"]] or "").strip() if "doi" in header else ""
                    accepted.append(LLESample(
                        record_id=f"{path.name}:{sheet.title}:{row_number}",
                        smiles=smiles, names=names, temperature_k=temperature + 273.15,
                        pressure_kpa=pressure * MMHG_TO_KPA, phase_alpha=alpha, phase_beta=beta,
                        quality_weight=weight, source=str(path), doi=doi,
                    ))
        finally:
            workbook.close()
    unique: dict[str, LLESample] = {}
    for sample in accepted:
        key = _identifier(sample)
        previous = unique.get(key)
        if previous is None or sample.quality_weight > previous.quality_weight:
            unique[key] = sample
    samples = tuple(unique.values())
    if not samples:
        raise RuntimeError("No valid LLE samples were loaded")
    return LLEDatasetLoadResult(samples=samples, audit={
        "raw_rows": raw_rows, "accepted_before_deduplication": len(accepted),
        "duplicates_removed": len(accepted) - len(samples), "loaded_samples": len(samples),
        "rejected": dict(sorted(rejected.items())), "component_count": component_count,
    })


def build_lle_split(
    samples: Sequence[LLESample], *, seed: int, protocol: str = "ternary_train_ternary_test",
    test_fraction: float = 0.15, validation_fraction: float = 0.15,
) -> LLESplit:
    """Create a system-disjoint split before directed augmentation."""
    if not 0.0 < test_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("LLE split fractions must lie in (0, 1)")
    systems: dict[tuple[str, ...], list[LLESample]] = {}
    for sample in samples:
        systems.setdefault(sample.system_key, []).append(sample)
    keys = sorted(systems)
    if len(keys) < 3:
        raise ValueError("LLE strict holdout needs at least three distinct systems")
    random.Random(seed).shuffle(keys)
    test_count = max(1, round(len(keys) * test_fraction))
    validation_count = max(1, round(len(keys) * validation_fraction))
    if test_count + validation_count >= len(keys):
        raise ValueError("LLE split leaves no training systems")
    test_keys = set(keys[:test_count])
    validation_keys = set(keys[test_count:test_count + validation_count])
    train_keys = set(keys[test_count + validation_count:])
    return LLESplit(
        train=tuple(sample for key in train_keys for sample in systems[key]),
        validation=tuple(sample for key in validation_keys for sample in systems[key]),
        test=tuple(sample for key in test_keys for sample in systems[key]),
        protocol=protocol, seed=seed,
    )


def save_lle_split(path: Path, dataset: Sequence[LLESample], split: LLESplit) -> None:
    payload = {
        "schema_version": 1, "protocol": split.protocol, "seed": split.seed,
        "dataset_sha256": lle_dataset_digest(dataset),
        "partitions": {
            name: [_identifier(sample) for sample in values]
            for name, values in (("train", split.train), ("validation", split.validation), ("test", split.test))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_lle_split(path: Path, dataset: Sequence[LLESample]) -> LLESplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != lle_dataset_digest(dataset):
        raise ValueError("LLE split dataset digest does not match the loaded dataset")
    by_id = {_identifier(sample): sample for sample in dataset}
    partitions = payload.get("partitions", {})
    values = {}
    for name in ("train", "validation", "test"):
        ids = partitions.get(name)
        if not isinstance(ids, list) or not all(identifier in by_id for identifier in ids):
            raise ValueError(f"Invalid LLE split partition: {name}")
        values[name] = tuple(by_id[identifier] for identifier in ids)
    if set(sample.system_key for sample in values["train"]) & set(sample.system_key for sample in values["validation"]):
        raise ValueError("LLE train/validation system overlap")
    if set(sample.system_key for sample in values["train"]) & set(sample.system_key for sample in values["test"]):
        raise ValueError("LLE train/test system overlap")
    if set(sample.system_key for sample in values["validation"]) & set(sample.system_key for sample in values["test"]):
        raise ValueError("LLE validation/test system overlap")
    return LLESplit(**values, protocol=str(payload["protocol"]), seed=int(payload["seed"]))


@dataclass
class LLEBatch:
    molecules: Tensor
    temperature_k: Tensor
    pressure_kpa: Tensor
    source_phase: Tensor
    target_phase: Tensor
    mask: Tensor
    quality_weight: Tensor
    record_ids: tuple[str, ...]

    def to(self, device: torch.device) -> "LLEBatch":
        return LLEBatch(**{name: value.to(device) if isinstance(value, Tensor) else value for name, value in vars(self).items()})


class LLETensorDataset(Dataset[tuple[Tensor, float, float, tuple[float, ...], tuple[float, ...], float, str]]):
    """Expand both directed representations only after a raw LLE split is fixed."""

    def __init__(self, samples: Sequence[LLESample], feature_map: dict[str, np.ndarray]) -> None:
        self.rows = []
        for sample in samples:
            feature = torch.from_numpy(np.stack([feature_map[smiles] for smiles in sample.smiles]).astype(np.float32))
            self.rows.extend((
                (feature, sample.temperature_k, sample.pressure_kpa, sample.phase_alpha, sample.phase_beta, sample.quality_weight, sample.record_id),
                (feature, sample.temperature_k, sample.pressure_kpa, sample.phase_beta, sample.phase_alpha, sample.quality_weight, sample.record_id),
            ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate_lle(rows) -> LLEBatch:
    if not rows:
        raise ValueError("Cannot collate an empty LLE batch")
    batch_size, feature_dim = len(rows), rows[0][0].shape[-1]
    molecules = torch.zeros(batch_size, 3, feature_dim)
    source = torch.zeros(batch_size, 3)
    target = torch.zeros_like(source)
    mask = torch.zeros_like(source)
    for index, (feature, _, _, source_phase, target_phase, _, _) in enumerate(rows):
        count = feature.shape[0]
        molecules[index, :count] = feature
        source[index, :count] = torch.tensor(source_phase)
        target[index, :count] = torch.tensor(target_phase)
        mask[index, :count] = 1.0
    return LLEBatch(
        molecules=molecules,
        temperature_k=torch.tensor([[row[1]] for row in rows]),
        pressure_kpa=torch.tensor([[row[2]] for row in rows]),
        source_phase=source, target_phase=target, mask=mask,
        quality_weight=torch.tensor([[row[5]] for row in rows]),
        record_ids=tuple(row[6] for row in rows),
    )
