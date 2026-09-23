"""Stable chemical identities and reusable experiment split artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem, RDLogger

from .loading import VLESample


SCHEMA_VERSION = 1
PROTOCOL_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


def validate_protocol_name(value: object) -> str:
    """Validate the protocol slug before it can participate in a filesystem path."""
    if not isinstance(value, str) or not PROTOCOL_PATTERN.fullmatch(value):
        raise ValueError(
            "Split protocol must be a lowercase filesystem-safe slug containing only "
            "letters, digits, '.', '_' or '-'"
        )
    return value


@lru_cache(maxsize=None)
def canonical_smiles(smiles: str) -> str:
    """Return an isomeric canonical SMILES, with a deterministic test-data fallback."""
    RDLogger.DisableLog("rdApp.error")
    try:
        molecule = Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog("rdApp.error")
    if molecule is None:
        normalized = smiles.strip()
        if not normalized:
            raise ValueError("Molecular identity cannot be empty")
        return normalized
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_order(sample: VLESample) -> list[tuple[str, int]]:
    return sorted(
        ((canonical_smiles(smiles), index) for index, smiles in enumerate(sample.smiles)),
        key=lambda item: (item[0], item[1]),
    )


def _digest(payload: object, length: int | None = None) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value = hashlib.sha256(encoded).hexdigest()
    return value if length is None else value[:length]


@lru_cache(maxsize=None)
def component_id(smiles: str) -> str:
    return f"mol_{_digest(canonical_smiles(smiles), 16)}"


@lru_cache(maxsize=None)
def system_id(sample: VLESample) -> str:
    components = [smiles for smiles, _ in _canonical_order(sample)]
    return f"{len(components)}c_{_digest(components, 20)}"


@lru_cache(maxsize=None)
def sample_id(sample: VLESample) -> str:
    order = _canonical_order(sample)
    payload = {
        "components": [smiles for smiles, _ in order],
        "temperature_k": round(sample.temperature_k, 7),
        "pressure_kpa": round(sample.pressure_kpa, 7),
        "x": [round(sample.liquid_composition[index], 9) for _, index in order],
        "y": [round(sample.vapor_composition[index], 9) for _, index in order],
    }
    return f"state_{_digest(payload, 24)}"


def dataset_digest(samples: Sequence[VLESample]) -> str:
    identifiers = sorted(sample_id(sample) for sample in samples)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Dataset contains duplicate stable sample identities")
    return _digest(identifiers)


@dataclass(frozen=True)
class DatasetPartitions:
    train: tuple[VLESample, ...]
    validation: tuple[VLESample, ...]
    test: tuple[VLESample, ...]
    protocol: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _partition_ids(split: DatasetPartitions) -> dict[str, list[str]]:
    return {
        "train": [sample_id(sample) for sample in split.train],
        "validation": [sample_id(sample) for sample in split.validation],
        "test": [sample_id(sample) for sample in split.test],
    }


def _validate_partition_ids(partitions: dict[str, list[str]]) -> None:
    expected = {"train", "validation", "test"}
    if set(partitions) != expected:
        raise ValueError("Split partitions must contain train, validation, and test")
    sets = {name: set(values) for name, values in partitions.items()}
    for name, values in partitions.items():
        if len(values) != len(sets[name]):
            raise ValueError(f"Split partition '{name}' contains duplicate sample IDs")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Split partitions overlap")


def save_split_assignment(
    path: Path,
    dataset: Sequence[VLESample],
    split: DatasetPartitions,
) -> None:
    """Atomically write the exact row assignment used by an experiment."""
    protocol = validate_protocol_name(split.protocol)
    partitions = _partition_ids(split)
    _validate_partition_ids(partitions)
    known = {sample_id(sample) for sample in dataset}
    assigned = set().union(*(set(values) for values in partitions.values()))
    unknown = assigned - known
    if unknown:
        raise ValueError(f"Split contains {len(unknown)} rows absent from the dataset")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "seed": split.seed,
        "dataset_sha256": dataset_digest(dataset),
        "dataset_rows": len(dataset),
        "assigned_rows": len(assigned),
        "metadata": split.metadata,
        "partitions": partitions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_split_assignment(
    path: Path,
    dataset: Sequence[VLESample],
) -> DatasetPartitions:
    """Load a split only if it matches the exact audited dataset."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported split schema version")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or any(
        not isinstance(values, list) or not all(isinstance(value, str) for value in values)
        for values in partitions.values()
    ):
        raise ValueError("Malformed split partitions")
    _validate_partition_ids(partitions)
    actual_digest = dataset_digest(dataset)
    if payload.get("dataset_sha256") != actual_digest:
        raise ValueError("Split dataset digest does not match the loaded dataset")
    by_id = {sample_id(sample): sample for sample in dataset}
    unknown = set().union(*(set(values) for values in partitions.values())) - set(by_id)
    if unknown:
        raise ValueError(f"Split references {len(unknown)} unknown sample IDs")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Split metadata must be a JSON object")
    protocol = validate_protocol_name(payload.get("protocol"))
    return DatasetPartitions(
        train=tuple(by_id[value] for value in partitions["train"]),
        validation=tuple(by_id[value] for value in partitions["validation"]),
        test=tuple(by_id[value] for value in partitions["test"]),
        protocol=protocol,
        seed=int(payload.get("seed")),
        metadata=metadata,
    )
