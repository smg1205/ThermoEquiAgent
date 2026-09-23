"""RDKit physicochemical and topological molecular descriptors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def rdkit_descriptor_definition_path() -> Path:
    return PROJECT_ROOT / 'datasets/molecular_features/rdkit_descriptors.json'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class RDKit2DEncoder:
    """Deterministic raw values for the frozen ThermoFormer RDKit-24 set."""

    _DESCRIPTORS = (
        ("MolWt", 500.0),
        ("MolLogP", 10.0),
        ("TPSA", 200.0),
        ("NumHDonors", 10.0),
        ("NumHAcceptors", 20.0),
        ("NumRotatableBonds", 20.0),
        ("RingCount", 10.0),
        ("NumAromaticRings", 10.0),
        ("NumAliphaticRings", 10.0),
        ("FractionCSP3", 1.0),
        ("HeavyAtomCount", 100.0),
        ("NHOHCount", 20.0),
        ("NOCount", 20.0),
        ("NumHeteroatoms", 50.0),
        ("NumValenceElectrons", 500.0),
        ("MolMR", 200.0),
        ("MaxPartialCharge", 2.0),
        ("MinPartialCharge", 2.0),
        ("NumRadicalElectrons", 5.0),
        ("BertzCT", 2000.0),
        ("BalabanJ", 10.0),
        ("Chi0v", 50.0),
        ("Kappa1", 50.0),
        ("LabuteASA", 500.0),
    )

    _MODEL_SIZE = "raw24_v1"

    def __init__(self, cache_path: Path, definition_path: Path | None = None) -> None:
        self.cache_path = cache_path
        self.definition_path = definition_path or rdkit_descriptor_definition_path()
        payload = json.loads(self.definition_path.read_text(encoding="utf-8"))
        self.feature_names = tuple(str(name) for name in payload["descriptors"])
        expected = tuple(name for name, _ in self._DESCRIPTORS)
        if self.feature_names != expected:
            raise ValueError("RDKit descriptor asset does not match the frozen RDKit-24 set")
        self.definition_sha256 = _sha256(self.definition_path)

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if str(cache["model"].item()) != "rdkit_2d":
                return {}
            if str(cache["model_size"].item()) != self._MODEL_SIZE:
                return {}
            if str(cache["definition_sha256"].item()) != self.definition_sha256:
                return {}
            smiles = cache["smiles"].astype(str).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
        if features.ndim != 2 or len(smiles) != len(features):
            raise ValueError(f"Invalid RDKit descriptor cache: {self.cache_path}")
        return {smile: features[index] for index, smile in enumerate(smiles)}

    @classmethod
    def _describe(cls, smiles: str) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")
        values = []
        for name, _ in cls._DESCRIPTORS:
            descriptor = getattr(Descriptors, name)
            value = float(descriptor(molecule))
            values.append(value if np.isfinite(value) else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _save_cache(self, features: dict[str, np.ndarray]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(features)
        np.savez_compressed(
            self.cache_path,
            smiles=np.asarray(ordered),
            features=np.stack([features[smile] for smile in ordered]).astype(np.float32),
            model=np.asarray("rdkit_2d"),
            model_size=np.asarray(self._MODEL_SIZE),
            descriptor_names=np.asarray(self.feature_names),
            definition_sha256=np.asarray(self.definition_sha256),
        )

    def encode(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        requested = sorted({value.strip() for value in smiles if value.strip()})
        if not requested:
            raise ValueError("At least one non-empty SMILES is required")
        cached = self._load_cache()
        missing = [value for value in requested if value not in cached]
        for value in missing:
            cached[value] = self._describe(value)
        if missing:
            self._save_cache(cached)
        return {smile: cached[smile] for smile in requested}


class LegacyFixedScaleRDKit2DEncoder:
    """Exact pre-multiview A1 descriptors divided by frozen hand scales."""

    _MODEL_SIZE = "scaled24_v1"

    def __init__(self, cache_path: Path, definition_path: Path | None = None) -> None:
        self.cache_path = cache_path
        self.definition_path = definition_path or rdkit_descriptor_definition_path()
        payload = json.loads(self.definition_path.read_text(encoding="utf-8"))
        self.feature_names = tuple(str(name) for name in payload["descriptors"])
        expected = tuple(name for name, _ in RDKit2DEncoder._DESCRIPTORS)
        if self.feature_names != expected:
            raise ValueError("Legacy RDKit asset does not match the frozen RDKit-24 set")
        self.definition_sha256 = _sha256(self.definition_path)

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if str(cache["model"].item()) != "rdkit_2d":
                return {}
            if str(cache["model_size"].item()) != self._MODEL_SIZE:
                return {}
            smiles = cache["smiles"].astype(str).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
        if features.ndim != 2 or len(smiles) != len(features):
            raise ValueError(f"Invalid legacy RDKit descriptor cache: {self.cache_path}")
        return {smile: features[index] for index, smile in enumerate(smiles)}

    @staticmethod
    def _describe(smiles: str) -> np.ndarray:
        raw = RDKit2DEncoder._describe(smiles)
        scales = np.asarray(
            [scale for _, scale in RDKit2DEncoder._DESCRIPTORS], dtype=np.float32
        )
        return raw / scales

    def _save_cache(self, features: dict[str, np.ndarray]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(features)
        np.savez_compressed(
            self.cache_path,
            smiles=np.asarray(ordered),
            features=np.stack([features[smile] for smile in ordered]).astype(np.float32),
            model=np.asarray("rdkit_2d"),
            model_size=np.asarray(self._MODEL_SIZE),
        )

    def encode(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        requested = sorted({value.strip() for value in smiles if value.strip()})
        if not requested:
            raise ValueError("At least one non-empty SMILES is required")
        cached = self._load_cache()
        missing = [value for value in requested if value not in cached]
        for value in missing:
            cached[value] = self._describe(value)
        if missing:
            self._save_cache(cached)
        return {smile: cached[smile] for smile in requested}


@dataclass(frozen=True)
class RDKitDescriptorScaler:
    """Train-partition-only standardization for the frozen descriptor vector."""

    mean: np.ndarray
    std: np.ndarray
    descriptor_names: tuple[str, ...]
    fit_smiles: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        raw_features: dict[str, np.ndarray],
        train_smiles: Sequence[str],
        descriptor_names: Sequence[str],
    ) -> "RDKitDescriptorScaler":
        fitted = tuple(sorted({value for value in train_smiles if value in raw_features}))
        if not fitted:
            raise ValueError("RDKit scaler requires at least one training molecule")
        matrix = np.stack([raw_features[value] for value in fitted]).astype(np.float64)
        if matrix.shape[1] != len(descriptor_names) or not np.isfinite(matrix).all():
            raise ValueError("RDKit scaler received invalid descriptor values")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            descriptor_names=tuple(descriptor_names),
            fit_smiles=fitted,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.descriptor_names):
            raise ValueError("RDKit descriptor dimension does not match the fitted scaler")
        transformed = (array - self.mean) / self.std
        if not np.isfinite(transformed).all():
            raise ValueError("RDKit standardization produced a non-finite value")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "descriptor_names": list(self.descriptor_names),
            "fit_smiles": list(self.fit_smiles),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}
