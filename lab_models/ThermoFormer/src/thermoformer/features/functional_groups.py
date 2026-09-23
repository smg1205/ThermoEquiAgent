"""SMARTS-based functional-group count features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def functional_group_vocabulary_path() -> Path:
    return PROJECT_ROOT / 'datasets/molecular_features/functional_groups.json'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class FunctionalGroupEncoder:
    """Audited SMARTS occurrence counts with a versioned, immutable vocabulary."""

    _MODEL_SIZE = "thermoformer_functional_groups_v1"

    def __init__(self, cache_path: Path, vocabulary_path: Path | None = None) -> None:
        from rdkit import Chem

        self.cache_path = cache_path
        self.vocabulary_path = vocabulary_path or functional_group_vocabulary_path()
        payload = json.loads(self.vocabulary_path.read_text(encoding="utf-8"))
        groups = payload.get("groups", [])
        self.feature_names = tuple(str(group["name"]) for group in groups)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Functional-group vocabulary names must be non-empty and unique")
        self.smarts = tuple(str(group["smarts"]) for group in groups)
        self.patterns = tuple(Chem.MolFromSmarts(value) for value in self.smarts)
        invalid = [name for name, pattern in zip(self.feature_names, self.patterns) if pattern is None]
        if invalid:
            raise ValueError(f"Invalid functional-group SMARTS: {', '.join(invalid)}")
        self.vocabulary_sha256 = _sha256(self.vocabulary_path)

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if str(cache["model"].item()) != "functional_groups":
                return {}
            if str(cache["model_size"].item()) != self._MODEL_SIZE:
                return {}
            if str(cache["vocabulary_sha256"].item()) != self.vocabulary_sha256:
                return {}
            names = tuple(cache["feature_names"].astype(str).tolist())
            if names != self.feature_names:
                return {}
            smiles = cache["smiles"].astype(str).tolist()
            features = np.asarray(cache["counts"], dtype=np.float32)
            presence = np.asarray(cache["presence"], dtype=np.uint8)
        expected = (len(smiles), len(self.feature_names))
        if features.shape != expected or presence.shape != expected:
            raise ValueError(f"Invalid functional-group cache: {self.cache_path}")
        if not np.array_equal(presence, features > 0.0):
            raise ValueError("Functional-group count and presence cache entries disagree")
        return {smile: features[index] for index, smile in enumerate(smiles)}

    def _describe(self, smiles: str) -> np.ndarray:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")
        values = [
            float(len(molecule.GetSubstructMatches(pattern, uniquify=True)))
            for pattern in self.patterns
        ]
        return np.asarray(values, dtype=np.float32)

    def _save_cache(self, features: dict[str, np.ndarray]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(features)
        counts = np.stack([features[smile] for smile in ordered]).astype(np.float32)
        np.savez_compressed(
            self.cache_path,
            smiles=np.asarray(ordered),
            features=counts,
            counts=counts,
            presence=(counts > 0.0).astype(np.uint8),
            feature_names=np.asarray(self.feature_names),
            smarts=np.asarray(self.smarts),
            vocabulary_sha256=np.asarray(self.vocabulary_sha256),
            model=np.asarray("functional_groups"),
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
