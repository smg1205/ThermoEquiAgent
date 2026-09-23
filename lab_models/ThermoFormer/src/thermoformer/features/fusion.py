"""Fusion and split-aware preprocessing of complementary molecular views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .functional_groups import FunctionalGroupEncoder, functional_group_vocabulary_path
from .rdkit_descriptors import (
    LegacyFixedScaleRDKit2DEncoder,
    RDKit2DEncoder,
    RDKitDescriptorScaler,
    rdkit_descriptor_definition_path,
)
from .unimol_v2 import UniMolV2Encoder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridMolecularEncoder:
    """Concatenate physical, pretrained 3D, and functional-group features."""

    _BRANCH_ORDER = ("rdkit_2d", "unimol_v2", "functional_groups")

    def __init__(
        self,
        cache_path: Path,
        batch_size: int = 16,
        model_size: str = "84m",
        use_cuda: bool | None = None,
        backend_factory: Callable[..., object] | None = None,
        use_rdkit_descriptors: bool = True,
        use_unimol: bool = True,
        use_functional_groups: bool = True,
    ) -> None:
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.model_size = model_size
        self.use_cuda = use_cuda
        self._backend_factory = backend_factory
        enabled = {
            "rdkit_2d": use_rdkit_descriptors,
            "unimol_v2": use_unimol,
            "functional_groups": use_functional_groups,
        }
        self.enabled_branches = tuple(name for name in self._BRANCH_ORDER if enabled[name])
        if not self.enabled_branches:
            raise ValueError("Hybrid molecular representation requires at least one branch")
        self.feature_block_sizes: dict[str, int] = {}

    @property
    def _signature(self) -> str:
        return "+".join(
            {
                "rdkit_2d": "rdkit_raw24_v1",
                "unimol_v2": f"unimolv2_{self.model_size}",
                "functional_groups": FunctionalGroupEncoder._MODEL_SIZE,
            }[name]
            for name in self.enabled_branches
        )

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if str(cache["model"].item()) != "hybrid_molecular":
                return {}
            if str(cache["model_size"].item()) != self._signature:
                return {}
            branch_names = tuple(cache["branch_names"].astype(str).tolist())
            branch_dims = cache["branch_dims"].astype(int).tolist()
            if branch_names != self.enabled_branches:
                return {}
            smiles = cache["smiles"].astype(str).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
        self.feature_block_sizes = dict(zip(branch_names, branch_dims))
        if features.shape != (len(smiles), sum(branch_dims)):
            raise ValueError(f"Invalid hybrid molecular cache: {self.cache_path}")
        return {smile: features[index] for index, smile in enumerate(smiles)}

    def _branch_encoders(self) -> dict[str, object]:
        cache_root = self.cache_path.parent
        return {
            "rdkit_2d": RDKit2DEncoder(cache_root / "rdkit_2d_raw24_v1.npz"),
            "unimol_v2": UniMolV2Encoder(
                cache_root / f"unimolv2_{self.model_size}.npz",
                batch_size=self.batch_size,
                model_size=self.model_size,
                use_cuda=self.use_cuda,
                backend_factory=self._backend_factory,
            ),
            "functional_groups": FunctionalGroupEncoder(
                cache_root / "functional_groups_thermoformer_v1.npz"
            ),
        }

    def _save_cache(self, features: dict[str, np.ndarray]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(features)
        np.savez_compressed(
            self.cache_path,
            smiles=np.asarray(ordered),
            features=np.stack([features[smile] for smile in ordered]).astype(np.float32),
            branch_names=np.asarray(tuple(self.feature_block_sizes)),
            branch_dims=np.asarray(tuple(self.feature_block_sizes.values()), dtype=np.int64),
            model=np.asarray("hybrid_molecular"),
            model_size=np.asarray(self._signature),
        )

    def encode(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        requested = sorted({value.strip() for value in smiles if value.strip()})
        if not requested:
            raise ValueError("At least one non-empty SMILES is required")
        cached = self._load_cache()
        missing = [value for value in requested if value not in cached]
        if missing:
            encoders = self._branch_encoders()
            branch_features = {
                name: encoders[name].encode(missing) for name in self.enabled_branches
            }
            self.feature_block_sizes = {
                name: int(branch_features[name][missing[0]].shape[0])
                for name in self.enabled_branches
            }
            for smile in missing:
                cached[smile] = np.concatenate(
                    [branch_features[name][smile] for name in self.enabled_branches]
                ).astype(np.float32)
            self._save_cache(cached)
        return {smile: cached[smile] for smile in requested}


@dataclass(frozen=True)
class PreparedMolecularFeatures:
    values: dict[str, np.ndarray]
    view_dimensions: dict[str, int]
    metadata: dict[str, object]


def prepare_partition_features(
    encoder: object,
    all_smiles: Sequence[str],
    train_smiles: Sequence[str],
) -> PreparedMolecularFeatures:
    """Encode all molecules while fitting RDKit statistics on training molecules only."""

    if not hasattr(encoder, "encode"):
        raise TypeError("Molecular encoder must expose encode(smiles)")
    raw = encoder.encode(all_smiles)
    if not raw:
        raise ValueError("Molecular feature map is empty")
    total_dim = int(next(iter(raw.values())).shape[0])
    view_dimensions = {"rdkit_2d": 0, "unimol_v2": 0, "functional_groups": 0}
    if isinstance(encoder, HybridMolecularEncoder):
        view_dimensions.update(encoder.feature_block_sizes)
        block_order = encoder.enabled_branches
    elif isinstance(encoder, RDKit2DEncoder):
        view_dimensions["rdkit_2d"] = total_dim
        block_order = ("rdkit_2d",)
    elif isinstance(encoder, LegacyFixedScaleRDKit2DEncoder):
        view_dimensions["rdkit_2d"] = total_dim
        block_order = ("rdkit_2d",)
    elif isinstance(encoder, UniMolV2Encoder):
        view_dimensions["unimol_v2"] = total_dim
        block_order = ("unimol_v2",)
    elif isinstance(encoder, FunctionalGroupEncoder):
        view_dimensions["functional_groups"] = total_dim
        block_order = ("functional_groups",)
    else:
        raise TypeError(f"Unsupported molecular encoder type: {type(encoder).__name__}")

    values = {name: np.asarray(vector, dtype=np.float32).copy() for name, vector in raw.items()}
    metadata: dict[str, object] = {
        "view_dimensions": view_dimensions,
        "block_order": list(block_order),
    }
    source_paths: dict[str, Path] = {}
    if isinstance(encoder, HybridMolecularEncoder):
        source_paths = {
            "rdkit_2d": encoder.cache_path.parent / "rdkit_2d_raw24_v1.npz",
            "unimol_v2": encoder.cache_path.parent / f"unimolv2_{encoder.model_size}.npz",
            "functional_groups": (
                encoder.cache_path.parent / "functional_groups_thermoformer_v1.npz"
            ),
        }
    elif isinstance(encoder, RDKit2DEncoder):
        source_paths = {"rdkit_2d": encoder.cache_path}
    elif isinstance(encoder, LegacyFixedScaleRDKit2DEncoder):
        source_paths = {"rdkit_2d": encoder.cache_path}
    elif isinstance(encoder, UniMolV2Encoder):
        source_paths = {"unimol_v2": encoder.cache_path}
    elif isinstance(encoder, FunctionalGroupEncoder):
        source_paths = {"functional_groups": encoder.cache_path}
    metadata["source_cache_sha256"] = {
        name: _sha256(path)
        for name, path in source_paths.items()
        if name in block_order and path.is_file()
    }
    if view_dimensions["rdkit_2d"] and not isinstance(
        encoder, LegacyFixedScaleRDKit2DEncoder
    ):
        offset = sum(
            view_dimensions[name]
            for name in block_order[: block_order.index("rdkit_2d")]
        )
        dimension = view_dimensions["rdkit_2d"]
        raw_rdkit = {name: vector[offset : offset + dimension] for name, vector in raw.items()}
        descriptor_encoder = (
            encoder
            if isinstance(encoder, RDKit2DEncoder)
            else RDKit2DEncoder(encoder.cache_path.parent / "rdkit_2d_raw24_v1.npz")
        )
        scaler = RDKitDescriptorScaler.fit(
            raw_rdkit,
            train_smiles,
            descriptor_encoder.feature_names,
        )
        for name, vector in values.items():
            vector[offset : offset + dimension] = scaler.transform(
                raw[name][offset : offset + dimension]
            )
        metadata["rdkit_descriptor_definition_sha256"] = descriptor_encoder.definition_sha256
        metadata["rdkit_scaler"] = scaler.to_dict()
    elif isinstance(encoder, LegacyFixedScaleRDKit2DEncoder):
        metadata["rdkit_legacy_fixed_scales"] = {
            name: scale for name, scale in RDKit2DEncoder._DESCRIPTORS
        }
        metadata["rdkit_descriptor_definition_sha256"] = encoder.definition_sha256
    if view_dimensions["functional_groups"]:
        group_encoder = (
            encoder
            if isinstance(encoder, FunctionalGroupEncoder)
            else FunctionalGroupEncoder(
                encoder.cache_path.parent / "functional_groups_thermoformer_v1.npz"
            )
        )
        metadata["functional_group_vocabulary_sha256"] = group_encoder.vocabulary_sha256
        metadata["functional_group_names"] = list(group_encoder.feature_names)
        metadata["functional_group_presence"] = "deterministic counts > 0; BASE/OTHER in model"
    definition_payload = {
        "view_dimensions": view_dimensions,
        "block_order": list(block_order),
        "rdkit_descriptor_definition_sha256": metadata.get(
            "rdkit_descriptor_definition_sha256"
        ),
        "functional_group_vocabulary_sha256": metadata.get(
            "functional_group_vocabulary_sha256"
        ),
    }
    metadata["feature_definition_sha256"] = hashlib.sha256(
        json.dumps(definition_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not all(np.isfinite(vector).all() for vector in values.values()):
        raise ValueError("Prepared molecular features contain NaN or Inf")
    return PreparedMolecularFeatures(values, view_dimensions, metadata)


def encoder_cache_filename(config: Any) -> str:
    """Return a cache identity that changes with every representation branch."""

    if config.representation == "unimol_v2":
        return f"unimolv2_{config.model_size}.npz"
    if config.representation == "rdkit_2d":
        return "rdkit_2d_raw24_v1.npz"
    if config.representation == "rdkit_2d_legacy_fixed":
        return "rdkit_2d_scaled24_v1.npz"
    if config.representation == "functional_groups":
        return "functional_groups_thermoformer_v1.npz"
    if config.representation not in ("hybrid", "multiview"):
        raise ValueError(f"Unsupported molecular representation: {config.representation}")
    branches = []
    if config.use_rdkit_descriptors:
        branches.append("rdkit")
    if config.use_unimol:
        branches.append(f"unimol_{config.model_size}")
    if config.use_functional_groups:
        branches.append("functional")
    return "hybrid_" + "_".join(branches) + "_v1.npz"


def feature_subset_sha256(feature_map: dict[str, np.ndarray]) -> str:
    """Hash the exact molecular feature subset consumed by a run."""
    digest = hashlib.sha256()
    for smiles in sorted(feature_map):
        array = np.ascontiguousarray(feature_map[smiles], dtype=np.float32)
        digest.update(smiles.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_molecular_encoder(
    config: Any,
    cache_path: Path,
    use_cuda: bool | None = None,
    backend_factory: Callable[..., object] | None = None,
) -> object:
    """Construct the configured molecular encoder behind one runner-facing seam."""

    if config.representation in ("hybrid", "multiview"):
        return HybridMolecularEncoder(
            cache_path,
            batch_size=config.batch_size,
            model_size=config.model_size,
            use_cuda=use_cuda,
            backend_factory=backend_factory,
            use_rdkit_descriptors=config.use_rdkit_descriptors,
            use_unimol=config.use_unimol,
            use_functional_groups=config.use_functional_groups,
        )
    if config.representation == "unimol_v2":
        return UniMolV2Encoder(
            cache_path,
            batch_size=config.batch_size,
            model_size=config.model_size,
            use_cuda=use_cuda,
            backend_factory=backend_factory,
        )
    if config.representation == "rdkit_2d":
        return RDKit2DEncoder(cache_path)
    if config.representation == "rdkit_2d_legacy_fixed":
        return LegacyFixedScaleRDKit2DEncoder(cache_path)
    if config.representation == "functional_groups":
        return FunctionalGroupEncoder(cache_path)
    raise ValueError(f"Unsupported molecular representation: {config.representation}")
