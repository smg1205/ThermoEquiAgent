"""Uni-Mol v2 molecular structure embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

class UniMolV2Encoder:
    """Encode unique SMILES with Uni-Mol v2 ``cls_repr`` vectors."""

    def __init__(
        self,
        cache_path: Path,
        batch_size: int = 16,
        model_size: str = "84m",
        use_cuda: bool | None = None,
        backend_factory: Callable[..., object] | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.model_size = model_size
        self.use_cuda = use_cuda
        self._backend_factory = backend_factory

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            model = str(cache["model"].item())
            model_size = str(cache["model_size"].item())
            if model != "unimolv2" or model_size != self.model_size:
                return {}
            smiles = cache["smiles"].astype(str).tolist()
            features = np.asarray(cache["features"], dtype=np.float32)
        if features.ndim != 2 or len(smiles) != len(features):
            raise ValueError(f"Invalid Uni-Mol cache: {self.cache_path}")
        return {smile: features[index] for index, smile in enumerate(smiles)}

    def _make_backend(self) -> object:
        if self._backend_factory is None:
            from unimol_tools import UniMolRepr

            factory: Callable[..., object] = UniMolRepr
        else:
            factory = self._backend_factory
        options = {
            "data_type": "molecule",
            "model_name": "unimolv2",
            "model_size": self.model_size,
            "batch_size": self.batch_size,
        }
        if self.use_cuda is not None:
            options["use_cuda"] = self.use_cuda
        return factory(**options)

    def _save_cache(self, features: dict[str, np.ndarray]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(features)
        matrix = np.stack([features[smile] for smile in ordered]).astype(np.float32)
        np.savez_compressed(
            self.cache_path,
            smiles=np.asarray(ordered),
            features=matrix,
            model=np.asarray("unimolv2"),
            model_size=np.asarray(self.model_size),
        )

    def encode(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        requested = sorted({value.strip() for value in smiles if value.strip()})
        if not requested:
            raise ValueError("At least one non-empty SMILES is required")
        cached = self._load_cache()
        missing = [value for value in requested if value not in cached]
        if missing:
            backend = self._make_backend()
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                result = backend.get_repr(chunk, return_atomic_reprs=False)
                # ``unimol_tools`` changed its public return type across releases:
                # older builds return {"cls_repr": ndarray}, newer ones return a
                # plain list of per-molecule ``cls_repr`` vectors.  Accept both so
                # the encoder is not pinned to one installed version.
                if isinstance(result, dict):
                    if "cls_repr" not in result:
                        raise TypeError(
                            "Uni-Mol v2 get_repr returned a dict without 'cls_repr'"
                        )
                    embeddings = np.asarray(result["cls_repr"], dtype=np.float32)
                elif isinstance(result, (list, tuple)):
                    embeddings = np.asarray(result, dtype=np.float32)
                    if embeddings.ndim == 1 and len(chunk) == 1:
                        embeddings = embeddings.reshape(1, -1)
                else:
                    raise TypeError(
                        "Uni-Mol v2 get_repr must return a dict containing 'cls_repr' "
                        f"or a list of vectors, got {type(result).__name__}"
                    )
                if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk):
                    raise ValueError(
                        f"Uni-Mol returned shape {embeddings.shape} for {len(chunk)} molecules"
                    )
                cached.update({smile: embeddings[index] for index, smile in enumerate(chunk)})
            self._save_cache(cached)
        return {smile: cached[smile] for smile in requested}
