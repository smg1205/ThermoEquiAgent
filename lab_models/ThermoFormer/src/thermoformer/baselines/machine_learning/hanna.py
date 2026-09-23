"""Official pretrained HANNA ensemble adapter for the common VLE solver."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from torch import Tensor, nn
from transformers import AutoModel

from ...data.loading import VLESample
from ...data.splitting import canonical_smiles


HANNA_SOURCE_REVISION = "6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365"
HANNA_EMBEDDING_DIMENSION = 384


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OfficialHANNAAssets:
    """Pinned files distributed by the MIT-licensed official HANNA repository."""

    root: Path

    @classmethod
    def default(cls, project_root: Path) -> "OfficialHANNAAssets":
        return cls(project_root / 'models/baselines/hanna_official')

    @property
    def source_path(self) -> Path:
        return self.root / "models" / "HANNA" / "HANNA.py"

    @property
    def chemberta_path(self) -> Path:
        return self.root / "models" / "ChemBERTa"

    @property
    def ensemble_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.root / "models" / "HANNA" / "ensemble" / f"HANNA_parameters_binary{index}.pt"
            for index in range(10)
        )

    @property
    def temperature_scaler_path(self) -> Path:
        return self.root / "utils" / "scalers" / "temperature_scaler.pkl"

    @property
    def embedding_scaler_path(self) -> Path:
        return self.root / "utils" / "scalers" / "bert_scaler.pkl"

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.root / "LICENSE",
            self.source_path,
            self.chemberta_path / "config.json",
            self.chemberta_path / "model.safetensors",
            self.chemberta_path / "vocab.json",
            self.temperature_scaler_path,
            self.embedding_scaler_path,
            *self.ensemble_paths,
        )

    def validate(self) -> None:
        missing = [str(path) for path in self.required_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Official HANNA assets are incomplete: " + ", ".join(missing))

    def hashes(self, project_root: Path) -> dict[str, str]:
        self.validate()
        return {
            path.resolve().relative_to(project_root.resolve()).as_posix(): _binary_sha256(path)
            for path in self.required_files()
        }


@lru_cache(maxsize=1)
def _load_official_source(path: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location("thermoformer_vendor_hanna", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load official HANNA source from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@dataclass
class HANNAModelOutputs:
    log_gamma: Tensor
    log_psat: Tensor
    attention_bias: None = None
    attention_bias_penalty: None = None


class HANNAThermodynamicAdapter(nn.Module):
    """Expose official HANNA log-gamma predictions to the shared VLE solver."""

    def __init__(self, assets: OfficialHANNAAssets, device: torch.device) -> None:
        super().__init__()
        assets.validate()
        official = _load_official_source(str(assets.source_path.resolve()))
        self.ensemble = official.HANNA_Ensemble_Multicomponent(
            [str(path) for path in assets.ensemble_paths],
            device=device,
        )
        self.ensemble.requires_grad_(False)
        temperature_scaler = pd.read_pickle(assets.temperature_scaler_path)
        self.register_buffer(
            "temperature_mean",
            torch.as_tensor(temperature_scaler.mean_, dtype=torch.float32).reshape(1, 1),
        )
        self.register_buffer(
            "temperature_scale",
            torch.as_tensor(temperature_scaler.scale_, dtype=torch.float32).reshape(1, 1),
        )
        self.to(device)
        self.eval()

    def forward(
        self,
        molecules: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> HANNAModelOutputs:
        del pressure_kpa
        if molecules.shape[-1] != HANNA_EMBEDDING_DIMENSION + 2:
            raise ValueError("HANNA molecule tensor must contain 384 embeddings and two Psat coefficients")
        embeddings = molecules[..., :HANNA_EMBEDDING_DIMENSION]
        intercept = molecules[..., HANNA_EMBEDDING_DIMENSION]
        inverse_temperature = molecules[..., HANNA_EMBEDDING_DIMENSION + 1]
        scaled_temperature = (temperature_k - self.temperature_mean) / self.temperature_scale
        independent_x = x[:, :-1]
        log_gamma, _ = self.ensemble(scaled_temperature, independent_x, embeddings)
        log_psat = intercept + inverse_temperature / temperature_k
        return HANNAModelOutputs(log_gamma=log_gamma * mask, log_psat=log_psat * mask)


class OfficialHANNAInference:
    """Load official molecular embeddings once and prepare solver molecule tensors."""

    def __init__(self, assets: OfficialHANNAAssets, device: torch.device) -> None:
        assets.validate()
        self.assets = assets
        self.device = device
        self.official = _load_official_source(str(assets.source_path.resolve()))
        self.chemberta = AutoModel.from_pretrained(
            str(assets.chemberta_path), local_files_only=True
        ).to(device)
        self.chemberta.requires_grad_(False)
        self.chemberta.eval()
        tokenizer = Tokenizer(
            WordLevel.from_file(
                str(assets.chemberta_path / "vocab.json"), unk_token="[UNK]"
            )
        )
        tokenizer.pre_tokenizer = Split(
            pattern=Regex(r"\[(.*?)\]|Br|Cl|."), behavior="isolated"
        )
        self.tokenizer = tokenizer
        self.embedding_scaler = pd.read_pickle(assets.embedding_scaler_path)
        self.model = HANNAThermodynamicAdapter(assets, device)
        self._embedding_cache: dict[str, Tensor] = {}

    def embedding(self, smiles: str) -> Tensor:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if canonical not in self._embedding_cache:
            raw = self.official.get_smiles_embedding(
                canonical,
                self.tokenizer,
                self.chemberta,
                self.device,
                max_length=512,
            )
            scaled = self.embedding_scaler.transform(raw)
            self._embedding_cache[canonical] = torch.as_tensor(
                scaled[0], dtype=torch.float32, device=self.device
            )
        return self._embedding_cache[canonical]

    def molecule_tensor(
        self,
        rows: Sequence[VLESample],
        vapor_pressure: Mapping[str, object],
    ) -> Tensor:
        values = []
        for row in rows:
            components = []
            for smiles in row.smiles:
                fitted = vapor_pressure[canonical_smiles(smiles)]
                coefficients = torch.tensor(
                    [fitted.intercept, fitted.inverse_temperature],
                    dtype=torch.float32,
                    device=self.device,
                )
                components.append(torch.cat([self.embedding(smiles), coefficients]))
            values.append(torch.stack(components))
        return torch.stack(values)


class OfficialHANNALogGammaPredictor:
    """Expose the frozen official ensemble as a statewise log-gamma predictor."""

    def __init__(self, assets: OfficialHANNAAssets, device: torch.device) -> None:
        self.inference = OfficialHANNAInference(assets, device)
        self.device = device

    def log_gamma(
        self,
        smiles: Sequence[str],
        temperature_k: float,
        composition: Sequence[float],
    ) -> Tensor:
        if len(smiles) not in (2, 3) or len(smiles) != len(composition):
            raise ValueError("HANNA requires a complete binary or ternary state")
        embeddings = torch.stack(
            [self.inference.embedding(value) for value in smiles]
        ).unsqueeze(0)
        temperature = torch.tensor(
            [[temperature_k]], dtype=torch.float32, device=self.device
        )
        scaled_temperature = (
            temperature - self.inference.model.temperature_mean
        ) / self.inference.model.temperature_scale
        independent_x = torch.tensor(
            [list(composition[:-1])], dtype=torch.float32, device=self.device
        )
        with torch.enable_grad():
            log_gamma, _ = self.inference.model.ensemble(
                scaled_temperature, independent_x, embeddings
            )
        return log_gamma[0].detach().cpu()

    def log_gamma_batch(self, samples: Sequence[VLESample], batch_size: int = 256) -> list[Tensor]:
        """Evaluate registered states in homogeneous binary/ternary batches."""
        outputs: list[Tensor | None] = [None] * len(samples)
        for count in (2, 3):
            indices = [index for index, sample in enumerate(samples) if sample.component_count == count]
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                batch = [samples[index] for index in selected]
                embeddings = torch.stack([
                    torch.stack([self.inference.embedding(value) for value in sample.smiles])
                    for sample in batch
                ])
                temperature = torch.tensor(
                    [[sample.temperature_k] for sample in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                scaled_temperature = (
                    temperature - self.inference.model.temperature_mean
                ) / self.inference.model.temperature_scale
                independent_x = torch.tensor(
                    [list(sample.liquid_composition[:-1]) for sample in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.enable_grad():
                    values, _ = self.inference.model.ensemble(
                        scaled_temperature, independent_x, embeddings
                    )
                for index, value in zip(selected, values.detach().cpu()):
                    outputs[index] = value
        if any(value is None for value in outputs):
            raise RuntimeError("HANNA batch prediction missed a registered state")
        return [value for value in outputs if value is not None]
