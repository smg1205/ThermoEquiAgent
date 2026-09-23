"""Lazy adapter for the official experimental fine-tuned TeNNet-SAC package."""

from __future__ import annotations

import importlib.metadata
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor


TENNETSAC_PACKAGE_VERSION = "0.1.10"
TENNETSAC_WHEEL_SHA256 = "0ec0e2724273730e4fd8dd70a2fc1c170c092f098462b1065eb20cf5110a05e5"
TENNETSAC_WHEEL_FILENAME = "tennetsac-0.1.10-py3-none-any.whl"
TENNETSAC_SOURCE_REVISION = "2367e89c87c335c4fd27ff7fa22c87f660f459ba"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_tennetsac_wheel(cache_root: Path) -> dict[str, str]:
    """Verify the exact published wheel used to provision the runtime package."""
    path = cache_root / TENNETSAC_WHEEL_FILENAME
    if not path.is_file():
        raise RuntimeError(
            f"Verified TeNNet-SAC wheel is missing: {path}. "
            "Download tennetsac==0.1.10 without dependencies before smoke/formal evaluation."
        )
    digest = _file_sha256(path)
    if digest != TENNETSAC_WHEEL_SHA256:
        raise RuntimeError("TeNNet-SAC wheel SHA-256 does not match the published package asset")
    return {"filename": path.name, "sha256": digest}


@lru_cache(maxsize=1)
def installed_tennetsac_asset_audit() -> dict[str, object]:
    """Hash every shipped TeNNet head and the two external encoder snapshots."""
    distribution = importlib.metadata.distribution("tennetsac")
    package_root = Path(distribution.locate_file("tennetsac"))
    checkpoints = sorted(package_root.joinpath("ckpt_files").rglob("*.ckpt"))
    if len(checkpoints) != 13:
        raise RuntimeError("TeNNet-SAC package must contain base/profile/geometry and ten tuned heads")
    audit: dict[str, object] = {
        "package_version": importlib.metadata.version("tennetsac"),
        "package_source": {
            path.relative_to(package_root).as_posix(): _file_sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        },
        "package_checkpoints": {
            path.relative_to(package_root).as_posix(): _file_sha256(path) for path in checkpoints
        },
    }
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    repositories = {
        "chemberta": "models--DeepChem--ChemBERTa-77M-MLM",
        "smi_ted": "models--ibm--materials.smi-ted",
    }
    encoder_assets = {}
    for name, folder in repositories.items():
        root = hub / folder
        reference = root / "refs" / "main"
        if not reference.is_file():
            raise RuntimeError(f"TeNNet-SAC external encoder is not materialized: {name}")
        revision = reference.read_text(encoding="utf-8").strip()
        snapshot = root / "snapshots" / revision
        files = sorted(
            path for path in snapshot.rglob("*")
            if path.is_file() and path.stat().st_size > 0
        )
        if not files:
            raise RuntimeError(f"TeNNet-SAC external encoder snapshot is empty: {name}")
        encoder_assets[name] = {
            "revision": revision,
            "files": {
                path.relative_to(snapshot).as_posix(): _file_sha256(path) for path in files
            },
        }
    audit["external_encoders"] = encoder_assets
    return audit


@lru_cache(maxsize=1)
def _official_predictors() -> tuple[Callable[..., object], Callable[..., object]]:
    installed = importlib.metadata.version("tennetsac")
    if installed != TENNETSAC_PACKAGE_VERSION:
        raise RuntimeError(
            f"TeNNet-SAC requires tennetsac=={TENNETSAC_PACKAGE_VERSION}; found {installed}"
        )
    from tennetsac import binary_lng, multi_lng
    import tennetsac.core as core
    if not getattr(core.sigma_profile_wrapper, "cache_info", None):
        # The official API recomputes two frozen language-model profiles at every
        # solver iteration. Memoization changes no numerical result and is essential
        # for repeated VLE root evaluations of the same molecules.
        core.sigma_profile_wrapper = lru_cache(maxsize=4096)(core.sigma_profile_wrapper)
    return binary_lng, multi_lng


class TeNNetSACPredictor:
    """Dispatch binary and multicomponent rows to the unchanged tuned public APIs."""

    def __init__(
        self,
        binary_predictor: Callable[..., object] | None = None,
        multi_predictor: Callable[..., object] | None = None,
    ) -> None:
        if (binary_predictor is None) != (multi_predictor is None):
            raise ValueError("Both TeNNet-SAC predictors must be supplied together")
        if binary_predictor is None:
            binary_predictor, multi_predictor = _official_predictors()
        self.binary_predictor = binary_predictor
        self.multi_predictor = multi_predictor

    @staticmethod
    def _validate(smiles: Sequence[str], composition: Sequence[float]) -> None:
        if len(smiles) not in (2, 3) or len(smiles) != len(composition):
            raise ValueError("TeNNet-SAC baseline supports complete binary or ternary states")
        if any(Chem.MolFromSmiles(value) is None for value in smiles):
            raise ValueError("TeNNet-SAC received an invalid SMILES")
        values = np.asarray(composition, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ValueError("TeNNet-SAC composition must be finite, nonnegative and normalized")

    def log_gamma(
        self,
        smiles: Sequence[str],
        temperature_k: float,
        composition: Sequence[float],
    ) -> Tensor:
        self._validate(smiles, composition)
        if not np.isfinite(temperature_k) or temperature_k <= 0.0:
            raise ValueError("TeNNet-SAC temperature must be positive")
        if len(smiles) == 2:
            first, second = self.binary_predictor(
                list(smiles), float(temperature_k), [float(composition[0])], version="tuned"
            )
            values = [first[0], second[0]]
        else:
            values = self.multi_predictor(
                list(smiles), float(temperature_k), list(map(float, composition)), version="tuned"
            )
        result = torch.as_tensor(values, dtype=torch.float32)
        if result.shape != (len(smiles),) or not torch.isfinite(result).all():
            raise RuntimeError("TeNNet-SAC returned invalid log activity coefficients")
        return result

class CudaTeNNetSACPredictor(TeNNetSACPredictor):
    """Numerically equivalent tuned-ensemble inference with segment heads on CUDA."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        if device.type != "cuda":
            raise ValueError("CudaTeNNetSACPredictor requires a CUDA device")
        import tennetsac.core as core
        self.core = core
        if not getattr(core.sigma_profile_wrapper, "cache_info", None):
            core.sigma_profile_wrapper = lru_cache(maxsize=4096)(core.sigma_profile_wrapper)
        self.models = [model.to(device).eval() for model in core.Gamma_finetuned_models]
        self.device = device
        self.combinatorial = core.calc_ln_gamma.__globals__["compute_SG_combinatorial_term"]

    def _segment_activity(self, sigma: Tensor, temperature_k: float) -> Tensor:
        values = []
        for model in self.models:
            local_sigma = sigma.to(self.device).clone().detach().requires_grad_(True)
            temperature = torch.tensor([temperature_k], dtype=torch.float32, device=self.device)
            with torch.enable_grad():
                _, segment = model(local_sigma, temperature)
            values.append(segment)
        return torch.stack(values).mean(dim=0)

    def log_gamma(
        self,
        smiles: Sequence[str],
        temperature_k: float,
        composition: Sequence[float],
    ) -> Tensor:
        self._validate(smiles, composition)
        profiles, areas, volumes = [], [], []
        for value in smiles:
            sigma, area, volume = self.core.sigma_profile_wrapper(value)
            profiles.append(sigma.to(self.device))
            areas.append(float(area))
            volumes.append(float(volume))
        mixture = sum(float(x) * sigma for x, sigma in zip(composition, profiles))
        sigmas = torch.cat([*profiles, mixture], dim=0)
        temperatures = torch.full(
            (len(smiles) + 1,), float(temperature_k), dtype=torch.float32, device=self.device
        )
        ensemble = []
        for model in self.models:
            local = sigmas.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                _, segment = model(local, temperatures)
            ensemble.append(segment)
        segments = torch.stack(ensemble).mean(dim=0)
        pure, mixed = segments[:-1], segments[-1]
        combinatorial = self.combinatorial(list(composition), areas, volumes)
        residual = []
        for sigma, area, pure_value in zip(profiles, areas, pure):
            normalized = sigma / area
            residual.append(area / 5.8447 * torch.sum(normalized * (mixed - pure_value)))
        result = torch.as_tensor(combinatorial, dtype=torch.float32, device=self.device)
        result = result + torch.stack(residual)
        if not torch.isfinite(result).all():
            raise RuntimeError("CUDA TeNNet-SAC returned invalid log activity coefficients")
        return result.detach().cpu()

    def log_gamma_batch(self, samples, batch_size: int = 256) -> list[Tensor]:
        """Batch pure and mixture segment evaluations on CUDA."""
        outputs: list[Tensor] = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            records = []
            sigma_rows = []
            temperatures = []
            for sample in batch:
                profiles, areas, volumes = [], [], []
                for value in sample.smiles:
                    sigma, area, volume = self.core.sigma_profile_wrapper(value)
                    profiles.append(sigma)
                    areas.append(float(area))
                    volumes.append(float(volume))
                mixture = sum(
                    float(x) * sigma for x, sigma in zip(sample.liquid_composition, profiles)
                )
                records.append((sample, profiles, areas, volumes))
                sigma_rows.extend([*profiles, mixture])
                temperatures.extend([sample.temperature_k] * (sample.component_count + 1))
            sigmas = torch.cat(sigma_rows, dim=0).to(self.device)
            t = torch.tensor(temperatures, dtype=torch.float32, device=self.device)
            ensemble = []
            for model in self.models:
                local = sigmas.clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    _, segments = model(local, t)
                ensemble.append(segments)
            segments = torch.stack(ensemble).mean(dim=0)
            offset = 0
            for sample, profiles, areas, volumes in records:
                count = sample.component_count
                pure = segments[offset : offset + count]
                mixed = segments[offset + count]
                offset += count + 1
                combinatorial = self.combinatorial(
                    list(sample.liquid_composition), areas, volumes
                )
                residual = []
                for sigma, area, pure_value in zip(profiles, areas, pure):
                    normalized = sigma.to(self.device) / area
                    residual.append(
                        area / 5.8447 * torch.sum(normalized * (mixed - pure_value))
                    )
                value = torch.as_tensor(
                    combinatorial, dtype=torch.float32, device=self.device
                ) + torch.stack(residual)
                outputs.append(value.detach().cpu())
        return outputs
