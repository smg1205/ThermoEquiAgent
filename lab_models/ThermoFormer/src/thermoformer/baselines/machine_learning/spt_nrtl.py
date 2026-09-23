"""Fixed-database SPT-NRTL lookup and multicomponent NRTL equations."""

from __future__ import annotations

import csv
import hashlib
import io
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ...data.splitting import canonical_smiles


SPT_NRTL_DATABASE_REVISION = "f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e"
SPT_NRTL_SOURCE_REVISION = "691d55ce07a860b9e700877c64b6342a6dc385e1"
SPT_NRTL_RAW_ROOT = (
    "https://raw.githubusercontent.com/ClapeyronThermo/spt-nrtl-db/"
    f"{SPT_NRTL_DATABASE_REVISION}/v3"
)


@dataclass(frozen=True)
class SPTNRTLPairParameters:
    alpha: tuple[float, float]
    tau_ij: tuple[float, float, float, float]
    tau_ji: tuple[float, float, float, float]
    source_url: str = ""
    source_sha256: str = ""
    selected_row_sha256: str = ""
    lookup_direction: str = "forward"

    def reversed(self) -> "SPTNRTLPairParameters":
        return SPTNRTLPairParameters(
            self.alpha,
            self.tau_ji,
            self.tau_ij,
            self.source_url,
            self.source_sha256,
            self.selected_row_sha256,
            "reverse" if self.lookup_direction == "forward" else "forward",
        )


class SPTParameterFileUnavailable(FileNotFoundError):
    """The fixed database has no source file for the requested component."""


class SPTParameterRowUnavailable(KeyError):
    """The fixed database file has no row for the requested pair."""


def _temperature_polynomial(coefficients: Sequence[float], temperature_k: Tensor) -> Tensor:
    t0, t1, t2, t3 = coefficients
    return t0 + t1 / temperature_k + t2 * torch.log(temperature_k) + t3 * temperature_k


def multicomponent_nrtl_log_gamma(
    composition: Tensor,
    temperature_k: float | Tensor,
    pairs: Mapping[tuple[int, int], SPTNRTLPairParameters],
) -> Tensor:
    """Evaluate the standard multicomponent NRTL equation using every binary pair."""
    x = torch.as_tensor(composition)
    if x.ndim != 1 or x.numel() < 2:
        raise ValueError("NRTL composition must be a one-dimensional mixture")
    if not torch.isfinite(x).all() or torch.any(x < 0.0) or not torch.isclose(x.sum(), x.new_tensor(1.0), atol=1e-8):
        raise ValueError("NRTL composition must be finite, nonnegative and normalized")
    temperature = torch.as_tensor(temperature_k, dtype=x.dtype, device=x.device)
    if temperature.ndim != 0 or not torch.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("NRTL temperature must be a positive scalar")
    count = x.numel()
    expected = {(i, j) for i in range(count) for j in range(i + 1, count)}
    if set(pairs) != expected:
        raise KeyError("Every constituent binary pair is required for multicomponent NRTL")
    tau = x.new_zeros((count, count))
    alpha = x.new_zeros((count, count))
    for (i, j), pair in pairs.items():
        alpha_value = pair.alpha[0] + pair.alpha[1] * temperature
        alpha[i, j] = alpha_value
        alpha[j, i] = alpha_value
        tau[i, j] = _temperature_polynomial(pair.tau_ij, temperature)
        tau[j, i] = _temperature_polynomial(pair.tau_ji, temperature)
    interaction = torch.exp(-alpha * tau)
    denominators = torch.einsum("k,kj->j", x, interaction).clamp_min(1e-15)
    weighted_tau = torch.einsum("m,mj,mj->j", x, tau, interaction)
    first = torch.einsum("j,ji,ji,i->i", x, tau, interaction, denominators.reciprocal())
    bracket = tau - (weighted_tau / denominators).unsqueeze(0)
    second = torch.einsum(
        "j,ij,ij,j->i", x, interaction, bracket, denominators.reciprocal()
    )
    result = first + second
    if not torch.isfinite(result).all():
        raise RuntimeError("SPT-NRTL produced non-finite activity coefficients")
    return result


class SPTNRTLDatabase:
    """Exact canonical-SMILES lookup against a fixed remote database revision."""

    columns = (
        "SMILES0", "SMILES1", "a_1", "a_2",
        "t_12_1", "t_12_2", "t_12_3", "t_12_4",
        "t_21_1", "t_21_2", "t_21_3", "t_21_4",
    )

    def __init__(
        self,
        cache_root: Path,
        timeout_seconds: float = 30.0,
        download_attempts: int = 4,
    ) -> None:
        if download_attempts < 1:
            raise ValueError("SPT-NRTL download attempts must be positive")
        self.cache_root = cache_root
        self.timeout_seconds = timeout_seconds
        self.download_attempts = download_attempts
        self.audit: dict[str, dict[str, str]] = {}
        self.pair_audit: dict[str, dict[str, object]] = {}

    @staticmethod
    def relative_path(smiles: str) -> Path:
        encoded = urllib.parse.quote(smiles, safe="")
        return Path(smiles[0]) / str(len(smiles)) / f"{encoded}.csv"

    def _load_file(self, smiles: str) -> tuple[str, str, str]:
        relative = self.relative_path(smiles)
        cache = self.cache_root / "v3" / relative
        url = f"{SPT_NRTL_RAW_ROOT}/{relative.as_posix()}"
        if cache.is_file():
            payload = cache.read_bytes()
        else:
            for attempt in range(1, self.download_attempts + 1):
                try:
                    with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
                        payload = response.read()
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 404:
                        raise SPTParameterFileUnavailable(
                            f"SPT-NRTL has no file for {smiles}"
                        ) from error
                    if attempt == self.download_attempts:
                        raise
                except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
                    if attempt == self.download_attempts:
                        raise
                time.sleep(float(attempt))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(cache.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(cache)
        digest = hashlib.sha256(payload).hexdigest()
        self.audit[smiles] = {
            "url": url,
            "sha256": digest,
            "cache_relative_path": cache.relative_to(self.cache_root).as_posix(),
        }
        return payload.decode("utf-8-sig"), url, digest

    @classmethod
    def _parse_row(cls, text: str, partner: str, url: str, digest: str) -> SPTNRTLPairParameters:
        reader = csv.DictReader(io.StringIO(text))
        if tuple(reader.fieldnames or ()) != cls.columns:
            raise ValueError("Malformed SPT-NRTL database header")
        for row in reader:
            if row["SMILES1"] == partner:
                try:
                    selected_row_sha256 = hashlib.sha256(
                        ",".join(row[column] for column in cls.columns).encode("utf-8")
                    ).hexdigest()
                    return SPTNRTLPairParameters(
                        (float(row["a_1"]), float(row["a_2"])),
                        tuple(float(row[f"t_12_{index}"]) for index in range(1, 5)),
                        tuple(float(row[f"t_21_{index}"]) for index in range(1, 5)),
                        url,
                        digest,
                        selected_row_sha256,
                        "forward",
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError("Malformed SPT-NRTL parameter row") from error
        raise SPTParameterRowUnavailable(f"SPT-NRTL pair row unavailable for {partner}")

    def _record_pair(
        self, first: str, second: str, parameters: SPTNRTLPairParameters
    ) -> SPTNRTLPairParameters:
        key = f"{first}|{second}"
        self.pair_audit[key] = {
            "requested_smiles": [first, second],
            "lookup_direction": parameters.lookup_direction,
            "source_url": parameters.source_url,
            "source_sha256": parameters.source_sha256,
            "selected_row_sha256": parameters.selected_row_sha256,
            "alpha": list(parameters.alpha),
            "tau_ij": list(parameters.tau_ij),
            "tau_ji": list(parameters.tau_ji),
        }
        return parameters

    def lookup(self, first: str, second: str) -> SPTNRTLPairParameters:
        left, right = canonical_smiles(first), canonical_smiles(second)
        try:
            text, url, digest = self._load_file(left)
            return self._record_pair(left, right, self._parse_row(text, right, url, digest))
        except (SPTParameterFileUnavailable, SPTParameterRowUnavailable):
            text, url, digest = self._load_file(right)
            return self._record_pair(
                left, right, self._parse_row(text, left, url, digest).reversed()
            )

    def system_pairs(self, smiles: Sequence[str]) -> dict[tuple[int, int], SPTNRTLPairParameters]:
        return {
            (i, j): self.lookup(smiles[i], smiles[j])
            for i in range(len(smiles))
            for j in range(i + 1, len(smiles))
        }


class SPTNRTLPredictor:
    def __init__(self, database: SPTNRTLDatabase) -> None:
        self.database = database
        self._pair_cache: dict[tuple[str, ...], dict[tuple[int, int], SPTNRTLPairParameters]] = {}

    def system_pairs(
        self, smiles: Sequence[str]
    ) -> dict[tuple[int, int], SPTNRTLPairParameters]:
        """Resolve each distinct system once, including unavailable-coverage lookups."""
        canonical = tuple(canonical_smiles(value) for value in smiles)
        if canonical not in self._pair_cache:
            self._pair_cache[canonical] = self.database.system_pairs(canonical)
        return self._pair_cache[canonical]

    def require_coverage(self, smiles: Sequence[str]) -> None:
        """Raise a typed unavailable exception unless every binary pair exists."""
        self.system_pairs(smiles)

    def log_gamma(
        self,
        smiles: Sequence[str],
        temperature_k: float,
        composition: Sequence[float],
    ) -> Tensor:
        return multicomponent_nrtl_log_gamma(
            torch.as_tensor(composition, dtype=torch.float64),
            temperature_k,
            self.system_pairs(smiles),
        ).to(torch.float32)
