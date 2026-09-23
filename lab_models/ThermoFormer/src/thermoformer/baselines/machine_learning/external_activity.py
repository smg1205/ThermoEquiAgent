"""Common bridge from fixed external log-gamma predictors to the VLE solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn


class LogGammaPredictor(Protocol):
    def log_gamma(
        self,
        smiles: Sequence[str],
        temperature_k: float,
        composition: Sequence[float],
    ) -> Tensor: ...


@dataclass
class ExternalActivityOutputs:
    log_gamma: Tensor
    log_psat: Tensor
    attention_bias: None = None
    attention_bias_penalty: None = None


class ExternalActivityThermodynamicAdapter(nn.Module):
    """Evaluate fixed Python predictors while retaining the shared solver contract."""

    def __init__(self, predictor: LogGammaPredictor, smiles_rows: Sequence[Sequence[str]]) -> None:
        super().__init__()
        self.predictor = predictor
        self.smiles_rows = tuple(tuple(row) for row in smiles_rows)
        self._prediction_cache: dict[tuple[tuple[str, ...], float, tuple[float, ...]], Tensor] = {}

    def forward(
        self,
        molecules: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> ExternalActivityOutputs:
        del pressure_kpa
        if molecules.shape[-1] != 2 or molecules.shape[:2] != x.shape:
            raise ValueError("External activity molecule tensor must contain two Psat coefficients")
        if len(self.smiles_rows) != x.shape[0]:
            raise ValueError("External activity SMILES batch does not match composition batch")
        values = []
        for index, smiles in enumerate(self.smiles_rows):
            active = int(mask[index].sum().item())
            temperature_value = float(temperature_k[index, 0].detach().cpu())
            composition_values = tuple(float(value) for value in x[index, :active].detach().cpu())
            key = (smiles[:active], temperature_value, composition_values)
            if key not in self._prediction_cache:
                self._prediction_cache[key] = self.predictor.log_gamma(
                    smiles[:active], temperature_value, composition_values
                ).detach().cpu()
            predicted = self._prediction_cache[key].to(device=x.device, dtype=x.dtype)
            padded = torch.zeros_like(x[index])
            padded[:active] = predicted
            values.append(padded)
        log_gamma = torch.stack(values) * mask
        intercept = molecules[..., 0]
        inverse_temperature = molecules[..., 1]
        log_psat = (intercept + inverse_temperature / temperature_k) * mask
        return ExternalActivityOutputs(log_gamma=log_gamma, log_psat=log_psat)


def psat_coefficient_tensor(
    smiles_rows: Sequence[Sequence[str]],
    vapor_pressure: dict[str, object],
    canonicalizer,
    device: torch.device,
) -> Tensor:
    rows = []
    for smiles in smiles_rows:
        rows.append(torch.tensor(
            [
                [
                    vapor_pressure[canonicalizer(value)].intercept,
                    vapor_pressure[canonicalizer(value)].inverse_temperature,
                ]
                for value in smiles
            ],
            dtype=torch.float32,
            device=device,
        ))
    return torch.stack(rows)
