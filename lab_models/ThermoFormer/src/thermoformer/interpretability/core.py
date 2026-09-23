"""Model-level primitives for physically constrained interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from ..models import ModelOutputs, ThermoFormer, ThermoFormerConfig


@dataclass(frozen=True)
class ModelBundle:
    model: ThermoFormer
    checkpoint: Path
    seed: int
    protocol: str
    best_validation_loss: float
    metadata: dict[str, Any]


def load_thermoformer_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> ModelBundle:
    """Load one formal checkpoint without accepting architecture mismatches."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_name") != "ThermoFormer":
        raise ValueError(f"Unsupported checkpoint model: {checkpoint}")
    config = ThermoFormerConfig(**payload["model_config"])
    model = ThermoFormer(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return ModelBundle(
        model=model,
        checkpoint=checkpoint.resolve(),
        seed=int(payload["seed"]),
        protocol=str(payload["protocol"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        metadata=payload,
    )


def simplex_response_directions(composition: Tensor) -> Tensor:
    """Return closed-simplex directions for increasing each component.

    Column ``j`` increases ``x_j`` by one differential unit and removes the
    same amount from all other components in proportion to their current mole
    fractions.  This yields a square, composition-conserving response matrix.
    """
    if composition.ndim != 1 or composition.numel() not in (2, 3):
        raise ValueError("composition must be a one-dimensional binary or ternary vector")
    if not bool(torch.isfinite(composition).all()):
        raise ValueError("composition must be finite")
    if bool((composition <= 0.0).any()) or bool((composition >= 1.0).any()):
        raise ValueError("simplex response directions require an interior composition")
    if not torch.isclose(
        composition.sum(),
        torch.ones((), dtype=composition.dtype, device=composition.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("composition must sum to one")
    count = composition.numel()
    directions = torch.empty(
        count,
        count,
        dtype=composition.dtype,
        device=composition.device,
    )
    for component in range(count):
        direction = -composition / (1.0 - composition[component])
        direction = direction.clone()
        direction[component] = 1.0
        directions[:, component] = direction
    return directions


def padded_state_tensors(
    smiles: Sequence[str],
    feature_map: dict[str, np.ndarray],
    temperature_k: np.ndarray,
    pressure_kpa: np.ndarray,
    compositions: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create padded tensors for a repeated binary or ternary system."""
    count = len(smiles)
    if count not in (2, 3):
        raise ValueError("ThermoFormer interpretation supports binary or ternary systems")
    compositions = np.asarray(compositions, dtype=np.float32)
    if compositions.ndim != 2 or compositions.shape[1] != count:
        raise ValueError("compositions must have shape [states, components]")
    states = len(compositions)
    features = np.stack([feature_map[value] for value in smiles]).astype(np.float32)
    molecules = torch.zeros(states, 3, features.shape[1], dtype=torch.float32, device=device)
    molecules[:, :count] = torch.from_numpy(features).to(device).unsqueeze(0)
    x = torch.zeros(states, 3, dtype=torch.float32, device=device)
    x[:, :count] = torch.from_numpy(compositions).to(device)
    mask = torch.zeros_like(x)
    mask[:, :count] = 1.0
    temperature = torch.as_tensor(temperature_k, dtype=torch.float32, device=device).reshape(-1, 1)
    pressure = torch.as_tensor(pressure_kpa, dtype=torch.float32, device=device).reshape(-1, 1)
    if len(temperature) == 1 and states > 1:
        temperature = temperature.expand(states, 1).clone()
    if len(pressure) == 1 and states > 1:
        pressure = pressure.expand(states, 1).clone()
    if len(temperature) != states or len(pressure) != states:
        raise ValueError("temperature and pressure must contain one value or one per state")
    return molecules, temperature, pressure, x, mask


def vapor_from_outputs(outputs: ModelOutputs, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return nonideal and ideal vapor compositions at the same model state."""
    log_partial = torch.log(x.clamp_min(1e-12)) + outputs.log_psat
    ideal = torch.softmax(log_partial.masked_fill(~mask.bool(), -1e9), dim=-1) * mask
    nonideal = torch.softmax(
        (log_partial + outputs.log_gamma).masked_fill(~mask.bool(), -1e9),
        dim=-1,
    ) * mask
    return nonideal, ideal


def pairwise_log_relative_volatility(
    outputs: ModelOutputs,
    first: int,
    second: int,
) -> Tensor:
    return (
        outputs.log_gamma[:, first]
        + outputs.log_psat[:, first]
        - outputs.log_gamma[:, second]
        - outputs.log_psat[:, second]
    )


def stable_pca_scores(matrix: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic PCA scores with a fixed sign convention."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("PCA requires a two-dimensional matrix with at least two rows")
    if components < 1 or components > min(values.shape):
        raise ValueError("invalid PCA component count")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    loadings = right[:components].copy()
    for index in range(components):
        pivot = int(np.argmax(np.abs(loadings[index])))
        if loadings[index, pivot] < 0.0:
            loadings[index] *= -1.0
    scores = centered @ loadings.T
    variance = singular_values[:components] ** 2
    total = float(np.sum(singular_values**2))
    explained = variance / total if total > 0.0 else np.zeros_like(variance)
    return scores, explained


def thermodynamic_response_sensitivity(
    model: ThermoFormer,
    molecules: Tensor,
    temperature_k: Tensor,
    pressure_kpa: Tensor,
    composition: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiate log-gamma along closed-simplex directions and temperature."""
    active = int(mask[0].sum().item())
    if composition.shape != (1, 3) or mask.shape != (1, 3):
        raise ValueError("sensitivity expects one padded ThermoFormer state")
    x = composition.detach().clone().requires_grad_(True)
    temperature = temperature_k.detach().clone().requires_grad_(True)
    outputs = model(molecules, temperature, pressure_kpa, x, mask)
    jacobian_rows = []
    temperature_rows = []
    for component in range(active):
        composition_gradient, temperature_gradient = torch.autograd.grad(
            outputs.log_gamma[0, component],
            (x, temperature),
            retain_graph=True,
            create_graph=False,
        )
        jacobian_rows.append(composition_gradient[0, :active])
        temperature_rows.append(temperature_gradient.reshape(()))
    jacobian = torch.stack(jacobian_rows)
    directions = simplex_response_directions(x[0, :active])
    closed_simplex_response = jacobian @ directions
    return closed_simplex_response.detach(), torch.stack(temperature_rows).detach()
