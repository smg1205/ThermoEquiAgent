"""Direct excess-Gibbs and derivative supervision from experimental VLE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..data import VLEBatch
from ..models import ModelOutputs
from ..thermodynamics.vapor_pressure import evaluate_pure_property_correlations


@dataclass(frozen=True)
class DirectThermodynamicTargets:
    excess_gibbs_rt: Tensor
    log_gamma: Tensor
    ge_mask: Tensor
    gamma_mask: Tensor
    covered_states: int
    covered_components: int
    total_states: int
    total_components: int

    @property
    def state_coverage(self) -> float:
        return self.covered_states / max(self.total_states, 1)

    @property
    def component_coverage(self) -> float:
        return self.covered_components / max(self.total_components, 1)


@dataclass(frozen=True)
class DirectThermodynamicLoss:
    total: Tensor
    excess_gibbs: Tensor
    activity_coefficient: Tensor
    targets: DirectThermodynamicTargets

    def detached(self) -> dict[str, float]:
        return {
            "direct_thermodynamic_total": float(self.total.detach().cpu()),
            "excess_gibbs": float(self.excess_gibbs.detach().cpu()),
            "activity_coefficient": float(self.activity_coefficient.detach().cpu()),
            "ge_state_coverage": self.targets.state_coverage,
            "gamma_component_coverage": self.targets.component_coverage,
        }


def build_direct_thermodynamic_targets(
    batch: VLEBatch,
    *,
    minimum_fraction: float = 1e-4,
    pure_endpoint_tolerance: float = 1e-7,
) -> DirectThermodynamicTargets:
    """Construct labels using validated external Psat correlations only."""
    if not 0.0 < minimum_fraction < 0.5:
        raise ValueError("minimum_fraction must lie between zero and 0.5")
    external = evaluate_pure_property_correlations(
        batch.temperature_k,
        batch.mask,
        batch.pure_property_parameters,
    )
    present = batch.mask.bool()
    identifiable = (
        present
        & (batch.x >= minimum_fraction)
        & (batch.y >= minimum_fraction)
        & external.valid
    )
    safe_x = batch.x.clamp_min(minimum_fraction)
    safe_y = batch.y.clamp_min(minimum_fraction)
    log_gamma = (
        torch.log(safe_y)
        + torch.log(batch.pressure_kpa.clamp_min(1e-12))
        - torch.log(safe_x)
        - external.log_psat_kpa
    )
    log_gamma = torch.where(identifiable, log_gamma, torch.zeros_like(log_gamma))

    component_count = present.sum(-1, keepdim=True)
    fully_identifiable = identifiable.sum(-1, keepdim=True) == component_count
    dominant_fraction, dominant_index = batch.x.max(-1, keepdim=True)
    vapor_dominant = batch.y.gather(1, dominant_index)
    pure_endpoint = (
        (dominant_fraction >= 1.0 - pure_endpoint_tolerance)
        & (vapor_dominant >= 1.0 - pure_endpoint_tolerance)
    )
    dominant_mask = torch.zeros_like(present).scatter(1, dominant_index, True)
    gamma_mask = torch.where(pure_endpoint, dominant_mask & present, identifiable)
    log_gamma = torch.where(
        pure_endpoint.expand_as(log_gamma) & dominant_mask,
        torch.zeros_like(log_gamma),
        log_gamma,
    )
    ge_mask = fully_identifiable | pure_endpoint
    excess_gibbs_rt = (batch.x * log_gamma * gamma_mask).sum(-1, keepdim=True)
    excess_gibbs_rt = torch.where(
        pure_endpoint, torch.zeros_like(excess_gibbs_rt), excess_gibbs_rt
    )
    return DirectThermodynamicTargets(
        excess_gibbs_rt=excess_gibbs_rt,
        log_gamma=log_gamma,
        ge_mask=ge_mask,
        gamma_mask=gamma_mask,
        covered_states=int(ge_mask.sum().detach().cpu()),
        covered_components=int(gamma_mask.sum().detach().cpu()),
        total_states=batch.x.shape[0],
        total_components=int(present.sum().detach().cpu()),
    )


def _masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    quality_weight: Tensor,
) -> Tensor:
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(element) * quality_weight.to(element)
    return (element * weights).sum() / weights.sum().clamp_min(1.0)


def direct_thermodynamic_supervision(
    outputs: ModelOutputs,
    batch: VLEBatch,
    *,
    excess_gibbs_weight: float = 1.0,
    activity_coefficient_weight: float = 0.5,
    minimum_fraction: float = 1e-4,
) -> DirectThermodynamicLoss:
    """Supervise both the scalar potential and its autograd derivatives."""
    if outputs.excess_gibbs_rt is None:
        raise ValueError("Direct GE supervision requires an excess-Gibbs model")
    if excess_gibbs_weight < 0.0 or activity_coefficient_weight < 0.0:
        raise ValueError("Direct thermodynamic loss weights cannot be negative")
    targets = build_direct_thermodynamic_targets(
        batch, minimum_fraction=minimum_fraction
    )
    ge_loss = _masked_huber(
        outputs.excess_gibbs_rt,
        targets.excess_gibbs_rt,
        targets.ge_mask,
        batch.quality_weight,
    )
    gamma_loss = _masked_huber(
        outputs.log_gamma,
        targets.log_gamma,
        targets.gamma_mask,
        batch.quality_weight,
    )
    return DirectThermodynamicLoss(
        total=(
            excess_gibbs_weight * ge_loss
            + activity_coefficient_weight * gamma_loss
        ),
        excess_gibbs=ge_loss,
        activity_coefficient=gamma_loss,
        targets=targets,
    )
