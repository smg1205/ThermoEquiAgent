"""Dimensionless mixing Gibbs energy, TPD diagnostics, and differentiable LLE solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class LLEEquilibriumState:
    x_target: Tensor
    log_fugacity_residual: Tensor
    equilibrium_residual: Tensor
    tpd: Tensor
    converged: Tensor
    nontrivial: Tensor
    iterations: int


def _simplex(logits: Tensor, mask: Tensor) -> Tensor:
    masked_logits = logits.masked_fill(~mask.bool(), -1e9)
    value = torch.softmax(masked_logits, dim=-1) * mask
    return value / value.sum(-1, keepdim=True).clamp_min(1e-12)


def gmix_dimless(
    model: nn.Module, molecules: Tensor, temperature_k: Tensor, pressure_kpa: Tensor,
    x: Tensor, mask: Tensor,
) -> Tensor:
    """Return g_mix/(RT)=sum_i x_i log(x_i)+gE/(RT), preserving AD gamma path."""
    composition = (x * mask) / (x.mul(mask).sum(-1, keepdim=True).clamp_min(1e-12))
    outputs = model(molecules, temperature_k, pressure_kpa, composition, mask)
    if outputs.excess_gibbs_rt is None:
        raise ValueError("LLE requires activity_mode=excess_gibbs")
    ideal = (composition.clamp_min(1e-12).log() * composition * mask).sum(-1, keepdim=True)
    return ideal + outputs.excess_gibbs_rt


def tpd(
    model: nn.Module, molecules: Tensor, temperature_k: Tensor, pressure_kpa: Tensor,
    reference: Tensor, candidate: Tensor, mask: Tensor,
) -> Tensor:
    """Tangent-plane distance of candidate relative to the observed source phase."""
    source = (reference * mask) / reference.mul(mask).sum(-1, keepdim=True).clamp_min(1e-12)
    target = (candidate * mask) / candidate.mul(mask).sum(-1, keepdim=True).clamp_min(1e-12)
    source_output = model(molecules, temperature_k, pressure_kpa, source, mask)
    target_output = model(molecules, temperature_k, pressure_kpa, target, mask)
    chemical_delta = (
        target.clamp_min(1e-12).log() + target_output.log_gamma
        - source.clamp_min(1e-12).log() - source_output.log_gamma
    )
    return (target * chemical_delta * mask).sum(-1, keepdim=True)


def _initial_logits(source: Tensor, mask: Tensor, multistarts: int) -> Sequence[Tensor]:
    count = mask.sum(-1, keepdim=True).clamp_min(2.0)
    reflected = ((1.0 - source) / (count - 1.0)) * mask
    reflected = reflected / reflected.sum(-1, keepdim=True).clamp_min(1e-12)
    uniform = mask / count
    values = [reflected, uniform]
    for index in range(max(0, multistarts - 2)):
        rolled = torch.roll(reflected, shifts=index + 1, dims=-1) * mask
        rolled = rolled / rolled.sum(-1, keepdim=True).clamp_min(1e-12)
        values.append(rolled)
    return [value.clamp_min(1e-6).log() for value in values[:multistarts]]


def solve_lle(
    model: nn.Module,
    molecules: Tensor,
    temperature_k: Tensor,
    pressure_kpa: Tensor,
    x_source: Tensor,
    mask: Tensor,
    *,
    multistarts: int = 8,
    iterations: int = 64,
    step_size: float = 0.2,
    phase_tol: float = 1e-3,
    equilibrium_tolerance: float = 1e-4,
    tpd_tolerance: float = 1e-4,
    continuation_hint: Tensor | None = None,
    differentiable: bool = True,
) -> LLEEquilibriumState:
    """Find a nontrivial liquid counterpart without accepting a target label."""
    if multistarts < 1 or iterations < 1:
        raise ValueError("multistarts and iterations must be positive")
    source = (x_source * mask) / x_source.mul(mask).sum(-1, keepdim=True).clamp_min(1e-12)
    source_output = model(molecules, temperature_k, pressure_kpa, source, mask)
    source_log_activity = source.clamp_min(1e-12).log() + source_output.log_gamma
    # The initial-condition axis is independent. Flatten it into the batch
    # dimension so all multistarts share each model evaluation, while retaining
    # the same per-start objective scaling as the serial implementation.
    initial_logits = torch.stack(_initial_logits(source, mask, multistarts), dim=1)
    batch_size, start_count, component_count = initial_logits.shape
    expanded_mask = mask.unsqueeze(1).expand(-1, start_count, -1).reshape(-1, component_count)
    expanded_source = source.unsqueeze(1).expand(-1, start_count, -1).reshape(-1, component_count)
    expanded_activity = source_log_activity.unsqueeze(1).expand(-1, start_count, -1).reshape(-1, component_count)
    expanded_molecules = molecules.unsqueeze(1).expand(-1, start_count, -1, -1).reshape(-1, molecules.shape[1], molecules.shape[2])
    expanded_temperature = temperature_k.unsqueeze(1).expand(-1, start_count, -1).reshape(-1, temperature_k.shape[1])
    expanded_pressure = pressure_kpa.unsqueeze(1).expand(-1, start_count, -1).reshape(-1, pressure_kpa.shape[1])
    logits = initial_logits.reshape(-1, component_count)
    for _ in range(iterations):
        logits = logits.requires_grad_(True)
        target = _simplex(logits, expanded_mask)
        output = model(expanded_molecules, expanded_temperature, expanded_pressure, target, expanded_mask)
        residual = (expanded_activity - (target.clamp_min(1e-12).log() + output.log_gamma)) * expanded_mask
        objective = residual.square().sum(-1).sum() / batch_size
        gradient = torch.autograd.grad(objective, logits, create_graph=differentiable, retain_graph=True)[0]
        logits = (logits - step_size * gradient).clamp(-20.0, 20.0)
        if not differentiable:
            logits = logits.detach()
    target = _simplex(logits, expanded_mask)
    output = model(expanded_molecules, expanded_temperature, expanded_pressure, target, expanded_mask)
    residual = (expanded_activity - (target.clamp_min(1e-12).log() + output.log_gamma)) * expanded_mask
    targets = target.reshape(batch_size, start_count, component_count)
    residuals = residual.reshape(batch_size, start_count, component_count)
    rms_values = (residuals.square().sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1.0)).sqrt()
    tpd_values = tpd(model, expanded_molecules, expanded_temperature, expanded_pressure, expanded_source, target, expanded_mask).reshape(batch_size, start_count)
    l1 = (targets - source.unsqueeze(1)).abs().mul(mask.unsqueeze(1)).sum(-1)
    nontrivial = l1 >= phase_tol
    valid = nontrivial & torch.isfinite(rms_values) & torch.isfinite(tpd_values)
    stable = tpd_values >= -tpd_tolerance
    score = rms_values.detach() + tpd_values.detach().abs().mul(1e-3)
    if continuation_hint is not None:
        score = score + (targets.detach() - continuation_hint.unsqueeze(1)).abs().sum(-1).mul(1e-6)
    # Prefer candidates that pass the TPD stability screen. If finite-step
    # optimization provides none, select the best nontrivial finite fallback
    # solely for differentiable supervision and mark it unconverged below.
    screened_score = score.masked_fill(~(valid & stable), float("inf"))
    fallback_score = score.masked_fill(~valid, float("inf"))
    has_stable = torch.isfinite(screened_score).any(dim=1, keepdim=True)
    score = torch.where(has_stable, screened_score, fallback_score)
    indices = score.argmin(dim=1)
    gather = indices[:, None, None].expand(-1, 1, targets.shape[-1])
    chosen_target = targets.gather(1, gather).squeeze(1)
    chosen_residual = residuals.gather(1, gather).squeeze(1)
    chosen_rms = rms_values.gather(1, indices[:, None]).squeeze(1).unsqueeze(-1)
    chosen_tpd = tpd_values.gather(1, indices[:, None]).squeeze(1).unsqueeze(-1)
    chosen_nontrivial = nontrivial.gather(1, indices[:, None]).squeeze(1).unsqueeze(-1)
    chosen_stable = stable.gather(1, indices[:, None]).squeeze(1).unsqueeze(-1)
    converged = chosen_nontrivial & chosen_stable & (chosen_rms <= equilibrium_tolerance) & (chosen_tpd.abs() <= tpd_tolerance)
    return LLEEquilibriumState(
        x_target=chosen_target, log_fugacity_residual=chosen_residual,
        equilibrium_residual=chosen_rms, tpd=chosen_tpd,
        converged=converged, nontrivial=chosen_nontrivial, iterations=iterations,
    )
