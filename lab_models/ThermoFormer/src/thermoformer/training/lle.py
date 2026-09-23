"""Standalone three-stage LLE training using the shared thermodynamic backbone."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..configuration import LLEConfig, PhysicsFineTuningConfig
from ..data import LLEBatch, LLESample, LLETensorDataset, collate_lle
from ..thermodynamics import solve_lle
from .supervised import TrainingConfig, _cpu_state, seed_everything


@dataclass
class LLEFitResult:
    state_dict: dict[str, Tensor]
    stage_states: dict[str, dict[str, Tensor]]
    stage_validation: dict[str, dict[str, float]]
    stage_best_epochs: dict[str, int]
    selected_stage: str
    history: list[dict[str, object]]
    parameter_summary: dict[str, object]


def lle_loader(samples: Sequence[LLESample], features: dict[str, np.ndarray], config: TrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        LLETensorDataset(samples, features), batch_size=config.batch_size, shuffle=shuffle,
        collate_fn=collate_lle, generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
    )


def _weighted_huber(prediction: Tensor, target: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    values = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(values) * weight.to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def lle_supervision(model: nn.Module, batch: LLEBatch, config: LLEConfig, *, iterations: int) -> tuple[Tensor, dict[str, Tensor]]:
    state = solve_lle(
        model, batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.source_phase, batch.mask,
        multistarts=config.multistarts, iterations=iterations, step_size=config.step_size,
        phase_tol=config.phase_tol, equilibrium_tolerance=config.equilibrium_tolerance,
        tpd_tolerance=config.tpd_tolerance,
    )
    loss = _weighted_huber(state.x_target, batch.target_phase, batch.mask, batch.quality_weight)
    return loss, {
        "composition": loss, "fugacity": loss * 0.0, "solver_residual": state.equilibrium_residual.mean(),
        "convergence_rate": state.converged.float().mean(), "nontrivial_rate": state.nontrivial.float().mean(),
    }


def lle_log_fugacity_residual(model: nn.Module, batch: LLEBatch, x_min: float) -> tuple[Tensor, Tensor]:
    """Observed-endpoint log-fugacity residual and its valid-component mask."""
    alpha = model(batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.source_phase, batch.mask)
    beta = model(batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.target_phase, batch.mask)
    residual = (
        batch.source_phase.clamp_min(x_min).log() + alpha.log_gamma
        - batch.target_phase.clamp_min(x_min).log() - beta.log_gamma
    )
    valid = batch.mask.bool() & (batch.source_phase >= x_min) & (batch.target_phase >= x_min)
    return residual, valid


def lle_fugacity_loss(model: nn.Module, batch: LLEBatch, x_min: float) -> Tensor:
    residual, valid = lle_log_fugacity_residual(model, batch, x_min)
    return _weighted_huber(residual, torch.zeros_like(residual), valid, batch.quality_weight)


def lle_stage3_objective(supervised: Tensor, fugacity: Tensor, config: LLEConfig) -> Tensor:
    """Retain LLE endpoint supervision during fugacity-constrained refinement."""
    return config.composition_weight * supervised + config.fugacity_weight * fugacity


def evaluate_lle(
    model: nn.Module, samples: Sequence[LLESample], features: dict[str, np.ndarray],
    training: TrainingConfig, config: LLEConfig, device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    loader = lle_loader(samples, features, training, False)
    model.eval()
    absolute = squared = count = fugacity_squared = 0.0
    fugacity_count = converged = nontrivial = records = 0
    predictions: list[dict[str, object]] = []
    for host_batch in loader:
        batch = host_batch.to(device)
        with torch.enable_grad():
            state = solve_lle(
                model, batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.source_phase, batch.mask,
                multistarts=config.multistarts, iterations=config.eval_iterations, step_size=config.step_size,
                phase_tol=config.phase_tol, equilibrium_tolerance=config.equilibrium_tolerance,
                tpd_tolerance=config.tpd_tolerance, differentiable=False,
            )
            observed_fugacity, observed_fugacity_valid = lle_log_fugacity_residual(model, batch, config.x_min)
        valid = batch.mask.bool()
        error = (state.x_target - batch.target_phase)[valid]
        absolute += float(error.abs().sum().detach().cpu())
        squared += float(error.square().sum().detach().cpu())
        count += int(valid.sum().detach().cpu())
        fugacity_error = observed_fugacity[observed_fugacity_valid]
        fugacity_squared += float(fugacity_error.square().sum().detach().cpu())
        fugacity_count += int(observed_fugacity_valid.sum().detach().cpu())
        converged += int(state.converged.sum().detach().cpu())
        nontrivial += int(state.nontrivial.sum().detach().cpu())
        for index, record_id in enumerate(batch.record_ids):
            predictions.append({
                "record_id": record_id, "source": batch.source_phase[index].detach().cpu().tolist(),
                "target_observed": batch.target_phase[index].detach().cpu().tolist(),
                "target_predicted": state.x_target[index].detach().cpu().tolist(),
                "mask": batch.mask[index].detach().cpu().tolist(),
                "temperature_k": float(batch.temperature_k[index].item()),
                "pressure_kpa": float(batch.pressure_kpa[index].item()),
                "equilibrium_residual": float(state.equilibrium_residual[index].item()),
                "tpd": float(state.tpd[index].item()), "converged": bool(state.converged[index].item()),
                "nontrivial": bool(state.nontrivial[index].item()),
            })
            records += 1
    if count == 0:
        raise ValueError("LLE evaluation has no valid compositions")
    actual = np.asarray([value for row in predictions for value, keep in zip(row["target_observed"], row["mask"]) if keep], dtype=float)
    predicted = np.asarray([value for row in predictions for value, keep in zip(row["target_predicted"], row["mask"]) if keep], dtype=float)
    denominator = float(((actual - actual.mean()) ** 2).sum())
    return {
        "composition_mae": absolute / count, "composition_rmse": math.sqrt(squared / count),
        "composition_r2": float(1.0 - ((actual - predicted) ** 2).sum() / denominator) if denominator > 0.0 else float("nan"),
        "log_fugacity_rmse": math.sqrt(fugacity_squared / fugacity_count) if fugacity_count else float("nan"),
        "solver_convergence_rate": converged / max(records, 1),
        "nontrivial_solution_rate": nontrivial / max(records, 1),
    }, predictions


def _stage3_optimizer(model: nn.Module, finetuning: PhysicsFineTuningConfig) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    allowed = ("pair_potential", "film", "mixture_token")
    parameters: list[dict[str, object]] = []
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in allowed}
    for name, parameter in model.named_parameters():
        group = next((candidate for candidate in allowed if name == candidate or name.startswith(candidate + ".")), None)
        parameter.requires_grad_(group is not None)
        if group is not None:
            groups[group].append(parameter)
    learning_rates = {"pair_potential": finetuning.pair_potential_lr, "film": finetuning.film_lr, "mixture_token": finetuning.mixture_token_lr}
    for name, values in groups.items():
        if values:
            parameters.append({"params": values, "lr": learning_rates[name], "name": name})
    if not parameters:
        raise ValueError("LLE Stage 3 has no trainable shared-thermodynamic parameters")
    summary = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for values in groups.values() for parameter in values),
        "groups": {name: {"parameters": sum(parameter.numel() for parameter in values), "lr": learning_rates[name]} for name, values in groups.items()},
    }
    return torch.optim.AdamW(parameters, weight_decay=0.0), summary


def fit_lle_stages(
    model: nn.Module, train_samples: Sequence[LLESample], validation_samples: Sequence[LLESample],
    features: dict[str, np.ndarray], training: TrainingConfig, lle: LLEConfig,
    finetuning: PhysicsFineTuningConfig, device: torch.device,
) -> LLEFitResult:
    """Stage 1 is the preloaded VLE GE checkpoint; no LLE labels are fabricated."""
    seed_everything(training.seed)
    model.to(device)
    stage1 = _cpu_state(model)
    stage1_metrics, _ = evaluate_lle(model, validation_samples, features, training, lle, device)
    train_loader = lle_loader(train_samples, features, training, True)
    history: list[dict[str, object]] = []
    stage_states = {"stage1": stage1}
    stage_validation = {"stage1": stage1_metrics}
    stage_epochs = {"stage1": 0}

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    best_state = None
    best_metrics = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, training.epochs_supervised + 1):
        model.train()
        for host_batch in train_loader:
            batch = host_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = lle_supervision(model, batch, lle, iterations=lle.train_iterations)
            total = lle.composition_weight * loss
            total.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], training.gradient_clip)
            optimizer.step()
        metrics, _ = evaluate_lle(model, validation_samples, features, training, lle, device)
        history.append({"stage": "stage2_lle", "epoch": epoch, "validation": metrics})
        if best_metrics is None or metrics["composition_mae"] < best_metrics["composition_mae"] - training.validation_min_delta:
            best_state, best_metrics, best_epoch, wait = _cpu_state(model), metrics, epoch, 0
        else:
            wait += 1
        if training.early_stopping_patience and epoch >= training.minimum_supervised_epochs and wait >= training.early_stopping_patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("LLE Stage 2 produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    stage_states["stage2"], stage_validation["stage2"], stage_epochs["stage2"] = best_state, best_metrics, best_epoch

    optimizer, parameter_summary = _stage3_optimizer(model, finetuning)
    best_stage3 = None
    best_stage3_metrics = None
    best_stage3_epoch = 0
    for epoch in range(1, 11):
        model.train()
        for host_batch in train_loader:
            batch = host_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            supervised, _ = lle_supervision(model, batch, lle, iterations=lle.train_iterations)
            fugacity = lle_fugacity_loss(model, batch, lle.x_min)
            total = lle_stage3_objective(supervised, fugacity, lle)
            total.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], training.gradient_clip)
            optimizer.step()
        metrics, _ = evaluate_lle(model, validation_samples, features, training, lle, device)
        history.append({"stage": "stage3_lle_fugacity", "epoch": epoch, "validation": metrics})
        if best_stage3_metrics is None or metrics["composition_mae"] < best_stage3_metrics["composition_mae"] - training.validation_min_delta:
            best_stage3, best_stage3_metrics, best_stage3_epoch = _cpu_state(model), metrics, epoch
    if best_stage3 is None or best_stage3_metrics is None:
        raise RuntimeError("LLE Stage 3 produced no validation-selected checkpoint")
    stage_states["stage3"], stage_validation["stage3"], stage_epochs["stage3"] = best_stage3, best_stage3_metrics, best_stage3_epoch
    selected = "stage3" if best_stage3_metrics["composition_mae"] < best_metrics["composition_mae"] - training.validation_min_delta else "stage2"
    model.load_state_dict(stage_states[selected], strict=True)
    return LLEFitResult(
        state_dict=stage_states[selected], stage_states=stage_states, stage_validation=stage_validation,
        stage_best_epochs=stage_epochs, selected_stage=selected, history=history, parameter_summary=parameter_summary,
    )
