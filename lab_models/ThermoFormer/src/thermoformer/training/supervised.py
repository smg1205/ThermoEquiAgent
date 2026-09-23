"""Supervised ThermoFormer training and shared objective evaluation."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import VLEBatch, VLESample, VLETensorDataset, collate_vle
from .losses import (
    Objective,
    direct_vle_objective,
    experimental_objective,
    with_teacher_forced_fugacity_equilibrium,
)
from ..evaluation.metrics import masked_r2
from ..thermodynamics.vapor_pressure import PurePropertyCatalog
from ..thermodynamics.vle_solver import equilibrium_at_tp, solve_batch_modes
from .checkpointing import cpu_state_dict as _cpu_state


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs_supervised: int = 80
    epochs_physics: int = 0
    early_stopping_patience: int = 12
    minimum_supervised_epochs: int = 10
    minimum_physics_epochs: int = 0
    validation_min_delta: float = 0.0
    pressure_weight: float = 1.0
    pure_weight: float = 0.5
    solver_iterations_eval: int = 48
    gradient_clip: float = 5.0
    seed: int = 42

    def __post_init__(self) -> None:
        integer_values = {
            "batch_size": self.batch_size,
            "epochs_supervised": self.epochs_supervised,
            "epochs_physics": self.epochs_physics,
            "early_stopping_patience": self.early_stopping_patience,
            "minimum_supervised_epochs": self.minimum_supervised_epochs,
            "minimum_physics_epochs": self.minimum_physics_epochs,
            "solver_iterations_eval": self.solver_iterations_eval,
            "seed": self.seed,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values.values()):
            raise ValueError("batch_size, epoch counts, and seed must be integers")
        numeric_values = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "pressure_weight": self.pressure_weight,
            "pure_weight": self.pure_weight,
            "gradient_clip": self.gradient_clip,
            "validation_min_delta": self.validation_min_delta,
        }
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric_values.values()
        ):
            raise ValueError("training floating-point hyperparameters must be finite numbers")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if any(
            value < 0
            for value in (
                self.epochs_supervised,
                self.epochs_physics,
                self.early_stopping_patience,
                self.minimum_supervised_epochs,
                self.minimum_physics_epochs,
            )
        ):
            raise ValueError("epoch counts cannot be negative")
        if self.solver_iterations_eval < 1:
            raise ValueError("solver evaluation iterations must be positive")
        if any(
            value < 0.0
            for value in (
                self.pressure_weight,
                self.pure_weight,
                self.validation_min_delta,
            )
        ):
            raise ValueError("loss weights cannot be negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")


@dataclass
class FitResult:
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, object]]
    best_validation_loss: float | None


def seed_everything(seed: int) -> None:
    # PyTorch deterministic CUDA GEMMs require this setting to exist before
    # the first cuBLAS-backed Uni-Mol/model operation in the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    config: TrainingConfig,
    shuffle: bool,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        VLETensorDataset(
            samples,
            feature_map,
            pure_property_catalog=pure_property_catalog,
        ),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_vle,
        generator=generator if shuffle else None,
    )


def _objective(
    model: nn.Module,
    batch: VLEBatch,
    config: TrainingConfig,
    physics_scale: float = 1.0,
    teacher_forced_fugacity_weight: float = 0.0,
) -> Objective:
    if getattr(getattr(model, "config", None), "decoder_mode", None) == "direct_vle":
        return direct_vle_objective(
            model,
            batch,
            pressure_weight=config.pressure_weight,
        )
    state = equilibrium_at_tp(
        model,
        batch.molecules,
        batch.temperature_k,
        batch.pressure_kpa,
        batch.x,
        batch.mask,
        batch.pure_property_parameters,
    )
    objective = experimental_objective(
        state,
        observed_y=batch.y,
        observed_pressure_kpa=batch.pressure_kpa,
        quality_weight=batch.quality_weight,
        mask=batch.mask,
        pressure_weight=config.pressure_weight,
        pure_weight=config.pure_weight,
    )
    return with_teacher_forced_fugacity_equilibrium(
        objective,
        state,
        batch,
        weight=teacher_forced_fugacity_weight * physics_scale,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    physics_scale: float = 1.0,
    teacher_forced_fugacity_weight: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    gradient_norm_total = 0.0
    gradient_norm_max = 0.0
    for batch_index, host_batch in enumerate(loader):
        batch = host_batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            objective = _objective(
                model, batch, config, physics_scale,
                teacher_forced_fugacity_weight,
            )
            if not bool(torch.isfinite(objective.total).all()):
                raise FloatingPointError(
                    f"Training produced a non-finite loss in batch {batch_index + 1}"
                )
            objective.total.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"Training produced non-finite gradients in batch {batch_index + 1}"
                )
            gradient_value = float(gradient_norm.detach().cpu())
            gradient_norm_total += gradient_value * batch.x.shape[0]
            gradient_norm_max = max(gradient_norm_max, gradient_value)
            optimizer.step()
        else:
            with torch.no_grad():
                objective = _objective(
                    model, batch, config, physics_scale,
                    teacher_forced_fugacity_weight,
                )
            if not bool(torch.isfinite(objective.total).all()):
                raise FloatingPointError(
                    f"Validation produced a non-finite loss in batch {batch_index + 1}"
                )
        size = batch.x.shape[0]
        sample_count += size
        for name, value in objective.detached().items():
            totals[name] = totals.get(name, 0.0) + value * size
    if sample_count == 0:
        raise ValueError("Training loader is empty")
    metrics = {name: value / sample_count for name, value in totals.items()}
    if training:
        metrics["gradient_norm_mean"] = gradient_norm_total / sample_count
        metrics["gradient_norm_max"] = gradient_norm_max
    if teacher_forced_fugacity_weight > 0.0:
        metrics["physics_weight_scale"] = physics_scale
    return metrics


def fit_model(
    model: nn.Module,
    train_samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
    validation_samples: Sequence[VLESample] | None = None,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> FitResult:
    """Fit the supervised stage; fugacity Stage 2 uses ``fit_physics_stage``."""
    if config.epochs_physics > 0:
        raise ValueError(
            "fit_model is supervised-only; use the C1 fugacity fine-tuning runner for Stage 2"
        )
    seed_everything(config.seed)
    model.to(device)
    train_loader = _loader(
        train_samples,
        feature_map,
        config,
        shuffle=True,
        pure_property_catalog=pure_property_catalog,
    )
    validation_loader = (
        _loader(
            validation_samples,
            feature_map,
            config,
            shuffle=False,
            pure_property_catalog=pure_property_catalog,
        )
        if validation_samples
        else None
    )
    history: list[dict[str, object]] = []
    best_state = _cpu_state(model)
    best_validation = math.inf

    def train_stage(
        name: str,
        epochs: int,
        minimum_epochs: int,
        epoch_offset: int,
    ) -> None:
        nonlocal best_state, best_validation
        if epochs == 0:
            return
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        stage_best_state = _cpu_state(model)
        stage_best_validation = best_validation
        epochs_without_improvement = 0
        stage_started = time.perf_counter()
        print(f"[stage {name}] starting {epochs} epoch(s)", flush=True)
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            set_training_epoch = getattr(model, "set_training_epoch", None)
            if callable(set_training_epoch):
                set_training_epoch(epoch_offset + epoch)
            train_metrics = _run_epoch(
                model, train_loader, device, config, optimizer
            )
            validation_metrics = (
                _run_epoch(
                    model,
                    validation_loader,
                    device,
                    config,
                    None,
                )
                if validation_loader is not None
                else None
            )
            history.append(
                {
                    "stage": name,
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            improved = validation_metrics is None
            if validation_metrics is None:
                stage_best_state = _cpu_state(model)
            elif (
                validation_metrics["total"]
                < stage_best_validation - config.validation_min_delta
            ):
                improved = True
                stage_best_validation = validation_metrics["total"]
                stage_best_state = _cpu_state(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            now = time.perf_counter()
            validation_total = (
                float("nan")
                if validation_metrics is None
                else float(validation_metrics["total"])
            )
            print(
                f"[stage {name}] [epoch {epoch}/{epochs}] "
                f"train_total={float(train_metrics['total']):.6f} "
                f"val_total={validation_total:.6f} "
                f"best={stage_best_validation:.6f} "
                f"improved={'yes' if improved else 'no'} "
                f"epoch_time={now - epoch_started:.1f}s "
                f"total_time={now - stage_started:.1f}s",
                flush=True,
            )
            if (
                validation_metrics is not None
                and config.early_stopping_patience > 0
                and epoch >= minimum_epochs
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                history[-1]["early_stopped"] = True
                print(
                    f"[stage {name}] early stopping at epoch {epoch}; "
                    f"best_validation={stage_best_validation:.6f}",
                    flush=True,
                )
                break
        model.load_state_dict(stage_best_state)
        best_state = stage_best_state
        best_validation = stage_best_validation

    train_stage(
        "experimental",
        config.epochs_supervised,
        config.minimum_supervised_epochs,
        0,
    )
    model.load_state_dict(best_state)
    return FitResult(
        state_dict=best_state,
        history=history,
        best_validation_loss=None if validation_loader is None else best_validation,
    )


def evaluate_model(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    solver_iterations: int = 48,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> dict[str, float]:
    config = TrainingConfig(batch_size=batch_size)
    loader = _loader(
        samples,
        feature_map,
        config,
        shuffle=False,
        pure_property_catalog=pure_property_catalog,
    )
    model.to(device).eval()
    actual_y: list[torch.Tensor] = []
    predicted_y: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    pressure_log_errors: list[torch.Tensor] = []
    isothermal_actual_y: list[torch.Tensor] = []
    isothermal_predicted_y: list[torch.Tensor] = []
    isothermal_masks: list[torch.Tensor] = []
    isothermal_pressure_log_errors: list[torch.Tensor] = []
    isothermal_converged: list[torch.Tensor] = []
    isobaric_actual_y: list[torch.Tensor] = []
    isobaric_predicted_y: list[torch.Tensor] = []
    isobaric_masks: list[torch.Tensor] = []
    isobaric_temperature_errors: list[torch.Tensor] = []
    isobaric_converged: list[torch.Tensor] = []
    with torch.no_grad():
        for host_batch in loader:
            batch = host_batch.to(device)
            state = equilibrium_at_tp(
                model,
                batch.molecules,
                batch.temperature_k,
                batch.pressure_kpa,
                batch.x,
                batch.mask,
                batch.pure_property_parameters,
            )
            actual_y.append(batch.y.cpu())
            predicted_y.append(state.y.cpu())
            masks.append(batch.mask.cpu())
            pressure_log_errors.append(
                (
                    torch.log(state.calculated_pressure_kpa.clamp_min(1e-12))
                    - torch.log(batch.pressure_kpa.clamp_min(1e-12))
                ).cpu()
            )
            solutions = solve_batch_modes(
                model,
                batch,
                iterations=solver_iterations,
                strict=False,
            )
            isothermal_rows = solutions.isothermal_rows
            if solutions.isothermal is not None:
                solved = solutions.isothermal
                isothermal_actual_y.append(batch.y[isothermal_rows].cpu())
                isothermal_predicted_y.append(solved.y.cpu())
                isothermal_masks.append(batch.mask[isothermal_rows].cpu())
                isothermal_pressure_log_errors.append(
                    (
                        torch.log(solved.pressure_kpa.clamp_min(1e-12))
                        - torch.log(batch.pressure_kpa[isothermal_rows].clamp_min(1e-12))
                    ).cpu()
                )
                isothermal_converged.append(solved.converged.float().cpu())
            isobaric_rows = solutions.isobaric_rows
            if solutions.isobaric is not None:
                solved = solutions.isobaric
                isobaric_actual_y.append(batch.y[isobaric_rows].cpu())
                isobaric_predicted_y.append(solved.y.cpu())
                isobaric_masks.append(batch.mask[isobaric_rows].cpu())
                isobaric_temperature_errors.append(
                    (solved.temperature_k - batch.temperature_k[isobaric_rows]).cpu()
                )
                isobaric_converged.append(solved.converged.float().cpu())
    actual = torch.cat(actual_y)
    predicted = torch.cat(predicted_y)
    mask = torch.cat(masks)
    log_pressure_error = torch.cat(pressure_log_errors)
    metrics = {
        "teacher_forced_vapor_r2": masked_r2(actual, predicted, mask),
        "teacher_forced_pressure_log_residual_rmse": float(
            torch.sqrt(torch.mean(log_pressure_error**2))
        ),
    }
    if isothermal_actual_y:
        metrics.update(
            {
                "isothermal_vapor_r2": masked_r2(
                    torch.cat(isothermal_actual_y),
                    torch.cat(isothermal_predicted_y),
                    torch.cat(isothermal_masks),
                ),
                "isothermal_pressure_log_rmse": float(
                    torch.sqrt(torch.mean(torch.cat(isothermal_pressure_log_errors) ** 2))
                ),
                "isothermal_convergence_rate": float(torch.cat(isothermal_converged).mean()),
                "isothermal_samples": float(sum(row.shape[0] for row in isothermal_actual_y)),
            }
        )
    if isobaric_actual_y:
        metrics.update(
            {
                "isobaric_vapor_r2": masked_r2(
                    torch.cat(isobaric_actual_y),
                    torch.cat(isobaric_predicted_y),
                    torch.cat(isobaric_masks),
                ),
                "isobaric_temperature_rmse_k": float(
                    torch.sqrt(torch.mean(torch.cat(isobaric_temperature_errors) ** 2))
                ),
                "isobaric_convergence_rate": float(torch.cat(isobaric_converged).mean()),
                "isobaric_samples": float(sum(row.shape[0] for row in isobaric_actual_y)),
            }
        )
    nonfinite = sorted(name for name, value in metrics.items() if not math.isfinite(value))
    if nonfinite:
        raise FloatingPointError(
            f"Evaluation produced non-finite metrics: {', '.join(nonfinite)}"
        )
    return metrics


def config_dict(config: TrainingConfig) -> dict[str, object]:
    return asdict(config)
