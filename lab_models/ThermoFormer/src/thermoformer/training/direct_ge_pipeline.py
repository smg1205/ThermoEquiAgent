"""Training utilities shared by the staged direct-GE experiment."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..configuration import DirectGESupervisionConfig, PhysicsFineTuningConfig
from ..data import VLEBatch, VLESample
from ..evaluation import predict_vle, prediction_metric_rows
from ..thermodynamics.vapor_pressure import PurePropertyCatalog
from .direct_ge import build_direct_thermodynamic_targets, direct_thermodynamic_supervision
from .fugacity_finetuning import configure_physics_finetuning, physics_warmup_scale
from .supervised import (
    TrainingConfig,
    _cpu_state,
    _loader,
    _objective,
    seed_everything,
)

_VALIDATION_MAE_KEYS = (
    "isothermal_pressure_mae_kpa",
    "isothermal_y_mae",
    "isobaric_temperature_mae_k",
    "isobaric_y_mae",
)
_VALIDATION_COVERAGE_KEYS = (
    "isothermal_coverage",
    "isobaric_coverage",
)


def configure_ge_pretraining(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Freeze the unidentifiable Psat branch and optimize the GE backbone only."""
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative and finite")

    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("vapor_pressure"))
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise ValueError("GE pretraining has no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def validation_composite_score(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    coverage_tolerance: float = 1e-12,
) -> float:
    """Equal-weight normalized validation MAE, gated by solver coverage."""
    for key in _VALIDATION_COVERAGE_KEYS:
        candidate_value = float(candidate[key])
        baseline_value = float(baseline[key])
        if not math.isfinite(candidate_value) or (
            candidate_value + coverage_tolerance < baseline_value
        ):
            return float("inf")

    ratios: list[float] = []
    for key in _VALIDATION_MAE_KEYS:
        candidate_value = float(candidate[key])
        baseline_value = float(baseline[key])
        if (
            not math.isfinite(candidate_value)
            or not math.isfinite(baseline_value)
            or baseline_value <= 0.0
        ):
            return float("inf")
        ratios.append(candidate_value / baseline_value)
    return float(sum(ratios) / len(ratios))


def format_direct_ge_progress(
    *,
    seed: int,
    stage_index: int,
    stage_count: int,
    stage_name: str,
    epoch: int,
    epoch_count: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    validation_composite: float,
    best_validation_composite: float,
    improved: bool,
    epoch_seconds: float,
    total_seconds: float,
) -> str:
    """Format one stable, human-readable epoch progress record."""
    return (
        f"[seed {seed}] [stage {stage_index}/{stage_count}: {stage_name}] "
        f"[epoch {epoch}/{epoch_count}] "
        f"train_total={float(train_metrics['total']):.6f} "
        f"val_score={float(validation_composite):.6f} "
        f"best={float(best_validation_composite):.6f} "
        f"P_MAE={float(validation_metrics['isothermal_pressure_mae_kpa']):.4f} kPa "
        f"y_iso_MAE={float(validation_metrics['isothermal_y_mae']):.6f} "
        f"T_MAE={float(validation_metrics['isobaric_temperature_mae_k']):.4f} K "
        f"y_isob_MAE={float(validation_metrics['isobaric_y_mae']):.6f} "
        f"improved={'yes' if improved else 'no'} "
        f"epoch_time={float(epoch_seconds):.1f}s "
        f"total_time={float(total_seconds):.1f}s"
    )


def _print_direct_ge_progress(**values: object) -> None:
    print(format_direct_ge_progress(**values), flush=True)


@dataclass
class DirectGEFitResult:
    state_dict: dict[str, torch.Tensor]
    stage_states: dict[str, dict[str, torch.Tensor]]
    history: list[dict[str, object]]
    selected_stage: str
    stage_validation_losses: dict[str, float]
    stage_validation_metrics: dict[str, dict[str, float]]
    stage_label_metrics: dict[str, dict[str, float]]
    parameter_summary: dict[str, object]
    best_validation_loss: float
    stage_best_epochs: dict[str, int]
    selection_partitions: tuple[str, ...] = ("validation",)


def _validation_summary(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
    pure_property_catalog: PurePropertyCatalog | None,
) -> dict[str, float]:
    records = predict_vle(
        model,
        samples,
        feature_map,
        batch_size=config.batch_size,
        device=device,
        solver_iterations=config.solver_iterations_eval,
        pure_property_catalog=pure_property_catalog,
    )
    rows = prediction_metric_rows(records)
    directions = {
        str(row["direction"]): row
        for row in rows
        if row["scope"] == "direction"
    }
    if "isothermal" not in directions or "isobaric" not in directions:
        raise ValueError("Validation split must contain isothermal and isobaric rows")
    isothermal = directions["isothermal"]
    isobaric = directions["isobaric"]

    def required(row: Mapping[str, object], key: str) -> float:
        value = row.get(key)
        return float(value) if value is not None else float("inf")

    return {
        "isothermal_pressure_mae_kpa": required(isothermal, "pressure_mae_kpa"),
        "isothermal_y_mae": required(isothermal, "y_mae"),
        "isobaric_temperature_mae_k": required(isobaric, "temperature_mae_k"),
        "isobaric_y_mae": required(isobaric, "y_mae"),
        "isothermal_coverage": required(isothermal, "valid_coverage"),
        "isobaric_coverage": required(isobaric, "valid_coverage"),
        "isothermal_solver_failure_rate": required(
            isothermal, "solver_failure_rate"
        ),
        "isobaric_solver_failure_rate": required(isobaric, "solver_failure_rate"),
    }


def evaluate_direct_label_metrics(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
    pure_property_catalog: PurePropertyCatalog | None,
    *,
    minimum_fraction: float,
) -> dict[str, float]:
    """Report label coverage and unweighted MAE without changing model state."""
    loader = _loader(samples, feature_map, config, False, pure_property_catalog)
    model.to(device).eval()
    ge_absolute = 0.0
    gamma_absolute = 0.0
    covered_states = 0
    covered_components = 0
    total_states = 0
    total_components = 0
    with torch.no_grad():
        for host_batch in loader:
            batch = host_batch.to(device)
            outputs = model(
                batch.molecules,
                batch.temperature_k,
                batch.pressure_kpa,
                batch.x,
                batch.mask,
            )
            targets = build_direct_thermodynamic_targets(
                batch, minimum_fraction=minimum_fraction
            )
            ge_mask = targets.ge_mask.to(outputs.excess_gibbs_rt)
            gamma_mask = targets.gamma_mask.to(outputs.log_gamma)
            ge_absolute += float(
                (
                    (outputs.excess_gibbs_rt - targets.excess_gibbs_rt).abs()
                    * ge_mask
                ).sum().cpu()
            )
            gamma_absolute += float(
                ((outputs.log_gamma - targets.log_gamma).abs() * gamma_mask).sum().cpu()
            )
            covered_states += targets.covered_states
            covered_components += targets.covered_components
            total_states += targets.total_states
            total_components += targets.total_components
    return {
        "excess_gibbs_rt_mae": ge_absolute / max(covered_states, 1),
        "log_gamma_mae": gamma_absolute / max(covered_components, 1),
        "ge_label_coverage": covered_states / max(total_states, 1),
        "gamma_label_coverage": covered_components / max(total_components, 1),
        "ge_labeled_states": float(covered_states),
        "gamma_labeled_components": float(covered_components),
    }


def _run_direct_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    supervision: DirectGESupervisionConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    include_vle: bool,
    fugacity_scale: float,
) -> dict[str, float]:
    model.train(True)
    totals: dict[str, float] = {}
    samples = 0
    covered_states = 0
    covered_components = 0
    total_components = 0
    for batch_index, host_batch in enumerate(loader):
        batch: VLEBatch = host_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch.molecules,
            batch.temperature_k,
            batch.pressure_kpa,
            batch.x,
            batch.mask,
        )
        direct = direct_thermodynamic_supervision(
            outputs,
            batch,
            excess_gibbs_weight=supervision.excess_gibbs_weight,
            activity_coefficient_weight=supervision.activity_coefficient_weight,
            minimum_fraction=supervision.minimum_fraction,
        )
        vle = None
        if include_vle:
            vle = _objective(
                model,
                batch,
                config,
                teacher_forced_fugacity_weight=(1.0 if fugacity_scale > 0.0 else 0.0),
            )
            base_vle = vle.total - vle.teacher_forced_fugacity
            total = (
                direct.total
                + supervision.vle_weight * base_vle
                + supervision.fugacity_weight
                * fugacity_scale
                * vle.teacher_forced_fugacity
            )
        else:
            total = direct.total
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(
                f"Direct-GE training produced a non-finite loss in batch {batch_index + 1}"
            )
        total.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.gradient_clip,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"Direct-GE training produced non-finite gradients in batch {batch_index + 1}"
            )
        optimizer.step()
        size = batch.x.shape[0]
        samples += size
        metrics = direct.detached()
        metrics["total"] = float(total.detach().cpu())
        metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
        if vle is not None:
            metrics.update({f"vle_{key}": value for key, value in vle.detached().items()})
        for name, value in metrics.items():
            if name not in {"ge_state_coverage", "gamma_component_coverage"}:
                totals[name] = totals.get(name, 0.0) + value * size
        covered_states += direct.targets.covered_states
        covered_components += direct.targets.covered_components
        total_components += direct.targets.total_components
    if samples == 0:
        raise ValueError("Training loader is empty")
    averaged = {name: value / samples for name, value in totals.items()}
    averaged["ge_state_coverage"] = covered_states / samples
    averaged["gamma_component_coverage"] = (
        covered_components / max(total_components, 1)
    )
    averaged["fugacity_warmup_scale"] = fugacity_scale
    return averaged


def fit_direct_ge_stages(
    model: nn.Module,
    train_samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    config: TrainingConfig,
    supervision: DirectGESupervisionConfig,
    finetuning: PhysicsFineTuningConfig,
    device: torch.device,
    *,
    validation_samples: Sequence[VLESample],
    pure_property_catalog: PurePropertyCatalog | None,
    baseline_state: Mapping[str, torch.Tensor],
) -> DirectGEFitResult:
    """Fit all three stages; the API intentionally has no test-set argument."""
    if not supervision.enabled:
        raise ValueError("direct_ge_supervision must be enabled")
    if not validation_samples:
        raise ValueError("Direct-GE checkpoint selection requires validation samples")
    seed_everything(config.seed)
    model.to(device)
    train_loader = _loader(
        train_samples, feature_map, config, True, pure_property_catalog
    )
    training_started = time.perf_counter()

    model.load_state_dict(baseline_state, strict=True)
    baseline_metrics = _validation_summary(
        model, validation_samples, feature_map, config, device, pure_property_catalog
    )
    baseline_labels = evaluate_direct_label_metrics(
        model,
        validation_samples,
        feature_map,
        config,
        device,
        pure_property_catalog,
        minimum_fraction=supervision.minimum_fraction,
    )
    stage_states = {"stage0": _cpu_state(model)}
    stage_metrics = {"stage0": baseline_metrics}
    stage_labels = {"stage0": baseline_labels}
    stage_scores = {"stage0": 1.0}
    # Epoch zero is the externally supplied supervised reference. Recording it
    # alongside the trainable stages makes each persisted state auditable
    # without using test-set metrics to reconstruct a selection decision.
    stage_best_epochs = {"stage0": 0}
    history: list[dict[str, object]] = []

    # This is a cumulative ablation: direct G^E supervision begins from the
    # validation-selected supervised checkpoint recorded as stage 0, rather
    # than reinitialising the model. Each later stage likewise begins from
    # the preceding stage's local best checkpoint.
    model.load_state_dict(baseline_state, strict=True)
    stage1_optimizer = configure_ge_pretraining(
        model,
        learning_rate=supervision.pretrain_learning_rate,
        weight_decay=config.weight_decay,
    )
    stage1_state: dict[str, torch.Tensor] | None = None
    stage1_score = math.inf
    stage1_metrics: dict[str, float] | None = None
    stage1_labels: dict[str, float] | None = None
    stage1_best_epoch = 0
    print(
        f"[seed {config.seed}] [stage 1/3: direct GE] "
        f"starting {supervision.pretrain_epochs} epoch(s)",
        flush=True,
    )
    for epoch in range(1, supervision.pretrain_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_direct_epoch(
            model,
            train_loader,
            config,
            supervision,
            device,
            stage1_optimizer,
            include_vle=False,
            fugacity_scale=0.0,
        )
        validation_metrics = _validation_summary(
            model, validation_samples, feature_map, config, device, pure_property_catalog
        )
        score = validation_composite_score(validation_metrics, baseline_metrics)
        label_metrics = evaluate_direct_label_metrics(
            model, validation_samples, feature_map, config, device,
            pure_property_catalog, minimum_fraction=supervision.minimum_fraction,
        )
        history.append({
            "stage": "stage1_direct_ge",
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_labels": label_metrics,
            "validation_composite": score,
        })
        improved = stage1_state is None or score < stage1_score - config.validation_min_delta
        if improved:
            stage1_score = score
            stage1_state = _cpu_state(model)
            stage1_metrics = validation_metrics
            stage1_labels = label_metrics
            stage1_best_epoch = epoch
        now = time.perf_counter()
        _print_direct_ge_progress(
            seed=config.seed,
            stage_index=1,
            stage_count=3,
            stage_name="direct GE",
            epoch=epoch,
            epoch_count=supervision.pretrain_epochs,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            validation_composite=score,
            best_validation_composite=stage1_score,
            improved=improved,
            epoch_seconds=now - epoch_started,
            total_seconds=now - training_started,
        )
    if stage1_state is None or stage1_metrics is None or stage1_labels is None:
        raise RuntimeError("Stage 1 produced no validation-selected checkpoint")
    model.load_state_dict(stage1_state, strict=True)
    stage_states["stage1"] = stage1_state
    stage_metrics["stage1"] = stage1_metrics
    stage_labels["stage1"] = stage1_labels
    stage_scores["stage1"] = stage1_score
    stage_best_epochs["stage1"] = stage1_best_epoch

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    stage2_optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Select only among checkpoints produced by this stage. Starting from the
    # stage-1 state is part of the trajectory, but retaining it as an
    # epoch-zero candidate would silently turn a stage-2 row into a rollback.
    stage2_state: dict[str, torch.Tensor] | None = None
    stage2_metrics: dict[str, float] | None = None
    stage2_labels: dict[str, float] | None = None
    stage2_score = math.inf
    stage2_best_epoch = 0
    epochs_without_improvement = 0
    print(
        f"[seed {config.seed}] [stage 2/3: joint VLE] "
        f"starting {config.epochs_supervised} epoch(s)",
        flush=True,
    )
    for epoch in range(1, config.epochs_supervised + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_direct_epoch(
            model, train_loader, config, supervision, device, stage2_optimizer,
            include_vle=True, fugacity_scale=0.0,
        )
        validation_metrics = _validation_summary(
            model, validation_samples, feature_map, config, device, pure_property_catalog
        )
        score = validation_composite_score(validation_metrics, baseline_metrics)
        label_metrics = evaluate_direct_label_metrics(
            model, validation_samples, feature_map, config, device,
            pure_property_catalog, minimum_fraction=supervision.minimum_fraction,
        )
        history.append({
            "stage": "stage2_joint_vle",
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_labels": label_metrics,
            "validation_composite": score,
        })
        improved = stage2_state is None or score < stage2_score - config.validation_min_delta
        if improved:
            stage2_score = score
            stage2_state = _cpu_state(model)
            stage2_metrics = validation_metrics
            stage2_labels = label_metrics
            stage2_best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        now = time.perf_counter()
        _print_direct_ge_progress(
            seed=config.seed,
            stage_index=2,
            stage_count=3,
            stage_name="joint VLE",
            epoch=epoch,
            epoch_count=config.epochs_supervised,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            validation_composite=score,
            best_validation_composite=stage2_score,
            improved=improved,
            epoch_seconds=now - epoch_started,
            total_seconds=now - training_started,
        )
        if (
            config.early_stopping_patience > 0
            and epoch >= config.minimum_supervised_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            history[-1]["early_stopped"] = True
            break
    if stage2_state is None or stage2_metrics is None or stage2_labels is None:
        raise RuntimeError("Stage 2 produced no validation-selected checkpoint")
    model.load_state_dict(stage2_state, strict=True)
    stage_states["stage2"] = stage2_state
    stage_metrics["stage2"] = stage2_metrics
    stage_labels["stage2"] = stage2_labels
    stage_scores["stage2"] = stage2_score
    stage_best_epochs["stage2"] = stage2_best_epoch

    setup = configure_physics_finetuning(model, config, finetuning)
    parameter_summary = {
        "total_parameters": setup.total_parameters,
        "trainable_parameters": setup.trainable_parameters,
        "trainable_fraction": setup.trainable_parameters / setup.total_parameters,
        "stage1_trainable_parameters": sum(
            value.numel() for name, value in model.named_parameters()
            if not name.startswith("vapor_pressure")
        ),
        "stage2_trainable_parameters": sum(
            value.numel() for value in model.parameters()
        ),
        "stage3": setup.record(),
    }
    print(json.dumps({"direct_ge_parameter_summary": parameter_summary}, sort_keys=True))
    # As above, the stage-2 state initializes fugacity fine-tuning but is not
    # eligible to be reported as a stage-3 checkpoint.
    stage3_state: dict[str, torch.Tensor] | None = None
    stage3_metrics: dict[str, float] | None = None
    stage3_labels: dict[str, float] | None = None
    stage3_score = math.inf
    stage3_best_epoch = 0
    print(
        f"[seed {config.seed}] [stage 3/3: fugacity] "
        f"starting {supervision.fugacity_epochs} epoch(s)",
        flush=True,
    )
    for epoch in range(1, supervision.fugacity_epochs + 1):
        epoch_started = time.perf_counter()
        scale = physics_warmup_scale(epoch, supervision.warmup_epochs)
        train_metrics = _run_direct_epoch(
            model, train_loader, config, supervision, device, setup.optimizer,
            include_vle=True, fugacity_scale=scale,
        )
        validation_metrics = _validation_summary(
            model, validation_samples, feature_map, config, device, pure_property_catalog
        )
        score = validation_composite_score(validation_metrics, baseline_metrics)
        label_metrics = evaluate_direct_label_metrics(
            model, validation_samples, feature_map, config, device,
            pure_property_catalog, minimum_fraction=supervision.minimum_fraction,
        )
        history.append({
            "stage": "stage3_fugacity",
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_labels": label_metrics,
            "validation_composite": score,
        })
        improved = stage3_state is None or score < stage3_score - config.validation_min_delta
        if improved:
            stage3_score = score
            stage3_state = _cpu_state(model)
            stage3_metrics = validation_metrics
            stage3_labels = label_metrics
            stage3_best_epoch = epoch
        now = time.perf_counter()
        _print_direct_ge_progress(
            seed=config.seed,
            stage_index=3,
            stage_count=3,
            stage_name="fugacity",
            epoch=epoch,
            epoch_count=supervision.fugacity_epochs,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            validation_composite=score,
            best_validation_composite=stage3_score,
            improved=improved,
            epoch_seconds=now - epoch_started,
            total_seconds=now - training_started,
        )
    if stage3_state is None or stage3_metrics is None or stage3_labels is None:
        raise RuntimeError("Stage 3 produced no validation-selected checkpoint")
    stage_states["stage3"] = stage3_state
    stage_metrics["stage3"] = stage3_metrics
    stage_labels["stage3"] = stage3_labels
    stage_scores["stage3"] = stage3_score
    stage_best_epochs["stage3"] = stage3_best_epoch
    selected_stage = min(stage_scores, key=stage_scores.get)
    selected_state = stage_states[selected_stage]
    model.load_state_dict(selected_state, strict=True)
    return DirectGEFitResult(
        state_dict=selected_state,
        stage_states=stage_states,
        history=history,
        selected_stage=selected_stage,
        stage_validation_losses=stage_scores,
        stage_validation_metrics=stage_metrics,
        stage_label_metrics=stage_labels,
        parameter_summary=parameter_summary,
        best_validation_loss=stage_scores[selected_stage],
        stage_best_epochs=stage_best_epochs,
    )
