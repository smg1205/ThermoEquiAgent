"""Supervised training and validation-gated fugacity fine-tuning."""

from .direct_ge import (
    DirectThermodynamicLoss,
    DirectThermodynamicTargets,
    build_direct_thermodynamic_targets,
    direct_thermodynamic_supervision,
)
from .losses import Objective, with_teacher_forced_fugacity_equilibrium
from .supervised import FitResult, TrainingConfig, evaluate_model, fit_model, seed_everything

train_supervised = fit_model


def __getattr__(name: str):
    if name in {
        "PhysicsFitResult",
        "configure_physics_finetuning",
        "finetune_with_fugacity",
        "fit_physics_stage",
        "load_stage1_checkpoint",
        "physics_finetune_objective",
    }:
        from . import fugacity_finetuning

        if name == "finetune_with_fugacity":
            return fugacity_finetuning.fit_physics_stage
        return getattr(fugacity_finetuning, name)
    raise AttributeError(name)

__all__ = [
    "FitResult",
    "DirectThermodynamicLoss",
    "DirectThermodynamicTargets",
    "Objective",
    "PhysicsFitResult",
    "TrainingConfig",
    "configure_physics_finetuning",
    "evaluate_model",
    "finetune_with_fugacity",
    "fit_model",
    "fit_physics_stage",
    "load_stage1_checkpoint",
    "physics_finetune_objective",
    "seed_everything",
    "train_supervised",
    "build_direct_thermodynamic_targets",
    "direct_thermodynamic_supervision",
    "with_teacher_forced_fugacity_equilibrium",
]
