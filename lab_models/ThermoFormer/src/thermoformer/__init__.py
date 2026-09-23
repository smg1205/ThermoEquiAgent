"""Public research interface for ThermoFormer's scientific modules."""

from .configuration import ExperimentConfig, load_experiment_config
from .data import VLEBatch, VLESample, load_vle_dataset
from .evaluation import evaluate_protocol
from .features import build_molecular_encoder, build_molecular_features, prepare_partition_features
from .models import ModelOutputs, ThermoFormer, ThermoFormerConfig, final_model_config
from .training import finetune_with_fugacity, train_supervised
from .thermodynamics import (
    ConvergenceError,
    EquilibriumState,
    equilibrium_at_tp,
    solve_isobaric,
    solve_isothermal,
)

__all__ = [
    "ConvergenceError",
    "EquilibriumState",
    "ExperimentConfig",
    "ModelOutputs",
    "ThermoFormer",
    "ThermoFormerConfig",
    "VLEBatch",
    "VLESample",
    "build_molecular_encoder",
    "build_molecular_features",
    "equilibrium_at_tp",
    "final_model_config",
    "finetune_with_fugacity",
    "load_experiment_config",
    "load_vle_dataset",
    "prepare_partition_features",
    "solve_isobaric",
    "solve_isothermal",
    "train_supervised",
    "evaluate_protocol",
]
