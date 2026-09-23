"""Typed experiment configuration for ThermoFormer training and ablations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal, Sequence

import yaml

from .models import ThermoFormerConfig
from .training.supervised import TrainingConfig


@dataclass(frozen=True)
class EncoderConfig:
    representation: Literal[
        "multiview", "hybrid", "unimol_v2", "rdkit_2d",
        "rdkit_2d_legacy_fixed", "functional_groups"
    ] = "unimol_v2"
    fusion_mode: Literal["legacy", "naive", "interaction_specific"] = "legacy"
    model_size: str = "84m"
    batch_size: int = 16
    use_rdkit_descriptors: bool = True
    use_unimol: bool = True
    use_functional_groups: bool = True
    chemical_attention_bias: bool = False
    context_pair_interaction: bool = False

    def __post_init__(self) -> None:
        allowed = (
            "multiview", "hybrid", "unimol_v2", "rdkit_2d",
            "rdkit_2d_legacy_fixed", "functional_groups",
        )
        if self.representation not in allowed:
            raise ValueError(f"encoder.representation must be one of {allowed}")
        if self.model_size not in ("84m", "164m", "310m", "570m", "1.1B"):
            raise ValueError(f"Unsupported Uni-Mol v2 model_size: {self.model_size}")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("encoder.batch_size must be an integer")
        if self.batch_size < 1:
            raise ValueError("encoder.batch_size must be positive")
        branch_values = (
            self.use_rdkit_descriptors,
            self.use_unimol,
            self.use_functional_groups,
        )
        if any(
            not isinstance(value, bool)
            for value in (
                *branch_values,
                self.chemical_attention_bias,
                self.context_pair_interaction,
            )
        ):
            raise ValueError("hybrid encoder branch switches must be boolean")
        if self.fusion_mode not in ("legacy", "naive", "interaction_specific"):
            raise ValueError("encoder.fusion_mode must be legacy, naive, or interaction_specific")
        multiview = self.representation in ("multiview", "hybrid")
        if multiview and not any(branch_values):
            raise ValueError("multiview encoder requires at least one branch")
        if multiview and self.fusion_mode == "legacy":
            raise ValueError("multiview representation requires naive or interaction_specific fusion")
        if not multiview and self.fusion_mode != "legacy":
            raise ValueError("single-view representations require legacy fusion")
        if self.fusion_mode == "interaction_specific" and not all(branch_values):
            raise ValueError("interaction-specific fusion requires all three molecular views")
        if self.chemical_attention_bias and (
            not multiview or not self.use_rdkit_descriptors or not self.use_unimol
        ):
            raise ValueError(
                "Chemical attention bias requires multiview RDKit and Uni-Mol branches"
            )
        if self.context_pair_interaction and not multiview:
            raise ValueError("Context pair interaction requires multiview representation")


@dataclass(frozen=True)
class DataConfig:
    root: str = "datasets/vle_reference"
    pure_property_catalog: str = ""
    source_filter: str = ""
    failed_weight: float = 0.0
    max_pressure_kpa: float | None = 500.0
    minimum_pure_anchor_temperatures: int = 2

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str)
            for value in (self.root, self.pure_property_catalog, self.source_filter)
        ):
            raise ValueError(
                "data.root, data.pure_property_catalog, and data.source_filter must be strings"
            )
        if (
            not isinstance(self.failed_weight, (int, float))
            or isinstance(self.failed_weight, bool)
            or not math.isfinite(self.failed_weight)
        ):
            raise ValueError("data.failed_weight must be a finite number")
        if self.max_pressure_kpa is not None and (
            not isinstance(self.max_pressure_kpa, (int, float))
            or isinstance(self.max_pressure_kpa, bool)
            or not math.isfinite(self.max_pressure_kpa)
        ):
            raise ValueError("data.max_pressure_kpa must be a finite number or null")
        if (
            not isinstance(self.minimum_pure_anchor_temperatures, int)
            or isinstance(self.minimum_pure_anchor_temperatures, bool)
        ):
            raise ValueError("data.minimum_pure_anchor_temperatures must be an integer")
        if not 0.0 <= self.failed_weight <= 1.0:
            raise ValueError("data.failed_weight must be between zero and one")
        if self.max_pressure_kpa is not None and self.max_pressure_kpa <= 0.0:
            raise ValueError("data.max_pressure_kpa must be positive or null")
        if self.minimum_pure_anchor_temperatures < 0:
            raise ValueError("data.minimum_pure_anchor_temperatures cannot be negative")


@dataclass(frozen=True)
class EvaluationConfig:
    mode: Literal["kfold", "holdout"] = "kfold"
    folds: int = 5
    test_fraction: float = 0.15
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.mode not in ("kfold", "holdout"):
            raise ValueError("evaluation.mode must be 'kfold' or 'holdout'")
        if not isinstance(self.folds, int) or isinstance(self.folds, bool):
            raise ValueError("evaluation.folds must be an integer")
        if self.folds < 2:
            raise ValueError("evaluation.folds must be at least two")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in (self.test_fraction, self.validation_fraction)
        ):
            raise ValueError("evaluation fractions must be finite numbers")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("evaluation.test_fraction must be between zero and one")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("evaluation.validation_fraction must be between zero and one")
        if self.test_fraction + self.validation_fraction >= 1.0:
            raise ValueError("test_fraction + validation_fraction must be below one")


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str = "experiments/run_records/thermoformer"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    results_file: str | None = None

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("runtime.device must be auto, cpu, or cuda")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("runtime.output_dir cannot be empty")
        if self.results_file is not None and (
            not isinstance(self.results_file, str) or not self.results_file.strip()
        ):
            raise ValueError("runtime.results_file must be a non-empty string or null")


@dataclass(frozen=True)
class PhysicsFineTuningConfig:
    enabled: bool = True
    warmup_epochs: int = 2
    trainable_modules: tuple[str, ...] = (
        "pair_potential",
        "vapor_pressure",
        "film",
        "mixture_token",
    )
    pair_potential_lr: float = 2e-5
    vapor_pressure_lr: float = 1e-5
    film_lr: float = 5e-6
    mixture_token_lr: float = 5e-6
    teacher_forced_fugacity_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("physics_finetuning.enabled must be boolean")
        if (
            not isinstance(self.warmup_epochs, int)
            or isinstance(self.warmup_epochs, bool)
            or self.warmup_epochs < 0
        ):
            raise ValueError("physics_finetuning.warmup_epochs must be non-negative")
        allowed = {"pair_potential", "vapor_pressure", "film", "mixture_token"}
        if not isinstance(self.trainable_modules, (tuple, list)) or any(
            not isinstance(value, str) or not value for value in self.trainable_modules
        ):
            raise ValueError("physics_finetuning.trainable_modules must contain names")
        unknown = set(self.trainable_modules) - allowed
        if unknown:
            raise ValueError(
                "Unknown physics_finetuning.trainable_modules: "
                + ", ".join(sorted(unknown))
            )
        learning_rates = (
            self.pair_potential_lr,
            self.vapor_pressure_lr,
            self.film_lr,
            self.mixture_token_lr,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in learning_rates
        ):
            raise ValueError("physics_finetuning learning rates must be positive and finite")
        if (
            not isinstance(self.teacher_forced_fugacity_weight, (int, float))
            or isinstance(self.teacher_forced_fugacity_weight, bool)
            or not math.isfinite(self.teacher_forced_fugacity_weight)
            or self.teacher_forced_fugacity_weight < 0.0
        ):
            raise ValueError(
                "physics_finetuning.teacher_forced_fugacity_weight must be non-negative and finite"
            )


@dataclass(frozen=True)
class DirectGESupervisionConfig:
    """Hyperparameters for staged direct excess-Gibbs supervision."""

    enabled: bool = True
    pretrain_epochs: int = 20
    pretrain_learning_rate: float = 2e-4
    fugacity_epochs: int = 10
    excess_gibbs_weight: float = 1.0
    activity_coefficient_weight: float = 0.5
    vle_weight: float = 1.0
    fugacity_weight: float = 0.01
    minimum_fraction: float = 1e-4
    warmup_epochs: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("direct_ge_supervision.enabled must be boolean")
        for name in ("pretrain_epochs", "fugacity_epochs", "warmup_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"direct_ge_supervision.{name} must be a non-negative integer"
                )
        for name in ("pretrain_learning_rate", "minimum_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"direct_ge_supervision.{name} must be positive and finite")
        for name in (
            "excess_gibbs_weight",
            "activity_coefficient_weight",
            "vle_weight",
            "fugacity_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"direct_ge_supervision.{name} must be non-negative and finite")
        if self.minimum_fraction >= 0.5:
            raise ValueError("direct_ge_supervision.minimum_fraction must be below 0.5")


@dataclass(frozen=True)
class LLEConfig:
    """Data, solver, and refinement settings for a standalone LLE task."""

    binary_workbook: str = "binary_lle_english.xlsx"
    ternary_workbook: str = "ternary_lle_english.xlsx"
    component_count: Literal[2, 3] = 3
    stage1_checkpoint_template: str = ""
    composition_weight: float = 1.0
    fugacity_weight: float = 0.01
    x_min: float = 1e-4
    multistarts: int = 8
    train_iterations: int = 24
    eval_iterations: int = 64
    step_size: float = 0.2
    phase_tol: float = 1e-3
    equilibrium_tolerance: float = 1e-4
    tpd_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.component_count not in (2, 3):
            raise ValueError("lle.component_count must be 2 or 3")
        if not isinstance(self.stage1_checkpoint_template, str):
            raise ValueError("lle.stage1_checkpoint_template must be a string")
        for name in ("binary_workbook", "ternary_workbook"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"lle.{name} must be a non-empty string")
        for name in ("multistarts", "train_iterations", "eval_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"lle.{name} must be a positive integer")
        for name in ("composition_weight", "fugacity_weight", "x_min", "step_size", "phase_tol", "equilibrium_tolerance", "tpd_tolerance"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"lle.{name} must be finite")
            if value < 0.0:
                raise ValueError(f"lle.{name} must be non-negative")
        if self.x_min <= 0.0 or self.x_min >= 0.5 or self.step_size <= 0.0:
            raise ValueError("lle.x_min must lie in (0, 0.5) and lle.step_size must be positive")
        if self.train_iterations > self.eval_iterations:
            raise ValueError("lle.train_iterations cannot exceed lle.eval_iterations")

@dataclass(frozen=True)
class ProtocolConfig:
    """Registered split family and random seeds represented by a configuration."""

    registered_splits: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    evaluation_partition: Literal["validation", "test"] = "test"

    def __post_init__(self) -> None:
        if not isinstance(self.registered_splits, (tuple, list)) or any(
            not isinstance(value, str) or not value for value in self.registered_splits
        ):
            raise ValueError("protocol.registered_splits must contain non-empty names")
        if not isinstance(self.seeds, (tuple, list)) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in self.seeds
        ):
            raise ValueError("protocol.seeds must contain integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("protocol.seeds must be unique")
        if self.evaluation_partition not in {"validation", "test"}:
            raise ValueError("protocol.evaluation_partition must be validation or test")
        object.__setattr__(self, "registered_splits", tuple(self.registered_splits))
        object.__setattr__(self, "seeds", tuple(self.seeds))

    def validate_request(self, split_protocol: str, seed: int, partition: str) -> None:
        """Reject a split, seed, or partition outside the declared protocol family."""
        if split_protocol not in self.registered_splits:
            raise ValueError(
                f"Split '{split_protocol}' is not declared by this protocol configuration"
            )
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} is not declared by this protocol configuration")
        if partition != self.evaluation_partition:
            raise ValueError(
                f"Evaluation partition '{partition}' does not match "
                f"'{self.evaluation_partition}'"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "thermoformer_base"
    seed: int = 42
    task_mode: Literal["vle", "lle"] = "vle"
    model: ThermoFormerConfig = field(default_factory=ThermoFormerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    protocol: ProtocolConfig | None = None
    physics_finetuning: PhysicsFineTuningConfig | None = None
    direct_ge_supervision: DirectGESupervisionConfig | None = None
    lle: LLEConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("experiment seed must be an integer")
        if self.task_mode not in ("vle", "lle"):
            raise ValueError("task_mode must be vle or lle; joint mode is unsupported")
        if self.task_mode == "lle" and self.lle is None:
            raise ValueError("task_mode=lle requires an lle configuration section")
        if self.task_mode == "lle" and self.training.epochs_supervised > 80:
            raise ValueError("LLE Stage 2 training.epochs_supervised cannot exceed 80")
        if (
            self.model.chemical_attention_bias
            and not self.encoder.chemical_attention_bias
        ):
            raise ValueError(
                "Configure chemical_attention_bias in the encoder section"
            )
        if (
            self.model.context_pair_interaction
            and not self.encoder.context_pair_interaction
        ):
            raise ValueError(
                "Configure context_pair_interaction in the encoder section"
            )
        if (
            self.model.chemical_bias_headwise
            or self.model.chemical_bias_shared_gate
            or self.model.chemical_bias_modality_gates
        ) and not self.encoder.chemical_attention_bias:
            raise ValueError("Chemical-bias controls require encoder.chemical_attention_bias")
        if (
            self.model.chemical_bias_modality_gates
            and not self.encoder.use_functional_groups
        ):
            raise ValueError("Chemical modality gates require the functional-group branch")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.physics_finetuning is None:
            payload.pop("physics_finetuning")
        if self.direct_ge_supervision is None:
            payload.pop("direct_ge_supervision")
        if self.lle is None:
            payload.pop("lle")
        if self.protocol is None:
            payload.pop("protocol")
        return payload


def _parse_override(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _with_overrides(payload: dict[str, object], overrides: Sequence[str]) -> dict[str, object]:
    updated = json.loads(json.dumps(payload))
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must use section.field=value: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        path = dotted_key.split(".")
        if len(path) != 2:
            raise ValueError(f"Override must address one section and field: {dotted_key}")
        section, field_name = path
        if section not in {"model", "encoder", "data", "evaluation", "training", "runtime", "protocol", "physics_finetuning", "direct_ge_supervision", "lle"}:
            raise ValueError(f"Unknown configuration section: {section}")
        section_payload = updated.setdefault(section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"Cannot override nested field below {section}")
        section_payload[field_name] = _parse_override(raw_value)
    return updated


def _section(section: str, cls: type, value: object) -> object:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} configuration fields: {', '.join(unknown)}")
    return cls(**value)


def _deep_merge(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_payload(path: Path, ancestors: tuple[Path, ...] = ()) -> dict[str, object]:
    resolved = path.resolve()
    if resolved in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, resolved))
        raise ValueError(f"Cyclic experiment configuration inheritance: {chain}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = (
            yaml.safe_load(handle)
            if resolved.suffix.lower() in {".yaml", ".yml"}
            else json.load(handle)
        )
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration root must be a JSON object")
    parent_reference = payload.pop("extends", None)
    if parent_reference is None:
        return payload
    if not isinstance(parent_reference, str) or not parent_reference.strip():
        raise ValueError("Configuration 'extends' must be a non-empty path string")
    parent_path = Path(parent_reference)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    parent = _load_payload(parent_path, (*ancestors, resolved))
    return _deep_merge(parent, payload)


def load_experiment_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    payload = _load_payload(path)
    payload = _with_overrides(payload, overrides)
    allowed_root = {"name", "seed", "task_mode", "model", "encoder", "data", "evaluation", "training", "runtime", "protocol", "physics_finetuning", "direct_ge_supervision", "lle"}
    unknown_root = sorted(set(payload) - allowed_root)
    if unknown_root:
        raise ValueError(
            f"Unknown experiment configuration sections: {', '.join(unknown_root)}"
        )
    seed = payload.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("experiment seed must be an integer")
    training_payload = payload.get("training", {})
    if not isinstance(training_payload, dict):
        raise ValueError("Configuration section 'training' must be an object")
    if "seed" in training_payload and training_payload["seed"] != seed:
        raise ValueError("training.seed must match the top-level experiment seed")
    training_payload = {**training_payload, "seed": seed}
    return ExperimentConfig(
        name=str(payload.get("name", "thermoformer_base")),
        seed=seed,
        task_mode=str(payload.get("task_mode", "vle")),
        model=_section("model", ThermoFormerConfig, payload.get("model", {})),
        encoder=_section("encoder", EncoderConfig, payload.get("encoder", {})),
        data=_section("data", DataConfig, payload.get("data", {})),
        evaluation=_section("evaluation", EvaluationConfig, payload.get("evaluation", {})),
        training=_section("training", TrainingConfig, training_payload),
        runtime=_section("runtime", RuntimeConfig, payload.get("runtime", {})),
        protocol=(
            _section("protocol", ProtocolConfig, payload["protocol"])
            if "protocol" in payload
            else None
        ),
        physics_finetuning=(
            _section(
                "physics_finetuning",
                PhysicsFineTuningConfig,
                payload["physics_finetuning"],
            )
            if "physics_finetuning" in payload
            else None
        ),
        direct_ge_supervision=(
            _section(
                "direct_ge_supervision",
                DirectGESupervisionConfig,
                payload["direct_ge_supervision"],
            )
            if "direct_ge_supervision" in payload
            else None
        ),
        lle=(
            _section("lle", LLEConfig, payload["lle"])
            if "lle" in payload
            else None
        ),
    )


def experiment_sha256(experiment: ExperimentConfig) -> str:
    """Hash a resolved experiment definition independently of replicate seed."""
    payload = experiment.to_dict()
    payload["seed"] = 0
    training = payload.get("training")
    if isinstance(training, dict):
        training["seed"] = 0
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
