"""Run the binary-only, three-stage ThermoFormer ablation campaign.

The campaign is intentionally isolated from the legacy joint binary--ternary
ablation artifacts.  Each structural variant first obtains a supervised Stage
0 reference on the registered ``overall_binary`` split, then runs the direct
GE, joint VLE, and fugacity-constrained stages through ``run_paper_experiment``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_c1_physics_finetune import recover_completed_seed_manifest
from src.thermoformer.configuration import ExperimentConfig, load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import (
    requested_run_fingerprint,
    result_protocol_name,
    run_paper_experiment,
)
from src.thermoformer.reporting.aggregation import aggregate_protocol_results
from src.thermoformer.reporting.artifacts import artifact_sha256


SPLIT_PROTOCOL = "overall_binary"
FORMAL_NAMESPACE = Path("configs/vle/ablation/studies/overall_binary_three_stage")
SMOKE_NAMESPACE = Path("experiments/run_records/smoke/overall_binary_three_stage")
STAGES = ("stage0", "stage1", "stage2", "stage3")
THREE_STAGE_ARTIFACTS = (
    tuple(f"{stage}_checkpoint" for stage in STAGES)
    + tuple(f"{stage}_predictions" for stage in STAGES)
    + ("stage_comparison",)
)


_STAGE0_RECIPE_SCHEMA = "thermoformer-stage0-recipe-v1"
_RUNTIME_MODEL_FIELDS = frozenset(
    {
        "feature_dim",
        "rdkit_feature_dim",
        "unimol_feature_dim",
        "functional_group_feature_dim",
    }
)
_CHECKPOINT_MANIFEST_PROVENANCE_FIELDS = (
    "dataset_sha256",
    "split_sha256",
    "feature_cache_sha256",
    "feature_subset_sha256",
    "feature_definition_sha256",
    "rdkit_descriptor_list_sha256",
    "rdkit_scaler_sha256",
    "functional_group_vocabulary_sha256",
    "pure_property_catalog_sha256",
    "unimol_cache_sha256",
)

@dataclass(frozen=True)
class AblationVariant:
    """Stable identifier and display label for one retained Table 2 variant."""

    identifier: str
    label: str


VARIANTS = {
    "c0_current_vanilla": AblationVariant(
        "c0_current_vanilla", "Uni-Mol v2 only"
    ),
    "v1_rdkit_only": AblationVariant("v1_rdkit_only", "RDKit descriptors only"),
    "v3_functional_group_only": AblationVariant(
        "v3_functional_group_only", "Functional-group features only"
    ),
    "v4_rdkit_unimol_naive": AblationVariant(
        "v4_rdkit_unimol_naive", "RDKit descriptors + Uni-Mol v2"
    ),
    "c1_three_view_vanilla": AblationVariant(
        "c1_three_view_vanilla", "Full three-view vanilla Transformer"
    ),
    "c2_chemical_bias_full": AblationVariant(
        "c2_chemical_bias_full",
        "Chemical-interaction-biased Transformer with context pair potential",
    ),
    "c3_no_pair_bias": AblationVariant(
        "c3_no_pair_bias", "Context-conditioned pair potential without attention bias"
    ),
}
VARIANT_IDS = tuple(VARIANTS)


@dataclass(frozen=True)
class Stage0Reference:
    """A checkpoint and optional same-campaign manifest used by Stage 1."""

    checkpoint: Path
    manifest: Path | None
    source: Literal["generated", "legacy_c1"]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--variant",
        action="append",
        choices=VARIANT_IDS,
        help="Run one retained structural variant. Repeat to run a subset.",
    )
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument(
        "--smoke",
        action="store_true",
        help="Run seed 0 only with one epoch in every training stage.",
    )
    value.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing Stage 0 or three-stage seed bundle.",
    )
    return value


def resolve_seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    """Restrict formal experiments to the registered five random seeds."""

    seeds = tuple(arguments.seeds or ((0,) if arguments.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if arguments.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    return seeds


def selected_variants(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Keep CLI order deterministic and reject accidental duplicate work."""

    variants = tuple(arguments.variant or VARIANT_IDS)
    if len(variants) != len(set(variants)):
        raise ValueError("Each --variant may be supplied at most once")
    return variants


def config_path(project_root: Path, variant: str) -> Path:
    """Return the formal three-stage configuration for one variant."""

    _require_variant(variant)
    return (
        project_root
        / FORMAL_NAMESPACE
        / "three_stage"
        / f"{variant}.yaml"
    )


def stage0_config_path(project_root: Path, variant: str) -> Path:
    """Return the supervised reference configuration for one variant."""

    _require_variant(variant)
    return project_root / FORMAL_NAMESPACE / "stage0" / f"{variant}.yaml"


def split_path(project_root: Path, seed: int) -> Path:
    return project_root / 'datasets/splits/vle' / SPLIT_PROTOCOL / f"seed_{seed}.json"


def protocol_name(variant: str) -> str:
    """Use the legacy-compatible artifact basename requested for reporting."""

    _require_variant(variant)
    return result_protocol_name(variant, SPLIT_PROTOCOL)


def artifact_roots(
    project_root: Path,
    *,
    stage: Literal["stage0", "three_stage"],
    smoke: bool,
) -> tuple[Path, Path, Path]:
    """Resolve strictly separate formal and diagnostic artifact namespaces."""

    if stage not in {"stage0", "three_stage"}:
        raise ValueError("stage must be stage0 or three_stage")
    if smoke:
        root = project_root / SMOKE_NAMESPACE / stage
        return root / 'experiments/run_records', root / 'models/vle', root / 'experiments/reference_results'

    relative = FORMAL_NAMESPACE if stage == "three_stage" else FORMAL_NAMESPACE / "stage0"
    return (
        project_root / 'experiments/run_records' / relative,
        project_root / 'models/vle' / relative,
        project_root / 'experiments/reference_results' / relative,
    )


def smoke_overrides(stage: Literal["stage0", "three_stage"]) -> tuple[str, ...]:
    """Shorten every trainable stage without changing formal configuration files."""

    common = (
        "protocol.evaluation_partition=validation",
        "training.epochs_supervised=1",
        "training.minimum_supervised_epochs=1",
        "training.batch_size=32",
        "training.solver_iterations_eval=8",
    )
    if stage == "stage0":
        return common
    if stage == "three_stage":
        return (
            *common,
            "direct_ge_supervision.pretrain_epochs=1",
            "direct_ge_supervision.fugacity_epochs=1",
        )
    raise ValueError("stage must be stage0 or three_stage")


def legacy_c1_stage0_paths(project_root: Path, seed: int) -> tuple[Path, Path]:
    """Locate the pre-existing binary C1 supervised checkpoint and manifest."""

    return (
        project_root / 'models/vle' / SPLIT_PROTOCOL / f"seed_{seed}" / "best_model.pt",
        project_root / 'experiments/vle/prediction/reference' / SPLIT_PROTOCOL / f"seed_{seed}" / "manifest.json",
    )


def _canonical_json_sha256(payload: object) -> str:
    """Return a stable digest for semantic provenance records."""

    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _semantic_model_recipe(
    model: Mapping[str, object], encoder: Mapping[str, object]
) -> dict[str, object]:
    """Drop runtime dimensions while retaining every architectural switch."""

    normalized = {
        str(name): value
        for name, value in model.items()
        if name not in _RUNTIME_MODEL_FIELDS
    }
    normalized["fusion_mode"] = encoder.get("fusion_mode")
    normalized["chemical_attention_bias"] = encoder.get(
        "chemical_attention_bias"
    )
    normalized["context_pair_interaction"] = encoder.get(
        "context_pair_interaction"
    )
    return normalized


def _configured_file(project_root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _current_c1_stage0_recipe(
    project_root: Path, seed: int
) -> dict[str, object] | None:
    """Resolve all C1 Stage 0 inputs before deciding legacy reuse."""

    try:
        experiment = load_experiment_config(
            stage0_config_path(project_root, "c1_three_view_vanilla")
        )
        _validate_config(
            experiment, variant="c1_three_view_vanilla", stage="stage0"
        )
        split_file = split_path(project_root, seed)
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(split_payload, Mapping):
        return None
    partitions = split_payload.get("partitions")
    if not isinstance(partitions, Mapping):
        return None
    partition_recipe: dict[str, dict[str, object]] = {}
    for partition in ("train", "validation", "test"):
        values = partitions.get(partition)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            return None
        partition_recipe[partition] = {
            "count": len(values),
            "sha256": _canonical_json_sha256(values),
        }
    dataset_sha256 = split_payload.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or not dataset_sha256:
        return None
    payload = experiment.to_dict()
    data = payload.get("data")
    encoder = payload.get("encoder")
    model = payload.get("model")
    training = payload.get("training")
    protocol = payload.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (data, encoder, model, training, protocol)
    ):
        return None
    cache_path = project_root / "cache" / encoder_cache_filename(experiment.encoder)
    catalog_path = _configured_file(
        project_root, experiment.data.pure_property_catalog
    )
    if not cache_path.is_file() or (
        catalog_path is not None and not catalog_path.is_file()
    ):
        return None
    effective_training = dict(training)
    effective_training["seed"] = seed
    return {
        "schema": _STAGE0_RECIPE_SCHEMA,
        "variant": "c1_three_view_vanilla",
        "protocol": {
            "split_protocol": SPLIT_PROTOCOL,
            "seed": seed,
            "registered_splits": list(protocol.get("registered_splits", ())),
            "split_sha256": artifact_sha256(split_file),
            "dataset_sha256": dataset_sha256,
            "partitions": partition_recipe,
            "metadata_sha256": _canonical_json_sha256(
                split_payload.get("metadata")
            ),
        },
        "data": {
            "config": dict(data),
            "dataset_sha256": dataset_sha256,
            "pure_property_catalog_sha256": (
                artifact_sha256(catalog_path) if catalog_path is not None else None
            ),
        },
        "features": {
            "encoder": dict(encoder),
            "cache_sha256": artifact_sha256(cache_path),
        },
        "model": _semantic_model_recipe(model, encoder),
        "training": effective_training,
        "loss": {
            "pressure_weight": effective_training.get("pressure_weight"),
            "pure_weight": effective_training.get("pure_weight"),
            "direct_ge_supervision": None,
            "physics_finetuning": None,
        },
        "selection": {
            "partition": "validation",
            "evaluation_partition": "validation",
            "test_metrics_used_for_selection": False,
        },
    }


def stage0_recipe_fingerprint(recipe: Mapping[str, object]) -> str:
    """Fingerprint the complete, versioned Stage 0 recipe."""

    return _canonical_json_sha256(dict(recipe))


def legacy_stage0_matches_current_recipe(
    legacy_recipe: object,
    legacy_recipe_sha256: object,
    current_recipe: Mapping[str, object],
) -> tuple[bool, tuple[str, ...]]:
    """Compare an auditable legacy recipe without accepting partial metadata."""

    if not isinstance(legacy_recipe, Mapping):
        return False, ("missing stage0_recipe",)
    if not isinstance(legacy_recipe_sha256, str) or not legacy_recipe_sha256:
        return False, ("missing stage0_recipe_sha256",)
    canonical_legacy = dict(legacy_recipe)
    if stage0_recipe_fingerprint(canonical_legacy) != legacy_recipe_sha256:
        return False, ("stage0_recipe_sha256 mismatch",)
    canonical_current = dict(current_recipe)
    if canonical_legacy == canonical_current:
        return True, ()
    changed = tuple(
        sorted(
            str(key)
            for key in set(canonical_legacy) | set(canonical_current)
            if canonical_legacy.get(key) != canonical_current.get(key)
        )
    )
    return False, changed or ("unknown recipe mismatch",)


def _resolved_config_matches_stage0_recipe(
    project_root: Path,
    artifacts: Mapping[str, object],
    recipe: Mapping[str, object],
) -> bool:
    """Bind a legacy recipe to the hash-verified resolved configuration."""

    record = artifacts.get("resolved_config")
    if not isinstance(record, Mapping):
        return False
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        return False
    resolved_path = (project_root / path_value).resolve()
    if not resolved_path.is_file() or artifact_sha256(resolved_path) != expected_hash:
        return False
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(resolved, Mapping):
        return False
    data = resolved.get("data")
    encoder = resolved.get("encoder")
    model = resolved.get("model")
    training = resolved.get("training")
    if not all(
        isinstance(value, Mapping) for value in (data, encoder, model, training)
    ):
        return False
    recipe_data = recipe.get("data")
    recipe_features = recipe.get("features")
    recipe_training = recipe.get("training")
    recipe_protocol = recipe.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (recipe_data, recipe_features, recipe_training, recipe_protocol)
    ):
        return False
    if dict(data) != recipe_data.get("config"):
        return False
    if dict(encoder) != recipe_features.get("encoder"):
        return False
    if _semantic_model_recipe(model, encoder) != recipe.get("model"):
        return False
    effective_training = dict(training)
    effective_training["seed"] = recipe_protocol.get("seed")
    if effective_training != dict(recipe_training):
        return False
    paper_protocol = resolved.get("paper_protocol")
    return (
        isinstance(paper_protocol, Mapping)
        and paper_protocol.get("split_protocol") == SPLIT_PROTOCOL
        and paper_protocol.get("seed") == recipe_protocol.get("seed")
        and paper_protocol.get("dataset_sha256")
        == recipe_data.get("dataset_sha256")
    )


def _checkpoint_matches_stage0_recipe(
    checkpoint: Path,
    manifest: Mapping[str, object],
    recipe: Mapping[str, object],
) -> bool:
    """Verify that the bytes declared by the manifest carry the same recipe."""

    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    protocol = recipe.get("protocol")
    data = recipe.get("data")
    features = recipe.get("features")
    selection = recipe.get("selection")
    training = recipe.get("training")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, data, features, selection, training)
    ):
        return False
    for field in _CHECKPOINT_MANIFEST_PROVENANCE_FIELDS:
        if field not in manifest or payload.get(field) != manifest.get(field):
            return False
    if any(
        (
            payload.get("dataset_sha256") != data.get("dataset_sha256"),
            payload.get("split_sha256") != protocol.get("split_sha256"),
            payload.get("feature_cache_sha256") != features.get("cache_sha256"),
            payload.get("pure_property_catalog_sha256")
            != data.get("pure_property_catalog_sha256"),
            payload.get("split_protocol") != SPLIT_PROTOCOL,
            payload.get("seed") != protocol.get("seed"),
            payload.get("run_kind") != "formal",
            payload.get("analysis_status") != "confirmatory",
            payload.get("evaluation_partition")
            != selection.get("evaluation_partition"),
            payload.get("selection_partition") != selection.get("partition"),
            payload.get("test_metrics_used_for_selection")
            != selection.get("test_metrics_used_for_selection"),
            payload.get("stage1_checkpoint") is not None,
        )
    ):
        return False
    model_config = payload.get("model_config")
    encoder = features.get("encoder")
    if not isinstance(model_config, Mapping) or not isinstance(encoder, Mapping):
        return False
    if _semantic_model_recipe(model_config, encoder) != recipe.get("model"):
        return False
    if payload.get("training_config") != dict(training):
        return False
    return payload.get("model") is not None


def legacy_c1_stage0_reference(project_root: Path, seed: int) -> Stage0Reference | None:
    """Reuse old C1 only after complete semantic and byte-level validation."""

    checkpoint, manifest = legacy_c1_stage0_paths(project_root, seed)
    if not checkpoint.is_file() or not manifest.is_file():
        return None
    if not _tracked_and_clean(project_root, checkpoint):
        return None
    current_recipe = _current_c1_stage0_recipe(project_root, seed)
    if current_recipe is None:
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "status": "completed",
        "run_kind": "formal",
        "analysis_status": "confirmatory",
        "protocol": SPLIT_PROTOCOL,
        "split_protocol": SPLIT_PROTOCOL,
        "seed": seed,
        "git_dirty": False,
        "selection_partition": "validation",
        "evaluation_partition": "validation",
        "test_metrics_used_for_selection": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        return None
    if payload.get("split_sha256") != artifact_sha256(split_path(project_root, seed)):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    recipe_matches, _ = legacy_stage0_matches_current_recipe(
        payload.get("stage0_recipe"),
        payload.get("stage0_recipe_sha256"),
        current_recipe,
    )
    if not recipe_matches:
        return None
    if not _resolved_config_matches_stage0_recipe(project_root, artifacts, current_recipe):
        return None
    record = artifacts.get("checkpoint")
    if not isinstance(record, dict):
        return None
    recorded_path = record.get("path")
    recorded_hash = record.get("sha256")
    if not isinstance(recorded_path, str) or not isinstance(recorded_hash, str):
        return None
    declared = (project_root / recorded_path).resolve()
    if declared != checkpoint.resolve() or artifact_sha256(checkpoint) != recorded_hash:
        return None
    with checkpoint.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            return None
    if not _checkpoint_matches_stage0_recipe(checkpoint, payload, current_recipe):
        return None
    return Stage0Reference(checkpoint, None, "legacy_c1")

def _tracked_and_clean(project_root: Path, path: Path) -> bool:
    """Require a legacy reuse input to remain byte-equivalent to HEAD."""

    try:
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return False
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return False
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=project_root,
        capture_output=True,
    )
    return clean.returncode == 0


def _require_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")


def _validate_config(
    config: ExperimentConfig,
    *,
    variant: str,
    stage: Literal["stage0", "three_stage"],
) -> None:
    """Make accidental protocol/architecture drift fail before training."""

    if config.name != variant:
        raise ValueError(f"{stage} configuration name must be {variant}")
    if config.protocol is None:
        raise ValueError(f"{stage} configuration must declare the registered protocol")
    if config.protocol.registered_splits != (SPLIT_PROTOCOL,):
        raise ValueError(f"{stage} configuration must use {SPLIT_PROTOCOL}")
    if config.protocol.seeds != tuple(range(5)):
        raise ValueError(f"{stage} configuration must declare seeds 0--4")
    expected_partition = "validation" if stage == "stage0" else "test"
    if config.protocol.evaluation_partition != expected_partition:
        raise ValueError(
            f"{stage} configuration must evaluate {expected_partition}, not "
            f"{config.protocol.evaluation_partition}"
        )
    if stage == "stage0":
        if config.direct_ge_supervision is not None:
            raise ValueError("Stage 0 must remain supervised-only")
        if config.training.epochs_physics != 0:
            raise ValueError("Stage 0 cannot enable physics fine-tuning")
        return
    if config.direct_ge_supervision is None or config.physics_finetuning is None:
        raise ValueError("The three-stage configuration requires direct-GE and fugacity stages")
    if config.direct_ge_supervision.pretrain_epochs != 20:
        raise ValueError("Stage 1 must use 20 direct-GE epochs")
    if config.training.epochs_supervised != 80:
        raise ValueError("Stage 2 must use the 80-epoch supervised budget")
    if config.direct_ge_supervision.fugacity_epochs != 10:
        raise ValueError("Stage 3 must use 10 fugacity epochs")


def _manifest_path(
    roots: tuple[Path, Path, Path], variant: str, seed: int
) -> Path:
    return roots[2] / protocol_name(variant) / f"seed_{seed}" / "manifest.json"


def _checkpoint_path(
    roots: tuple[Path, Path, Path], variant: str, seed: int
) -> Path:
    return roots[1] / protocol_name(variant) / f"seed_{seed}" / "best_model.pt"


def _read_recorded_commit(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("git_commit")
    return value if isinstance(value, str) and value else None


def _recover_if_complete(
    *,
    manifest_path: Path,
    config_path: Path,
    split: Path,
    feature_cache: Path,
    seed: int,
    device: str,
    overrides: tuple[str, ...],
    run_kind: str,
    evaluation_partition: str,
    stage1_checkpoint: Path | None,
    aggregate_expected: bool,
    analysis_status: str,
    variant: str,
    stage: Literal["stage0", "three_stage"],
) -> dict[str, object] | None:
    """Reuse only an artifact bundle with matching hashes and request inputs."""

    recorded_commit = _read_recorded_commit(manifest_path)
    request_sha256 = requested_run_fingerprint(
        config_path,
        split,
        seed,
        feature_cache,
        device,
        overrides,
        run_kind,
        evaluation_partition,
        stage1_checkpoint,
        aggregate_expected,
        recorded_commit,
        analysis_status,
    )
    recovered = recover_completed_seed_manifest(
        manifest_path,
        expected_status="smoke" if run_kind == "smoke" else "completed",
        expected_protocol=protocol_name(variant),
        expected_seed=seed,
        expected_evaluation_partition=evaluation_partition,
        expected_request_sha256=request_sha256,
        expected_analysis_status=analysis_status,
    )
    if recovered is not None and stage == "three_stage":
        _require_three_stage_artifacts(recovered, manifest_path)
    return recovered


def _resolve_project_artifact_path(value: object, name: str) -> Path:
    """Resolve a recorded artifact path without allowing project-root escape."""

    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Three-stage {name} has no artifact path")
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Three-stage {name} escapes the project root") from error
    return resolved


def _require_manifest_artifact(
    artifacts: dict[str, object],
    name: str,
    *,
    expected: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Require one manifest artifact and verify its recorded SHA-256."""

    record = artifacts.get(name)
    if not isinstance(record, dict):
        raise RuntimeError(f"Three-stage manifest lacks artifact metadata: {name}")
    path = _resolve_project_artifact_path(record.get("path"), name)
    digest = record.get("sha256")
    if not isinstance(digest, str) or not digest:
        raise RuntimeError(f"Three-stage {name} has no SHA-256")
    if expected is not None and path != expected.resolve():
        raise RuntimeError(f"Three-stage {name} points to an unexpected location")
    if not path.is_file() or artifact_sha256(path) != digest:
        raise RuntimeError(f"Three-stage {name} failed SHA-256 verification")
    return path, record


def _require_stage_artifact_link(
    stage: str,
    payload: dict[str, object],
    field: str,
    artifact_path: Path,
    artifact_record: dict[str, object],
) -> None:
    """Require stage-comparison provenance to match the manifest artifact."""

    name = f"{stage} {field}"
    linked_path = _resolve_project_artifact_path(payload.get(field), name)
    linked_digest = payload.get(f"{field}_sha256")
    if linked_path != artifact_path:
        raise RuntimeError(f"Three-stage {name} path does not match the manifest")
    if (
        not isinstance(linked_digest, str)
        or linked_digest != artifact_record.get("sha256")
        or artifact_sha256(linked_path) != linked_digest
    ):
        raise RuntimeError(
            f"Three-stage {name} SHA-256 does not match the manifest artifact"
        )


def _require_three_stage_artifacts(
    manifest: dict[str, object], manifest_path: Path
) -> None:
    """Reject a result that lacks verified checkpoint and prediction evidence."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Three-stage manifest has no artifact index")
    missing = [name for name in THREE_STAGE_ARTIFACTS if name not in artifacts]
    if missing:
        raise RuntimeError(
            "Three-stage manifest lacks independent stage artifacts: " + ", ".join(missing)
        )
    comparison, _ = _require_manifest_artifact(
        artifacts,
        "stage_comparison",
        expected=manifest_path.parent / "stage_comparison.json",
    )
    try:
        payload = json.loads(comparison.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Stage comparison is unreadable: {comparison}") from error
    stages = payload.get("stages")
    if not isinstance(stages, dict) or not set(STAGES).issubset(stages):
        raise RuntimeError(
            "Stage comparison must retain Stage 0, Stage 1, Stage 2, and Stage 3"
        )
    if manifest_path.resolve().parent != comparison.parent:
        raise RuntimeError("Stage comparison must reside beside its seed manifest")
    for stage in STAGES:
        checkpoint, checkpoint_record = _require_manifest_artifact(
            artifacts, f"{stage}_checkpoint"
        )
        predictions, prediction_record = _require_manifest_artifact(
            artifacts,
            f"{stage}_predictions",
            expected=manifest_path.parent / f"{stage}_predictions.csv",
        )
        stage_payload = stages.get(stage)
        if not isinstance(stage_payload, dict):
            raise RuntimeError(f"Stage comparison has malformed {stage} evidence")
        _require_stage_artifact_link(
            stage, stage_payload, "checkpoint", checkpoint, checkpoint_record
        )
        _require_stage_artifact_link(
            stage, stage_payload, "predictions", predictions, prediction_record
        )


def _run_one(
    *,
    project_root: Path,
    variant: str,
    stage: Literal["stage0", "three_stage"],
    seed: int,
    device: str,
    smoke: bool,
    overwrite: bool,
    stage1_reference: Stage0Reference | None = None,
    aggregate_expected: bool = False,
) -> tuple[dict[str, object], Path, Path]:
    """Run or hash-validate a single stage/seed artifact bundle."""

    path = stage0_config_path(project_root, variant) if stage == "stage0" else config_path(project_root, variant)
    overrides = smoke_overrides(stage) if smoke else ()
    formal_experiment = load_experiment_config(path)
    _validate_config(formal_experiment, variant=variant, stage=stage)
    experiment = (
        load_experiment_config(path, overrides) if overrides else formal_experiment
    )
    roots = artifact_roots(project_root, stage=stage, smoke=smoke)
    split = split_path(project_root, seed)
    if not split.is_file():
        raise FileNotFoundError(f"Missing registered split: {split}")
    feature_cache = project_root / "cache" / encoder_cache_filename(experiment.encoder)
    evaluation_partition = "validation" if smoke or stage == "stage0" else "test"
    run_kind = "smoke" if smoke else "formal"
    analysis_status = "diagnostic" if smoke else "confirmatory"
    checkpoint = stage1_reference.checkpoint if stage1_reference is not None else None
    generated_manifest = stage1_reference.manifest if stage1_reference is not None else None
    manifest_path = _manifest_path(roots, variant, seed)
    recovered = None
    if not overwrite:
        recovered = _recover_if_complete(
            manifest_path=manifest_path,
            config_path=path,
            split=split,
            feature_cache=feature_cache,
            seed=seed,
            device=device,
            overrides=overrides,
            run_kind=run_kind,
            evaluation_partition=evaluation_partition,
            stage1_checkpoint=checkpoint,
            aggregate_expected=aggregate_expected,
            analysis_status=analysis_status,
            variant=variant,
            stage=stage,
        )
    manifest = recovered or run_paper_experiment(
        config_path=path,
        split_path=split,
        seed=seed,
        run_root=roots[0],
        checkpoint_root=roots[1],
        results_root=roots[2],
        feature_cache=feature_cache,
        device_name=device,
        overrides=overrides,
        allow_overwrite=overwrite,
        run_kind=run_kind,
        evaluation_partition=evaluation_partition,
        stage1_checkpoint=checkpoint,
        generated_stage1_manifest=generated_manifest,
        aggregate_expected=aggregate_expected,
        analysis_status=analysis_status,
    )
    if stage == "three_stage":
        _require_three_stage_artifacts(manifest, manifest_path)
    return manifest, _checkpoint_path(roots, variant, seed), manifest_path


def _stage0_reference(
    *,
    project_root: Path,
    variant: str,
    seed: int,
    device: str,
    smoke: bool,
    overwrite: bool,
) -> Stage0Reference:
    """Use auditable legacy C1 only for formal work; otherwise train Stage 0."""

    if variant == "c1_three_view_vanilla" and not smoke and not overwrite:
        legacy = legacy_c1_stage0_reference(project_root, seed)
        if legacy is not None:
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "stage": "stage0",
                        "status": "reused",
                        "source": legacy.source,
                        "checkpoint": str(legacy.checkpoint),
                    }
                ),
                flush=True,
            )
            return legacy
    _, checkpoint, manifest = _run_one(
        project_root=project_root,
        variant=variant,
        stage="stage0",
        seed=seed,
        device=device,
        smoke=smoke,
        overwrite=overwrite,
    )
    if not checkpoint.is_file() or not manifest.is_file():
        raise RuntimeError("Stage 0 did not materialize its checkpoint and manifest")
    return Stage0Reference(checkpoint, manifest, "generated")


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    seeds = resolve_seeds(arguments)
    variants = selected_variants(arguments)
    complete_formal_campaign = not arguments.smoke and seeds == tuple(range(5))
    final_roots = artifact_roots(
        PROJECT_ROOT, stage="three_stage", smoke=arguments.smoke
    )

    for variant in variants:
        for seed in seeds:
            reference = _stage0_reference(
                project_root=PROJECT_ROOT,
                variant=variant,
                seed=seed,
                device=arguments.device,
                smoke=arguments.smoke,
                overwrite=arguments.overwrite,
            )
            try:
                manifest, _, _ = _run_one(
                    project_root=PROJECT_ROOT,
                    variant=variant,
                    stage="three_stage",
                    seed=seed,
                    device=arguments.device,
                    smoke=arguments.smoke,
                    overwrite=arguments.overwrite,
                    stage1_reference=reference,
                    aggregate_expected=False,
                )
            finally:
                _release_accelerator_memory()
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "status": manifest["status"],
                        "selected_stage": manifest.get("selected_stage"),
                        "stage0_source": reference.source,
                    }
                ),
                flush=True,
            )

        if complete_formal_campaign:
            protocol_dir = final_roots[2] / protocol_name(variant)
            aggregate_protocol_results(protocol_dir, expected_seeds=tuple(range(5)))
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "status": "aggregated",
                        "aggregate_manifest": str(protocol_dir / "aggregate_manifest.json"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
