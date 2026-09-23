"""Evaluate a trained ThermoFormer checkpoint on a registered split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import experiment_sha256, load_experiment_config
from src.thermoformer.data import load_vle_dataset, retain_pure_anchored_systems
from src.thermoformer.data.splitting import dataset_digest, load_split_assignment
from src.thermoformer.evaluation import evaluate_protocol, write_prediction_csv
from src.thermoformer.features import (
    build_molecular_encoder,
    encoder_cache_filename,
    feature_subset_sha256,
    prepare_partition_features,
)
from src.thermoformer.models import ThermoFormer, ThermoFormerConfig
from src.thermoformer.reporting import artifact_sha256, atomic_write_json
from src.thermoformer.thermodynamics.vapor_pressure import (
    empty_pure_property_catalog,
    load_pure_property_catalog,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--split", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--partition", choices=("validation", "test"), default="test")
    value.add_argument("--feature-cache", type=Path, default=None)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / 'experiments/run_records/evaluation')
    value.add_argument("--overwrite", action="store_true")
    return value


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def validate_checkpoint_provenance(
    checkpoint: dict[str, object],
    expected_identities: dict[str, object],
    expected_model_config: ThermoFormerConfig,
) -> None:
    """Reject evaluation inputs that are not the checkpoint's recorded inputs."""
    for key, expected in expected_identities.items():
        if expected is not None and checkpoint.get(key) != expected:
            raise RuntimeError(f"Checkpoint provenance mismatch for {key}")
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise RuntimeError("Checkpoint model configuration does not match the experiment")


def validate_checkpoint_selection(
    checkpoint: dict[str, object],
    partition: str,
) -> None:
    """Prevent an unselected Stage-2 candidate from being evaluated on test data."""
    if (
        partition == "test"
        and checkpoint.get("checkpoint_role") == "best_physics_epoch_by_validation"
        and checkpoint.get("selected_as_final") is not True
    ):
        raise RuntimeError(
            "The Stage-2 checkpoint was not selected on validation and cannot be "
            "evaluated on the test partition"
        )


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    experiment = load_experiment_config(args.config)
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)

    loaded = load_vle_dataset(
        _resolve(experiment.data.root),
        source_filter=experiment.data.source_filter,
        failed_weight=experiment.data.failed_weight,
        max_pressure_kpa=experiment.data.max_pressure_kpa,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=experiment.data.minimum_pure_anchor_temperatures,
    )
    split = load_split_assignment(args.split, samples)
    if experiment.protocol is not None:
        experiment.protocol.validate_request(split.protocol, split.seed, args.partition)
    cache = args.feature_cache or PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    encoder = build_molecular_encoder(experiment.encoder, cache, use_cuda=device.type == "cuda")
    all_smiles = sorted({smiles for sample in samples for smiles in sample.smiles})
    train_smiles = sorted({smiles for sample in split.train for smiles in sample.smiles})
    features = prepare_partition_features(encoder, all_smiles, train_smiles)
    catalog_path = (
        _resolve(experiment.data.pure_property_catalog)
        if experiment.data.pure_property_catalog
        else None
    )

    view_dimensions = features.view_dimensions
    multiview = experiment.encoder.fusion_mode != "legacy"
    expected_model_config = replace(
        experiment.model,
        feature_dim=int(next(iter(features.values.values())).shape[0]),
        fusion_mode=experiment.encoder.fusion_mode,
        rdkit_feature_dim=view_dimensions["rdkit_2d"] if multiview else 0,
        unimol_feature_dim=view_dimensions["unimol_v2"] if multiview else 0,
        functional_group_feature_dim=(view_dimensions["functional_groups"] if multiview else 0),
        chemical_attention_bias=experiment.encoder.chemical_attention_bias,
        context_pair_interaction=experiment.encoder.context_pair_interaction,
    )
    resolved_experiment = replace(
        experiment,
        seed=split.seed,
        training=replace(experiment.training, seed=split.seed),
        model=expected_model_config,
    )
    checkpoint_sha256 = artifact_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_selection(checkpoint, args.partition)
    expected_identities = {
        "dataset_sha256": dataset_digest(samples),
        "split_sha256": artifact_sha256(args.split),
        "split_protocol": split.protocol,
        "seed": split.seed,
        "resolved_config_sha256": experiment_sha256(resolved_experiment),
        "feature_subset_sha256": feature_subset_sha256(features.values),
        "feature_definition_sha256": features.metadata["feature_definition_sha256"],
        "rdkit_descriptor_list_sha256": features.metadata.get(
            "rdkit_descriptor_definition_sha256"
        ),
        "functional_group_vocabulary_sha256": features.metadata.get(
            "functional_group_vocabulary_sha256"
        ),
        "rdkit_scaler_sha256": features.metadata.get("rdkit_scaler", {}).get("sha256"),
        "feature_cache_sha256": artifact_sha256(cache),
        "pure_property_catalog_sha256": (
            artifact_sha256(catalog_path) if catalog_path is not None else None
        ),
    }
    validate_checkpoint_provenance(
        checkpoint,
        expected_identities,
        expected_model_config,
    )
    model = ThermoFormer(expected_model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    catalog = (
        load_pure_property_catalog(catalog_path)
        if catalog_path is not None
        else empty_pure_property_catalog()
    )
    result = evaluate_protocol(
        model,
        samples,
        split,
        features.values,
        partition=args.partition,
        batch_size=experiment.training.batch_size,
        device=device,
        solver_iterations=experiment.training.solver_iterations_eval,
        pure_property_catalog=catalog,
    )
    output_dir = (
        args.output_dir
        / result.protocol
        / f"seed_{result.seed}"
        / result.partition
        / checkpoint_sha256[:16]
    )
    completion = output_dir / "evaluation_manifest.json"
    if completion.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation already exists: {completion}")
    prediction_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    write_prediction_csv(prediction_path, result.predictions)
    atomic_write_json(metrics_path, result.metrics)
    atomic_write_json(
        completion,
        {
            "status": "completed_diagnostic_evaluation",
            "protocol": result.protocol,
            "seed": result.seed,
            "partition": result.partition,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": experiment_sha256(resolved_experiment),
            "split_sha256": artifact_sha256(args.split),
            "feature_subset_sha256": expected_identities["feature_subset_sha256"],
            "pure_property_catalog_sha256": expected_identities[
                "pure_property_catalog_sha256"
            ],
            "predictions_sha256": artifact_sha256(prediction_path),
            "metrics_sha256": artifact_sha256(metrics_path),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
