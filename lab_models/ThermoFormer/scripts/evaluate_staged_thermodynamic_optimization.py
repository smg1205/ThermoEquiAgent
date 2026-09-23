"""Recompute validation-only diagnostics for saved VLE Stage 0--3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.data import (
    load_vle_dataset,
    retain_pure_anchored_systems,
)
from src.thermoformer.data.splitting import load_split_assignment
from src.thermoformer.features import (
    build_molecular_encoder,
    encoder_cache_filename,
    prepare_partition_features,
)
from src.thermoformer.models import ThermoFormer, ThermoFormerConfig
from src.thermoformer.reporting.artifacts import artifact_sha256, atomic_write_text
from src.thermoformer.thermodynamics.vapor_pressure import load_pure_property_catalog
from src.thermoformer.training.direct_ge import build_direct_thermodynamic_targets
from src.thermoformer.training.supervised import TrainingConfig, _loader, seed_everything


STAGES = ("stage0", "stage1", "stage2", "stage3")
METRICS = (
    "mean_abs_log_fugacity_residual",
    "p95_abs_log_fugacity_residual",
    "relative_pressure_closure_residual",
    "log_gamma_reference_error",
)


def _sha256_json(value: object) -> str:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _evaluate(model, samples, feature_map, training, catalog, device, minimum_fraction):
    loader = _loader(samples, feature_map, training, False, catalog)
    fugacity_residuals: list[float] = []
    pressure_closures: list[float] = []
    gamma_errors: list[float] = []
    model.to(device).eval()
    with torch.no_grad():
        for host_batch in loader:
            batch = host_batch.to(device)
            output = model(
                batch.molecules,
                batch.temperature_k,
                batch.pressure_kpa,
                batch.x,
                batch.mask,
            )
            psat = output.log_psat.exp()
            liquid = batch.x * output.log_gamma.exp() * psat
            vapor = batch.y * batch.pressure_kpa
            identifiable = (
                batch.mask.bool()
                & (batch.x >= minimum_fraction)
                & (batch.y >= minimum_fraction)
            )
            residual = torch.log(liquid.clamp_min(1e-12)) - torch.log(vapor.clamp_min(1e-12))
            fugacity_residuals.extend(residual[identifiable].abs().cpu().tolist())
            closure = ((liquid * batch.mask).sum(-1) / batch.pressure_kpa.squeeze(-1) - 1.0).abs()
            pressure_closures.extend(closure.cpu().tolist())

            targets = build_direct_thermodynamic_targets(
                batch, minimum_fraction=minimum_fraction
            )
            gamma_errors.extend(
                (output.log_gamma - targets.log_gamma)[targets.gamma_mask.bool()]
                .abs().cpu().tolist()
            )
    if not fugacity_residuals or not pressure_closures or not gamma_errors:
        raise RuntimeError("Validation diagnostics produced an empty metric population")
    return {
        "mean_abs_log_fugacity_residual": float(np.mean(fugacity_residuals)),
        "p95_abs_log_fugacity_residual": float(np.percentile(fugacity_residuals, 95.0)),
        "relative_pressure_closure_residual": float(np.mean(pressure_closures)),
        "log_gamma_reference_error": float(np.mean(gamma_errors)),
        "validation_states": len(samples),
        "fugacity_components": len(fugacity_residuals),
        "gamma_reference_components": len(gamma_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = root / "configs/vle/prediction/studies/vle/three_stage.yaml"
    experiment = load_experiment_config(config_path)
    data_root = root / experiment.data.root
    loaded = load_vle_dataset(
        data_root,
        source_filter=experiment.data.source_filter,
        failed_weight=experiment.data.failed_weight,
        max_pressure_kpa=experiment.data.max_pressure_kpa,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=experiment.data.minimum_pure_anchor_temperatures,
    )
    catalog = load_pure_property_catalog(root / experiment.data.pure_property_catalog)
    cache = root / "cache" / encoder_cache_filename(experiment.encoder)
    device = torch.device(args.device)
    output = root / "experiments/vle/ablation/staged_validation"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []

    for seed in args.seeds:
        checkpoint_dir = root / (
            "experiments/vle/training_records/v1/three/checkpoints/"
            "thermoformer_vle_three_stage.on.vle_overall_binary_ternary"
        ) / f"seed_{seed}"
        stage0_payload = torch.load(
            checkpoint_dir / "stage0_best_model.pt", map_location="cpu", weights_only=False
        )
        recorded_split_path = Path(stage0_payload["split_file"]).resolve()
        split_path = (
            recorded_split_path
            if recorded_split_path.is_file()
            else output / "frozen_splits" / f"seed_{seed}.json"
        )
        if artifact_sha256(split_path) != stage0_payload["split_sha256"]:
            raise RuntimeError(f"Frozen split SHA mismatch: seed {seed}")
        split = load_split_assignment(split_path, samples)
        seed_everything(seed)
        encoder = build_molecular_encoder(experiment.encoder, cache, use_cuda=device.type == "cuda")
        train_smiles = sorted({smiles for row in split.train for smiles in row.smiles})
        all_smiles = sorted({smiles for row in samples for smiles in row.smiles})
        feature_map = prepare_partition_features(encoder, all_smiles, train_smiles).values
        result_dir = root / (
            "experiments/vle/training_records/v1/three/results/"
            "thermoformer_vle_three_stage.on.vle_overall_binary_ternary"
        ) / f"seed_{seed}"
        comparison = json.loads((result_dir / "stage_comparison.json").read_text(encoding="utf-8"))
        if comparison.get("selection_partition") != "validation":
            raise RuntimeError(f"Seed {seed} checkpoints were not selected on validation")
        for stage in STAGES:
            checkpoint = checkpoint_dir / f"{stage}_best_model.pt"
            expected = comparison["stages"][stage]["checkpoint_sha256"]
            actual = artifact_sha256(checkpoint)
            if actual != expected:
                raise RuntimeError(f"Checkpoint SHA mismatch: seed {seed} {stage}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = ThermoFormer(ThermoFormerConfig(**payload["model_config"]))
            model.load_state_dict(payload["model"])
            training = TrainingConfig(**payload["training_config"])
            metrics = _evaluate(
                model, split.validation, feature_map, training, catalog, device,
                experiment.direct_ge_supervision.minimum_fraction,
            )
            row = {"seed": seed, "stage": stage, **metrics}
            rows.append(row)
            provenance.append({
                "seed": seed,
                "stage": stage,
                "checkpoint": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": actual,
                "split": str(split_path),
                "split_sha256": artifact_sha256(split_path),
                "selection_partition": "validation",
                "evaluation_partition": "validation",
            })
            print(json.dumps(row), flush=True)

    csv_path = output / "per_seed_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage]
        item: dict[str, object] = {"stage": stage, "seeds": len(stage_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in stage_rows]
            item[f"{metric}_mean"] = fmean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else None
        summary.append(item)
    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "status": "completed",
        "scope": "saved-checkpoint validation-only diagnostics",
        "selection_partition": "validation",
        "evaluation_partition": "validation",
        "retraining": False,
        "checkpoint_reselection": False,
        "test_set_evaluation": False,
        "seeds": args.seeds,
        "metric_definitions": {
            "mean_abs_log_fugacity_residual": "mean component-wise |ln(x_i gamma_i Psat_i)-ln(y_i P)| at observed validation states",
            "p95_abs_log_fugacity_residual": "95th percentile of the same component-wise absolute residual",
            "relative_pressure_closure_residual": "mean |sum_i(x_i gamma_i Psat_i)/P-1| over validation states",
            "log_gamma_reference_error": "mean absolute ln(gamma) error where external Psat makes the experimental reference identifiable",
        },
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": artifact_sha256(config_path),
        "dataset_files": [
            {"path": path.relative_to(root).as_posix(), "sha256": artifact_sha256(path)}
            for path in sorted(data_root.glob("*.xlsx"))
        ],
        "provenance": provenance,
        "per_seed_metrics_sha256": artifact_sha256(csv_path),
        "summary_sha256": artifact_sha256(summary_path),
        "request_sha256": _sha256_json({"seeds": args.seeds, "stages": STAGES, "metrics": METRICS}),
    }
    atomic_write_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()







