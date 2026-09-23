"""Evaluate a frozen task_mode=lle checkpoint on a registered ternary LLE test split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.data import load_lle_dataset, load_lle_split
from src.thermoformer.features import build_molecular_encoder, encoder_cache_filename, prepare_partition_features
from src.thermoformer.models import ThermoFormer
from src.thermoformer.training.lle import evaluate_lle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    if experiment.task_mode != "lle" or experiment.lle is None or experiment.lle.component_count != 3:
        raise ValueError("This evaluator requires task_mode=lle with three components")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    data_root = PROJECT_ROOT / experiment.data.root
    loaded = load_lle_dataset(data_root, binary_workbook=experiment.lle.binary_workbook,
                              ternary_workbook=experiment.lle.ternary_workbook, component_count=3,
                              failed_weight=experiment.data.failed_weight)
    split = load_lle_split(args.split, loaded.samples)
    if split.seed != args.seed:
        raise ValueError("Split seed does not match --seed")
    cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    encoder = build_molecular_encoder(experiment.encoder, cache, use_cuda=device.type == "cuda")
    all_smiles = sorted({smiles for sample in loaded.samples for smiles in sample.smiles})
    train_smiles = sorted({smiles for sample in split.train for smiles in sample.smiles})
    features = prepare_partition_features(encoder, all_smiles, train_smiles)
    dimensions = features.view_dimensions
    model_config = replace(experiment.model, feature_dim=int(next(iter(features.values.values())).shape[0]),
                           fusion_mode=experiment.encoder.fusion_mode,
                           rdkit_feature_dim=dimensions["rdkit_2d"], unimol_feature_dim=dimensions["unimol_v2"],
                           functional_group_feature_dim=dimensions["functional_groups"],
                           chemical_attention_bias=experiment.encoder.chemical_attention_bias,
                           context_pair_interaction=experiment.encoder.context_pair_interaction)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("task_mode") != "lle" or payload.get("model_config") != asdict(model_config):
        raise ValueError("Checkpoint is not compatible with this LLE configuration")
    model = ThermoFormer(model_config)
    model.load_state_dict(payload["model"], strict=True)
    metrics, predictions = evaluate_lle(model.to(device), split.test, features.values,
                                        replace(experiment.training, seed=args.seed), experiment.lle, device)
    result = {"partition": "test", "metrics": metrics, "prediction_count": len(predictions)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
