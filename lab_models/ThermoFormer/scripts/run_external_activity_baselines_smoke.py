"""Run validation-only interface/coverage smoke tests for fixed activity baselines."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.machine_learning.formal import (
    FormalConfig,
    _fixed_external_activity_result,
)
from src.thermoformer.baselines.machine_learning.protocols import load_registered_vle_dataset
from src.thermoformer.baselines.machine_learning.spt_nrtl import (
    SPT_NRTL_DATABASE_REVISION,
    SPTNRTLDatabase,
    SPTNRTLPredictor,
)
from src.thermoformer.baselines.machine_learning.tennet_sac import (
    CudaTeNNetSACPredictor,
    TeNNetSACPredictor,
    installed_tennetsac_asset_audit,
    verified_tennetsac_wheel,
)
from src.thermoformer.data.splitting import canonical_smiles, load_split_assignment
from src.thermoformer.reporting.artifacts import artifact_sha256, atomic_write_json


SMOKE_SYSTEMS = {
    (2, "isothermal"): frozenset(("CC(C)O", "CC(C)CC(C)(C)C")),
    (2, "isobaric"): frozenset(("CC(C)O", "CC(C)CC(C)(C)C")),
    (3, "isothermal"): frozenset(("CC(C)CO", "CC(C)CC(C)(C)C", "CC(=O)CC(C)C")),
    (3, "isobaric"): frozenset(("CO", "COC(=O)OC", "CC(=O)CC(C)C")),
}

SMOKE_IMPLEMENTATION_FILES = (
    "src/thermoformer/baselines/machine_learning/external_activity.py",
    "src/thermoformer/baselines/machine_learning/formal.py",
    "src/thermoformer/baselines/machine_learning/spt_nrtl.py",
    "src/thermoformer/baselines/machine_learning/tennet_sac.py",
    "scripts/run_external_activity_baselines_smoke.py",
    "scripts/run_ml_baselines_formal.py",
)


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def _validation_rows():
    samples = load_registered_vle_dataset(PROJECT_ROOT)
    split = load_split_assignment(
        PROJECT_ROOT / 'datasets/splits/vle/vle_overall_binary_ternary/seed_0.json',
        samples,
    )
    selected = []
    for (count, direction), system in SMOKE_SYSTEMS.items():
        candidates = [
            row for row in split.validation
            if row.component_count == count
            and row.experiment_mode == direction
            and frozenset(map(canonical_smiles, row.smiles)) == system
            and min(row.liquid_composition) >= 0.02
        ]
        if not candidates:
            raise RuntimeError(f"Registered validation smoke cell is empty: {count}, {direction}")
        selected.append(candidates[len(candidates) // 2])
    return split.train, tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("tennet_sac", "spt_nrtl"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/vle/comparison/diagnostic_records/external_activity_baselines_smoke"),
    )
    args = parser.parse_args()
    train, validation = _validation_rows()
    if args.baseline == "tennet_sac":
        predictor = TeNNetSACPredictor()
        reference = predictor.log_gamma(("CCCC", "CCO"), 330.0, (0.5, 0.5))
        expected_reference = torch.tensor([0.6033937335014343, 0.3612835109233856])
        if not torch.allclose(reference, expected_reference, atol=2e-6, rtol=2e-6):
            raise RuntimeError("Official TeNNet-SAC binary reference prediction changed")
        cuda_reference = None
        if torch.cuda.is_available():
            cuda_predictor = CudaTeNNetSACPredictor(torch.device("cuda"))
            cuda_reference = cuda_predictor.log_gamma(
                ("CCCC", "CCO"), 330.0, (0.5, 0.5)
            )
            if not torch.allclose(cuda_reference, reference, atol=1.1e-4, rtol=1.1e-4):
                raise RuntimeError("CUDA TeNNet-SAC adapter is not equivalent to the official API")
            predictor = cuda_predictor
        assets = {
            "package": "tennetsac==0.1.10",
            "model_version": "tuned_mean_of_10_experimental_finetuned_heads",
            "verified_wheel": verified_tennetsac_wheel(
                PROJECT_ROOT / 'experiments/run_records/cache/tennetsac'
            ),
            "installed_assets": installed_tennetsac_asset_audit(),
            "official_api_reference": {
                "smiles": ["CCCC", "CCO"],
                "temperature_k": 330.0,
                "composition": [0.5, 0.5],
                "log_gamma": reference.tolist(),
                "cuda_log_gamma": cuda_reference.tolist() if cuda_reference is not None else None,
                "cuda_equivalence_tolerance": 1.1e-4,
            },
        }
    else:
        database = SPTNRTLDatabase(PROJECT_ROOT / 'experiments/run_records/cache/spt_nrtl', 90.0)
        predictor = SPTNRTLPredictor(database)
        assets = {
            "database_revision": SPT_NRTL_DATABASE_REVISION,
            "database_download_attempts": database.download_attempts,
            "database_license": "not_declared_at_fixed_revision_publication_use_requires_review",
            "database_query_audit": database.audit,
            "database_pair_audit": database.pair_audit,
        }
    result = _fixed_external_activity_result(
        args.baseline,
        predictor,
        train,
        validation,
        "external_fixed_to_joint_test",
        FormalConfig(device="cpu"),
        assets,
    )
    payload = {
        "schema_version": 1,
        "status": "smoke_passed" if len(result.predictions) == 4 else "smoke_partial_coverage",
        "run_kind": "smoke",
        "analysis_status": "diagnostic",
        "selection_partition": "not_applicable_fixed_external_model",
        "evaluation_partition": "validation",
        "test_rows_used": False,
        "baseline": args.baseline,
        "registered_cells": 4,
        "valid_predictions": len(result.predictions),
        "metrics": list(result.rows),
        "model_audit": result.checkpoint,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "requirements_sha256": artifact_sha256(PROJECT_ROOT / "requirements.txt"),
        "environment_sha256": artifact_sha256(PROJECT_ROOT / "environment.yml"),
        "implementation_sha256": {
            path: artifact_sha256(PROJECT_ROOT / path) for path in SMOKE_IMPLEMENTATION_FILES
        },
    }
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    destination = output_root / args.baseline / "smoke_result.json"
    payload = _json_finite(payload)
    atomic_write_json(destination, payload)
    print(json.dumps({"output": str(destination), **{key: payload[key] for key in ("status", "valid_predictions")}}, indent=2))


if __name__ == "__main__":
    main()

