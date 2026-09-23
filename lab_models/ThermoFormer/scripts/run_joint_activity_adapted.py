"""Run HANNA/TeNNet-SAC residual adaptations on a registered component scope."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.machine_learning.artifacts import (
    aggregate_seed_metrics,
    write_seed_metrics,
)
from src.thermoformer.baselines.machine_learning.formal import (
    FormalConfig,
    _atomic_csv,
    _fixed_external_activity_result,
)
from src.thermoformer.baselines.machine_learning.hanna import (
    OfficialHANNAAssets,
    OfficialHANNALogGammaPredictor,
)
from src.thermoformer.baselines.machine_learning.joint_activity_adapted import (
    JointActivityConfig,
    JointAdaptedActivityPredictor,
    train_joint_activity_head,
)
from src.thermoformer.baselines.machine_learning.protocols import load_registered_vle_dataset
from src.thermoformer.baselines.machine_learning.tennet_sac import (
    CudaTeNNetSACPredictor,
    TeNNetSACPredictor,
)
from src.thermoformer.data.splitting import (
    canonical_smiles,
    dataset_digest,
    load_split_assignment,
)
from src.thermoformer.reporting.artifacts import artifact_sha256


class MemoizedPredictor:
    def __init__(self, predictor: object) -> None:
        self.predictor = predictor
        self.cache: dict[tuple[tuple[str, ...], float, tuple[float, ...]], torch.Tensor] = {}

    @staticmethod
    def _key(smiles, temperature_k, composition):
        return (
            tuple(smiles),
            float(temperature_k),
            tuple(float(value) for value in composition),
        )

    def log_gamma(self, smiles, temperature_k, composition):
        key = self._key(smiles, temperature_k, composition)
        if key not in self.cache:
            self.cache[key] = self.predictor.log_gamma(*key).detach().cpu()
        return self.cache[key].clone()

    def log_gamma_many(self, samples):
        missing = [
            sample for sample in samples
            if self._key(sample.smiles, sample.temperature_k, sample.liquid_composition)
            not in self.cache
        ]
        batch_method = getattr(self.predictor, "log_gamma_batch", None)
        if missing and batch_method is not None:
            values = batch_method(missing)
            for sample, value in zip(missing, values):
                self.cache[self._key(
                    sample.smiles, sample.temperature_k, sample.liquid_composition
                )] = value.detach().cpu()
        elif missing:
            for sample in missing:
                self.log_gamma(sample.smiles, sample.temperature_k, sample.liquid_composition)
        return [
            self.log_gamma(sample.smiles, sample.temperature_k, sample.liquid_composition)
            for sample in samples
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=("hanna_joint_adapted", "tennet_sac_joint_adapted"),
        default=("hanna_joint_adapted", "tennet_sac_joint_adapted"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--benchmark",
        choices=("binary_train_binary_test", "ternary_train_ternary_test", "joint_train_joint_test"),
        default="joint_train_joint_test",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/vle/comparison/v1/machine_learning"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/vle/comparison/v1/machine_learning"),
    )
    return parser.parse_args()


def benchmark_partitions(split, benchmark: str):
    """Return train/validation/test rows for the requested registered scope."""
    if benchmark == "joint_train_joint_test":
        return split.train, split.validation, split.test
    if benchmark in {"binary_train_binary_test", "ternary_train_ternary_test"}:
        component_count = 2 if benchmark == "binary_train_binary_test" else 3
        partitions = tuple(
            tuple(row for row in rows if row.component_count == component_count)
            for rows in (split.train, split.validation, split.test)
        )
        if any(not rows for rows in partitions):
            raise ValueError(f"{component_count}-component benchmark produced an empty partition")
        return partitions
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def psat_calibration_partition(split):
    """Return registered training rows used only to recover pure-component Psat.

    Interaction adaptation continues to use benchmark_partitions. The
    vapor-pressure fitter itself accepts only near-pure endpoints (x >= 0.999),
    so binary mixture activity labels never enter ternary-only adaptation.
    """

    return tuple(split.train)


def main() -> None:
    args = parse_args()
    if any(seed not in range(5) for seed in args.seeds):
        raise ValueError("Seeds are frozen to 0--4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    samples = load_registered_vle_dataset(PROJECT_ROOT)
    dataset_sha256 = dataset_digest(samples)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result_root = args.result_root if args.result_root.is_absolute() else PROJECT_ROOT / args.result_root
    checkpoint_root = (
        args.checkpoint_root
        if args.checkpoint_root.is_absolute()
        else PROJECT_ROOT / args.checkpoint_root
    )
    if args.smoke:
        result_root = (
            PROJECT_ROOT / 'experiments/vle/comparison/diagnostic_records/activity_adapted_smoke' / args.benchmark
        )
        checkpoint_root = result_root / 'models/vle'

    for baseline in args.baselines:
        if baseline == "hanna_joint_adapted":
            base = MemoizedPredictor(
                OfficialHANNALogGammaPredictor(
                    OfficialHANNAAssets.default(PROJECT_ROOT), device
                )
            )
        else:
            base = MemoizedPredictor(
                CudaTeNNetSACPredictor(device) if device.type == "cuda" else TeNNetSACPredictor()
            )
        seed_files = []
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            split_protocol = (
                "vle_overall_binary"
                if args.benchmark == "binary_train_binary_test"
                else "vle_overall_binary_ternary"
            )
            split_path = PROJECT_ROOT / 'datasets/splits/vle' / split_protocol / f"seed_{seed}.json"
            split = load_split_assignment(split_path, samples)
            train, validation, test = benchmark_partitions(split, args.benchmark)
            psat_train = psat_calibration_partition(split)
            if args.smoke:
                def interior(rows, limit):
                    return tuple(
                        row for row in rows
                        if min(row.liquid_composition) > 1.0e-4
                        and min(row.vapor_composition) > 1.0e-4
                    )[:limit]
                train, validation = interior(train, 24), interior(validation, 12)
                test = interior(split.validation[12:], 12)
            from src.thermoformer.baselines.machine_learning.formal import fit_vapor_pressure_correlations
            vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(psat_train)
            print(
                f"[activity-adapted] baseline={baseline} seed={seed} "
                f"interaction_train_rows={len(train)} "
                f"interaction_validation_rows={len(validation)} "
                f"psat_calibration_rows={len(psat_train)} "
                f"psat_components={len(vapor_pressure)}",
                flush=True,
            )
            if args.smoke:
                def covered(rows, limit):
                    return tuple(
                        row for row in rows
                        if all(canonical_smiles(value) in vapor_pressure for value in row.smiles)
                    )[:limit]
                train = covered(interior(train, 256), 24)
                validation = covered(interior(validation, 256), 12)
                test = validation
            adaptation = JointActivityConfig(
                epochs=2 if args.smoke else args.epochs,
                patience=2 if args.smoke else args.patience,
            )
            trained = train_joint_activity_head(
                train, validation, vapor_pressure, base, adaptation, device
            )
            predictor = JointAdaptedActivityPredictor(base, trained.head, device)
            official_key = baseline.replace("_joint_adapted", "")
            evaluated = _fixed_external_activity_result(
                official_key,
                predictor,
                psat_train,
                test,
                args.benchmark,
                FormalConfig(device=args.device),
                {"base_model": official_key, "frozen": True},
            )
            scope = {
                "binary_train_binary_test": "binary",
                "ternary_train_ternary_test": "ternary",
                "joint_train_joint_test": "joint",
            }[args.benchmark]
            artifact_baseline = baseline.replace("_joint_adapted", f"_{scope}_adapted")
            evaluated = replace(
                evaluated,
                rows=tuple({**row, "baseline": artifact_baseline, "benchmark": args.benchmark} for row in evaluated.rows),
                predictions=tuple({**row, "baseline": artifact_baseline, "benchmark": args.benchmark} for row in evaluated.predictions),
            )
            if args.smoke and not evaluated.predictions:
                raise RuntimeError("Smoke test produced no valid validation predictions")
            seed_root = result_root / f"{artifact_baseline}.on.{args.benchmark}" / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            metrics_path = seed_root / "metrics.csv"
            write_seed_metrics(metrics_path, evaluated.rows, seed)
            _atomic_csv(seed_root / "predictions.csv", evaluated.predictions or ({"status": "no_predictions"},))
            _atomic_csv(seed_root / "history.csv", trained.history)
            checkpoint_path = (
                checkpoint_root / f"{artifact_baseline}.on.{args.benchmark}" / f"seed_{seed}" / "best_model.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": artifact_baseline,
                    "head_state_dict": trained.head.state_dict(),
                    "config": asdict(adaptation),
                    "best_epoch": trained.best_epoch,
                    "train_rows": trained.train_rows,
                    "validation_rows": trained.validation_rows,
                    "vapor_pressure_audit": vapor_audit,
                    "vapor_pressure_calibration": {
                        "partition": "registered_training_only",
                        "eligible_rows": "near-pure endpoints with x >= 0.999",
                        "calibration_rows_scanned": len(psat_train),
                        "binary_mixture_activity_labels_used": False,
                    },
                    "selection_partition": "validation",
                    "evaluation_partition": "test" if not args.smoke else "validation_smoke",
                    "test_labels_used_for_selection": False,
                    "parameter_accounting": {
                        "residual_head_trainable": sum(p.numel() for p in trained.head.parameters()),
                        "official_base_optimized": 0,
                    },
                },
                checkpoint_path,
            )
            manifest = {
                "status": "smoke_passed" if args.smoke else "completed",
                "baseline": artifact_baseline,
                "benchmark": args.benchmark,
                "seed": seed,
                "selection_partition": "validation",
                "evaluation_partition": "validation_smoke" if args.smoke else "test",
                "test_labels_used_for_selection": False,
                "config": asdict(adaptation),
                "train_rows": trained.train_rows,
                "validation_rows": trained.validation_rows,
                "vapor_pressure_calibration": {
                    "partition": "registered_training_only",
                    "eligible_rows": "near-pure endpoints with x >= 0.999",
                    "calibration_rows_scanned": len(psat_train),
                    "binary_mixture_activity_labels_used": False,
                },
                "base_predictions_cached": len(base.cache),
                "source_commit": source_commit,
                "dataset_sha256": dataset_sha256,
                "split_protocol": split_protocol,
                "split_sha256": artifact_sha256(split_path),
                "checkpoint": {
                    "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": artifact_sha256(checkpoint_path),
                },
            }
            (seed_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            seed_files.append(metrics_path)
        if not args.smoke and set(args.seeds) == set(range(5)):
            aggregate_seed_metrics(
                seed_files,
                result_root / f"{artifact_baseline}.on.{args.benchmark}" / "aggregate.csv",
            )
        del base
        torch.cuda.empty_cache()
    print(json.dumps({
        "status": "completed",
        "benchmark": args.benchmark,
        "baselines": args.baselines,
        "seeds": args.seeds,
    }))


if __name__ == "__main__":
    main()

