"""Run one release-data VLE protocol through the audited C1 three-stage recipe."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import result_protocol_name, run_paper_experiment

CONFIG_ROOT = PROJECT_ROOT / "configs/vle/prediction/studies/vle"


def complete(path: Path, status: str) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == status


def run_stage(
    config_path: Path,
    split_path: Path,
    seed: int,
    roots: dict[str, Path],
    label: str,
    device: str,
    smoke: bool,
    stage0_checkpoint: Path | None = None,
    stage0_manifest: Path | None = None,
) -> tuple[Path, Path]:
    experiment = load_experiment_config(config_path)
    protocol = result_protocol_name(experiment.name, split_path.parent.name)
    result = roots[f"{label}_results"] / protocol / f"seed_{seed}"
    checkpoint = roots[f"{label}_checkpoints"] / protocol / f"seed_{seed}/best_model.pt"
    manifest = result / "manifest.json"
    expected = "smoke" if smoke else "completed"
    if complete(manifest, expected) and checkpoint.is_file():
        print(f"[resume] {label} {split_path.parent.name} seed {seed}", flush=True)
        return manifest, checkpoint
    if manifest.exists():
        raise RuntimeError(f"Stale or interrupted result requires a separate output directory: {manifest}")
    overrides = []
    if smoke:
        overrides += [
            "protocol.evaluation_partition=validation",
            "training.epochs_supervised=1",
            "training.minimum_supervised_epochs=1",
            "training.batch_size=32",
            "training.solver_iterations_eval=8",
        ]
        if label == "three":
            overrides += [
                "direct_ge_supervision.pretrain_epochs=1",
                "direct_ge_supervision.fugacity_epochs=1",
            ]
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    run_paper_experiment(
        config_path=config_path,
        split_path=split_path,
        seed=seed,
        run_root=roots[f"{label}_runs"],
        checkpoint_root=roots[f"{label}_checkpoints"],
        results_root=roots[f"{label}_results"],
        feature_cache=feature_cache,
        device_name=device,
        overrides=tuple(overrides),
        run_kind="smoke" if smoke else "formal",
        evaluation_partition="validation" if smoke or label == "stage0" else "test",
        stage1_checkpoint=stage0_checkpoint,
        generated_stage1_manifest=stage0_manifest if not smoke else None,
        aggregate_expected=False,
        analysis_status="diagnostic" if smoke else "confirmatory",
    )
    return manifest, checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or not set(args.seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    base = args.output.resolve()
    roots = {
        f"{stage}_{kind}": base / "vle" / stage / kind
        for stage in ("stage0", "three")
        for kind in ("runs", "checkpoints", "results")
    }
    for seed in args.seeds:
        split = PROJECT_ROOT / 'datasets/splits/vle' / args.protocol / f"seed_{seed}.json"
        if not split.is_file():
            raise FileNotFoundError(split)
        stage0_manifest, stage0_checkpoint = run_stage(
            CONFIG_ROOT / "stage0.yaml", split, seed, roots, "stage0",
            args.device, args.smoke,
        )
        try:
            final_manifest, _ = run_stage(
                CONFIG_ROOT / "three_stage.yaml", split, seed, roots, "three",
                args.device, args.smoke, stage0_checkpoint, stage0_manifest,
            )
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        print(json.dumps({"protocol": args.protocol, "seed": seed, "manifest": str(final_manifest)}), flush=True)


if __name__ == "__main__":
    main()
