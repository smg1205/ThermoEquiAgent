"""Run C1 three-stage state, unseen-component, or transfer campaigns."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import result_protocol_name, run_paper_experiment
from src.thermoformer.reporting.aggregation import aggregate_protocol_results


BINARY_STATE_PROTOCOLS = (
    "binary_state_composition_interpolation",
    "binary_state_composition_edge_extrapolation",
    "binary_state_temperature_low_extrapolation",
    "binary_state_temperature_high_extrapolation",
    "binary_state_pressure_low_extrapolation",
    "binary_state_pressure_high_extrapolation",
)
BINARY_UNSEEN_PROTOCOLS = ("binary_unseen_component",)
BINARY_TRANSFER_PROTOCOLS = (
    "binary_to_ternary_zero_shot",
    "binary_to_ternary_scale_0.05",
    "binary_to_ternary_scale_0.1",
    "binary_to_ternary_scale_0.25",
    "binary_to_ternary_scale_0.5",
    "binary_to_ternary_scale_1",
)
GROUP_FOLDERS = {
    "state": "binary_state_generalization",
    "unseen": "binary_unseen_components",
    "transfer": "binary_to_ternary",
}
GROUP_PROTOCOLS = {
    "state": BINARY_STATE_PROTOCOLS,
    "unseen": BINARY_UNSEEN_PROTOCOLS,
    "transfer": BINARY_TRANSFER_PROTOCOLS,
}
REPORT_TITLES = {
    "state": "# Binary-only state-space generalization",
    "unseen": "# Binary-only unseen-component generalization",
    "transfer": "# Binary-to-ternary transfer with three-stage training",
}
REPORT_DESCRIPTIONS = {
    "state": (
        "All values are mean ± standard deviation across seeds 0--4. Training, validation, and test contain binary VLE rows only. Checkpoint selection uses validation data; test data are evaluated afterward."
    ),
    "unseen": (
        "All values are mean ± standard deviation across seeds 0--4. Training, validation, and test contain binary VLE rows only. Checkpoint selection uses validation data; test data are evaluated afterward."
    ),
    "transfer": (
        "All values are mean ± standard deviation across seeds 0--4. The ternary test partition is fixed across training fractions for each seed. Checkpoint selection uses validation data; test data are evaluated afterward."
    ),
}
DISPLAY_NAMES = {
    "binary_state_composition_interpolation": "Composition interpolation",
    "binary_state_composition_edge_extrapolation": "Composition-edge extrapolation",
    "binary_state_temperature_low_extrapolation": "Low-temperature extrapolation",
    "binary_state_temperature_high_extrapolation": "High-temperature extrapolation",
    "binary_state_pressure_low_extrapolation": "Low-pressure extrapolation",
    "binary_state_pressure_high_extrapolation": "High-pressure extrapolation",
    "binary_unseen_component": "Unseen components",
    "binary_to_ternary_zero_shot": "0% ternary (zero-shot)",
    "binary_to_ternary_scale_0.05": "5.56% ternary",
    "binary_to_ternary_scale_0.1": "10% ternary",
    "binary_to_ternary_scale_0.25": "25% ternary",
    "binary_to_ternary_scale_0.5": "50% ternary",
    "binary_to_ternary_scale_1": "100% ternary",
}


def campaign_config_path(project_root: Path, group: str) -> Path:
    return (
        project_root
        / "configs/vle/prediction/studies"
        / GROUP_FOLDERS[group]
        / "config.yaml"
    )


def supervised_config_path(project_root: Path, group: str) -> Path:
    return campaign_config_path(project_root, group).with_name("stage0_supervised.yaml")


def artifact_roots(
    project_root: Path,
    group: str,
    variant: str,
    *,
    smoke: bool,
) -> tuple[Path, Path, Path]:
    if variant not in {"stage0", "three_stage"}:
        raise ValueError("variant must be stage0 or three_stage")
    folder = GROUP_FOLDERS[group]
    if smoke:
        base = project_root / "experiments/run_records/smoke" / folder / variant
        return base / 'experiments/run_records', base / 'models/vle', base / 'experiments/reference_results'
    relative = Path("configs/vle/prediction/studies") / folder / variant
    return (
        project_root / 'experiments/run_records' / relative,
        project_root / 'models/vle' / relative,
        project_root / 'experiments/reference_results' / relative,
    )


def smoke_overrides(variant: str) -> tuple[str, ...]:
    common = (
        "protocol.evaluation_partition=validation",
        "training.epochs_supervised=1",
        "training.minimum_supervised_epochs=1",
        "training.batch_size=32",
        "training.solver_iterations_eval=8",
    )
    if variant == "stage0":
        return common
    if variant == "direct":
        return (
            *common,
            "direct_ge_supervision.pretrain_epochs=1",
            "direct_ge_supervision.fugacity_epochs=1",
        )
    raise ValueError("variant must be stage0 or direct")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--group", choices=tuple(GROUP_PROTOCOLS), required=True)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument(
        "--compatible-training-commits",
        nargs="+",
        default=None,
        help="Explicit commits approved for a provenance-preserving aggregate.",
    )
    value.add_argument(
        "--mixed-commit-justification",
        default=None,
        help="Reason that the listed commits are scientifically equivalent.",
    )
    return value


def _seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    seeds = tuple(arguments.seeds or ((0,) if arguments.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if arguments.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    return seeds


def aggregation_provenance_overrides(arguments: argparse.Namespace) -> dict[str, object]:
    commits = arguments.compatible_training_commits
    justification = arguments.mixed_commit_justification
    if commits is None and justification is None:
        return {}
    if commits is None or not isinstance(justification, str) or not justification.strip():
        raise ValueError("Mixed training commits require a non-empty justification")
    return {
        "compatible_training_git_commits": tuple(dict.fromkeys(commits)),
        "mixed_commit_justification": justification,
    }



def _run(
    *,
    config_path: Path,
    split_protocol: str,
    seed: int,
    roots: tuple[Path, Path, Path],
    device: str,
    smoke: bool,
    overwrite: bool,
    variant: str,
    stage0_checkpoint: Path | None = None,
    stage0_manifest: Path | None = None,
) -> tuple[Path, Path, str]:
    experiment = load_experiment_config(config_path)
    result_protocol = result_protocol_name(experiment.name, split_protocol)
    run_root, checkpoint_root, results_root = roots
    result_dir = results_root / result_protocol / f"seed_{seed}"
    manifest = result_dir / "manifest.json"
    checkpoint = checkpoint_root / result_protocol / f"seed_{seed}/best_model.pt"
    expected_status = "smoke" if smoke else "completed"
    if manifest.is_file() and checkpoint.is_file() and not overwrite:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") == expected_status:
            print(f"[{split_protocol}] {variant} seed {seed}: verified; skipping", flush=True)
            return manifest, checkpoint, result_protocol
        raise FileExistsError(f"Incomplete result exists: {manifest}; pass --overwrite")
    split = PROJECT_ROOT / f"datasets/splits/vle/{split_protocol}/seed_{seed}.json"
    if not split.is_file():
        raise FileNotFoundError(
            f"Missing registered split: {split}. Run scripts/generate_splits.py first."
        )
    overrides = smoke_overrides(variant) if smoke else ()
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    evaluation_partition = (
        "validation" if variant == "stage0" or smoke else "test"
    )
    run_paper_experiment(
        config_path=config_path,
        split_path=split,
        seed=seed,
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        results_root=results_root,
        feature_cache=feature_cache,
        device_name=device,
        overrides=overrides,
        allow_overwrite=overwrite,
        run_kind="smoke" if smoke else "formal",
        evaluation_partition=evaluation_partition,
        stage1_checkpoint=stage0_checkpoint,
        generated_stage1_manifest=stage0_manifest,
        aggregate_expected=(variant == "direct" and not smoke),
        analysis_status="diagnostic" if smoke else "confirmatory",
    )
    return manifest, checkpoint, result_protocol


def _metric(row: dict[str, object], field: str, digits: int) -> str:
    mean = row.get(f"{field}_mean")
    std = row.get(f"{field}_std")
    if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _write_report(
    group: str,
    summaries: dict[str, list[dict[str, object]]],
    result_protocols: dict[str, str],
) -> Path:
    lines = [
        REPORT_TITLES[group],
        "",
        REPORT_DESCRIPTIONS[group],
        "",
        "| Evaluation setting | Task | State MAE | State RMSE | State R² | y MAE | y RMSE | y R² | Solver failure rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    protocols = GROUP_PROTOCOLS[group]
    selected: Counter[str] = Counter()
    direct_results = artifact_roots(PROJECT_ROOT, group, "three_stage", smoke=False)[2]
    for split_protocol in protocols:
        for row in summaries[split_protocol]:
            if row.get("scope") != "direction":
                continue
            direction = str(row.get("direction"))
            if direction == "isothermal":
                task, state, unit, digits = "molecules,T,x → P,y", "pressure", "kPa", 2
                suffix = "kpa"
            elif direction == "isobaric":
                task, state, unit, digits = "molecules,P,x → T,y", "temperature", "K", 2
                suffix = "k"
            else:
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        DISPLAY_NAMES[split_protocol],
                        task,
                        f"{_metric(row, f'{state}_mae_{suffix}', digits)} {unit}",
                        f"{_metric(row, f'{state}_rmse_{suffix}', digits)} {unit}",
                        _metric(row, f"{state}_r2", 3),
                        _metric(row, "y_mae", 4),
                        _metric(row, "y_rmse", 4),
                        _metric(row, "y_r2", 3),
                        _metric(row, "solver_failure_rate", 4),
                    )
                )
                + " |"
            )
        protocol_dir = direct_results / result_protocols[split_protocol]
        for seed in range(5):
            comparison = json.loads(
                (protocol_dir / f"seed_{seed}/stage_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            selected[str(comparison["selected_stage"])] += 1
    lines.extend(
        (
            "",
            "Validation-selected checkpoints: "
            + ", ".join(f"{key}={value}" for key, value in sorted(selected.items())),
            "",
            "The model uses Stage 1 direct GE/RT and ln(gamma) supervision, Stage 2 joint VLE supervision, and Stage 3 ten-epoch fugacity fine-tuning.",
        )
    )
    report = (
        PROJECT_ROOT
        / "configs/vle/prediction/studies"
        / GROUP_FOLDERS[group]
        / "results.md"
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    seeds = _seeds(arguments)
    aggregation_options = aggregation_provenance_overrides(arguments)
    protocols = GROUP_PROTOCOLS[arguments.group]
    stage0_config = supervised_config_path(PROJECT_ROOT, arguments.group)
    direct_config = campaign_config_path(PROJECT_ROOT, arguments.group)
    stage0_roots = artifact_roots(PROJECT_ROOT, arguments.group, "stage0", smoke=arguments.smoke)
    direct_roots = artifact_roots(PROJECT_ROOT, arguments.group, "three_stage", smoke=arguments.smoke)
    result_protocols: dict[str, str] = {}
    for split_protocol in protocols:
        for seed in seeds:
            stage0_manifest, stage0_checkpoint, _ = _run(
                config_path=stage0_config,
                split_protocol=split_protocol,
                seed=seed,
                roots=stage0_roots,
                device=arguments.device,
                smoke=arguments.smoke,
                overwrite=arguments.overwrite,
                variant="stage0",
            )
            _, _, direct_protocol = _run(
                config_path=direct_config,
                split_protocol=split_protocol,
                seed=seed,
                roots=direct_roots,
                device=arguments.device,
                smoke=arguments.smoke,
                overwrite=arguments.overwrite,
                variant="direct",
                stage0_checkpoint=stage0_checkpoint,
                stage0_manifest=stage0_manifest,
            )
            result_protocols[split_protocol] = direct_protocol
            print(json.dumps({"protocol": split_protocol, "seed": seed, "status": "completed"}), flush=True)
    if not arguments.smoke and seeds == tuple(range(5)):
        summaries = {
            split_protocol: aggregate_protocol_results(
                direct_roots[2] / result_protocols[split_protocol],
                expected_seeds=seeds,
                **aggregation_options,
            )
            for split_protocol in protocols
        }
        print(json.dumps({"report": str(_write_report(arguments.group, summaries, result_protocols))}))


if __name__ == "__main__":
    main()
