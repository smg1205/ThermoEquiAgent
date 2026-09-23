"""Run structural VLE ablations on the release joint binary--ternary split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_overall_binary_three_stage_ablations as campaign


SPLIT_PROTOCOL = "vle_overall_binary_ternary"
CONFIG_NAMESPACE = Path("configs/vle/ablation/studies/vle_joint")
ARTIFACT_NAMESPACE = Path("benchmarks/vle/ablation/joint")
TRAINED_VARIANTS = tuple(
    variant for variant in campaign.VARIANT_IDS
    if variant not in {"c1_three_view_vanilla", "v3_functional_group_only"}
)


def _artifact_roots(
    project_root: Path, *, stage: str, smoke: bool
) -> tuple[Path, Path, Path]:
    if stage not in {"stage0", "three_stage"}:
        raise ValueError("stage must be stage0 or three_stage")
    namespace = ARTIFACT_NAMESPACE / ("smoke" if smoke else "formal") / stage
    return (project_root / "experiments/vle/training_records/reference/ablation/joint" / namespace.relative_to(ARTIFACT_NAMESPACE), project_root / "models/vle/benchmarks/vle/ablation/joint" / namespace.relative_to(ARTIFACT_NAMESPACE), project_root / "experiments/vle/ablation/joint" / namespace.relative_to(ARTIFACT_NAMESPACE))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--variant", action="append", choices=TRAINED_VARIANTS)
    value.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    variants = tuple(arguments.variant or TRAINED_VARIANTS)
    campaign.SPLIT_PROTOCOL = SPLIT_PROTOCOL
    campaign.FORMAL_NAMESPACE = CONFIG_NAMESPACE
    campaign.SMOKE_NAMESPACE = Path("runs") / ARTIFACT_NAMESPACE / "smoke"
    campaign.artifact_roots = _artifact_roots
    forwarded: list[str] = []
    for variant in variants:
        forwarded += ["--variant", variant]
    forwarded += ["--seeds", *map(str, arguments.seeds), "--device", arguments.device]
    if arguments.smoke:
        forwarded.append("--smoke")
    if arguments.overwrite:
        forwarded.append("--overwrite")
    campaign.main(forwarded)


if __name__ == "__main__":
    main()


