"""Generate exploratory phase diagrams from the joint binary--ternary test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.overall_phase_diagrams import (  # noqa: E402
    run_overall_phase_diagram_study,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/vle/prediction/studies/overall_binary_ternary/phase_diagrams/figure_config.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    binary, ternary, report = run_overall_phase_diagram_study(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        overwrite=args.overwrite,
    )
    print(f"Binary figure: {binary}")
    print(f"Ternary figure: {ternary}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
