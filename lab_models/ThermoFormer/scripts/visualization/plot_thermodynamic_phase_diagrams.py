"""Generate test-exposed low- and high-error VLE phase-diagram cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.phase_diagrams import run_phase_diagram_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    figure, report, manifest = run_phase_diagram_study(
        project_root=PROJECT_ROOT,
        settings_path=(
            PROJECT_ROOT
            / "configs/vle/comparison/studies/thermodynamic_models/phase_diagrams/settings.json"
        ),
        overwrite=args.overwrite,
    )
    print(f"Figure: {figure}")
    print(f"Report: {report}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
