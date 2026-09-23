"""Build a registered C1 ablation report."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.reporting.c1_ablation import (
    write_binary_c1_ablation_outputs,
    write_joint_c1_ablation_outputs,
)


def main() -> None:
    """Build the selected C1 ablation tables and reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("binary", "joint"),
        default="binary",
        help="Dataset scope of the registered ablation report.",
    )
    args = parser.parse_args()
    writer = write_binary_c1_ablation_outputs if args.scope == "binary" else write_joint_c1_ablation_outputs
    for output in writer(PROJECT_ROOT):
        print(output)


if __name__ == "__main__":
    main()
