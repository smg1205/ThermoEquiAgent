"""Build manuscript-facing ThermoFormer result reports."""

from __future__ import annotations

import argparse

from build_c1_ablation_report import main as build_ablation_report
from build_c1_generalization_report import main as build_generalization_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=("generalization", "ablation", "all"))
    args = parser.parse_args()
    if args.report in ("generalization", "all"):
        build_generalization_report()
    if args.report in ("ablation", "all"):
        build_ablation_report()


if __name__ == "__main__":
    main()
