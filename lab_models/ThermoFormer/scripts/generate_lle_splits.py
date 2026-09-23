"""Generate registered strict LLE splits before directed augmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.data import build_lle_split, load_lle_dataset, save_lle_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if config.task_mode != "lle" or config.lle is None:
        raise ValueError("LLE split generation requires task_mode=lle")
    dataset = load_lle_dataset(
        PROJECT_ROOT / config.data.root, binary_workbook=config.lle.binary_workbook,
        ternary_workbook=config.lle.ternary_workbook, component_count=config.lle.component_count,
        failed_weight=config.data.failed_weight,
    )
    split = build_lle_split(dataset.samples, seed=args.seed)
    save_lle_split(args.output, dataset.samples, split)
    print(args.output)


if __name__ == "__main__":
    main()
