"""Create ternary-only projections of the registered joint paper splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_splits import _ternary_only_projection
from src.thermoformer.data import load_vle_dataset, retain_pure_anchored_systems
from src.thermoformer.data.splitting import load_split_assignment, save_split_assignment
from src.thermoformer.protocols.registry import PAPER_SEEDS


def main() -> None:
    loaded = load_vle_dataset(
        PROJECT_ROOT / 'datasets/vle_reference',
        failed_weight=0.0,
        max_pressure_kpa=500.0,
    )
    samples = retain_pure_anchored_systems(loaded.samples, minimum_temperatures=2)
    rows = []
    for seed in PAPER_SEEDS:
        source_path = PROJECT_ROOT / f"datasets/splits/vle/overall_binary_ternary/seed_{seed}.json"
        source = load_split_assignment(source_path, samples)
        projected = _ternary_only_projection(source)
        output_path = PROJECT_ROOT / f"datasets/splits/vle/overall_ternary/seed_{seed}.json"
        save_split_assignment(output_path, samples, projected)
        restored = load_split_assignment(output_path, samples)
        if restored != projected:
            raise RuntimeError(f"Split round-trip mismatch: {output_path}")
        rows.append(
            {
                "seed": seed,
                "train_rows": len(projected.train),
                "validation_rows": len(projected.validation),
                "test_rows": len(projected.test),
                "path": str(output_path),
            }
        )
    print(json.dumps({"status": "completed", "protocol": "overall_ternary", "splits": rows}, indent=2))


if __name__ == "__main__":
    main()
