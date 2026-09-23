"""Run leakage-safe diagnostic smoke tests for the audited ML VLE baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.machine_learning import SmokeConfig, run_smoke_suite, write_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("configs/vle/comparison/studies/machine_learning/settings.json"),
    )
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--split", type=Path, default=Path("datasets/splits/vle/vle_overall_binary_ternary/seed_0.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/run_records/smoke/ml_vle_baselines/seed_0"))
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else PROJECT_ROOT / args.settings
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1 or settings.get("formal", {}).get("enabled") is not False:
        raise ValueError("Smoke settings must use schema 1 with formal.enabled=false")
    smoke = settings.get("smoke")
    if not isinstance(smoke, dict) or int(smoke.get("seed", -1)) != args.seed:
        raise ValueError("Smoke settings do not match the requested seed")
    split = args.split if args.split.is_absolute() else PROJECT_ROOT / args.split
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    manifest = run_smoke_suite(
        PROJECT_ROOT,
        split,
        output,
        SmokeConfig(
            seed=args.seed,
            max_train_rows=args.max_train_rows or int(smoke["max_train_rows"]),
            max_validation_rows=args.max_validation_rows or int(smoke["max_validation_rows"]),
            epochs=int(smoke["epochs"]),
            settings_sha256=hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        ),
    )
    write_smoke_report(
        output,
        output / "results.md",
    )
    print(json.dumps({"status": manifest["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

