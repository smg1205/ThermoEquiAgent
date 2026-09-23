"""Generate the reproducible ThermoFormer dataset audit artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auditing import write_data_audit_report, write_ternary_subsystem_csv
from src.thermoformer.data import load_vle_dataset, retain_pure_anchored_systems


def main() -> None:
    dataset_root = PROJECT_ROOT / 'datasets/vle_reference'
    reports = PROJECT_ROOT / "reports"
    combined = load_vle_dataset(dataset_root, failed_weight=0.0, max_pressure_kpa=500.0)
    per_workbook = {
        name: load_vle_dataset(
            dataset_root,
            source_filter=Path(name).stem,
            failed_weight=0.0,
            max_pressure_kpa=500.0,
        )
        for name in ("binary_vle_english.xlsx", "ternary_vle_english.xlsx")
    }
    modeling = retain_pure_anchored_systems(
        combined.samples,
        minimum_temperatures=2,
    )
    payload = write_data_audit_report(
        reports / "data_audit.md",
        combined,
        modeling,
        per_workbook,
    )
    write_ternary_subsystem_csv(
        reports / "ternary_binary_subsystem_coverage.csv",
        modeling,
    )
    print(json.dumps(payload["modeling"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
