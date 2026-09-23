"""Write the formal ThermoFormer-vs-experiment table for heptane/nonane.

The source is the locked ``vle_overall_binary`` seed-2 test prediction file.
It contains NIST ThermoML experimental labels for DOI
10.1016/j.fluid.2013.05.016; no values are re-predicted or fitted here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SYSTEM_ID = "2c_be94156040a0f89bf1c4"
INPUT = REPO / "experiments" / "vle" / "prediction" / "reference" / "overall_binary" / "seed_2" / "predictions.csv"
OUTPUT = ROOT / "report" / "heptane_nonane_thermoformer_vs_experiment.csv"


def main() -> int:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["system_id"] == SYSTEM_ID]
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 locked test rows for {SYSTEM_ID}, found {len(rows)}")

    output: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["x_1"])):
        t_exp = float(row["target_temperature_k"]) - 273.15
        t_tf = float(row["predicted_temperature_k"]) - 273.15
        y_exp = float(row["y_true_1"])
        y_tf = float(row["y_pred_1"])
        output.append({
            "pressure_kPa": 101.325,
            "x_heptane": float(row["x_1"]),
            "T_experiment_C": round(t_exp, 6),
            "T_thermoformer_C": round(t_tf, 6),
            "y_heptane_experiment": y_exp,
            "y_heptane_thermoformer": round(y_tf, 6),
            "abs_temperature_error_C": round(abs(t_tf - t_exp), 6),
            "abs_y_error": round(abs(y_tf - y_exp), 6),
            "doi": row["doi"],
            "quality_status": row["quality_status"],
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
