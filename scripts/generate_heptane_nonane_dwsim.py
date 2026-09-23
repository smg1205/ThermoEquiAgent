"""Write five DWSIM UNIQUAC near-bubble flashes for heptane/nonane.

The selected 101.325 kPa points are read from the formal experimental / locked
ThermoFormer comparison CSV.  DWSIM alone calculates the flash temperature and
vapor composition; it never substitutes for the experimental or model columns.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_ternary_dwsim_demonstration as demo
from thermo_engine import dwsim_export as ded

SYSTEM = {
    "name": "heptane_nonane",
    "compounds": [
        ("heptane", ["n-Heptane", "Heptane", "n-heptane"]),
        ("nonane", ["n-Nonane", "Nonane", "n-nonane"]),
    ],
    "t_min_c": 80.0,
    "t_max_c": 170.0,
}
REPRESENTATIVE_X = (0.117, 0.359, 0.466, 0.633, 0.837)
P_KPA = 101.325


def _reference_rows() -> list[dict[str, str]]:
    path = ROOT / "report" / "heptane_nonane_thermoformer_vs_experiment.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    return [
        min(all_rows, key=lambda row: abs(float(row["x_heptane"]) - target))
        for target in REPRESENTATIVE_X
    ]


def main() -> int:
    outdir = ROOT / "report" / "success" / "正庚烷-正壬烷"
    outdir.mkdir(parents=True, exist_ok=True)
    factory, object_type = ded._automation_factory()
    demo._GLOBALS.automation = factory()
    demo._GLOBALS.object_type = object_type
    demo._GLOBALS.system = SYSTEM
    demo._GLOBALS.property_package = "UNIQUAC"

    results: list[dict[str, object]] = []
    for reference in _reference_rows():
        x_heptane = float(reference["x_heptane"])
        tag = f"heptane_x{x_heptane:.3f}".replace(".", "p")
        t_c, y, filename = demo._bubble_with_file([x_heptane, 1.0 - x_heptane], tag, outdir)
        vapor_flow, y_saved = demo._flash_at([x_heptane, 1.0 - x_heptane], t_c)
        y_heptane = float(y_saved[0]) if len(y_saved) == 2 else float(y[0])
        results.append({
            "pressure_kPa": P_KPA,
            "x_heptane": x_heptane,
            "T_experiment_C": reference["T_experiment_C"],
            "T_thermoformer_C": reference["T_thermoformer_C"],
            "T_dwsim_C": round(t_c, 6),
            "y_heptane_experiment": reference["y_heptane_experiment"],
            "y_heptane_thermoformer": reference["y_heptane_thermoformer"],
            "y_heptane_dwsim": round(y_heptane, 6),
            "vapor_flow_mol_s": f"{vapor_flow:.8g}",
            "file": filename,
        })
        print(f"x_heptane={x_heptane:.3f}; T_DWSIM={t_c:.4f} C; y_heptane={y_heptane:.6f}; file={filename}")

    output = outdir / "heptane_nonane_three_source_bubble.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
