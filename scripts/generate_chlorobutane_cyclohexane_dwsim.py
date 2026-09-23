"""Write v5-style DWSIM near-bubble flashes for 1-chlorobutane/cyclohexane.

The five representative liquid compositions are selected from the NIST
ThermoML 101.30 kPa isobar and agree with the comparison table in
``report/Agent整合ThermoFormer进度与三源验证报告v5.md``.  Experimental and
ThermoFormer values are read from the paired report CSV; DWSIM alone supplies
the flash temperature and vapor composition.

Run this only in a full local process with DWSIM and pythonnet available:

    python scripts/generate_chlorobutane_cyclohexane_dwsim.py
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


# Component order deliberately follows the experimental CSV and v5 table.
SYSTEM = {
    "name": "chlorobutane_cyclohexane",
    "compounds": [
        (
            "1-chlorobutane",
            [
                # DWSIM's current compound catalogue uses IUPAC-style
                # ``chloranyl`` nomenclature for haloalkanes (CAS 109-69-3).
                "1-Chloranylbutane",
                "1-Chlorobutane",
                "n-Butyl chloride",
                "Butyl chloride",
            ],
        ),
        ("cyclohexane", ["Cyclohexane"]),
    ],
    "t_min_c": 40.0,
    "t_max_c": 100.0,
}

# The same five-point presentation convention as v5 §2.1.
REPRESENTATIVE_X = (0.1005, 0.2796, 0.4705, 0.6967, 0.8960)
P_KPA = 101.30
P_TOL_KPA = 0.05


def _reference_rows() -> list[dict[str, str]]:
    path = ROOT / "report" / "chlorobutane_cyclohexane_thermoformer_vs_experiment.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for x_target in REPRESENTATIVE_X:
        candidates = [
            row
            for row in records
            if abs(float(row["pressure_kPa"]) - P_KPA) <= P_TOL_KPA
        ]
        selected.append(
            min(candidates, key=lambda row: abs(float(row["x_1_chlorobutane"]) - x_target))
        )
    return selected


def main() -> int:
    outdir = ROOT / "report" / "success" / "1-氯丁烷-环己烷"
    outdir.mkdir(parents=True, exist_ok=True)

    # These globals are the tested v5 flash implementation.  It uses DWSIM's
    # Automation API to solve each PT flash; this driver merely fixes the system
    # and picks the five report compositions.
    factory, object_type = ded._automation_factory()
    demo._GLOBALS.automation = factory()
    demo._GLOBALS.object_type = object_type
    demo._GLOBALS.system = SYSTEM
    demo._GLOBALS.property_package = "UNIQUAC"

    rows: list[dict[str, object]] = []
    for reference in _reference_rows():
        x1 = float(reference["x_1_chlorobutane"])
        liquid = [x1, 1.0 - x1]
        tag = f"chlorobutane_x{x1:.4f}".replace(".", "p")
        t_c, y, filename = demo._bubble_with_file(liquid, tag, outdir)
        vf, y_saved = demo._flash_at(liquid, t_c)
        y1 = float(y_saved[0]) if len(y_saved) == 2 else float(y[0])
        rows.append(
            {
                "pressure_kPa": P_KPA,
                "x_1_chlorobutane": x1,
                "T_experiment_C": reference["T_experiment_C"],
                "T_thermoformer_C": reference["T_thermoformer_C"],
                "T_dwsim_C": round(t_c, 4),
                "y_1_chlorobutane_experiment": reference["y_1_chlorobutane_experiment"],
                "y_1_chlorobutane_thermoformer": reference["y_1_chlorobutane_thermoformer"],
                "y_1_chlorobutane_dwsim": round(y1, 6),
                "vapor_flow_mol_s": f"{vf:.8g}",
                "file": filename,
            }
        )
        print(
            f"x_1-chlorobutane={x1:.4f}; T_DWSIM={t_c:.4f} C; "
            f"y_1-chlorobutane={y1:.6f}; V/F={vf:.3e}; file={filename}"
        )

    csv_path = outdir / "chlorobutane_cyclohexane_three_source_bubble.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
