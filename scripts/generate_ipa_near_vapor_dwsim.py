"""Generate 2-propanol / water DWSIM files just above the bubble point.

The strict bubble-point files can show zero vapor in DWSIM after rounding or
reload, which is not visually useful for SI screenshots.  This script writes
near-bubble files with a small but nonzero vapor outlet flow.
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


TARGET_VAPOR_MOL_S = 1.0e-4
MAX_DELTA_C = 1.0


def _vapor_flow_at(x: list[float], t_c: float) -> tuple[float, list[float], object]:
    fs, vapor = demo._build_flash(x, t_c)
    demo._GLOBALS.automation.CalculateFlowsheet2(fs)
    flow = float(vapor.GetMolarFlow())
    y = [float(v) for v in list(vapor.GetOverallComposition())]
    return flow, y, fs


def _near_vapor_temperature(x: list[float], bubble_c: float) -> tuple[float, float, list[float], object]:
    lo = bubble_c
    hi = bubble_c + MAX_DELTA_C
    hi_flow, _, _ = _vapor_flow_at(x, hi)
    while hi_flow < TARGET_VAPOR_MOL_S and hi < bubble_c + 5.0:
        hi += MAX_DELTA_C
        hi_flow, _, _ = _vapor_flow_at(x, hi)

    best_t = hi
    best_flow, best_y, best_fs = _vapor_flow_at(x, best_t)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        flow, y, fs = _vapor_flow_at(x, mid)
        if flow >= TARGET_VAPOR_MOL_S:
            best_t, best_flow, best_y, best_fs = mid, flow, y, fs
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-5:
            break
    return best_t, best_flow, best_y, best_fs


def main() -> int:
    outdir = ROOT / "report" / "dwsim"
    outdir.mkdir(parents=True, exist_ok=True)

    factory, object_type = ded._automation_factory()
    demo._GLOBALS.automation = factory()
    demo._GLOBALS.object_type = object_type
    demo._GLOBALS.system = demo.IPA_SYSTEM
    demo._GLOBALS.property_package = "NRTL"

    bubble_rows = {}
    with open(ROOT / "docs" / "dwsim_ipa_bubble.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            bubble_rows[float(row["x_ipa"])] = float(row["T_dwsim_C"])

    rows: list[dict[str, object]] = []
    for x_ipa in demo.IPA_SYSTEM["xs"]:
        x = [x_ipa, 1.0 - x_ipa]
        bubble_c = bubble_rows[x_ipa]
        t_c, vap_flow, y, fs = _near_vapor_temperature(x, bubble_c)
        x_tag = str(x_ipa).replace(".", "p")
        t_tag = f"{t_c:.4f}".replace(".", "p")
        v_tag = f"{vap_flow:.1e}".replace(".", "p")
        fname = f"ipa_x{x_tag}_2comp_near_vapor_{t_tag}C_v{v_tag}.dwxmz"
        destination = outdir / fname
        ded._save_flowsheet_via_temp(demo._GLOBALS.automation, fs, destination)
        if not destination.exists():
            raise RuntimeError(f"DWSIM save returned without writing {destination}")
        rows.append({
            "x_ipa": x_ipa,
            "bubble_T_C": bubble_c,
            "near_vapor_T_C": round(t_c, 5),
            "delta_T_C": round(t_c - bubble_c, 5),
            "vapor_mol_s": f"{vap_flow:.8g}",
            "y_ipa_vapor": round(y[0], 6),
            "file": fname,
        })
        print(
            f"x_ipa={x_ipa:g} bubble={bubble_c:.2f} C near={t_c:.5f} C "
            f"delta={t_c - bubble_c:.5f} C vapor={vap_flow:.6g} mol/s y_ipa={y[0]:.6f}"
        )

    csv_path = outdir / "ipa_water_near_vapor_files.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
