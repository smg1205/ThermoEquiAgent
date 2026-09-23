"""Ternary VLE isobaric bubble points via DWSIM, by bisection on vapor flow.

This reproduces the method used for the v4/v5 three-source tables
(``report/Agent整合ThermoFormer进度与三源验证报告v4.md`` §2.2,
``...v5.md`` §2.1/§2.2) for an arbitrary three-component liquid mixture.

Method (identical to ``scripts/dwsim_ipa_bubble.py`` for the binary case and
``scripts/dwsim_ternary_eche_bubble.py`` / ``scripts/dwsim_dmso_bubble.py`` for
the ternary case -- "对汽相摩尔流量二分求泡点"):

    For a fixed liquid composition x at fixed pressure, bisect on temperature.
    Each trial runs a TP flash of the feed and reads the **Vapor product molar
    flow**:

        vapor flow > 0  ->  the feed is at or above its bubble point -> hi = T
        vapor flow = 0  ->  the feed is still all liquid            -> lo = T

    The converged ``hi`` is the bubble temperature (the point where the first
    bubble forms).  The vapor composition y is the Vapor stream's overall
    composition at the last temperature that produced vapor.

Why the vapor-flow criterion: it needs no activity model or Antoine coefficients
from this project.  DWSIM supplies its own built-in property data and binary
interaction parameters, so the equilibrium numbers are DWSIM's, never ours.

Feed topology: ``Feed -> Vessel (Flash) -> Vapor / Liquid``.

Run with a DWSIM + pythonnet capable interpreter:

    python scripts/dwsim_ternary_vle_bubble.py
    python scripts/dwsim_ternary_vle_bubble.py --x1 0.374 --x2 0.024 --tmin 60 --tmax 130
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: DWSIM's internal pressure API takes Pa.  101.30 kPa is the pressure used by
#: the v5 report for the butanol / water / toluene ternary VLE system.
P_PA = 101.30 * 1000.0

#: Default bisection bracket, in degrees Celsius.  Matches the bracket used by
#: the existing per-system scripts (60..110 for EtOH/cyclohexane/ester,
#: 60..130 for EtOAc/nPrOAc/DMSO).
T_MIN_C = 60.0
T_MAX_C = 130.0

#: Bisection iterations.  With the (hi - lo) < 1e-6 C stop this converges to well
#: below any reporting precision.
MAX_ITER = 60
TOL_C = 1e-6

OUTDIR = ROOT / "report" / "success"

#: The three components of the v5 report's ternary VLE system (§2.2), as exact
#: DWSIM keys.  Order matches the report: 1-butanol / water / toluene.
DEFAULT_COMPONENTS = ["1-butanol", "Water", "Toluene"]

#: Short labels for reporting and for the saved-file names.
COMPONENT_LABELS = ["butanol", "water", "toluene"]

#: The five representative liquid compositions archived by the v5 report
#: (table 2.2-3 and the ``bwt_*`` file list), as x = [butanol, water, toluene].
#: ``x1``/``x2`` below are the butanol/water fractions the report tabulates;
#: toluene is the remainder.
DEFAULT_CASES: list[tuple[float, float, float]] = [
    (0.633, 0.022, 0.345),
    (0.800, 0.030, 0.170),
    (0.837, 0.050, 0.113),
    (0.359, 0.057, 0.584),
    (0.601, 0.060, 0.339),
    (0.915, 0.029, 0.056),
]


def _resolve_compounds(flowsheet: object, components: list[str]) -> None:
    """Add the compounds, failing loudly on an unmapped name.

    DWSIM's dictionary keys are case-sensitive; a silent fallback spelling would
    hide a genuine mismatch, so an unknown name is an error.
    """
    from thermo_engine.dwsim_export import _dwsim_compound_name

    for name in components:
        canonical = _dwsim_compound_name(name)
        try:
            flowsheet.AddCompound(canonical)
        except Exception as exc:  # noqa: BLE001 - DWSIM dictionary mismatch
            raise SystemExit(
                f"DWSIM does not know the compound {canonical!r} (from {name!r}): {type(exc).__name__}. "
                "Add an explicit mapping in thermo_engine.dwsim_export._DWSIM_COMPOUND_MAP."
            ) from exc


def build_flash(
    automation: object,
    object_type: object,
    x: list[float],
    t_c: float,
    components: list[str],
    *,
    pin_vessel: bool = False,
):
    """Build ``Feed -> Vessel (Flash) -> Vapor / Liquid`` at ``t_c`` Celsius.

    ``pin_vessel`` additionally writes the explicit ``FlashTemperature`` /
    ``FlashPressure`` on the vessel, which is what the archived report files
    (``report/dwsim/bwt_*.dwxmz``) carry.  The bisection trials leave it off
    (the feed state drives the flash), and only the saved file turns it on.
    """
    from thermo_engine.dwsim_export import _add_property_package, _composition_argument, _simulation_object

    flowsheet = automation.CreateFlowsheet()
    _resolve_compounds(flowsheet, components)
    _add_property_package(flowsheet, "NRTL")

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel = flowsheet.AddObject(object_type.Vessel, 450, 0, "Flash")
    vapor = flowsheet.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
    liquid = flowsheet.AddObject(object_type.MaterialStream, 750, 80, "Liquid")

    feed_stream = _simulation_object(feed)
    feed_stream.SetTemperature(t_c + 273.15)
    feed_stream.SetPressure(P_PA)
    feed_stream.SetMolarFlow(1.0)
    feed_stream.SetOverallComposition(_composition_argument(list(x)))

    if pin_vessel:
        vessel_object = _simulation_object(vessel)
        for attribute, value in (
            ("FlashTemperature", t_c + 273.15),
            ("FlashPressure", P_PA),
        ):
            try:
                setattr(vessel_object, attribute, value)
            except Exception:  # noqa: BLE001 - non-fatal on some DWSIM builds
                pass

    flowsheet.ConnectObjects(feed.GraphicObject, vessel.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(vessel.GraphicObject, vapor.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(vessel.GraphicObject, liquid.GraphicObject, 1, 0)
    return flowsheet, _simulation_object(vapor), _simulation_object(liquid)


def bubble_temperature(
    automation: object,
    object_type: object,
    x: list[float],
    components: list[str],
    *,
    t_min_c: float = T_MIN_C,
    t_max_c: float = T_MAX_C,
) -> dict[str, object]:
    """Isobaric bubble point of liquid composition ``x`` by bisecting vapor flow.

    Returns DWSIM's own answer: the bubble temperature in C and K, the vapor
    composition at that temperature, the number of flashes used, and the final
    bracket width (a convergence indicator).
    """
    from thermo_engine.dwsim_export import _simulation_object  # noqa: F401

    # The upper bound must sit above the bubble point, otherwise no trial ever
    # produces vapor and bisection has nothing to bracket.
    flowsheet, vapor, _ = build_flash(automation, object_type, x, t_max_c, components)
    automation.CalculateFlowsheet2(flowsheet)
    if float(vapor.GetMolarFlow()) <= 0.0:
        return {
            "converged": False,
            "reason": f"no vapor at the upper bound ({t_max_c:g} C); widen --tmax",
            "t_bubble_C": None,
            "t_bubble_K": None,
            "y": None,
            "flashes": 1,
            "bracket_C": None,
        }

    lo, hi = t_min_c, t_max_c
    t_bubble_c: float | None = None
    y: list[float] | None = None
    flashes = 1

    for _ in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        flowsheet, vapor, _ = build_flash(automation, object_type, x, mid, components)
        automation.CalculateFlowsheet2(flowsheet)
        flashes += 1
        if float(vapor.GetMolarFlow()) > 0.0:
            # Vapor present -> at or above the bubble point: capture and move down.
            hi = mid
            t_bubble_c = mid
            try:
                y = [float(c) for c in list(vapor.GetOverallComposition())]
            except Exception:  # noqa: BLE001 - composition unavailable on some builds
                pass
        else:
            lo = mid
        if hi - lo < TOL_C:
            break

    return {
        "converged": t_bubble_c is not None,
        "reason": "" if t_bubble_c is not None else "no vapor found within the bracket",
        "t_bubble_C": t_bubble_c,
        "t_bubble_K": None if t_bubble_c is None else t_bubble_c + 273.15,
        "y": y,
        "flashes": flashes,
        "bracket_C": hi - lo,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs=3, default=DEFAULT_COMPONENTS, help="three DWSIM compound names")
    parser.add_argument("--x1", type=float, default=None, help="liquid mole fraction of component 1")
    parser.add_argument("--x2", type=float, default=None, help="liquid mole fraction of component 2")
    parser.add_argument("--tmin", type=float, default=T_MIN_C, help="lower bracket, C")
    parser.add_argument("--tmax", type=float, default=T_MAX_C, help="upper bracket, C")
    parser.add_argument("--outdir", default=str(OUTDIR), help="output directory")
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="only write the CSV/JSON tables; do not render .dwxmz flowsheets",
    )
    args = parser.parse_args()

    components = list(args.components)
    if args.x1 is not None and args.x2 is not None:
        if not (0.0 <= args.x1 <= 1.0 and 0.0 <= args.x2 <= 1.0 and args.x1 + args.x2 <= 1.0):
            raise SystemExit("--x1 and --x2 must be fractions with x1 + x2 <= 1")
        cases = [(args.x1, args.x2, round(1.0 - args.x1 - args.x2, 10))]
    else:
        cases = list(DEFAULT_CASES)

    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print(f"=== ternary VLE bubble points (DWSIM, NRTL, {P_PA / 1000:.3f} kPa) ===")
    print(f"components: {' / '.join(components)}")
    print(f"method    : bisection on Vapor molar flow over [{args.tmin:g}, {args.tmax:g}] C\n")

    rows: list[dict[str, object]] = []
    for x1, x2, x3 in cases:
        result = bubble_temperature(
            automation, object_type, [x1, x2, x3], components, t_min_c=args.tmin, t_max_c=args.tmax
        )
        y = result["y"]
        y_text = " / ".join(f"{v:.4f}" for v in y) if isinstance(y, list) else "n/a"
        tb = result["t_bubble_C"]
        if tb is None:
            print(f"x=[{x1:.3f}, {x2:.3f}, {x3:.3f}]  *** FAILED: {result['reason']} ***")
        else:
            print(
                f"x=[{x1:.3f}, {x2:.3f}, {x3:.3f}]  T_bubble={tb:.2f} C "
                f"({result['t_bubble_K']:.2f} K)  y = {y_text}  "
                f"[{result['flashes']} flashes, bracket {result['bracket_C']:.2e} C]"
            )
        row: dict[str, object] = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "T_bubble_C": None if tb is None else round(float(tb), 4),
            "T_bubble_K": None if result["t_bubble_K"] is None else round(float(result["t_bubble_K"]), 4),
            "converged": result["converged"],
            "flashes": result["flashes"],
        }
        for index, value in enumerate(y if isinstance(y, list) else []):
            row[f"y{index + 1}"] = round(value, 6)
        rows.append(row)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Render one .dwxmz per converged case, with the Feed pinned exactly at the
    # bubble point so the file opens on the two-phase boundary -- this is the
    # structure the archived v5 files (``bwt_*_3comp_bubble_*.dwxmz``) carry.
    if not args.no_files:
        from thermo_engine.dwsim_export import _save_flowsheet_via_temp

        for row in rows:
            tb = row["T_bubble_C"]
            if tb is None:
                continue
            x = [float(row["x1"]), float(row["x2"]), float(row["x3"])]
            flowsheet, _, _ = build_flash(
                automation, object_type, x, float(tb), components, pin_vessel=True
            )
            automation.CalculateFlowsheet2(flowsheet)
            # The saved file names carry the bubble temperature in KELVIN, e.g.
            # ``bwt_x1_0p633_x2_0p022_3comp_bubble_379.1K.dwxmz``.
            t_k = float(row["T_bubble_K"])
            if len(components) == 3:
                name = (
                    f"bwt_x1_{str(row['x1']).replace('.', 'p')}"
                    f"_x2_{str(row['x2']).replace('.', 'p')}"
                    f"_3comp_bubble_{t_k:.1f}K.dwxmz"
                )
            else:
                name = f"ternary_bubble_{t_k:.1f}K.dwxmz"
            destination = outdir / name
            _save_flowsheet_via_temp(automation, flowsheet, destination)
            row["file"] = name
            print(f"  saved: {name}")

    csv_path = outdir / f"ternary_vle_bubble_{stamp}.csv"
    json_path = outdir / f"ternary_vle_bubble_{stamp}.json"

    fieldnames = ["x1", "x2", "x3", "T_bubble_C", "T_bubble_K", "converged", "flashes"] + [
        f"y{i + 1}" for i in range(len(components))
    ] + (["file"] if not args.no_files else [])
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    payload = {
        "method": "bisection on DWSIM Vapor molar flow (isobaric bubble point)",
        "property_package": "NRTL",
        "components": components,
        "pressure_Pa": P_PA,
        "pressure_kPa": round(P_PA / 1000.0, 6),
        "bracket_C": [args.tmin, args.tmax],
        "tolerance_C": TOL_C,
        "generated_at": datetime.now().isoformat(),
        "dwsim_home": os.getenv("DWSIM_HOME"),
        "cases": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for row in rows if row["converged"])
    print(f"\n{ok}/{len(rows)} cases converged")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
