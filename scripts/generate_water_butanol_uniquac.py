"""Water + 1-butanol binary LLE generated with the **UNIQUAC** property package.

This is the UNIQUAC counterpart of ``scripts/generate_water_butanol_default.py``
(which uses NRTL).  It builds the same ``report/dwsim`` binary-LLE template
topology::

    Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid

but carries a ``UNIQUAC`` property package instead of ``NRTL``.

Why both packages are worth keeping
-----------------------------------
DWSIM 9.0.5 ships built-in interaction parameters for both NRTL and UNIQUAC for
water / 1-butanol, but they do **not** agree.  Measured on this machine at
101.325 kPa (see the case table printed by ``main``):

    T=298.15 K, z=[0.3, 0.7]   NRTL: splits 0.748/0.252   UNIQUAC: splits 0.476/0.524
    T=298.15 K, z=[0.5, 0.5]   NRTL: SINGLE phase         UNIQUAC: splits 0.804/0.196

So the property package is a first-class modelling choice for this system, not a
detail: at z = [0.5, 0.5] NRTL does not predict a split at all while UNIQUAC
does.  Neither set is validated against experiment here; both are DWSIM's own
built-ins and every number is solved by DWSIM, never by this project.

The flowsheet is solved with ``CalculateFlowsheet4`` **before** saving, matching
``generate_water_butanol_default.py`` / ``generate_water_butanol_dwsim.py``.  A
file saved without solving opens in the DWSIM GUI with every object marked
``NotCalculated`` and empty phase fractions.

Run with a DWSIM + pythonnet capable interpreter:

    conda activate thermo
    python scripts/generate_water_butanol_uniquac.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402

#: DWSIM's internal pressure API takes Pa.
P_PA = 101325.0

#: Property package for this script.  Kept as a module constant so the NRTL and
#: UNIQUAC generators stay diff-able against one another.
PACKAGE = "UNIQUAC"

#: (label, temperature K, overall feed mole fractions [1-butanol, water]).
#: z = [0.5, 0.5] is included deliberately: it is the composition where NRTL and
#: UNIQUAC disagree most starkly (single phase vs split).
CASES: list[tuple[str, float, list[float]]] = [
    ("T25", 298.15, [0.3, 0.7]),
    ("T25b", 298.15, [0.5, 0.5]),
    ("T25c", 298.15, [0.1, 0.9]),
    ("T40", 313.15, [0.3, 0.7]),
    ("T60", 333.15, [0.3, 0.7]),
]

OUTDIR = ROOT / "lunwen" / "dwsim_demonstration" / "butanol_test"
OUTFILE = "water_butanol_UNIQUAC_default.dwxmz"

#: The single case rendered to a file.  Mirrors the NRTL generator, which saves
#: z = [0.3, 0.7] at 298.15 K.
SAVED_TEMPERATURE_K = 298.15
SAVED_FEED = [0.3, 0.7]


def phases(feed: object) -> dict[str, object]:
    """Read the per-phase molar fractions DWSIM computed for a stream."""
    out: dict[str, object] = {}
    for name in ("Vapor", "Liquid1", "Liquid2"):
        try:
            prop = feed.GetType().GetProperty(name).GetValue(feed, None)
            out[name] = float(prop.Properties.molarfraction) if prop is not None else None
        except Exception:
            out[name] = None
    try:
        out["ids"] = list(feed.PhaseIds) if feed.PhaseIds else None
    except Exception:
        out["ids"] = None
    return out


def composition_of(feed: object) -> list[float]:
    """Overall mole fractions of a material stream."""
    return [float(v) for v in feed.GetType().GetMethod("GetOverallComposition").Invoke(feed, None)]


def add_compounds(flowsheet: object) -> None:
    """Add water / 1-butanol using DWSIM's exact (case-sensitive) dictionary keys.

    DWSIM 9.0.5 stores the alcohol as the lower-case ``1-butanol``.  The
    spellings ``1-Butanol``, ``n-Butanol`` and ``N-butanol`` all raise
    ``KeyNotFoundException``, so there is deliberately **no** fallback spelling
    here -- a silent fallback would hide a real dictionary mismatch.
    """
    flowsheet.AddCompound("1-butanol")
    flowsheet.AddCompound("Water")


def run_case_table(automation: object, object_type: object) -> None:
    """Sweep the case table and print whether each case splits into two liquids."""
    print(f"=== water / 1-butanol, DEFAULT flash path, package={PACKAGE} ===")
    for label, temperature, z in CASES:
        flowsheet = automation.CreateFlowsheet()
        add_compounds(flowsheet)
        try:
            ded._add_property_package(flowsheet, PACKAGE)
        except Exception as exc:  # noqa: BLE001 - property package may be unavailable
            print(f"  {label:4s} {PACKAGE:8s}: package failed ({type(exc).__name__})")
            continue

        feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        feed = ded._simulation_object(feed_go)
        feed.SetTemperature(temperature)
        feed.SetPressure(P_PA)
        feed.SetMolarFlow(1.0)
        feed.SetOverallComposition(ded._composition_argument(z))

        errors = automation.CalculateFlowsheet4(flowsheet)
        error = str(errors[0])[:40] if errors is not None and errors.Count else ""
        state = phases(feed)
        liquid2 = state.get("Liquid2")
        split = isinstance(liquid2, float) and liquid2 > 1e-8
        print(
            f"  {label:4s} {PACKAGE:8s} T={temperature} z={z} -> "
            f"L1={state.get('Liquid1')} L2={liquid2} ids={state.get('ids')} "
            f"{'*** SPLIT ***' if split else 'single'}"
            f"{'  ERR:' + error if error else ''}"
        )


def build_saved_flowsheet(automation: object, object_type: object) -> tuple[object, object]:
    """Build the ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid`` case."""
    flowsheet = automation.CreateFlowsheet()
    add_compounds(flowsheet)
    ded._add_property_package(flowsheet, PACKAGE)

    feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = flowsheet.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vapor_go = flowsheet.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = flowsheet.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = flowsheet.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(SAVED_TEMPERATURE_K)
    feed.SetPressure(P_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(SAVED_FEED))

    # Pin the vessel flash to the feed conditions.  A DWSIM Vessel otherwise
    # defaults to its own 298.15 K flash and would evaluate the split at the
    # wrong temperature.
    vessel = ded._simulation_object(vessel_go)
    try:
        vessel.FlashTemperature = SAVED_TEMPERATURE_K
        vessel.FlashPressure = P_PA
    except Exception:  # noqa: BLE001 - non-fatal: feed conditions still drive the flash
        pass

    # Product streams carry the flash temperature so the saved file reports
    # consistent phase temperatures.
    for go in (light_go, heavy_go, vapor_go):
        try:
            ded._simulation_object(go).SetTemperature(SAVED_TEMPERATURE_K)
        except Exception:  # noqa: BLE001 - non-fatal on some DWSIM builds
            pass

    for source, target, source_index, target_index in (
        (feed_go, vessel_go, 0, 0),
        (vessel_go, vapor_go, 0, 0),
        (vessel_go, light_go, 1, 0),
        (vessel_go, heavy_go, 2, 0),
    ):
        try:
            flowsheet.ConnectObjects(source.GraphicObject, target.GraphicObject, source_index, target_index)
        except Exception:  # noqa: BLE001 - surface, don't abort
            pass

    # Solve BEFORE saving: an unsolved file opens with empty phase results.
    automation.CalculateFlowsheet4(flowsheet)
    return flowsheet, feed


def verify_saved_file(path: Path) -> bool:
    """Re-load the written file and confirm it really carries a solved split.

    Guards against the failure mode where a structurally correct file is saved
    with every object marked ``NotCalculated``.
    """
    import zipfile

    with zipfile.ZipFile(path) as archive:
        xml = archive.read(archive.namelist()[0]).decode("utf-8")
    calculated = xml.count("<Calculated>true</Calculated>")
    not_calculated = xml.count("<Status>NotCalculated</Status>")
    equilibrium = xml.count("<AtEquilibrium>true</AtEquilibrium>")
    ok = calculated > 0 and not_calculated == 0
    print(
        f"\n[verify] {path.name}: Calculated=true {calculated}, "
        f"Status=NotCalculated {not_calculated}, AtEquilibrium {equilibrium} "
        f"-> {'SOLVED' if ok else '*** UNSOLVED SKELETON ***'}"
    )
    return ok


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()

    run_case_table(automation, object_type)

    flowsheet, feed = build_saved_flowsheet(automation, object_type)
    state = phases(feed)
    print(
        f"\n[saved case] T={SAVED_TEMPERATURE_K} K z={SAVED_FEED} -> "
        f"L1={state.get('Liquid1')} L2={state.get('Liquid2')} ids={state.get('ids')}"
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    destination = OUTDIR / OUTFILE
    ded._save_flowsheet_via_temp(automation, flowsheet, destination)
    print(f"[OK] saved: {destination}")

    verify_saved_file(destination)


if __name__ == "__main__":
    main()
