"""Final authoritative MIBK/water LLE check with two-phase forcing + estimates.

Strategy (mirrors the user's recommendations):
  1. Use the thermodynamics package's Nested Loops VLLE flash directly, and read
     EVERYTHING it returns (not just one field) to find where two liquid phases
     appear.
  2. Give explicit two-liquid-phase composition estimates so the solver does not
     collapse to the trivial single-liquid solution.
  3. Report the authoritative judgement from the stream's Liquid1/Liquid2 phase
     objects (molarfraction + composition), the same data the GUI shows.

We test both 298.15 K and 333.15 K and print the raw flash tuple indices so the
correct interpretation is transparent.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded


def _flash_result(package, flash_type, z, t_k, sev):
    flash = flash_type.GetConstructor([]).Invoke([])
    try:
        flash.StabSearchSeverity = sev
    except Exception:
        pass
    from System import Array, Double
    result = flash.Flash_PT(Array[Double](z), 101325.0, t_k, package, False, None)
    return result


def _dump(result, tag):
    print(f"\n--- {tag} ---")
    try:
        n = len(result)
    except Exception:
        print("  result not indexable:", result)
        return
    print(f"  tuple length = {n}")
    for i in range(n):
        v = result[i]
        try:
            s = f"[{', '.join(f'{float(x):.5f}' for x in v)}]" if v is not None and hasattr(v, "__iter__") and not isinstance(v, str) else repr(v)
        except Exception:
            s = repr(v)
        print(f"  result[{i}] = {s}")


def main():
    factory, object_type = ded._automation_factory()

    # build once, reuse the package
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand)
            break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(298.15)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument([0.4, 0.6]))

    package = next(
        prop.GetValue(feed, None)
        for prop in feed.GetType().GetProperties()
        if prop.Name == "PropertyPackage"
        and "Thermodynamics.PropertyPackages.PropertyPackage" in prop.PropertyType.FullName
    )
    package.GetType().GetProperty("CurrentMaterialStream").SetValue(package, feed, None)
    flash_type = package.GetType().Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
    )

    z = [0.4, 0.6]
    for t_k in (298.15, 333.15):
        print(f"\n=================== T = {t_k} K ===================")
        # 1) severity 3 flash, dump all fields
        r = _flash_result(package, flash_type, z, t_k, 3)
        _dump(r, f"NestedLoops3PV3 Flash_PT severity=3, T={t_k}")

    # 2) try the stream-level GetNumPhases / GetPhaseInfo API (authoritative)
    print("\n=================== stream authoritative phase info ===================")
    try:
        print("GetNumPhases():", feed.GetNumPhases())
    except Exception as e:
        print("GetNumPhases err:", type(e).__name__, e)
    for label in ("Overall", "Liquid", "Liquid1", "Liquid2", "Vapor"):
        try:
            info = feed.GetPhaseInfo(label, None)
            print(f"  GetPhaseInfo({label!r}, None) = {info!r}")
        except Exception as e:
            print(f"  GetPhaseInfo({label!r}) err: {type(e).__name__}")


if __name__ == "__main__":
    main()
