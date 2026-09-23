"""Diagnose MIBK/water LLE at 298.15 K: does the Nested Loops VLLE return one phase?

Compares the flash result at 298.15 K against the known good 333.15 K case, and
probes whether the trivial-solution / no-split behaviour reproduces the user's
observation, plus whether a stability-test severity tweak changes the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402

MIBK_CANDIDATES = ["Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"]
WATER = "Water"
COMPOSITION = [0.4, 0.6]  # x_MIBK, x_water
PRESSURE_PA = 101325.0


def _build(factory, object_type):
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in MIBK_CANDIDATES:
        try:
            fs.AddCompound(cand)
            break
        except Exception:
            continue
    fs.AddCompound(WATER)
    ded._add_property_package(fs, "UNIQUAC")
    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed_MIBK_Water")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(298.15)
    feed.SetPressure(PRESSURE_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(COMPOSITION))
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
    return automation, package, flash_type


def _flash(package, flash_type, t_k):
    flash = flash_type.GetConstructor([]).Invoke([])
    flash.StabSearchSeverity = 3
    from System import Array, Double

    result = flash.Flash_PT(Array[Double](COMPOSITION), PRESSURE_PA, t_k, package, False, None)
    frac = float(result[5])
    comp = [float(v) for v in result[6]]
    return frac, comp


def main():
    factory, object_type = ded._automation_factory()
    automation, package, flash_type = _build(factory, object_type)

    for t in (333.15, 298.15):
        frac, comp = _flash(package, flash_type, t)
        print(f"T={t} K: organic fraction={frac:.6f}  organic comp=[{comp[0]:.5f},{comp[1]:.5f}]")

    # Also test different severity values at 298.15 K
    print("\n--- severity sweep at 298.15 K ---")
    from System import Array, Double

    for sev in (1, 2, 3):
        flash = flash_type.GetConstructor([]).Invoke([])
        try:
            flash.StabSearchSeverity = sev
        except Exception as e:
            print(f"  severity {sev}: set failed {type(e).__name__}")
            continue
        r = flash.Flash_PT(Array[Double](COMPOSITION), PRESSURE_PA, 298.15, package, False, None)
        print(f"  severity {sev}: fraction={float(r[5]):.6f}  comp={[float(v) for v in r[6]]}")


if __name__ == "__main__":
    main()
