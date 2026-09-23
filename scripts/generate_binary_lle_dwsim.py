"""DWSIM binary LLE demonstration: water + MIBK (partial miscibility).

Shows, concretely, how to KNOW a binary system is phase-split (two liquid
phases) in DWSIM: run a liquid-liquid flash with a UNIQUAC/NRTL package + the
Nested Loops (VLLE) algorithm, then read the SECOND LIQUID PHASE fraction from
the flash result tuple.

    result = flash.Flash_PT(composition, P, T, package, False, None)
    organic_fraction = result[5]   # mole fraction of the second (organic) liquid
    organic_composition = result[6]

 * organic_fraction ~ 0   -> only ONE liquid phase -> NOT split
 * organic_fraction ~ 1   -> only ONE liquid phase -> NOT split
 * 0 < organic_fraction < 1 -> TWO liquid phases -> SPLIT (LLE)

The feed composition (x_MIBK = 0.4) falls inside the two-phase region at
333.15 K, so the flash MUST return a fraction strictly between 0 and 1.  All
numbers are computed by DWSIM, never hard-coded.

Run (full-access terminal so pythonnet can init):

    conda activate thermo
    python scripts/generate_binary_lle_dwsim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402

# MIBK (4-methyl-2-pentanone) / water, a classic partially-miscible binary pair.
# Candidate DWSIM compound keys (first resolvable wins).
MIBK_CANDIDATES = [
    "Methyl isobutyl ketone",
    "4-Methyl-2-pentanone",
    "MIBK",
]
WATER = "Water"
FEED_COMPOSITION = [0.4, 0.6]  # x_MIBK, x_water -- inside the two-phase region
TEMPERATURE_K = 333.15
PRESSURE_PA = 101325.0
OUTPUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_binary_lle_flash.dwxmz"


def _resolve_mibk(fs) -> str:
    for cand in MIBK_CANDIDATES:
        try:
            fs.AddCompound(cand)
            return cand
        except Exception:
            continue
    raise RuntimeError(f"could not resolve MIBK compound: tried {MIBK_CANDIDATES}")


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()

    mibk_key = _resolve_mibk(flowsheet)
    print(f"[OK] added MIBK as {mibk_key!r}")
    flowsheet.AddCompound(WATER)

    # UNIQUAC has built-in MIBK/water LLE parameters; NRTL also works.
    ded._add_property_package(flowsheet, "UNIQUAC")

    feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed_MIBK_Water")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(TEMPERATURE_K)
    feed.SetPressure(PRESSURE_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_COMPOSITION))

    # Bind the feed's property package and run a manual liquid-liquid flash with
    # the Nested Loops (VLLE) algorithm + three-phase stability severity 3.
    package = next(
        prop.GetValue(feed, None)
        for prop in feed.GetType().GetProperties()
        if prop.Name == "PropertyPackage"
        and "Thermodynamics.PropertyPackages.PropertyPackage" in prop.PropertyType.FullName
    )
    package_type = package.GetType()
    package_type.GetProperty("CurrentMaterialStream").SetValue(package, feed, None)
    flash_type = package_type.Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
    )
    flash = flash_type.GetConstructor([]).Invoke([])
    flash.StabSearchSeverity = 3

    from System import Array, Double  # type: ignore[import-not-found]

    result = flash.Flash_PT(
        Array[Double](FEED_COMPOSITION),
        PRESSURE_PA,
        TEMPERATURE_K,
        package,
        False,
        None,
    )

    organic_fraction = float(result[5])
    organic_composition = [float(v) for v in result[6]]

    print("\n================ binary LLE flash result ================")
    print(f"feed (overall)    : x_MIBK = {FEED_COMPOSITION[0]}, x_water = {FEED_COMPOSITION[1]}")
    print(f"T = {TEMPERATURE_K} K, P = {PRESSURE_PA} Pa")
    print(f"second-liquid (organic) fraction = {organic_fraction:.6f}")
    print(f"organic-phase composition (x_MIBK, x_water) = "
          f"[{organic_composition[0]:.5f}, {organic_composition[1]:.5f}]")

    tol = 1.0e-8
    if organic_fraction < tol or organic_fraction > 1.0 - tol:
        print("\n[JUDGEMENT] organic fraction ~ 0 or ~1 => ONLY ONE liquid phase => NOT split.")
        print("            (feed composition is outside the two-phase region, or wrong package/flash)")
    else:
        # second phase composition = (z - frac*organic)/ (1-frac) by mass balance
        aqueous = [
            (FEED_COMPOSITION[i] - organic_fraction * organic_composition[i]) / (1.0 - organic_fraction)
            for i in range(2)
        ]
        print(f"aqueous-phase composition (x_MIBK, x_water) = "
              f"[{aqueous[0]:.5f}, {aqueous[1]:.5f}]")
        print("\n[JUDGEMENT] 0 < organic fraction < 1 => TWO liquid phases => SPLIT (LLE).")
        print("            organic phase is MIBK-rich, aqueous phase is water-rich.")

    # Save a recalculable flowsheet (feed stream + flash results) for the GUI.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    automation.SaveFlowsheet2(flowsheet, str(OUTPUT))
    print(f"\n[OK] saved flowsheet: {OUTPUT}")


if __name__ == "__main__":
    main()
