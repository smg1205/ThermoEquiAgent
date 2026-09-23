"""Export the binary water / 1-butanol LLE DWSIM flowsheet for the given request.

Request: 水 0.3 / 正丁醇 0.7, 298.15 K, 101.325 kPa, 1.0 mol/s.

Uses the repository's binary-LLE exporter (native Vessel TP flash), then reads the
flowsheet back and reports every stream so the two liquid phases can be inspected.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402
from thermo_engine.dwsim_export import export_dwsim_binary_lle_flowsheet  # noqa: E402

DEST = ROOT / "report" / "dwsim" / "water_butanol_LLE_298p15K_from_agent.dwxmz"

FEED = [0.3, 0.7]      # water, 1-butanol
T_K = 298.15
P_KPA = 101.325
FLOW = 1.0


def main() -> int:
    print("=== export ===")
    print(f"  components: water / 1-butanol")
    print(f"  feed      : {FEED}  T={T_K} K  P={P_KPA} kPa  F={FLOW} mol/s")

    try:
        p = export_dwsim_binary_lle_flowsheet(
            components=["water", "1-butanol"],
            feed_composition=FEED,
            temperature_K=T_K,
            pressure_kPa=P_KPA,
            feed_flow_mol_s=FLOW,
            property_package="NRTL",
            destination=DEST,
        )
    except TypeError as exc:
        print("  signature mismatch:", exc)
        # Fall back to inspecting the real signature.
        import inspect

        print(inspect.signature(export_dwsim_binary_lle_flowsheet))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"  export FAILED {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    print(f"  wrote: {p}  ({Path(p).stat().st_size} bytes)")

    # --- read back ------------------------------------------------------
    print("\n=== readback ===")
    factory, object_type = ded._automation_factory()
    a = factory()
    fs = a.LoadFlowsheet2(str(DEST))
    comps = [str(c).split(",")[0].strip().lstrip("[") for c in fs.SelectedCompounds]
    pkgs = [str(pp.GetType().Name) for pp in fs.PropertyPackages.Values]
    print(f"  compounds: {comps}")
    print(f"  package  : {pkgs}")

    errs = a.CalculateFlowsheet2(fs)
    print(f"  calculate error: {str(errs[0])[:120] if errs and errs.Count else ''!r}")

    for item in fs.SimulationObjects.Values:
        cls = item.GetType().Name
        tag = str(item.GraphicObject.Tag)
        if cls in ("Vessel", "Heater", "Separator"):
            v = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            try:
                print(f"  [{cls}] {tag}: FlashT={float(v.FlashTemperature)-273.15:.2f} C  "
                      f"FlashP={float(v.FlashPressure)/1000:.3f} kPa")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{cls}] {tag}: attr read failed {type(exc).__name__}")

    for item in fs.SimulationObjects.Values:
        if item.GetType().Name != "MaterialStream":
            continue
        tag = str(item.GraphicObject.Tag)
        s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        try:
            F = float(s.GetMolarFlow()); T = float(s.GetTemperature())
            P = float(s.GetPressure()) / 1000.0
            z = [float(x) for x in s.GetOverallComposition()]
            print(f"  {tag:<14} F={F:8.5f} mol/s  T={T-273.15:7.2f} C  P={P:8.3f} kPa  "
                  f"z={[round(v, 4) for v in z]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:<14} read failed: {type(exc).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
