"""Inspect DWSIM's native Recycle block ports (OT_Recycle) so we can break the
counter-current cascade loop properly."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    fs.AddCompound("Water")

    rc = getattr(object_type, "OT_Recycle", None)
    print("OT_Recycle member:", rc)
    if rc is None:
        return
    try:
        go = fs.AddObject(rc, 300, 0, "Recycle")
        sim = go.GetAsObject() if hasattr(go, "GetAsObject") else go
        print("class:", sim.GetType().FullName)
        for p in sim.GetConnectionPortsList():
            print("  ", p)
        print("\nproperties:")
        for pr in sim.GetType().GetProperties():
            n = pr.Name.lower()
            if any(k in n for k in ("max", "iter", "toleran", "accel", "converg", "spec")):
                try:
                    print(f"   {pr.Name} : {pr.PropertyType.Name} = {pr.GetValue(sim, None)!r}")
                except Exception:
                    pass
    except Exception as e:
        print("ERR:", type(e).__name__, e)


if __name__ == "__main__":
    main()
