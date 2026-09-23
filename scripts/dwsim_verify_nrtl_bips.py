"""Verify the NRTL BIPs actually landed in the property package (read back + activity check)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))
    pps = list(fs.PropertyPackages.Values)
    pp = pps[0]
    print("PP type:", pp.GetType().FullName)

    # read back InteractionParameters dictionary
    try:
        m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
        ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
        print("\nInteractionParameters keys:", list(ips.Keys))
        for k in list(ips.Keys):
            inner = ips[k]
            for k2 in list(inner.Keys):
                d = inner[k2]
                print(f"  {k} -> {k2}: A12={d.A12:.4f} A21={d.A21:.4f} alpha12={d.alpha12:.4f}")
    except Exception as e:
        print("readback err:", type(e).__name__, e)

    # also check the package's active NRTL parameter arrays (AUX or internal)
    try:
        for pn in ("NRTL_A12", "nrA12", "A12", "IPData"):
            pr = pp.GetType().GetProperty(pn)
            if pr is not None:
                try:
                    print(f"  {pn} =", pr.GetValue(pp, None))
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
