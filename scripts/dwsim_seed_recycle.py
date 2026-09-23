"""Seed the recycle stream so the cascade's recycle loop actually converges.

The Recycle_to_S1 stream runs a PH flash; without a good starting point the Gibbs
PH solver fails (PT Flash: Invalid solution) and the loop never closes.  We seed
its temperature, pressure, flow and composition with physically sensible values.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_vessel_cascade_2stage.dwxmz"
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_cascade_2stage_converged.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))

    def find(tag):
        for it in fs.SimulationObjects.Values:
            if str(it.GraphicObject.Tag) == tag:
                return it
        return None

    def vals(o):
        t = o.GetType()
        return (float(t.GetMethod("GetMolarFlow").Invoke(o, None)),
                [float(v) for v in t.GetMethod("GetOverallComposition").Invoke(o, None)],
                float(t.GetMethod("GetTemperature").Invoke(o, None)))

    # seed the recycle stream with a plausible aqueous return
    rec = find("Recycle_to_S1")
    if rec is not None:
        try:
            rec.SetTemperature(T)
            rec.SetPressure(P)
            rec.SetMolarFlow(0.6)
            rec.SetOverallComposition(ded._composition_argument([0.0, 1.0]))
            print("[OK] seeded Recycle_to_S1 (T, P, flow, composition)")
        except Exception as e:
            print("[WARN] seed:", type(e).__name__, e)

    print("recalculating...")
    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("errors:", str(e[0])[:140])
    else:
        print("no errors")

    print("\n--- streams ---")
    for tag in ("Feed", "Solvent", "Stage_1_light", "Stage_1_heavy",
                "Stage_2_light", "Stage_2_heavy", "Recycle_to_S1"):
        o = find(tag)
        if o is None:
            continue
        f, z, t = vals(o)
        print(f"  {tag:16s} flow={f:.5f} T={t:.2f} z=[MIBK {z[0]:.5f}, water {z[1]:.5f}]")

    # mass balance
    fF, zF, _ = vals(find("Feed"))
    fS, zS, _ = vals(find("Solvent"))
    fL, zL, _ = vals(find("Stage_2_light"))
    fH, zH, _ = vals(find("Stage_1_heavy"))
    print("\n--- mass balance ---")
    for i, name in enumerate(("MIBK", "water")):
        i_in = fF * zF[i] + fS * zS[i]
        o_out = fL * zL[i] + fH * zH[i]
        print(f"  {name:5s} in={i_in:.5f} out={o_out:.5f} diff={i_in-o_out:+.5f}")

    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
