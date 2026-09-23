"""MIBK/water binary LLE: UNIFAC-LL (no binary params) + NestedLoopsImmiscible kernel.

Combines the two findings that actually work:
  * UNIFAC-LL group-contribution package -> needs NO regressed binary parameters
  * NestedLoopsImmiscible flash kernel   -> gives the physically correct strong
    split for an immiscible pair (organic ~pure MIBK, aqueous ~pure water)

The kernel is invoked from a single-stage decanter unit (the only carrier that
reliably calls the chosen kernel on every solve), and the two equilibrium phases
are written to two product streams.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_unifacll_decanter.dwxmz"

DECANTER_SCRIPT = r'''from System import Activator, Array, Double

if ims1 is None or oms1 is None or oms2 is None:
    raise Exception("Decanter needs one inlet and two outlets.")

feed_flow = float(ims1.GetMolarFlow())
if feed_flow <= 0.0:
    raise Exception("Feed flow must be positive.")
z = [float(v) for v in ims1.GetOverallComposition()]
temperature = float(ims1.GetTemperature())
pressure = float(ims1.GetPressure())

pp = Me.PropertyPackage
if pp is None:
    pp = ims1.PropertyPackage
pp.CurrentMaterialStream = ims1

flash_type = pp.GetType().Assembly.GetType(
    "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoopsImmiscible"
)
flash = Activator.CreateInstance(flash_type)
try:
    flash.StabSearchSeverity = 3
except Exception:
    pass
result = flash.Flash_PT(Array[Double](z), pressure, temperature, pp, False, None)

phase1_frac = float(result[0]); phase1_x = [float(v) for v in result[2]]
phase2_frac = float(result[5]); phase2_x = [float(v) for v in result[6]]
if phase1_frac <= 1.0e-10 or phase2_frac <= 1.0e-10:
    raise Exception("Single liquid phase (no LLE split) at this feed condition.")

# organic = MIBK-richer phase (index 0 larger)
if phase1_x[0] >= phase2_x[0]:
    organic_frac, organic_x = phase1_frac, phase1_x
    aqueous_frac, aqueous_x = phase2_frac, phase2_x
else:
    organic_frac, organic_x = phase2_frac, phase2_x
    aqueous_frac, aqueous_x = phase1_frac, phase1_x

oms1.SetTemperature(temperature)
oms1.SetPressure(pressure)
oms1.SetMolarFlow(feed_flow * organic_frac)
oms1.SetOverallComposition(Array[Double](organic_x))

oms2.SetTemperature(temperature)
oms2.SetPressure(pressure)
oms2.SetMolarFlow(feed_flow * aqueous_frac)
oms2.SetOverallComposition(Array[Double](aqueous_x))
'''


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def main():
    factory, object_type = ded._automation_factory()
    from System import Array, Object, Double  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    pkg = None
    for p in ("UNIFAC-LL", "UNIFAC LL", "UNIFACLL"):
        try:
            ded._add_property_package(fs, p); pkg = p; break
        except Exception:
            continue
    print("[OK] package:", pkg)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed_MIBK_Water")
    dec_go = fs.AddObject(object_type.CustomUO, 350, 0, "LLE_Decanter_UNIFACLL")
    org_go = fs.AddObject(object_type.MaterialStream, 700, -70, "Organic_Phase")
    aq_go = fs.AddObject(object_type.MaterialStream, 700, 70, "Aqueous_Phase")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))

    dec = ded._simulation_object(dec_go)
    dec.ScriptText = DECANTER_SCRIPT
    dec.ComponentDescription = "MIBK/water LLE decanter: UNIFAC-LL + NestedLoopsImmiscible (no binary params)."

    for f, t, fi, ti, label in ((feed_go, dec_go, 0, 0, "feed"), (dec_go, org_go, 0, 0, "organic"),
                                (dec_go, aq_go, 1, 0, "aqueous")):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            print("[OK]", label)
        except Exception as e:
            print("[WARN]", label, type(e).__name__)

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        if e and e.Count:
            print(f"[{label}] ERR:", str(e[0])[:140]); return
        o = _vals(ded._simulation_object(org_go))
        a = _vals(ded._simulation_object(aq_go))
        print(f"[{label}] Organic: flow=%.5f z=[MIBK %.5f, water %.5f]" % (o["flow"], o["z"][0], o["z"][1]))
        print(f"        Aqueous: flow=%.5f z=[MIBK %.5f, water %.5f]" % (a["flow"], a["z"][0], a["z"][1]))

    print("\n===== UNIFAC-LL + NestedLoopsImmiscible decanter =====")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
