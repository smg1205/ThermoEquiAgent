"""Native Vessel LLE at 333.15 K with the NRTL parameters regressed FROM 333-353 K
tie-lines -- i.e. at a temperature inside the fitted range (interpolation, not
extrapolation), so the model should reproduce the experimental binodal:

    organic x_MIBK = 0.8109 , aqueous x_MIBK = 0.0023   (experiment, 333.15 K)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 333.15
P = 101325.0
Z = [0.4, 0.6]
FIT = ROOT / "lunwen" / "dwsim_demonstration" / "mibk_water_nrtl_fitted.json"
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle_333K_fitted.dwxmz"


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def main():
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    A12, A21, alpha = fit["A12_cal_per_mol"], fit["A21_cal_per_mol"], fit["alpha12"]
    factory, object_type = ded._automation_factory()
    from System import Activator, Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]

    # write fitted NRTL (cal/mol), both directions
    selected = getattr(fs, "SelectedCompounds", None)
    comps = {str(k): selected[k] for k in list(selected.Keys)}
    mibk_key = [k for k in comps if k != "Water"][0]
    water_key = "Water"
    m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
    ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
    inner_type = ips.GetType().GetGenericArguments()[1]
    data_type = m_uni.GetType().Assembly.GetType("DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData")

    def put(outer_key, inner_key, a12, a21, id1, id2):
        if not ips.ContainsKey(outer_key):
            ips.Add(outer_key, Activator.CreateInstance(inner_type))
        outer = ips[outer_key]
        if not inner.ContainsKey(inner_key) if False else not outer.ContainsKey(inner_key):
            outer.Add(inner_key, Activator.CreateInstance(data_type))
        d = outer[inner_key]
        d.ID1 = id1; d.ID2 = id2
        d.A12 = float(a12); d.A21 = float(a21); d.alpha12 = float(alpha)
        d.B12 = d.B21 = d.C12 = d.C21 = 0.0

    put(mibk_key, water_key, A12, A21, comps[mibk_key], comps[water_key])
    put(water_key, mibk_key, A21, A12, comps[water_key], comps[mibk_key])
    try:
        pp.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pp, False, None)
        pp.GetType().GetProperty("AreModelParametersDirty").SetValue(pp, True, None)
        pp.GetType().GetMethod("ConfigParameters").Invoke(pp, None)
    except Exception:
        pass
    print(f"[OK] fitted NRTL written: A12={A12:.2f} A21={A21:.2f} alpha={alpha:.4f} cal/mol")

    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    vessel = ded._simulation_object(vessel_go)
    try:
        vessel.FlashTemperature = T; vessel.FlashPressure = P
    except Exception:
        pass
    for f, t, fi, ti in ((feed_go, vessel_go, 0, 0), (vessel_go, vapor_go, 0, 0),
                         (vessel_go, light_go, 1, 0), (vessel_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass
    for go in (light_go, heavy_go, vapor_go):
        try:
            ded._simulation_object(go).SetTemperature(T)
        except Exception:
            pass

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        err = str(e[0])[:100] if e and e.Count else "none"
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        print(f"[{label}] err={err}")
        print(f"   Light: flow={l['flow']:.5f} [MIBK {l['z'][0]:.5f}, water {l['z'][1]:.5f}]")
        print(f"   Heavy: flow={h['flow']:.5f} [MIBK {h['z'][0]:.5f}, water {h['z'][1]:.5f}]")
        for i, nm in enumerate(("MIBK", "water")):
            cout = l["flow"] * l["z"][i] + h["flow"] * h["z"][i]
            print(f"   {nm}: in={Z[i]:.5f} out={cout:.5f} diff={Z[i]-cout:+.5f}")

    print(f"\n===== fitted NRTL @ {T} K (interpolation) =====")
    print("experiment: organic x_MIBK=0.8109, aqueous x_MIBK=0.0023")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
