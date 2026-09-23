"""Final correct MIBK/water LLE: fitted NRTL + native Vessel, verify physical binodal.

1) writes the regressed NRTL BIPs (from doi 10.1016/j.fluid.2016.11.005 tie-lines)
   into DWSIM's NRTL_IPData (A12/A21 in cal/mol = J/mol / 4.184).
2) sets GibbsMinimization + immiscible-water switch.
3) runs the native Vessel with Light/Heavy liquid outlets, seeds outlet T.
4) reports the two phase compositions so we can check the physical mutual
   solubility (organic x_water should be small, aqueous x_MIBK should be ~0).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]
FIT = ROOT / "lunwen" / "dwsim_demonstration" / "mibk_water_nrtl_fitted.json"
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def _flow(s):
    return float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))
def _z(s):
    return [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)]


def main():
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    A12_cal = fit["A12_cal_per_mol"]
    A21_cal = fit["A21_cal_per_mol"]
    alpha = fit["alpha12"]

    factory, object_type = ded._automation_factory()
    from System import Enum, Activator  # noqa: E402
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

    # write fitted NRTL BIPs (cal/mol) into NRTL_IPData
    try:
        from thermo_engine import dwsim_export as _ded
        selected = getattr(fs, "SelectedCompounds", None)
        comps = {str(k): selected[k] for k in list(selected.Keys)}
        m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
        ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
        inner_type = ips.GetType().GetGenericArguments()[1]
        data_type = m_uni.GetType().Assembly.GetType(
            "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData"
        )
        # MIBK = component 0, Water = component 1 (in the order added)
        mibk_key = None
        for k in comps:
            if "Methyl" in k or "isobutyl" in k.lower() or "MIBK" in k or "pentanone" in k.lower():
                mibk_key = k
        water_key = "Water"
        # build both directions into the dict keyed by first compound
        for (i_name, j_name, a12, a21, al) in [
            (mibk_key, water_key, A12_cal, A21_cal, alpha),
        ]:
            ci, cj = comps[i_name], comps[j_name]
            if not ips.ContainsKey(i_name):
                ips.Add(i_name, Activator.CreateInstance(inner_type))
            inner = ips[i_name]
            if not inner.ContainsKey(j_name):
                d = Activator.CreateInstance(data_type)
                inner.Add(j_name, d)
            d = inner[j_name]
            d.ID1 = ci; d.ID2 = cj
            d.A12 = float(a12); d.A21 = float(a21); d.alpha12 = float(al)
            d.B12 = 0.0; d.B21 = 0.0; d.C12 = 0.0; d.C21 = 0.0
            d.comment = "fitted from doi 10.1016/j.fluid.2016.11.005 MIBK-water tie-lines"
        # also symmetric direction (Water -> MIBK)
        if not ips.ContainsKey(water_key):
            ips.Add(water_key, Activator.CreateInstance(inner_type))
        w_inner = ips[water_key]
        if not w_inner.ContainsKey(mibk_key):
            d2 = Activator.CreateInstance(data_type); w_inner.Add(mibk_key, d2)
        d2 = w_inner[mibk_key]
        d2.ID1 = comps[water_key]; d2.ID2 = comps[mibk_key]
        d2.A12 = float(A21_cal); d2.A21 = float(A12_cal); d2.alpha12 = float(alpha)
        d2.B12 = 0.0; d2.B21 = 0.0; d2.C12 = 0.0; d2.C21 = 0.0
        d2.comment = "fitted NRTL (reverse direction)"
        # disable auto-estimate so DWSIM doesn't overwrite, mark dirty, reconfigure
        for pn in ("AutoEstimateMissingNRTLUNIQUACParameters",):
            try:
                pp.GetType().GetProperty(pn).SetValue(pp, False, None)
            except Exception:
                pass
        try:
            pp.GetType().GetProperty("AreModelParametersDirty").SetValue(pp, True, None)
        except Exception:
            pass
        cfg = pp.GetType().GetMethod("ConfigParameters")
        if cfg is not None:
            cfg.Invoke(pp, None)
        print(f"[OK] NRTL BIPs written: A12={A12_cal:.2f} A21={A21_cal:.2f} alpha={alpha:.4f} (cal/mol)")
    except Exception as e:
        print("[WARN] BIP write:", type(e).__name__, e)

    # GibbsMinimization + immiscible water
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)
    print("[OK] GibbsMinimization + immiscible-water on")

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

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:", str(e[0])[:160])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    print("\n===== MIBK/water LLE @ 298.15 K, fitted NRTL =====")
    print(f"Light_Liquid : flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(light), _z(light)[0], _z(light)[1]))
    print(f"Heavy_Liquid : flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(heavy), _z(heavy)[0], _z(heavy)[1]))
    print("\n[physical check] expected: organic x_water ~ 0.02, aqueous x_MIBK ~ 0.002")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
