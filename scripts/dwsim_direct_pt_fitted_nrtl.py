"""Direct PT-flash at 298.15 K with the fitted NRTL, bypassing the Vessel.

This isolates whether the fitted NRTL actually produces a two-liquid split (and
what compositions), independent of the Vessel unit's PH-flash path.
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


def _dump(result, tag):
    print(f"--- {tag} ---")
    n = len(result)
    print(f"  tuple length={n}")
    for i in range(n):
        v = result[i]
        try:
            s = f"[{', '.join(f'{float(x):.5f}' for x in v)}]" if hasattr(v, "__iter__") and not isinstance(v, str) else repr(v)
        except Exception:
            s = repr(v)
        print(f"  result[{i}] = {s}")


def main():
    factory, object_type = ded._automation_factory()
    from System import Activator, Array, Double, Enum  # noqa: E402
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

    # write fitted NRTL (cal/mol)
    from thermo_engine import dwsim_export as _ded
    selected = getattr(fs, "SelectedCompounds", None)
    comps = {str(k): selected[k] for k in list(selected.Keys)}
    mibk_key = [k for k in comps if k != "Water" and "Water" != k]
    # the MIBK key is the non-water one
    mibk_key = [k for k in comps if k != "Water"][0]
    water_key = "Water"
    m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
    ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
    inner_type = ips.GetType().GetGenericArguments()[1]
    data_type = m_uni.GetType().Assembly.GetType("DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData")
    def ensure(outer, k):
        if not outer.ContainsKey(k):
            outer.Add(k, Activator.CreateInstance(inner_type))
        return outer[k]
    # MIBK -> Water: A12 (MIBK->Water)=1747.85, A21=3601.07
    mw = ensure(ips, mibk_key)
    if not mw.ContainsKey(water_key):
        d = Activator.CreateInstance(data_type); mw.Add(water_key, d)
    d = mw[water_key]
    d.ID1 = comps[mibk_key]; d.ID2 = comps[water_key]
    d.A12 = 1747.8506; d.A21 = 3601.0666; d.alpha12 = 0.3798
    d.B12 = d.B21 = d.C12 = d.C21 = 0.0
    # Water -> MIBK
    wm = ensure(ips, water_key)
    if not wm.ContainsKey(mibk_key):
        d2 = Activator.CreateInstance(data_type); wm.Add(mibk_key, d2)
    d2 = wm[mibk_key]
    d2.ID1 = comps[water_key]; d2.ID2 = comps[mibk_key]
    d2.A12 = 3601.0666; d2.A21 = 1747.8506; d2.alpha12 = 0.3798
    d2.B12 = d2.B21 = d2.C12 = d2.C21 = 0.0

    try:
        pp.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pp, False, None)
    except Exception:
        pass
    try:
        pp.GetType().GetProperty("AreModelParametersDirty").SetValue(pp, True, None)
    except Exception:
        pass
    cfg = pp.GetType().GetMethod("ConfigParameters")
    if cfg is not None:
        cfg.Invoke(pp, None)
    print("[OK] NRTL written + ConfigParameters() invoked")

    # bind a feed stream so CurrentMaterialStream is not null
    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    pp.GetType().GetProperty("CurrentMaterialStream").SetValue(pp, feed, None)
    print("[OK] CurrentMaterialStream bound")

    # try both GibbsMinimizationMulti.Flash_PT and NestedLoops3PV3.Flash_PT
    asm = pp.GetType().Assembly
    for cls in ("GibbsMinimizationMulti", "NestedLoops3PV3"):
        try:
            ft = asm.GetType(f"DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.{cls}")
            flash = Activator.CreateInstance(ft)
            flash.StabSearchSeverity = 3
            r = flash.Flash_PT(Array[Double](Z), P, T, pp, False, None)
            _dump(r, f"{cls}.Flash_PT @ 298.15K")
        except Exception as e:
            print(f"[ERR] {cls}: {type(e).__name__} {str(e)[:100]}")


if __name__ == "__main__":
    main()
