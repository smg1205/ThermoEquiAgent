"""Native Liquid-Liquid Extractor (AbsorptionColumn + OperationMode=Extractor) for MIBK/water.

This is DWSIM's genuine native LLE unit (GUI: Columns -> Liquid-Liquid Extractor),
not a CustomUO script.  Property package: UNIFAC-LL (group contribution, no binary
parameters needed).

Flowsheet: Feed (MIBK+water) + Solvent (water) -> Extractor -> Raffinate (aqueous)
+ Extract (organic, MIBK-rich).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
STAGES = 6
FEED_FLOW = 1.0
FEED_Z = [0.4, 0.6]          # MIBK, water
SOLVENT_FLOW = 1.0           # pure water solvent
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_extractor.dwxmz"


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def main():
    factory, object_type = ded._automation_factory()
    from System import Enum, Array, Double  # noqa: E402
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

    # Force the package's flash to a multi-phase kernel (the column's stage
    # flashes go through PropertyPackage.GetFlash(), which only returns a
    # multi-phase kernel when FlashCalculationApproach = GibbsMinimization).
    _pp = list(fs.PropertyPackages.Values)[0]
    try:
        from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
        ap = _pp.GetType().GetProperty("FlashCalculationApproach")
        ap.SetValue(_pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
        fsp = _pp.GetType().GetProperty("FlashSettings")
        s = fsp.GetValue(_pp, None)
        s[FlashSetting.ImmiscibleWaterOption] = "True"
        s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(_pp, s, None)
        print("[OK] flash -> GibbsMinimization + immiscible-water")
    except Exception as e:
        print("[WARN] flash approach:", type(e).__name__, e)

    col_go = fs.AddObject(object_type.AbsorptionColumn, 400, 0, "Liquid-Liquid Extractor")
    col = ded._simulation_object(col_go)

    # switch to Extractor mode (native Liquid-Liquid Extractor)
    try:
        prop = col.GetType().GetProperty("OperationMode")
        col_type = prop.PropertyType
        prop.SetValue(col, Enum.Parse(col_type, "Extractor"), None)
        print("[OK] OperationMode = Extractor")
    except Exception as e:
        print("[WARN] OperationMode:", type(e).__name__, e)

    try:
        col.SetNumberOfStages(STAGES)
        col.NumberOfStages = STAGES
        col.MaxIterations = 500
        col.ColumnPressureDrop = 1000.0
        print("[OK] stages =", STAGES)
    except Exception as e:
        print("[WARN] stages:", type(e).__name__)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, -90, "Feed")
    solv_go = fs.AddObject(object_type.MaterialStream, 0, 90, "Solvent_Water")
    raff_go = fs.AddObject(object_type.MaterialStream, 800, -90, "Raffinate")
    ext_go = fs.AddObject(object_type.MaterialStream, 800, 90, "Extract")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(FEED_FLOW)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))

    solv = ded._simulation_object(solv_go)
    solv.SetTemperature(T); solv.SetPressure(P); solv.SetMolarFlow(SOLVENT_FLOW)
    solv.SetOverallComposition(ded._composition_argument([0.0, 1.0]))  # pure water

    # register feeds on stages (counter-current: solvent top stage 0, feed bottom)
    for stream, stage, port, label in (
        (solv_go, 0, 0, "solvent"),
        (feed_go, STAGES - 1, 1, "feed"),
    ):
        sim = ded._simulation_object(stream)
        for fn, args, nm in (
            (col.ConnectFeed, (sim, port), f"ConnectFeed {label} port {port}"),
            (col.SetStreamFeedStage, (sim, stage), f"SetStreamFeedStage {label} -> {stage}"),
        ):
            try:
                fn(*args); print("[OK]", nm)
            except Exception as e:
                print("[WARN]", nm, type(e).__name__)

    for fn, args, nm in (
        (col.ConnectTopProduct, (ded._simulation_object(raff_go),), "top->Raffinate"),
        (col.ConnectBottoms, (ded._simulation_object(ext_go),), "bottom->Extract"),
    ):
        try:
            fn(*args); print("[OK]", nm)
        except Exception as e:
            print("[WARN]", nm, type(e).__name__)

    # solver + initial estimates
    try:
        asm = col.GetType().Assembly
        st = asm.GetType("DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps.SolvingMethods.BurninghamOttoMethod")
        if st is not None:
            col.SetColumnSolver(st.GetConstructor([]).Invoke([]))
            print("[OK] Burningham-Otto solver")
    except Exception as e:
        print("[WARN] solver:", type(e).__name__)

    try:
        from System import Array as A, Double as D
        col.UseLiquidFlowEstimates = True
        col.UseVaporFlowEstimates = True
        col.UseTemperatureEstimates = True
        col.UseCompositionEstimates = True
        col.AutoUpdateInitialEstimates = False
        col.SetInitialLiquidMolarFlowEstimates(A[D]([SOLVENT_FLOW] * STAGES))
        col.SetInitialVaporMolarFlowEstimates(A[D]([FEED_FLOW] * STAGES))
        col.SetInitialTemperatureEstimates(A[D]([T] * STAGES))
        # two distinct liquid-phase composition estimates
        liq = [[0.02, 0.98] for _ in range(STAGES)]     # water-rich
        vap = [[0.98, 0.02] for _ in range(STAGES)]     # MIBK-rich
        inner = A[D]
        col.SetInitialMolarCompositionEstimates(
            A[inner]([inner(r) for r in liq]),
            A[inner]([inner(r) for r in vap]),
        )
        print("[OK] two-phase initial estimates set")
    except Exception as e:
        print("[WARN] estimates:", type(e).__name__, str(e)[:80])

    # graphical connections too
    for f, t, fi, ti in ((feed_go, col_go, 0, 0), (solv_go, col_go, 0, 1),
                         (col_go, raff_go, 0, 0), (col_go, ext_go, 1, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:")
        for i in range(e.Count):
            print("  ", str(e[i])[:180])

    print("\n===== native Liquid-Liquid Extractor (UNIFAC-LL) =====")
    for tag, go in (("Raffinate (aqueous)", raff_go), ("Extract (organic)", ext_go)):
        v = _vals(ded._simulation_object(go))
        print(f"  {tag}: flow={v['flow']:.5f} z=[MIBK {v['z'][0]:.5f}, water {v['z'][1]:.5f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
