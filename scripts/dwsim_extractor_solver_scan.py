"""Systematically try to converge DWSIM's native Liquid-Liquid Extractor for MIBK/water.

Scans solver methods x flash tags x estimate providers, with UNIFAC-LL, and reports
which combination (if any) converges the rigorous extractor into two liquid phases.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
STAGES = 5
FEED_Z = [0.4, 0.6]
SOLVENT_Z = [0.0, 1.0]
FEED_FLOW = 1.0
SOLV_FLOW = 1.0
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_extractor.dwxmz"


def _vals(s):
    return (float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
            [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)])


def attempt(automation, object_type, solver_name, tag, provider, sysmod):
    from System import Array, Double, Object
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    for p in ("UNIFAC-LL", "UNIFAC LL", "UNIFACLL"):
        try:
            ded._add_property_package(fs, p); break
        except Exception:
            continue

    col_go = fs.AddObject(object_type.AbsorptionColumn, 400, 0, "Extractor")
    col = ded._simulation_object(col_go)
    from System import Enum
    prop = col.GetType().GetProperty("OperationMode")
    prop.SetValue(col, Enum.Parse(prop.PropertyType, "Extractor"), None)
    col.SetNumberOfStages(STAGES)
    col.MaxIterations = 500
    col.InternalLoopTolerance = 1e-4
    col.ExternalLoopTolerance = 1e-4

    if tag:
        try:
            col.PreferredFlashAlgorithmTag = tag
        except Exception:
            pass
    if provider:
        try:
            col.InitialEstimatesProvider = provider
        except Exception:
            pass
    if sysmod:
        try:
            col.SolverScheme = Enum.Parse(col.GetType().GetProperty("SolverScheme").PropertyType, sysmod)
        except Exception:
            pass

    # solver
    asm = col.GetType().Assembly
    try:
        st = asm.GetType(f"DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps.SolvingMethods.{solver_name}")
        col.SetColumnSolver(st.GetConstructor([]).Invoke([]))
    except Exception as e:
        return f"solver-set-fail {type(e).__name__}"

    feed_go = fs.AddObject(object_type.MaterialStream, 0, -90, "Feed")
    solv_go = fs.AddObject(object_type.MaterialStream, 0, 90, "Solvent")
    raff_go = fs.AddObject(object_type.MaterialStream, 800, -90, "Raffinate")
    ext_go = fs.AddObject(object_type.MaterialStream, 800, 90, "Extract")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(FEED_FLOW)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))
    solv = ded._simulation_object(solv_go)
    solv.SetTemperature(T); solv.SetPressure(P); solv.SetMolarFlow(SOLV_FLOW)
    solv.SetOverallComposition(ded._composition_argument(SOLVENT_Z))

    try:
        col.ConnectFeed(solv, 0); col.SetStreamFeedStage(solv, 0)
        col.ConnectFeed(feed, 1); col.SetStreamFeedStage(feed, STAGES - 1)
        col.ConnectTopProduct(ded._simulation_object(raff_go))
        col.ConnectBottoms(ded._simulation_object(ext_go))
    except Exception as e:
        return f"connect-fail {type(e).__name__}"

    # estimates
    try:
        col.UseLiquidFlowEstimates = True
        col.UseVaporFlowEstimates = True
        col.UseTemperatureEstimates = True
        col.UseCompositionEstimates = True
        col.AutoUpdateInitialEstimates = False
        col.SetInitialLiquidMolarFlowEstimates(Array[Double]([SOLV_FLOW] * STAGES))
        col.SetInitialVaporMolarFlowEstimates(Array[Double]([FEED_FLOW] * STAGES))
        col.SetInitialTemperatureEstimates(Array[Double]([T] * STAGES))
        inner = Array[Double]
        liq = [inner([0.02, 0.98]) for _ in range(STAGES)]
        vap = [inner([0.98, 0.02]) for _ in range(STAGES)]
        col.SetInitialMolarCompositionEstimates(Array[inner](liq), Array[inner](vap))
    except Exception as e:
        return f"estimate-fail {type(e).__name__}"

    for f, t, fi, ti in ((feed_go, col_go, 0, 0), (solv_go, col_go, 0, 1),
                         (col_go, raff_go, 0, 0), (col_go, ext_go, 1, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        return f"FAIL: {str(e[0])[:70]}"
    rf, rz = _vals(ded._simulation_object(raff_go))
    ef, ez = _vals(ded._simulation_object(ext_go))
    ok = rf > 1e-6 and ef > 1e-6
    return f"OK={ok}  Raffinate flow={rf:.4f} z=[{rz[0]:.4f},{rz[1]:.4f}]  Extract flow={ef:.4f} z=[{ez[0]:.4f},{ez[1]:.4f}]"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    combos = []
    for solver in ("NaphtaliSandholmMethod", "BurninghamOttoMethod", "Tomich", "WangHenkeMethod", "WangHenkeMethod2"):
        for tag in (None, "Nested_Loops_Immiscible_VLLE", "Nested Loops (Immiscible)"):
            combos.append((solver, tag, None, None))
    for solver, tag, provider, sysmod in combos:
        try:
            r = attempt(automation, object_type, solver, tag, provider, sysmod)
        except Exception as e:
            r = f"EXC {type(e).__name__}: {str(e)[:60]}"
        print(f"{solver:24s} tag={str(tag):32s} -> {r}")


if __name__ == "__main__":
    main()
