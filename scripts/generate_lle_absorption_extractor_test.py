"""Build a DWSIM LLE extractor-column test case with AbsorptionColumn.Extractor.

Case:
    ethanol / ethyl acetate feed, water solvent.
    feed: ethanol 40 mol%, ethyl acetate 60 mol%, 1 mol/s
    solvent: water, solvent/feed molar ratio 1.5

This is intentionally a standalone probe script and is not wired into the web
backend yet.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402


def _try(label: str, func: Any, *args: object) -> bool:
    try:
        func(*args)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {label}")
    return True


def _try_property_package(flowsheet: Any, candidates: list[str]) -> str:
    last: Exception | None = None
    for name in candidates:
        try:
            ded._add_property_package(flowsheet, name)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[WARN] property package {name!r} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"[OK] property package: {name}")
        return name
    raise RuntimeError(f"no candidate property package could be added; last error: {last!r}")


def _ensure_ethanol_eac_water_nrtl_bips(flowsheet: Any) -> None:
    """Ensure the classic ethanol/ethyl acetate/water NRTL BIPs are present.

    The literature values supplied by the user are in the same scale as the
    DWSIM UI/exported table before conversion; this DWSIM build stores the
    ``A12``/``A21`` fields as value/4.184.  The existing built-in database uses
    that exact conversion, so keep the same scale when explicitly overriding.
    """

    pairs = [
        ("Ethyl acetate", "Ethanol", 1278.65 / 4.184, 1382.86 / 4.184, 0.2988),
        ("Ethyl acetate", "Water", 5380.57 / 4.184, 6719.58 / 4.184, 0.4393),
        ("Ethanol", "Water", -242.505 / 4.184, 5195.44 / 4.184, 0.2937),
    ]

    packages = list(flowsheet.PropertyPackages.Values)
    if not packages:
        print("[WARN] NRTL BIP setup skipped: no property package")
        return
    pp = packages[0]
    pp_type = pp.GetType()
    if "NRTLPropertyPackage" not in pp_type.FullName:
        print(f"[WARN] NRTL BIP setup skipped: active package is {pp_type.FullName}")
        return

    m_uni_prop = pp_type.GetProperty("m_uni")
    m_uni = m_uni_prop.GetValue(pp, None)
    ips_prop = m_uni.GetType().GetProperty("InteractionParameters")
    ips = ips_prop.GetValue(m_uni, None)
    inner_dict_type = ips.GetType().GetGenericArguments()[1]
    data_type = m_uni.GetType().Assembly.GetType("DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData")

    from System import Activator  # type: ignore[import-not-found]

    for comp_i, comp_j, a12, a21, alpha in pairs:
        if not ips.ContainsKey(comp_i):
            ips.Add(comp_i, Activator.CreateInstance(inner_dict_type))
        inner = ips[comp_i]
        if inner.ContainsKey(comp_j):
            data = inner[comp_j]
        else:
            data = Activator.CreateInstance(data_type)
            inner.Add(comp_j, data)
        data.ID1 = comp_i
        data.ID2 = comp_j
        data.A12 = float(a12)
        data.A21 = float(a21)
        data.alpha12 = float(alpha)
        data.B12 = 0.0
        data.B21 = 0.0
        data.C12 = 0.0
        data.C21 = 0.0
        data.comment = "ThermoAgent explicit ethanol/ethyl acetate/water LLE NRTL BIP"
        print(
            "[OK] NRTL BIP "
            f"{comp_i} / {comp_j}: A12={data.A12:.6g}, A21={data.A21:.6g}, alpha={data.alpha12:.4g}"
        )

    pp_type.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pp, False, None)
    pp_type.GetProperty("AreModelParametersDirty").SetValue(pp, True, None)


def _set_extractor_mode(column: Any) -> None:
    from System import Enum  # type: ignore[import-not-found]

    prop = column.GetType().GetProperty("OperationMode")
    op_type = prop.PropertyType
    extractor = Enum.Parse(op_type, "Extractor")
    prop.SetValue(column, extractor, None)
    print(f"[OK] OperationMode = {column.OperationMode}")


def _net_double_array(values: list[float]) -> Any:
    from System import Array, Double  # type: ignore[import-not-found]

    return Array[Double]([float(v) for v in values])


def _net_jagged_double_array(rows: list[list[float]]) -> Any:
    from System import Array, Double  # type: ignore[import-not-found]

    inner = Array[Double]
    outer = Array[inner]
    return outer([inner([float(v) for v in row]) for row in rows])


def _set_lle_initial_estimates(
    column: Any,
    stages: int,
    organic_flow: float,
    solvent_flow: float,
    strong: bool = False,
) -> None:
    """Seed the rigorous extractor with simple LLE initial estimates."""

    # DWSIM reuses L/V names for the two counter-current liquid phases:
    # L is the heavy water-rich phase flowing down from the top, while V is
    # the light ester-rich phase flowing up from the bottom.
    liquid_flows = [solvent_flow] * stages
    vapor_flows = [organic_flow] * stages
    temperatures = [298.15] * stages

    # In Extractor mode, DWSIM's rigorous-column data model still uses the
    # historical liquid/vapor estimate fields. Seed them as two distinct liquid
    # phases to avoid the LLE solver collapsing to the trivial single-phase
    # solution. Component order: ethanol, ethyl acetate, water.
    top_liq = [0.01, 0.01, 0.98] if strong else [0.03, 0.02, 0.95]
    bottom_liq = [0.12, 0.04, 0.84] if strong else [0.10, 0.05, 0.85]
    liquid_comps: list[list[float]] = []
    for i in range(stages):
        frac = i / max(stages - 1, 1)
        row = [(1.0 - frac) * top_liq[j] + frac * bottom_liq[j] for j in range(3)]
        total = sum(row)
        liquid_comps.append([v / total for v in row])
    top_vap = [0.25, 0.72, 0.03] if strong else [0.22, 0.73, 0.05]
    bottom_vap = [0.40, 0.59, 0.01] if strong else [0.38, 0.60, 0.02]
    vapor_comps: list[list[float]] = []
    for i in range(stages):
        frac = i / max(stages - 1, 1)
        row = [(1.0 - frac) * top_vap[j] + frac * bottom_vap[j] for j in range(3)]
        total = sum(row)
        vapor_comps.append([v / total for v in row])

    column.UseLiquidFlowEstimates = True
    column.UseVaporFlowEstimates = True
    column.UseTemperatureEstimates = True
    column.UseCompositionEstimates = True
    column.AutoUpdateInitialEstimates = False

    _try(
        "SetInitialLiquidMolarFlowEstimates",
        column.SetInitialLiquidMolarFlowEstimates,
        _net_double_array(liquid_flows),
    )
    _try(
        "SetInitialVaporMolarFlowEstimates",
        column.SetInitialVaporMolarFlowEstimates,
        _net_double_array(vapor_flows),
    )
    _try(
        "SetInitialTemperatureEstimates",
        column.SetInitialTemperatureEstimates,
        _net_double_array(temperatures),
    )
    _try(
        "SetInitialMolarCompositionEstimates",
        column.SetInitialMolarCompositionEstimates,
        _net_jagged_double_array(liquid_comps),
        _net_jagged_double_array(vapor_comps),
    )


def _set_burningham_otto_solver(column: Any) -> None:
    asm = column.GetType().Assembly
    solver_type = asm.GetType(
        "DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps.SolvingMethods.BurninghamOttoMethod"
    )
    if solver_type is None:
        print("[WARN] BurninghamOttoMethod type not found")
        return
    solver = solver_type.GetConstructor([]).Invoke([])
    try:
        solver.RelaxCompositionUpdates = True
        solver.RelaxTemperatureUpdates = True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] solver relaxation flags: {type(exc).__name__}: {exc}")
    _try("SetColumnSolver Burningham-Otto", column.SetColumnSolver, solver)
    try:
        column.Solver = solver
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] set Solver property: {type(exc).__name__}: {exc}")
    try:
        solver_name = solver.Name
    except Exception:  # noqa: BLE001
        solver_name = solver.GetType().FullName
    print(f"[OK] solver = {solver_name}")


def _connect_feed_stage(column: Any, stream: Any, stage: int, label: str) -> None:
    sim = ded._simulation_object(stream)
    _try(f"ConnectFeed {label} stage {stage}", column.ConnectFeed, sim, stage)
    # Register the stream in the rigorous column's internal stream dictionary.
    # The graphical connection alone is not sufficient for AbsorptionColumn.
    _try(f"ConnectFeedMaterialStream {label}", column.ConnectFeedMaterialStream, sim, 0 if label == "feed" else 1)
    _try(f"SetStreamFeedStage {label} stage {stage}", column.SetStreamFeedStage, sim, stage)


def _connect_product_streams(column: Any, top_stream: Any, bottom_stream: Any) -> None:
    """Register extractor products internally, in addition to drawing the links."""

    _try("ConnectTopProduct", column.ConnectTopProduct, ded._simulation_object(top_stream))
    _try("ConnectBottoms", column.ConnectBottoms, ded._simulation_object(bottom_stream))
    # Keep the port-level registration as a compatibility fallback for DWSIM
    # builds where the named methods do not populate the product dictionary.
    _try("ConnectProductMaterialStream top", column.ConnectProductMaterialStream, ded._simulation_object(top_stream), 0)
    _try("ConnectProductMaterialStream bottom", column.ConnectProductMaterialStream, ded._simulation_object(bottom_stream), 1)


def _calculate(automation: Any, flowsheet: Any, column: Any) -> None:
    for name in ("CalculateFlowsheet2", "CalculateFlowsheet", "CalculateFlowsheet3", "CalculateFlowsheet4"):
        method = getattr(automation, name, None)
        if not callable(method):
            continue
        try:
            method(flowsheet)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {name}: {type(exc).__name__}: {exc}")
            continue
        print(f"[OK] {name}")
        return
    try:
        column.Calculate(None)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] column.Calculate(None): {type(exc).__name__}: {exc}")
        return
    print("[OK] column.Calculate(None)")


def _save_verbose(automation: Any, flowsheet: Any, destination: Path) -> None:
    def _attempt(path: Path) -> bool:
        attempts = (
            (automation, "SaveFlowsheet2", (flowsheet, str(path))),
            (automation, "SaveFlowsheet2", (flowsheet, str(path), True)),
            (automation, "SaveFlowsheet", (flowsheet, str(path), True)),
            (automation, "SaveFlowsheet", (flowsheet, str(path))),
            (flowsheet, "SaveToXML", (str(path),)),
            (flowsheet, "SaveToFile", (str(path),)),
        )
        nonlocal last
        for owner, name, args in attempts:
            method = getattr(owner, name, None)
            if not callable(method):
                print(f"[WARN] save {owner.GetType().FullName}.{name}: missing")
                continue
            try:
                method(*args)
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[WARN] save {name}{tuple(type(a).__name__ for a in args)}: {type(exc).__name__}: {exc}")
                continue
            print(f"[OK] save via {name}{tuple(type(a).__name__ for a in args)}")
            return True
        return False

    last: Exception | None = None
    if _attempt(destination):
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="dwsim-lle-export-"))
    temp_destination = tmpdir / destination.name
    print(f"[INFO] retry save through temp path: {temp_destination}")
    if _attempt(temp_destination):
        shutil.copyfile(temp_destination, destination)
        print("[OK] copied temp export back to requested path")
        return
    raise RuntimeError(f"all DWSIM save attempts failed; last error: {last!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / ".pytest-tmp" / "ethanol_eac_water_lle_absorption_extractor.dwxmz"),
        help="Output .dwxmz path.",
    )
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--feed-stage", type=int, default=4)
    parser.add_argument("--solvent-stage", type=int, default=1)
    parser.add_argument("--feed-flow", type=float, default=1.0)
    parser.add_argument("--solvent-flow", type=float, default=1.5)
    parser.add_argument(
        "--feed-ethanol-fraction",
        type=float,
        default=0.4,
        help="Ethanol mole fraction in the ethanol/ethyl-acetate organic feed.",
    )
    parser.add_argument(
        "--strong-lle-seed",
        action="store_true",
        help="Use strongly separated water-rich and organic-rich phase estimates.",
    )
    parser.add_argument(
        "--property-package",
        default="NRTL",
        help="DWSIM property package to try first. Use NRTL for parameterized LLE tests.",
    )
    parser.add_argument("--no-calc", action="store_true")
    args = parser.parse_args()

    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()

    compounds = ["Ethanol", "Ethyl acetate", "Water"]
    for name in compounds:
        flowsheet.AddCompound(name)
        print(f"[OK] compound: {name}")

    package = _try_property_package(
        flowsheet,
        [
            args.property_package,
            "NRTL",
            "UNIQUAC",
            "UNIFAC-LL",
            "UNIFAC LL",
            "UNIFACLL",
        ],
    )
    if package == "NRTL":
        _ensure_ethanol_eac_water_nrtl_bips(flowsheet)

    column_go = flowsheet.AddObject(object_type.AbsorptionColumn, 300, 0, "Liquid-Liquid Extractor")
    column = ded._simulation_object(column_go)
    _set_extractor_mode(column)
    _try("SetNumberOfStages", column.SetNumberOfStages, int(args.stages))
    column.NumberOfStages = int(args.stages)
    column.SelectedEquipmentType = "Tray Column"
    column.MaxIterations = 500
    column.InternalLoopTolerance = 1.0e-4
    column.ExternalLoopTolerance = 1.0e-4
    column.ColumnPressureDrop = 1000.0
    column.InitialEstimatesProvider = "Internal 2 (Experimental)"
    _set_burningham_otto_solver(column)
    print(f"[OK] stages = {column.NumberOfStages}")
    print(f"[OK] equipment = {column.SelectedEquipmentType!r}")

    feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, -90, "Feed_EtOH_EtAc")
    solvent_go = flowsheet.AddObject(object_type.MaterialStream, 0, 90, "Solvent_Water")
    top_go = flowsheet.AddObject(object_type.MaterialStream, 620, -90, "Raffinate_Top")
    bottom_go = flowsheet.AddObject(object_type.MaterialStream, 620, 90, "Extract_Bottom")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(298.15)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(float(args.feed_flow))
    feed.SetOverallComposition(
        ded._composition_argument(
            [float(args.feed_ethanol_fraction), 1.0 - float(args.feed_ethanol_fraction), 0.0]
        )
    )
    print(
        f"[OK] feed: {args.feed_flow:g} mol/s, "
        f"x=[EtOH {args.feed_ethanol_fraction:g}, "
        f"EtAc {1.0 - args.feed_ethanol_fraction:g}, Water 0.0]"
    )

    solvent = ded._simulation_object(solvent_go)
    solvent.SetTemperature(298.15)
    solvent.SetPressure(101325.0)
    solvent.SetMolarFlow(float(args.solvent_flow))
    solvent.SetOverallComposition(ded._composition_argument([0.0, 0.0, 1.0]))
    print(f"[OK] solvent: {args.solvent_flow:g} mol/s, pure water")

    _set_lle_initial_estimates(
        column,
        int(args.stages),
        organic_flow=float(args.feed_flow),
        solvent_flow=float(args.solvent_flow),
        strong=args.strong_lle_seed,
    )

    _connect_feed_stage(column, feed_go, int(args.feed_stage), "feed")
    _connect_feed_stage(column, solvent_go, int(args.solvent_stage), "solvent")
    _connect_product_streams(column, top_go, bottom_go)

    for from_go, to_go, fidx, tidx, label in (
        (feed_go, column_go, 0, 0, "feed -> extractor port 0"),
        (solvent_go, column_go, 0, 1, "solvent -> extractor port 1"),
        (column_go, top_go, 0, 0, "extractor outlet 0 -> top"),
        (column_go, bottom_go, 1, 0, "extractor outlet 1 -> bottom"),
    ):
        _try(label, flowsheet.ConnectObjects, from_go.GraphicObject, to_go.GraphicObject, fidx, tidx)

    if not args.no_calc:
        _calculate(automation, flowsheet, column)

    _save_verbose(automation, flowsheet, destination)
    print(f"[OK] saved: {destination}")
    print(f"[OK] size: {destination.stat().st_size} bytes")
    print(f"[INFO] package used: {package}")


if __name__ == "__main__":
    main()
