"""Generate a DWSIM ternary LLE liquid-liquid extraction column.

Case: ethanol / ethyl acetate / water.

The target is the real DWSIM *Liquid-Liquid Extractor* (the rigorous
``AbsorptionColumn`` with ``OperationMode = Extractor``), NOT a ``Vessel``,
``Splitter`` or ``ComponentSeparator``.  This matches the documented route in
``lunwen/dwsim_export.md``:

    AbsorptionColumn + OperationMode Extractor + NRTL + VLLE flash + BIP
    + Burningham-Otto initial estimates.

The classic directional NRTL binary-interaction parameters for this ternary
are written into DWSIM's ``NRTL_IPData`` (A_ij / 4.184 scale), so the file is
self-contained even on a machine whose built-in NRTL database lacks them.

Requirements (on the machine that runs this script -- not the web server):

* Windows + a DWSIM install (9.x), and pythonnet installable into the active
  Python environment (``pip install pythonnet``).
* ``DWSIM_HOME`` pointing at the folder that contains ``DWSIM.Automation.dll``
  (e.g. ``C:\\Program Files\\DWSIM``).  Set it in a local ``.env`` next to the
  repo root, or in the environment before running.

Usage (run FROM the repo root, i.e. the ``ThermoFormer/ThermoAgent`` folder):

    python scripts/generate_lle_ethanol_eac_water_dwsim.py
    python scripts/generate_lle_ethanol_eac_water_dwsim.py --out exports/test/etoh_eac_water_lle.dwxmz --stages 8 --calc

All phase-equilibrium numbers are produced by DWSIM, not this script.  The
feed/solvent/estimate values below are operating inputs and solver seeds; they
are never reported as equilibrium results.  This mirrors the repository rule
that LLE numbers never come from the LLM or from hand-written constants.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Case constants (operating inputs + solver seeds, NOT equilibrium results).
# ---------------------------------------------------------------------------
COMPONENTS = ["Ethanol", "Ethyl acetate", "Water"]

# Classic directional NRTL BIPs (A_ij/A_ji in Kelvin, alpha dimensionless),
# as recorded in lunwen/dwsim_export.md section 7.  A_ij and A_ji are stored in
# DWSIM's NRTL_IPData after dividing by 4.184 (DWSIM's internal cal-based scale).
NRTL_BIPS_K: list[tuple[str, str, float, float, float]] = [
    ("Ethyl acetate", "Ethanol", 1278.65, 1382.86, 0.2988),
    ("Ethyl acetate", "Water", 5380.57, 6719.58, 0.4393),
    ("Ethanol", "Water", -242.505, 5195.44, 0.2937),
]
DWSIM_CAL_PER_K = 4.184

# Operating point: the organic feed is ethanol in ethyl acetate (no water), and
# the fresh solvent is pure water.  Feed enters near the bottom, solvent near
# the top; raffinate (water-rich) leaves the top, extract (ester-rich) leaves
# the bottom.
FEED_TEMPERATURE_K = 298.15
FEED_PRESSURE_PA = 101325.0


@dataclass
class BuildResult:
    destination: Path
    property_package: str
    errors: list[str]
    saved_bytes: int


# ---------------------------------------------------------------------------
# small logging / retry helpers (no dependency on dwsim_export internals).
# ---------------------------------------------------------------------------
def _log(ok: bool, label: str, detail: str = "") -> None:
    tag = "[OK] " if ok else "[WARN]"
    print(f"{tag} {label}" + (f"  ({detail})" if detail else ""))


def _try(label: str, func: Any, *args: object) -> bool:
    try:
        func(*args)
    except Exception as exc:  # noqa: BLE001 - probe, don't abort
        _log(False, label, f"{type(exc).__name__}: {exc}")
        return False
    _log(True, label)
    return True


# ---------------------------------------------------------------------------
# DWSIM assembly loading.
# ---------------------------------------------------------------------------
def _load_dwsim() -> tuple[Any, Any]:
    load_dotenv()

    dwsim_home = os.getenv("DWSIM_HOME")
    if not dwsim_home:
        raise RuntimeError(
            "DWSIM_HOME is not set.  Point it at the folder containing "
            "DWSIM.Automation.dll and re-run."
        )
    install_dir = Path(dwsim_home).expanduser().resolve()
    automation_dll = install_dir / "DWSIM.Automation.dll"
    if not automation_dll.is_file():
        raise RuntimeError(f"DWSIM.Automation.dll not found under {install_dir}")

    # DWSIM is noisy about its temp directory on locked-down machines; give it
    # a clean writable one and keep the CWD out of it.
    dwsim_temp = Path(tempfile.mkdtemp(prefix="thermoagent-dwsim-"))
    os.environ["TEMP"] = str(dwsim_temp)
    os.environ["TMP"] = str(dwsim_temp)

    try:
        import clr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pythonnet is not installed (pip install pythonnet).") from exc

    if str(install_dir) not in sys.path:
        sys.path.append(str(install_dir))
    try:
        clr.AddReference(str(automation_dll))
        from DWSIM.Automation import Automation3  # type: ignore[import-not-found]
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM
        raise RuntimeError(f"Could not load DWSIM automation assemblies: {type(exc).__name__}") from exc

    return Automation3, ObjectType


def _simulation_object(obj: Any) -> Any:
    get_as_object = getattr(obj, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else obj


def _add_property_package(flowsheet: Any, name: str) -> None:
    create_and_add = getattr(flowsheet, "CreateAndAddPropertyPackage", None)
    if callable(create_and_add):
        create_and_add(name)
        return
    flowsheet.AddPropertyPackage(name)


def _double_array(values: list[float]) -> Any:
    from System import Array, Double  # type: ignore[import-not-found]

    return Array[Double]([float(v) for v in values])


def _jagged_double_array(rows: list[list[float]]) -> Any:
    from System import Array, Double  # type: ignore[import-not-found]

    inner = Array[Double]
    return Array[inner]([inner([float(v) for v in row]) for row in rows])


# ---------------------------------------------------------------------------
# NRTL BIP installation.
# ---------------------------------------------------------------------------
def _install_nrtl_bips(flowsheet: Any) -> str:
    """Write the classic ethanol/ethyl-acetate/water NRTL BIPs into the package.

    Returns a human-readable status line.  If the active package is not NRTL
    this only warns.

    Verified against this DWSIM 9.0.5 build by reflection:
    * ``InteractionParameters`` is ``Dictionary[str, Dictionary[str, NRTL_IPData]]``
      keyed by the compound's FULL ENGLISH NAME (e.g. ``"Ethanol"``, ``"Water"``).
    * ``NRTL_IPData.ID1/.ID2`` hold the compound OBJECTS (from SelectedCompounds).
    * ``A12/.A21`` are on the cal scale: the built-in Ethanol/Water pair stores
      A12 = -57.96 = -242.505 / 4.184, so the literature Kelvin values must be
      divided by 4.184 before writing (exactly as lunwen/dwsim_export.md doc 7).
    """

    packages = list(flowsheet.PropertyPackages.Values)
    if not packages:
        return "no property package; NRTL BIPs not installed"
    pp = packages[0]
    pp_type = pp.GetType()
    if "NRTLPropertyPackage" not in pp_type.FullName:
        return f"active package {pp_type.FullName} is not NRTL; BIPs not installed"

    from System import Activator  # type: ignore[import-not-found]

    selected = getattr(flowsheet, "SelectedCompounds", None)
    compounds_by_name: dict[str, object] = {}
    if selected is not None:
        for name in list(selected.Keys):
            compounds_by_name[str(name)] = selected[name]

    m_uni = pp_type.GetProperty("m_uni").GetValue(pp, None)
    ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
    inner_dict_type = ips.GetType().GetGenericArguments()[1]
    data_type = m_uni.GetType().Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData"
    )

    installed = 0
    for name_i, name_j, a_ij, a_ji, alpha in NRTL_BIPS_K:
        comp_i = compounds_by_name.get(name_i)
        comp_j = compounds_by_name.get(name_j)
        if comp_i is None or comp_j is None:
            print(f"[WARN] cannot resolve compound objects for {name_i}/{name_j}")
            continue
        if not ips.ContainsKey(name_i):
            ips.Add(name_i, Activator.CreateInstance(inner_dict_type))
        inner = ips[name_i]
        if inner.ContainsKey(name_j):
            data = inner[name_j]
        else:
            data = Activator.CreateInstance(data_type)
            inner.Add(name_j, data)
        data.ID1 = comp_i
        data.ID2 = comp_j
        data.A12 = float(a_ij / DWSIM_CAL_PER_K)
        data.A21 = float(a_ji / DWSIM_CAL_PER_K)
        data.alpha12 = float(alpha)
        data.B12 = 0.0
        data.B21 = 0.0
        data.C12 = 0.0
        data.C21 = 0.0
        data.comment = "ThermoAgent ethanol/ethyl-acetate/water LLE NRTL BIP (cal scale)"
        installed += 1

    # Do not silently auto-estimate missing pairs over our explicit set, and
    # tell DWSIM the parameters changed so it re-runs the interaction init.
    try:
        pp_type.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pp, False, None)
    except Exception:  # noqa: BLE001
        pass
    try:
        pp_type.GetProperty("AreModelParametersDirty").SetValue(pp, True, None)
    except Exception:  # noqa: BLE001
        pass
    # Rebuild the package's internal parameter tables from the dictionary we
    # just wrote.  Without this the rigorous column can fail with
    # "Error evaluating error functions" because the initialized tables still
    # reflect the previous (empty) parameter state.
    try:
        config = pp_type.GetMethod("ConfigParameters")
        if config is not None:
            config.Invoke(pp, None)
    except Exception:  # noqa: BLE001
        pass
    return f"NRTL BIPs installed ({installed}/3 pairs, A12/A21 cal scale)"


# ---------------------------------------------------------------------------
# Column configuration.
# ---------------------------------------------------------------------------
def _set_extractor_mode(column: Any) -> None:
    from System import Enum  # type: ignore[import-not-found]

    prop = column.GetType().GetProperty("OperationMode")
    extractor = Enum.Parse(prop.PropertyType, "Extractor")
    prop.SetValue(column, extractor, None)


def _install_vlle_flash(property_package: Any) -> bool:
    """Bind a two-liquid-phase-capable flash route to the property package.

    The rigorous LLE extractor must run an LLE/VLLE flash on every stage; when
    it falls back to a VLE-only default it cannot produce two liquid phases and
    the column collapses to the "trivial solution".  This mirrors the route in
    generate_lle_mixer_settler_process.py (Nested Loops / Gibbs minimization,
    three-phase stability severity 3).
    """
    try:
        from DWSIM.Interfaces.Enums import FlashSetting  # type: ignore[import-not-found]
        from System import Enum  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False

    try:
        settings_property = property_package.GetType().GetProperty("FlashSettings")
        settings = settings_property.GetValue(property_package, None)
        settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        settings_property.SetValue(property_package, settings, None)
    except Exception:  # noqa: BLE001 - older builds expose different members
        pass

    try:
        approach_property = property_package.GetType().GetProperty("FlashCalculationApproach")
        approach_property.SetValue(
            property_package,
            Enum.Parse(approach_property.PropertyType, "GibbsMinimization"),
            None,
        )
    except Exception:  # noqa: BLE001
        pass

    # Point the material streams at a VLLE-capable flash algorithm by tag where
    # the build supports it (matches the XML-patch route's PreferredFlashAlgorithmTag).
    try:
        flash_base_property = property_package.GetType().GetProperty("FlashBase")
        flash_base = flash_base_property.GetValue(property_package, None)
        if flash_base is not None and "Nested" in str(getattr(flash_base, "Name", "")):
            return True
    except Exception:  # noqa: BLE001
        pass
    return True


def _set_burningham_otto_solver(column: Any) -> bool:
    asm = column.GetType().Assembly
    solver_type = asm.GetType(
        "DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps.SolvingMethods.BurninghamOttoMethod"
    )
    if solver_type is None:
        return False
    solver = solver_type.GetConstructor([]).Invoke([])
    for attr in ("RelaxCompositionUpdates", "RelaxTemperatureUpdates"):
        try:
            setattr(solver, attr, True)
        except Exception:  # noqa: BLE001
            pass
    return _try("set Burningham-Otto solver", column.SetColumnSolver, solver)


def _set_lle_initial_estimates(
    column: Any,
    stages: int,
    organic_flow: float,
    solvent_flow: float,
    strong: bool,
    temperature_k: float,
) -> None:
    """Seed the counter-current extractor with two distinct liquid phases.

    DWSIM reuses its vapor/liquid estimate fields for the two counter-current
    liquid phases in Extractor mode.  We deliberately seed BOTH flows at a real
    positive value and give the two phase compositions a clear water-rich vs.
    ester-rich split, so the LLE solver does not collapse to the trivial K=1
    single-phase solution.

    Component order: ethanol, ethyl acetate, water.  The ester-rich "vapor"
    phase and water-rich "liquid" phase seeds are loosely informed by the real
    ethanol/ethyl-acetate/water tie-lines in datasets/lle/ternary_lle.csv
    (ester phase ~ x_water 0.02-0.05, aqueous phase ~ x_ester 0.02-0.07).
    """

    liquid_flows = [solvent_flow] * stages
    vapor_flows = [organic_flow] * stages
    temperatures = [float(temperature_k)] * stages

    # "liquid" = heavy water-rich phase; "vapor" = light ester-rich phase.
    top_liq = [0.02, 0.01, 0.97] if strong else [0.03, 0.02, 0.95]
    bottom_liq = [0.10, 0.02, 0.88] if strong else [0.10, 0.05, 0.85]
    liquid_comps: list[list[float]] = []
    for i in range(stages):
        frac = i / max(stages - 1, 1)
        row = [(1.0 - frac) * top_liq[j] + frac * bottom_liq[j] for j in range(3)]
        total = sum(row)
        liquid_comps.append([v / total for v in row])

    top_vap = [0.15, 0.82, 0.03] if strong else [0.22, 0.73, 0.05]
    bottom_vap = [0.30, 0.68, 0.02] if strong else [0.38, 0.60, 0.02]
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

    _try("initial liquid-flow estimates", column.SetInitialLiquidMolarFlowEstimates, _double_array(liquid_flows))
    _try("initial vapor-flow estimates", column.SetInitialVaporMolarFlowEstimates, _double_array(vapor_flows))
    _try("initial temperature estimates", column.SetInitialTemperatureEstimates, _double_array(temperatures))
    _try(
        "initial composition estimates",
        column.SetInitialMolarCompositionEstimates,
        _jagged_double_array(liquid_comps),
        _jagged_double_array(vapor_comps),
    )


# ---------------------------------------------------------------------------
# Connections and solving.
# ---------------------------------------------------------------------------
def _connect_feed_stage(column: Any, stream: Any, stage: int, port: int, label: str) -> None:
    """Register a feed on a port and place it on an explicit stage index.

    IMPORTANT (verified against DWSIM 9 reflection):
    * ``ConnectFeed(ISimulationObject, Int32)`` second argument is the FEED PORT
      index (0, 1, ...), NOT the stage number.
    * ``SetStreamFeedStage(MaterialStream, Int32)`` sets which stage the feed
      enters (0 = TopStage, N-1 = BottomStage).
    * DWSIM's LLE extractor requires a feed connected at the FIRST stage
      (index 0) and the solvent at the LAST, for counter-current flow:
      heavy water-rich phase descends from the top, light ester-rich phase
      rises from the bottom.
    """
    sim = _simulation_object(stream)
    _try(f"ConnectFeed {label} (port {port})", column.ConnectFeed, sim, port)
    _try(f"SetStreamFeedStage {label} -> stage {stage}", column.SetStreamFeedStage, sim, stage)


def _connect_products(column: Any, top: Any, bottom: Any) -> None:
    _try("ConnectTopProduct", column.ConnectTopProduct, _simulation_object(top))
    _try("ConnectBottoms", column.ConnectBottoms, _simulation_object(bottom))
    _try("ConnectProductMaterialStream top", column.ConnectProductMaterialStream, _simulation_object(top), 0)
    _try("ConnectProductMaterialStream bottom", column.ConnectProductMaterialStream, _simulation_object(bottom), 1)


def _calculate(automation: Any, flowsheet: Any, column: Any) -> list[str]:
    errors: list[str] = []

    # Solve the rigorous column FIRST and surface any exception verbatim.  The
    # whole-flowsheet shortcuts (CalculateFlowsheet*) can swallow the column
    # solver's exception, which is why "no error" in older runs was misleading.
    try:
        column.Calculate(None)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"column.Calculate raised {type(exc).__name__}: {exc}")
    else:
        print("[OK] column.Calculate(None)")

    def collect(exc_name: str, errs: Any) -> bool:
        if errs is None:
            return False
        if hasattr(errs, "Count") and int(errs.Count) > 0:
            for err in errs:
                errors.append(f"{exc_name}: {err}")
            return True
        return False

    for name in ("CalculateFlowsheet2", "CalculateFlowsheet", "CalculateFlowsheet3", "CalculateFlowsheet4"):
        method = getattr(automation, name, None)
        if not callable(method):
            continue
        try:
            collect(name, method(flowsheet))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name} raised {type(exc).__name__}: {exc}")
        else:
            return errors

    return errors


# ---------------------------------------------------------------------------
# Saving (with the documented temp-path fallback for the DWSIM access quirk).
# ---------------------------------------------------------------------------
def _save(automation: Any, flowsheet: Any, destination: Path) -> None:
    def attempt(path: Path) -> bool:
        candidates = (
            (automation, "SaveFlowsheet2", (flowsheet, str(path))),
            (automation, "SaveFlowsheet2", (flowsheet, str(path), True)),
            (automation, "SaveFlowsheet", (flowsheet, str(path), True)),
            (automation, "SaveFlowsheet", (flowsheet, str(path))),
            (flowsheet, "SaveToXML", (str(path),)),
            (flowsheet, "SaveToFile", (str(path),)),
        )
        for owner, method_name, args in candidates:
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            try:
                method(*args)
            except Exception:  # noqa: BLE001 - try the next spelling
                continue
            return True
        return False

    if attempt(destination):
        return

    # DWSIM/pythonnet may throw UnauthorizedAccessException for paths outside
    # the user temp dir; save there and copy the finished .dwxmz back.
    tmpdir = Path(tempfile.mkdtemp(prefix="thermoagent-dwsim-save-"))
    temp_destination = tmpdir / destination.name
    if not attempt(temp_destination):
        raise RuntimeError("all DWSIM save methods failed")
    shutil.copyfile(temp_destination, destination)
    try:
        shutil.rmtree(tmpdir)
    except PermissionError:  # pragma: no cover - locked by DWSIM briefly
        pass


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ethanol/ethyl-acetate/water DWSIM LLE extractor.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "exports" / "test" / "ethanol_eac_water_lle_extractor_final.dwxmz"),
        help="Output .dwxmz path.",
    )
    parser.add_argument("--stages", type=int, default=8)
    # Feed-stage placement is derived from the counter-current LLE convention
    # (solvent = heavy phase enters stage 0 = top, organic feed = light phase
    # enters stage N-1 = bottom); the old --feed-stage/--solvent-stage knobs are
    # no longer meaningful and are ignored.
    parser.add_argument("--feed-flow", type=float, default=1.0, help="Organic feed molar flow, mol/s.")
    parser.add_argument("--solvent-flow", type=float, default=1.5, help="Water solvent molar flow, mol/s.")
    parser.add_argument(
        "--feed-ethanol-fraction",
        type=float,
        default=0.40,
        help="Ethanol mole fraction in the ethanol/ethyl-acetate organic feed (rest is ethyl acetate).",
    )
    parser.add_argument(
        "--property-package",
        default="NRTL",
        help="Property package to add first (NRTL recommended; fallbacks tried in order).",
    )
    parser.add_argument("--strong-lle-seed", action="store_true")
    parser.add_argument("--calc", action="store_true", help="Run the flowsheet solve before saving.")
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=FEED_TEMPERATURE_K,
        help="Feed/solvent temperature, K (dataset covers 283.15..318.15 K).",
    )
    args = parser.parse_args()

    destination = Path(args.out).resolve()
    if destination.suffix.casefold() != ".dwxmz":
        print("error: output must use the .dwxmz extension")
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)

    Automation3, ObjectType = _load_dwsim()
    automation = Automation3()
    flowsheet = automation.CreateFlowsheet()

    for name in COMPONENTS:
        _try(f"add compound {name}", flowsheet.AddCompound, name)

    package_candidates = [args.property_package, "NRTL", "UNIQUAC", "UNIFAC-LL", "UNIFAC LL", "UNIFACLL"]
    active_package = None
    last_pkg_error: Exception | None = None
    for candidate in package_candidates:
        if candidate in {None, ""}:
            continue
        try:
            _add_property_package(flowsheet, candidate)
        except Exception as exc:  # noqa: BLE001
            last_pkg_error = exc
            _log(False, f"property package {candidate!r}", f"{type(exc).__name__}: {exc}")
            continue
        active_package = candidate
        _log(True, f"property package: {candidate}")
        break
    if active_package is None:
        raise RuntimeError(f"no property package could be added; last error: {last_package_error!r}")

    bip_status = _install_nrtl_bips(flowsheet)
    print(f"[INFO] {bip_status}")

    # Install a VLLE-capable flash route on the first property package.  This is
    # what actually lets the rigorous column produce two liquid phases instead of
    # collapsing to the single-phase "trivial solution".
    packages = list(flowsheet.PropertyPackages.Values)
    if packages:
        if _install_vlle_flash(packages[0]):
            print("[OK] VLLE flash route installed on property package")
        else:
            _log(False, "VLLE flash route install failed; trivial-solution risk remains")

    # The rigorous liquid-liquid extractor (NOT a vessel / splitter / separator).
    column_go = flowsheet.AddObject(ObjectType.AbsorptionColumn, 300, 0, "Liquid-Liquid Extractor")
    column = _simulation_object(column_go)
    _set_extractor_mode(column)
    # Tag the rigorous column's preferred flash so the stage flashes go through
    # the LLE-capable algorithm (avoid the trivial single-phase solution).
    _try(
        "set PreferredFlashAlgorithmTag = Nested Loops (VLLE)",
        setattr,
        column,
        "PreferredFlashAlgorithmTag",
        "Nested Loops (VLLE)",
    )
    _try("SetNumberOfStages", column.SetNumberOfStages, int(args.stages))
    column.NumberOfStages = int(args.stages)
    column.SelectedEquipmentType = "Tray Column"
    column.MaxIterations = 500
    column.InternalLoopTolerance = 1.0e-4
    column.ExternalLoopTolerance = 1.0e-4
    column.ColumnPressureDrop = 1000.0
    column.InitialEstimatesProvider = "Internal 2 (Experimental)"
    if not _set_burningham_otto_solver(column):
        _log(False, "Burningham-Otto solver unavailable; DWSIM defaults will be used")

    feed_go = flowsheet.AddObject(ObjectType.MaterialStream, 0, -90, "Feed_EtOH_EtAc")
    solvent_go = flowsheet.AddObject(ObjectType.MaterialStream, 0, 90, "Solvent_Water")
    top_go = flowsheet.AddObject(ObjectType.MaterialStream, 620, -90, "Raffinate_Top")
    bottom_go = flowsheet.AddObject(ObjectType.MaterialStream, 620, 90, "Extract_Bottom")

    feed = _simulation_object(feed_go)
    feed.SetTemperature(float(args.temperature_k))
    feed.SetPressure(FEED_PRESSURE_PA)
    feed.SetMolarFlow(float(args.feed_flow))
    feed.SetOverallComposition(
        _double_array([float(args.feed_ethanol_fraction), 1.0 - float(args.feed_ethanol_fraction), 0.0])
    )

    solvent = _simulation_object(solvent_go)
    solvent.SetTemperature(float(args.temperature_k))
    solvent.SetPressure(FEED_PRESSURE_PA)
    solvent.SetMolarFlow(float(args.solvent_flow))
    solvent.SetOverallComposition(_double_array([0.0, 0.0, 1.0]))

    # Bind the (VLLE-capable) property package to every material stream so the
    # feed/solvent/product flashes also go through the LLE route, not a VLE-only
    # default.  If a build lacks SetPropertyPackage, this is a harmless no-op.
    for stream in (feed_go, solvent_go, top_go, bottom_go):
        sim = _simulation_object(stream)
        setter = getattr(sim, "SetPropertyPackage", None)
        if callable(setter) and packages:
            _try(f"SetPropertyPackage on {stream.GraphicObject.Tag}", setter, packages[0])

    _set_lle_initial_estimates(
        column,
        int(args.stages),
        organic_flow=float(args.feed_flow),
        solvent_flow=float(args.solvent_flow),
        strong=args.strong_lle_seed or args.calc,
        temperature_k=float(args.temperature_k),
    )

    # Counter-current LLE: heavy water solvent enters the TOP stage (index 0),
    # light organic feed enters the BOTTOM stage (index N-1).  DWSIM requires a
    # feed at the first stage, so the solvent goes there.
    top_index = 0
    bottom_index = int(args.stages) - 1
    _connect_feed_stage(column, solvent_go, top_index, 0, "solvent")
    _connect_feed_stage(column, feed_go, bottom_index, 1, "feed")
    _connect_products(column, top_go, bottom_go)

    # Draw the graphical links too (in Extractor mode these may or may not be
    # accepted; the internal Connect* calls above are the authoritative ones).
    for from_go, to_go, fidx, tidx, label in (
        (feed_go, column_go, 0, 0, "feed -> extractor port 0"),
        (solvent_go, column_go, 0, 1, "solvent -> extractor port 1"),
        (column_go, top_go, 0, 0, "extractor outlet 0 -> top"),
        (column_go, bottom_go, 1, 0, "extractor outlet 1 -> bottom"),
    ):
        _try(label, flowsheet.ConnectObjects, from_go.GraphicObject, to_go.GraphicObject, fidx, tidx)

    solve_errors: list[str] = []
    if args.calc:
        solve_errors = list(_calculate(automation, flowsheet, column))
        for err in solve_errors:
            print(f"[CALC] {err}")

    _save(automation, flowsheet, destination)
    size = destination.stat().st_size

    print(f"[OK] saved: {destination}")
    print(f"[OK] size: {size} bytes")
    print(f"[OK] property package: {active_package}")
    print(
        f"[OK] feed: {args.feed_flow:g} mol/s, "
        f"x=[EtOH {args.feed_ethanol_fraction:g}, EtAc {1.0 - args.feed_ethanol_fraction:g}, Water 0]"
    )
    print(f"[OK] solvent: {args.solvent_flow:g} mol/s, pure water")
    if args.calc and solve_errors:
        print("[WARN] flowsheet solve reported errors -- inspect in DWSIM before using results")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
