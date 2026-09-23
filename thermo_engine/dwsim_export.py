"""DWSIM Automation export for validated non-electrolyte equilibrium runs.

The module imports pythonnet only when an export is requested.  This keeps the
core calculation package runnable on hosts that do not have DWSIM installed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from schemas.column_design import ExtractiveColumnDesign
from schemas.domain import FailureType, RunRecord
from thermo_engine.errors import ThermoEquiError


class DWSIMAutomation(Protocol):
    """Small adapter surface used from DWSIM's ``Automation3`` API."""

    def CreateFlowsheet(self) -> Any: ...


AutomationFactory = Callable[[], DWSIMAutomation]

_PROPERTY_PACKAGES = {
    "ideal/raoult": "Raoult's Law",
    "peng-robinson": "Peng-Robinson (PR)",
    "phasepy/peng-robinson": "Peng-Robinson (PR)",
    "clapeyron/peng-robinson": "Peng-Robinson (PR)",
    "nrtl": "NRTL",
    "wilson": "Wilson",
    "uniquac": "UNIQUAC",
}

#: DWSIM property-package names for the extractive column export, keyed by the
#: schema-level :class:`~schemas.column_design.PropertyPackage` literal.
_COLUMN_PROPERTY_PACKAGES = {
    "NRTL": "NRTL",
    "UNIQUAC": "UNIQUAC",
    "Wilson": "Wilson",
    "Peng-Robinson (PR)": "Peng-Robinson (PR)",
    "Ideal": "Raoult's Law",
}

#: Map ChemSep/model-side component names (and their lowercase variants) to the
#: exact, case-sensitive compound keys that DWSIM's internal dictionary expects.
#: ``AddCompound`` raises ``KeyNotFoundException`` for any name whose key is not
#: present, so names must be canonicalised before being sent to DWSIM.  Keys are
#: any commonly used model spelling (lowercased); values are the DWSIM key.
_DWSIM_COMPOUND_MAP: dict[str, str] = {
    "ethanol": "Ethanol",
    "water": "Water",
    "methanol": "Methanol",
    "toluene": "Toluene",
    "glycerol": "Glycerol",
    "glycerine": "Glycerol",
    "glycerin": "Glycerol",
    "ethylene glycol": "Ethylene glycol",
    "propane": "Propane",
    "n-propane": "Propane",
    # 2-propanol (isopropanol), IPA.  DWSIM's compound dictionary keys its
    # alcohol under the ISO name "Isopropanol" (same capitalization style as
    # Ethanol/Methanol/Glycerol).  Add DWSIM's key here for the IPA feed used by
    # the plain distillation example flowsheet.
    "isopropanol": "Isopropanol",
    "2-propanol": "Isopropanol",
    "isopropyl alcohol": "Isopropanol",
    "propan-2-ol": "Isopropanol",
    "异丙醇": "Isopropanol",
    "2-丙醇": "Isopropanol",
    # 1-chlorobutane / cyclohexane VLE reference system (NIST ThermoML
    # 10.1016/j.fluid.2006.02.009). DWSIM 9.0.5 indexes CAS 109-69-3 under
    # the IUPAC-style catalogue key below, rather than the common name.
    "1-chlorobutane": "1-Chloranylbutane",
    "n-butyl chloride": "1-Chloranylbutane",
    "butyl chloride": "1-Chloranylbutane",
    "1-chloranylbutane": "1-Chloranylbutane",
    "环己烷": "Cyclohexane",
    "cyclohexane": "Cyclohexane",
    # Straight-chain alkane VLE reference system: DWSIM's catalogue keys these
    # compounds with the lowercase ``n-`` prefix.
    "heptane": "n-Heptane",
    "n-heptane": "n-Heptane",
    "正庚烷": "n-Heptane",
    "nonane": "n-Nonane",
    "n-nonane": "n-Nonane",
    "正壬烷": "n-Nonane",
    # Liquid-liquid extraction (DMSO solvent) compounds.
    # Note: DWSIM's compound dictionary keys are case-sensitive and store this
    # ester as the capitalised "N-propyl acetate" (uppercase N, lowercase p).
    "n-propyl acetate": "N-propyl acetate",
    "propyl acetate": "N-propyl acetate",
    "ethyl acetate": "Ethyl acetate",
    "ethylacetate": "Ethyl acetate",
    "dimethyl sulfoxide": "Dimethyl sulfoxide",
    "dimethylsulfoxide": "Dimethyl sulfoxide",
    "dmso": "Dimethyl sulfoxide",
    "二甲基亚砜": "Dimethyl sulfoxide",
    # Acetic acid (water / acetic acid + DMSO extractive system).  Verified
    # against the DWSIM 9.0.5 dictionary: the key is "Acetic acid" with a
    # lowercase 'a'; its absence previously made AddCompound raise
    # KeyNotFoundException and abort the whole export.
    "acetic acid": "Acetic acid",
    "ethanoic acid": "Acetic acid",
    "aceticacid": "Acetic acid",
    "乙酸": "Acetic acid",
    "醋酸": "Acetic acid",
    "乙酸正丙酯": "N-propyl acetate",
    "乙酸乙酯": "Ethyl acetate",
    # Acetonitrile / toluene / tetrahydrofuran system (report §1.5 case 2).
    "acetonitrile": "Acetonitrile",
    "methyl cyanide": "Acetonitrile",
    "乙腈": "Acetonitrile",
    "tetrahydrofuran": "Tetrahydrofuran",
    "thf": "Tetrahydrofuran",
    "四氢呋喃": "Tetrahydrofuran",
    # Binary liquid-liquid (partially miscible) pairs, e.g. the report's
    # water / 1-butanol system.  DWSIM keys this alcohol in lower case
    # ("1-butanol"), unlike Ethanol/Methanol/Glycerol -- verified against the
    # DWSIM 9.0.5 dictionary, where "1-Butanol"/"n-Butanol" raise
    # KeyNotFoundException.
    "1-butanol": "1-butanol",
    "n-butanol": "1-butanol",
    "butanol": "1-butanol",
    "正丁醇": "1-butanol",
    # The identity resolver returns the IUPAC name for MIBK; DWSIM stores it
    # under the common name.
    "4-methyl-2-pentanone": "Methyl isobutyl ketone",
    "methyl isobutyl ketone": "Methyl isobutyl ketone",
    "mibk": "Methyl isobutyl ketone",
    "甲基异丁基酮": "Methyl isobutyl ketone",
    "chloroform": "Chloroform",
    "trichloromethane": "Chloroform",
    "三氯甲烷": "Chloroform",
    "chloroformum": "Chloroform",
    "phenol": "Phenol",
    "苯酚": "Phenol",
    "diethyl ether": "Diethyl ether",
    "ethoxyethane": "Diethyl ether",
    "乙醚": "Diethyl ether",
}


def _dwsim_compound_name(name: str) -> str:
    """Resolve a model-side component name to a DWSIM-recognised compound key.

    DWSIM's ``AddCompound`` is case-sensitive and raises
    ``KeyNotFoundException`` when the key is not in its dictionary.  We therefore
    map common model spellings to the exact DWSIM key.  When no alias is known we
    fall back to the name as given so well-formed names still work unchanged.
    """
    resolved = _DWSIM_COMPOUND_MAP.get(name.lower())
    if resolved is not None:
        return resolved
    return name


def missing_dwsim_compound_mappings(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return model-side compound names that have no explicit DWSIM mapping.

    DWSIM's dictionary is case-sensitive and installation-dependent enough that
    user-facing exports should fail early with a clear mapping/import request
    instead of discovering the problem inside ``AddCompound``.
    """
    return [name for name in names if name.casefold() not in _DWSIM_COMPOUND_MAP]


def _runtime_error(message: str, *, details: dict[str, object] | None = None) -> ThermoEquiError:
    return ThermoEquiError(
        FailureType.MISSING_DATA,
        message,
        "Install DWSIM and pythonnet, then set DWSIM_HOME to DWSIM's installation directory.",
        details,
    )


def _automation_factory() -> tuple[AutomationFactory, Any]:
    """Load DWSIM assemblies only for an explicit export request."""

    # Ensure the project's `.env` values (DWSIM_HOME and friends) are visible in
    # this process regardless of how the server was launched.  load_dotenv() is
    # idempotent and, by default, does not override variables already set.
    from dotenv import load_dotenv

    load_dotenv()

    dwsim_home = os.getenv("DWSIM_HOME")
    if not dwsim_home:
        raise _runtime_error("DWSIM_HOME is not configured.", details={"environment_variable": "DWSIM_HOME"})

    install_dir = Path(dwsim_home).expanduser().resolve()
    automation_dll = install_dir / "DWSIM.Automation.dll"
    if not automation_dll.is_file():
        raise _runtime_error(
            "DWSIM.Automation.dll was not found in DWSIM_HOME.",
            details={"dwsim_home": str(install_dir), "required_file": "DWSIM.Automation.dll"},
        )

    dwsim_temp = Path(os.getenv("DWSIM_TEMP_DIR") or Path.cwd() / ".tmp" / "dwsim").resolve()
    dwsim_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(dwsim_temp)
    os.environ["TMP"] = str(dwsim_temp)

    try:
        import clr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise _runtime_error("pythonnet is not installed.", details={"required_package": "pythonnet"}) from exc

    if str(install_dir) not in sys.path:
        sys.path.append(str(install_dir))
    try:
        clr.AddReference(str(automation_dll))
        from DWSIM.Automation import Automation3  # type: ignore[import-not-found]
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - requires a local DWSIM installation
        raise _runtime_error(
            "DWSIM assemblies could not be loaded.",
            details={"dwsim_home": str(install_dir), "exception": type(exc).__name__},
        ) from exc
    return Automation3, ObjectType


def _first_value(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _flowsheet_values(run: RunRecord) -> tuple[list[str], list[float], float, float, str]:
    snapshot = run.input_snapshot
    components = snapshot.get("components", [])
    names: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("The run snapshot does not contain valid DWSIM component names.")
        name = component.get("name")
        if not isinstance(name, str):
            raise ValueError("The run snapshot does not contain valid DWSIM component names.")
        names.append(name)
    if not names:
        raise ValueError("The run snapshot does not contain valid DWSIM component names.")

    conditions = snapshot.get("conditions", {})
    result = run.result
    points = result.get("points", [])
    first_point = points[0] if points and isinstance(points[0], dict) else {}
    composition = _first_composition(
        conditions.get("feed_composition"),
        conditions.get("liquid_composition"),
        conditions.get("vapor_composition"),
        first_point.get("liquid_composition"),
        first_point.get("vapor_composition"),
    )
    if composition is None or len(composition) != len(names):
        raise ValueError("The run does not contain a complete feed composition for DWSIM export.")

    temperature_K = _first_value(
        result.get("temperature_K"), conditions.get("temperature_K"), first_point.get("temperature_K")
    )
    pressure_kPa = _first_value(
        result.get("pressure_kPa"), conditions.get("pressure_kPa"), first_point.get("pressure_kPa")
    )
    if temperature_K is None or pressure_kPa is None:
        raise ValueError("The run does not contain both temperature and pressure for DWSIM export.")
    model_name = result.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError("The run does not contain a thermodynamic model name.")
    property_package = _PROPERTY_PACKAGES.get(model_name.casefold())
    if property_package is None:
        raise ValueError(f"DWSIM export does not support model '{model_name}'.")
    return names, composition, temperature_K, pressure_kPa, property_package


def _first_composition(*values: object) -> list[float] | None:
    for value in values:
        if isinstance(value, list) and value and all(isinstance(item, float | int) for item in value):
            return [float(item) for item in value]
    return None


def _save_flowsheet(automation: DWSIMAutomation, flowsheet: Any, destination: Path) -> None:
    """Save through the public Automation API, supporting maintained DWSIM variants."""

    last_error: Exception | None = None
    for owner, method_name, args in (
        (automation, "SaveFlowsheet2", (flowsheet, str(destination))),
        (automation, "SaveFlowsheet", (flowsheet, str(destination), True)),
        (automation, "SaveFlowsheet", (flowsheet, str(destination))),
        (flowsheet, "SaveToXML", (str(destination),)),
        (flowsheet, "SaveToFile", (str(destination),)),
    ):
        method = getattr(owner, method_name, None)
        if callable(method):
            try:
                method(*args)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
            else:
                return
    if last_error is not None:
        raise last_error
    raise RuntimeError("The installed DWSIM Automation API has no supported flowsheet save method.")


def _save_flowsheet_via_temp(automation: DWSIMAutomation, flowsheet: Any, destination: Path) -> None:
    """Save a flowsheet, falling back through a local temp file on access errors.

    Some DWSIM/pythonnet combinations throw ``UnauthorizedAccessException`` when
    saving directly into project workspaces, even when Python itself can write
    there.  Saving under the user's temp directory and copying the finished
    ``.dwxmz`` back keeps the export deterministic while avoiding that DWSIM-side
    path permission quirk.
    """
    try:
        _save_flowsheet(automation, flowsheet, destination)
        return
    except Exception as exc:
        if "UnauthorizedAccess" not in type(exc).__name__ and "UnauthorizedAccess" not in str(exc):
            raise

    tmpdir = Path(tempfile.mkdtemp(prefix="thermoagent-dwsim-"))
    temp_destination = tmpdir / destination.name
    _save_flowsheet(automation, flowsheet, temp_destination)
    shutil.copyfile(temp_destination, destination)
    try:
        shutil.rmtree(tmpdir)
    except PermissionError:
        # DWSIM can keep the just-saved package locked for a short period.  The
        # exported copy is already in ``destination``; a stale temp folder is
        # less harmful than reporting a failed export to the UI.
        pass


def _add_property_package(flowsheet: Any, property_package: str) -> None:
    """Add a property package through the API supported by the installed DWSIM version."""

    create_and_add = getattr(flowsheet, "CreateAndAddPropertyPackage", None)
    if callable(create_and_add):
        create_and_add(property_package)
        return
    flowsheet.AddPropertyPackage(property_package)


def _simulation_object(automation_object: Any) -> Any:
    """Unwrap DWSIM 9's generic simulation-object interface when available."""

    get_as_object = getattr(automation_object, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else automation_object


def _composition_argument(composition: list[float]) -> Any:
    """Convert compositions to the .NET array required by DWSIM's API."""

    try:
        from System import Array, Double  # type: ignore[import-not-found]
    except ImportError:
        return composition
    return Array[Double](composition)


def export_dwsim_flowsheet(
    run: RunRecord,
    destination: Path,
    *,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a DWSIM TP-flash flowsheet from an immutable validated run snapshot.

    The material feed is normalized to a total molar-flow basis of 1.0.
    Temperature is in K and the internal DWSIM pressure call receives Pa,
    converted from the API's kPa.
    """

    names, composition, temperature_K, pressure_kPa, property_package = _flowsheet_values(run)
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()
    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in names:
            flowsheet.AddCompound(_dwsim_compound_name(name))
        _add_property_package(flowsheet, property_package)

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        separator = flowsheet.AddObject(object_type.Vessel, 250, 0, "Equilibrium Flash")
        vapor = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Vapor Product")
        liquid = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Liquid Product")
        feed_stream = _simulation_object(feed)
        feed_stream.SetTemperature(temperature_K)
        feed_stream.SetPressure(pressure_kPa * 1000.0)
        feed_stream.SetMolarFlow(1.0)
        feed_stream.SetOverallComposition(_composition_argument(composition))
        flowsheet.ConnectObjects(feed.GraphicObject, separator.GraphicObject, 0, 0)
        flowsheet.ConnectObjects(separator.GraphicObject, vapor.GraphicObject, 0, 0)
        flowsheet.ConnectObjects(separator.GraphicObject, liquid.GraphicObject, 1, 0)
        _save_flowsheet_via_temp(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the phase-equilibrium flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination


def _extraction_solver_member(object_type: Any) -> Any:
    """Resolve a *real* liquid-liquid extraction unit operation.

    Do not fall back to Vessel, Splitter, or ComponentSeparator: those objects
    can create connected flowsheets, but they are not liquid-liquid extraction
    towers and will misrepresent the simulation.
    """
    for name in (
        "LiquidLiquidExtractor",
        "LiquidLiquidExtractionColumn",
        "ExtractionColumn",
        "LiquidLiquidColumn",
    ):
        member = getattr(object_type, name, None)
        if member is not None:
            return member
    raise _runtime_error(
        "The installed DWSIM Automation API exposes no real liquid-liquid extraction tower object.",
        details={
            "required_object_aliases": [
                "LiquidLiquidExtractor",
                "LiquidLiquidExtractionColumn",
                "ExtractionColumn",
                "LiquidLiquidColumn",
            ],
            "rejected_fallbacks": ["Vessel", "Splitter", "ComponentSeparator"],
        },
    )

def export_dwsim_lle_extraction(
    components: list[str],
    feed_composition: list[float],
    feed_flow_mol_s: float,
    feed_temperature_K: float,
    feed_pressure_kPa: float,
    solvent: str,
    solvent_ratio: float,
    property_package: str = "NRTL",
    destination: Path | None = None,
    raffinate_composition: list[float] | None = None,
    raffinate_flow_mol_s: float | None = None,
    extract_composition: list[float] | None = None,
    extract_flow_mol_s: float | None = None,
    *,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a DWSIM **liquid-liquid extraction** (decanter) flowsheet.

    This is a distinct export path from :func:`export_dwsim_flowsheet` (TP flash)
    and :func:`export_dwsim_extractive_column` (extractive *distillation*).  It
    models a liquid-liquid extraction with a single equilibrium decanter:

    * ``components`` must be exactly 3 — the two feed solutes plus the ``solvent``.
    * a ``Feed`` stream (solutes) and a ``Solvent`` stream (pure extraction solvent,
      molar flow = ``solvent_ratio`` × feed flow) enter the decanter;
    * the decanter splits into a ``Raffinate`` and an ``Extract`` product stream.

    **Scientific boundary:** this system's production backends cannot numerically
    compute liquid-liquid equilibrium, so no binary parameters are fabricated or
    written out here.  The flowsheet carries the NRTL property package and lets
    DWSIM supply its own built-in binary-interaction parameters; the equilibrium
    split is therefore computed by DWSIM, not by this project.  Always verify the
    solvent selectivity and the phase split in DWSIM before using the result for
    design.  This mirrors the documented limitation that LLE numbers never come
    from the LLM.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(components) != 3 or len(feed_composition) != 2:
        raise ValueError("LLE export requires three components (two solutes + solvent) and two feed fractions")
    if abs(sum(feed_composition) - 1.0) > 1e-6:
        raise ValueError("feed_composition must sum to one within 1e-6")
    if solvent not in components:
        raise ValueError("the extraction solvent must be one of the three components")

    canonical_names = [_dwsim_compound_name(name) for name in components]
    solvent_canonical = _dwsim_compound_name(solvent)
    preset_products = False

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in canonical_names:
            flowsheet.AddCompound(name)
        _add_property_package(flowsheet, property_package)

        extractor_type = _extraction_solver_member(object_type)
        decanter = flowsheet.AddObject(extractor_type, 250, 0, "Tie-Line LLE Splitter" if preset_products else "Liquid-Liquid Extractor")

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, -80, "Feed")
        solvent_stream = flowsheet.AddObject(object_type.MaterialStream, 0, 80, "Solvent")
        mixed_feed = (
            flowsheet.AddObject(object_type.MaterialStream, 170, 0, "Mixed Feed") if preset_products else None
        )
        raffinate = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Raffinate")
        extract = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Extract")

        # Feed: the two solutes only (solvent fraction zero).
        feed_stream = _simulation_object(feed)
        feed_stream.SetTemperature(float(feed_temperature_K))
        feed_stream.SetPressure(float(feed_pressure_kPa) * 1000.0)
        feed_stream.SetMolarFlow(float(feed_flow_mol_s))
        feed_comp = list(feed_composition) + [0.0]
        feed_stream.SetOverallComposition(_composition_argument(feed_comp))

        # Solvent: pure extraction solvent, molar flow = ratio × feed flow.
        solvent_obj = _simulation_object(solvent_stream)
        solvent_obj.SetTemperature(float(feed_temperature_K))
        solvent_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
        solvent_obj.SetMolarFlow(float(solvent_ratio) * float(feed_flow_mol_s))
        solvent_index = canonical_names.index(solvent_canonical)
        solvent_comp = [0.0, 0.0, 0.0]
        solvent_comp[solvent_index] = 1.0
        solvent_obj.SetOverallComposition(_composition_argument(solvent_comp))

        if preset_products:
            total_flow = float(feed_flow_mol_s) * (1.0 + float(solvent_ratio))
            overall = [
                float(feed_flow_mol_s) * float(feed_composition[0]) / total_flow,
                float(feed_flow_mol_s) * float(feed_composition[1]) / total_flow,
                float(solvent_ratio) * float(feed_flow_mol_s) / total_flow,
            ]
            mixed_obj = _simulation_object(mixed_feed)
            mixed_obj.SetTemperature(float(feed_temperature_K))
            mixed_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
            mixed_obj.SetMolarFlow(total_flow)
            mixed_obj.SetOverallComposition(_composition_argument(overall))

            raffinate_obj = _simulation_object(raffinate)
            raffinate_obj.SetTemperature(float(feed_temperature_K))
            raffinate_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
            raffinate_obj.SetMolarFlow(float(raffinate_flow_mol_s))
            raffinate_obj.SetOverallComposition(_composition_argument([float(v) for v in raffinate_composition]))

            extract_obj = _simulation_object(extract)
            extract_obj.SetTemperature(float(feed_temperature_K))
            extract_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
            extract_obj.SetMolarFlow(float(extract_flow_mol_s))
            extract_obj.SetOverallComposition(_composition_argument([float(v) for v in extract_composition]))

            separator_obj = _simulation_object(decanter)
            try:
                separator_obj.SpecifiedStreamIndex = 0
                from System import Enum  # type: ignore[import-not-found]

                spec_type = separator_obj.ComponentSepSpecs.GetType().GetGenericArguments()[1]
                sep_enum = spec_type.GetProperty("SepSpec").PropertyType
                percent_molar = Enum.Parse(sep_enum, "PercentInletMolarFlow")
                specs = separator_obj.ComponentSepSpecs
                for idx, compound in enumerate(canonical_names):
                    inlet_component_flow = total_flow * overall[idx]
                    fraction_to_raffinate = (
                        float(raffinate_flow_mol_s) * float(raffinate_composition[idx]) / inlet_component_flow
                        if inlet_component_flow > 0.0
                        else 0.0
                    )
                    value = max(0.0, min(100.0, 100.0 * fraction_to_raffinate))
                    specs[compound] = spec_type(compound, percent_molar, value, "%")
                separator_obj.ComponentSepSpecs = specs
            except Exception:
                pass

        if preset_products:
            mixer = flowsheet.AddObject(object_type.Mixer, 90, 0, "Feed Solvent Mixer")
            connection_attempts: list[tuple[Any, Any, int, int, str]] = [
                (feed.GraphicObject, mixer.GraphicObject, 0, 0, "feed->mixer"),
                (solvent_stream.GraphicObject, mixer.GraphicObject, 0, 1, "solvent->mixer"),
                (mixer.GraphicObject, mixed_feed.GraphicObject, 0, 0, "mixer->mixed-feed"),
                (mixed_feed.GraphicObject, decanter.GraphicObject, 0, 0, "mixed-feed->separator"),
                (decanter.GraphicObject, raffinate.GraphicObject, 0, 0, "separator->raffinate"),
                (decanter.GraphicObject, extract.GraphicObject, 1, 0, "separator->extract"),
            ]
        else:
            connection_attempts = [
                (feed.GraphicObject, decanter.GraphicObject, 0, 0, "feed->extractor"),
                (solvent_stream.GraphicObject, decanter.GraphicObject, 0, 1, "solvent->extractor"),
                (decanter.GraphicObject, raffinate.GraphicObject, 0, 0, "extractor->raffinate"),
                (decanter.GraphicObject, extract.GraphicObject, 1, 0, "extractor->extract"),
            ]
        for from_obj, to_obj, fidx, tidx, _label in connection_attempts:
            try:
                flowsheet.ConnectObjects(from_obj, to_obj, fidx, tidx)
            except Exception:  # noqa: BLE001 - surface, don't abort
                pass

        _save_flowsheet_via_temp(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the liquid-liquid extraction flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination


def _build_lle_vessel_flowsheet(
    automation: Any,
    object_type: Any,
    canonical_names: list[str],
    feed_composition: list[float],
    temperature_K: float,
    pressure_kPa: float,
    feed_flow_mol_s: float,
    property_package: str,
) -> tuple[Any, Any, Any]:
    """Build the ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid`` case.

    Shared by the binary and ternary LLE exports so both carry an identical,
    template-matched property set.  Returns ``(flowsheet, light, heavy)``.

    The vessel flash is pinned to the feed conditions and given the same
    attribute set as ``report/dwsim/water_butanol_NRTL_default.dwxmz``
    (Legacy calculation mode, OverrideT/P off, minimum pressure = operating
    pressure).  Without the pin a DWSIM ``Vessel`` would flash at its own
    298.15 K default and evaluate the split at the wrong temperature.
    """
    flowsheet = automation.CreateFlowsheet()
    # Each flowsheet needs its own AddCompound calls; compounds are not cached
    # across flowsheets (otherwise the flash collapses to one phase).
    for name in canonical_names:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, property_package)

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel = flowsheet.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vapor = flowsheet.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light = flowsheet.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy = flowsheet.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    pressure_pa = float(pressure_kPa) * 1000.0

    feed_stream = _simulation_object(feed)
    feed_stream.SetTemperature(float(temperature_K))
    feed_stream.SetPressure(pressure_pa)
    feed_stream.SetMolarFlow(float(feed_flow_mol_s))
    feed_stream.SetOverallComposition(_composition_argument([float(v) for v in feed_composition]))

    vessel_object = _simulation_object(vessel)
    for attribute, value in (
        ("FlashTemperature", float(temperature_K)),
        ("FlashPressure", pressure_pa),
        ("OverrideT", False),
        ("OverrideP", False),
        ("CalculationMode", "Legacy"),
        ("MinimumPressure", pressure_pa),
    ):
        try:
            setattr(vessel_object, attribute, value)
        except Exception:  # noqa: BLE001 - non-fatal: feed conditions still drive the flash
            pass

    # The product streams inherit the flash temperature so the saved file
    # reports consistent phase temperatures.
    for product in (vapor, light, heavy):
        try:
            _simulation_object(product).SetTemperature(float(temperature_K))
        except Exception:  # noqa: BLE001 - non-fatal on some DWSIM builds
            pass

    for from_obj, to_obj, fidx, tidx, label in (
        (feed.GraphicObject, vessel.GraphicObject, 0, 0, "feed->vessel"),
        (vessel.GraphicObject, vapor.GraphicObject, 0, 0, "vessel->vapor"),
        (vessel.GraphicObject, light.GraphicObject, 1, 0, "vessel->light-liquid"),
        (vessel.GraphicObject, heavy.GraphicObject, 2, 0, "vessel->heavy-liquid"),
    ):
        try:
            flowsheet.ConnectObjects(from_obj, to_obj, fidx, tidx)
        except Exception as exc:  # noqa: BLE001 - surface, don't abort
            print(f"[WARN] LLE: DWSIM rejected the {label} connection ({type(exc).__name__})")

    return flowsheet, light, heavy


def _solve_and_report_split(
    automation: Any,
    flowsheet: Any,
    light: Any,
    heavy: Any,
    canonical_names: list[str],
    temperature_K: float,
    split_out: dict[str, object] | None,
) -> dict[str, object]:
    """Solve the flowsheet, then read DWSIM's own phase split back out.

    Solving *before* saving is mandatory: a file saved without it is an unsolved
    skeleton whose objects all keep ``<Calculated>false</Calculated>``, so the
    GUI shows empty phase fractions.  The report templates
    (``report/dwsim/water_butanol_*.dwxmz``) are produced by scripts that call
    ``CalculateFlowsheet4`` before saving.

    A solve failure must NOT discard the file -- the structure is still correct
    and the user can re-run it in the GUI -- so it is surfaced, not raised.
    """
    solve_error = ""
    calculate = getattr(automation, "CalculateFlowsheet4", None)
    if callable(calculate):
        try:
            errors = calculate(flowsheet)
            if errors is not None and getattr(errors, "Count", 0):
                solve_error = str(errors[0])[:200]
        except Exception as exc:  # noqa: BLE001 - depends on installed DWSIM assemblies
            solve_error = f"{type(exc).__name__}: {exc}"[:200]
    if solve_error:
        print(f"[WARN] LLE: flowsheet did not converge ({solve_error})")

    split = _binary_lle_split(light, heavy)
    if split_out is not None:
        split_out.clear()
        split_out.update(split)

    if not split.get("separated", False):
        print(
            "[WARN] LLE: DWSIM computed a single liquid phase for "
            f"{' + '.join(canonical_names)} at {temperature_K:.2f} K; "
            f"the heavy phase carries {split.get('heavy_flow_mol_s')} mol/s. "
            "The file is still written, but it does not show a two-liquid split."
        )
    return split


def export_dwsim_binary_lle_flowsheet(
    components: list[str],
    feed_composition: list[float],
    temperature_K: float,
    pressure_kPa: float = 101.325,
    *,
    feed_flow_mol_s: float = 1.0,
    property_package: str = "NRTL",
    destination: Path,
    split_out: dict[str, object] | None = None,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a binary **liquid-liquid** DWSIM flowsheet from the report template.

    This reproduces the topology of ``report/dwsim/water_butanol_LLE_<T>K.dwxmz``
    (built by ``scripts/generate_water_butanol_dwsim.py``) for an arbitrary
    two-component partially miscible pair::

        Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid

    with the NRTL property package.  ``Light_Liquid`` is the organic-rich phase
    and ``Heavy_Liquid`` is the aqueous (water-rich) phase, matching the phase
    naming used in the report.

    **Scientific boundary:** this project never computes LLE numbers.  The
    flowsheet carries only structure plus the feed state; DWSIM resolves the
    liquid-liquid split with its own built-in binary-interaction parameters (or
    with ``AutoEstimateMissingNRTLUNIQUACParameters`` for pairs it does not
    have).  Always open the file in DWSIM and check the phase split before using
    the result for design.

    ``temperature_K`` is the flash temperature.  It is written to BOTH the feed
    stream and the vessel's explicit ``FlashTemperature``/``FlashPressure``
    because a DWSIM ``Vessel`` otherwise defaults to a 298.15 K flash and would
    evaluate the split at the wrong temperature.

    The flowsheet is solved with ``CalculateFlowsheet4`` **before** saving so the
    written file carries real phase results rather than an unsolved skeleton.

    ``split_out``, when supplied, is filled with DWSIM's own two-liquid result:
    ``light_flow_mol_s`` / ``heavy_flow_mol_s``, the matching
    ``*_composition`` arrays, and a ``separated`` flag that is ``False`` when the
    heavy phase carries no flow (i.e. DWSIM predicted a single liquid phase).
    A non-separating result is still written and reported, never silently passed
    off as a two-liquid split.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(components) != 2 or len(feed_composition) != 2:
        raise ValueError("binary LLE export requires exactly two components and two feed fractions")
    if abs(sum(feed_composition) - 1.0) > 1e-6:
        raise ValueError("feed_composition must sum to one within 1e-6")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if feed_flow_mol_s <= 0:
        raise ValueError("feed_flow_mol_s must be positive")

    canonical_names = [_dwsim_compound_name(name) for name in components]
    if len({name.casefold() for name in canonical_names}) != 2:
        raise ValueError("a binary LLE case requires two distinct components")

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    try:
        automation = factory()
        flowsheet, light, heavy = _build_lle_vessel_flowsheet(
            automation,
            object_type,
            canonical_names,
            [float(feed_composition[0]), float(feed_composition[1])],
            float(temperature_K),
            float(pressure_kPa),
            float(feed_flow_mol_s),
            property_package,
        )
        split = _solve_and_report_split(
            automation, flowsheet, light, heavy, canonical_names, float(temperature_K), split_out
        )
        _save_flowsheet_via_temp(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the binary liquid-liquid flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")

    del split  # reported via ``split_out`` and the warnings above
    return destination


def export_dwsim_ternary_lle_flowsheet(
    components: list[str],
    feed_composition: list[float],
    temperature_K: float,
    pressure_kPa: float = 101.325,
    *,
    feed_flow_mol_s: float = 1.0,
    property_package: str = "NRTL",
    destination: Path,
    split_out: dict[str, object] | None = None,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a three-component **liquid-liquid** DWSIM flowsheet.

    Same topology and property set as the binary export -- ``Feed ->
    Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid`` -- generalised to three
    components::

        e.g. Ethanol + Ethyl acetate + Water, NRTL

    ``Light_Liquid`` is the organic-rich (ester-rich) phase and ``Heavy_Liquid``
    is the aqueous phase.

    **Known DWSIM 9.0.5 limitation -- please read.**  Unlike the binary case,
    which DWSIM solves correctly out of the box, the *ternary* liquid-liquid
    split does **not** resolve through the Automation API on this DWSIM build.
    Measured on the local install with the feed ``z = [0.129, 0.188, 0.683]``:

    * all three ``FlashCalculationApproach`` kernels (``NestedLoops``,
      ``InsideOut``, ``GibbsMinimization``) return a single liquid phase;
    * all probed ``PreferredFlashAlgorithmTag`` values -- including
      ``Nested Loops (Immiscible)``, ``Nested Loops (VLLE)`` and ``Simple LLE``
      -- also return a single liquid phase;
    * the ``FlashSettings`` entries that would enable the immiscible kernel
      (``ImmiscibleWaterOption`` etc.) are read-only here and raise ``TypeError``.

    So this function writes a structurally correct, fully solved file whose
    ``Heavy_Liquid`` phase is typically **empty**; DWSIM is reporting, honestly,
    that its default kernel found one liquid.  The caller must surface that (see
    ``split_out['separated']``) rather than present it as a working extraction.
    Opening the file in the DWSIM GUI and selecting an immiscible flash kernel
    interactively is currently the way to obtain a split.

    **Scientific boundary:** this project never computes LLE numbers.  No binary
    parameters are written; DWSIM resolves the flash with its own built-ins.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(components) != 3 or len(feed_composition) != 3:
        raise ValueError("ternary LLE export requires exactly three components and three feed fractions")
    if abs(sum(feed_composition) - 1.0) > 1e-6:
        raise ValueError("feed_composition must sum to one within 1e-6")
    if any(value < 0 for value in feed_composition):
        raise ValueError("feed_composition must not contain negative fractions")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if feed_flow_mol_s <= 0:
        raise ValueError("feed_flow_mol_s must be positive")

    canonical_names = [_dwsim_compound_name(name) for name in components]
    if len({name.casefold() for name in canonical_names}) != 3:
        raise ValueError("a ternary LLE case requires three distinct components")

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    try:
        automation = factory()
        flowsheet, light, heavy = _build_lle_vessel_flowsheet(
            automation,
            object_type,
            canonical_names,
            [float(v) for v in feed_composition],
            float(temperature_K),
            float(pressure_kPa),
            float(feed_flow_mol_s),
            property_package,
        )
        _solve_and_report_split(
            automation, flowsheet, light, heavy, canonical_names, float(temperature_K), split_out
        )
        _save_flowsheet_via_temp(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the ternary liquid-liquid flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination


def _binary_lle_split(light_object: Any, heavy_object: Any) -> dict[str, object]:
    """Read the two-liquid split DWSIM computed for the LLE product streams.

    Returns a JSON-friendly dict with the molar flow and overall composition of
    the light (organic-rich) and heavy (water-rich) phases.  Every value is
    DWSIM's own result; nothing here is computed by this project.

    The ``separated`` flag is the important one: it is ``False`` when the heavy
    phase carries no flow, which means the feed did **not** actually split into
    two liquids.  A rendered file in that state opens correctly but shows an
    empty heavy phase, so callers must surface it rather than claim success.
    """
    report: dict[str, object] = {}
    for label, handle in (("light", light_object), ("heavy", heavy_object)):
        try:
            stream = _simulation_object(handle)
            stream_type = stream.GetType()
            flow = float(stream_type.GetMethod("GetMolarFlow").Invoke(stream, None))
            composition = [
                float(value)
                for value in stream_type.GetMethod("GetOverallComposition").Invoke(stream, None)
            ]
            report[f"{label}_flow_mol_s"] = flow
            report[f"{label}_composition"] = composition
        except Exception:  # noqa: BLE001 - depends on installed DWSIM assemblies
            report[f"{label}_flow_mol_s"] = None
            report[f"{label}_composition"] = None
    heavy_flow = report.get("heavy_flow_mol_s")
    report["separated"] = isinstance(heavy_flow, float) and heavy_flow > 1e-10
    return report


def _column_object_type(object_type: Any) -> Any:
    """Resolve a DWSIM ``ObjectType`` member for the extractive-distillation column.

    An extractive column needs TWO feeds (feed + entrainer) entering at different
    stages, which only a ``DistillationColumn`` (rigorous column) expresses.  A
    ``ShortcutColumn`` accepts a single feed, so it is used only as a last resort
    when the rigorous column is unavailable in the installed ObjectType enum.
    """
    for name in ("DistillationColumn", "ShortcutColumn", "Column"):
        member = getattr(object_type, name, None)
        if member is not None:
            return member
    raise _runtime_error(
        "The installed DWSIM Automation API exposes no column object type.",
        details={"available_column_aliases": ["DistillationColumn", "ShortcutColumn", "Column"]},
    )


def _set_column_product_specs(
    column: Any,
    *,
    distillate_flow_mol_s: float,
    bottoms_flow_mol_s: float,
) -> list[str]:
    """Write the product-flow specification onto a DWSIM rigorous column.

    A rigorous ``DistillationColumn`` carries two specification slots in
    ``column.Specs``, keyed ``"C"`` (condenser) and ``"R"`` (reboiler).  A freshly
    created column already has both entries, pre-configured as::

        "C": Stream_Ratio              (fed by RefluxRatio)
        "R": Product_Molar_Flow_Rate = 0.0

    The condenser slot is already correct and is therefore left untouched.  The
    reboiler slot, however, defaults to a **zero** product flow; leaving it at zero
    removes the degree of freedom the solver needs to close the component balance,
    and DWSIM then aborts with::

        Failed to fulfill mass balance for <component>: Relative Error = ~1.0

    Writing the true bottoms flow closes the balance.  The spec object exposes
    ``SType``, ``SpecValue`` and ``SpecUnit`` as writable properties.  Failures are
    reported as warnings rather than raised, so a partially configured file is still
    produced for inspection in the GUI.
    """
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import (  # type: ignore[import-not-found]
        ColumnSpec as _ColumnSpec,
    )

    warnings: list[str] = []
    specs = getattr(column, "Specs", None)
    if specs is None:
        return ["column exposes no Specs collection; product flows were not specified"]

    try:
        spec = specs["R"]
    except Exception:  # noqa: BLE001 - key absent in this build
        return [
            "column reboiler spec 'R' is missing; bottoms flow "
            f"{bottoms_flow_mol_s:.4f} mol/s not written"
        ]

    try:
        spec.SType = _ColumnSpec.SpecType.Product_Molar_Flow_Rate
        spec.SpecValue = float(bottoms_flow_mol_s)
        spec.SpecUnit = "mol/s"
        spec.ComponentID = ""
        spec.ComponentIndex = -1
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"column reboiler spec could not be set ({type(exc).__name__}); "
            "the rigorous solver may report a mass-balance failure"
        )

    # Informational: record the intended distillate rate.  The condenser stays on the
    # reflux Stream_Ratio, so the column remains exactly specified.
    try:
        column.DistillateFlowRate = float(distillate_flow_mol_s)
    except Exception:  # noqa: BLE001 - informational only
        pass
    return warnings


def _set_column_pressure_drop(column: Any, pressure_drop_kPa: float | None) -> list[str]:
    """Apply the column pressure drop (kPa) to a DWSIM rigorous column.

    DWSIM stores the rigorous-column pressure drop in Pa under
    ``ColumnPressureDrop``.  The value is supplied by the caller (request
    parameter); when absent, DWSIM keeps its own default.  Several member
    spellings are probed because the property is exposed differently across
    DWSIM builds.
    """
    warnings: list[str] = []
    if pressure_drop_kPa is None:
        return warnings
    if pressure_drop_kPa < 0:
        raise ValueError("pressure_drop_kPa must be non-negative")

    value_pa = float(pressure_drop_kPa) * 1000.0
    for attribute in ("ColumnPressureDrop", "SetColumnPressureDrop", "PressureDrop", "ColumnDeltaP"):
        target = getattr(column, attribute, None)
        if target is None:
            continue
        try:
            if callable(target):
                target(value_pa)
            else:
                setattr(column, attribute, value_pa)
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(f"column pressure drop could not be set via {attribute}: {type(exc).__name__}")
        else:
            return warnings
    warnings.append(
        "column pressure drop could not be set: no supported member found "
        f"(tried {', '.join(('ColumnPressureDrop', 'SetColumnPressureDrop', 'PressureDrop', 'ColumnDeltaP'))})"
    )
    return warnings


def _register_feed_on_stage(
    column: Any,
    stream: Any,
    stage: int,
    label: str,
) -> list[str]:
    """Bind a feed stream to an explicit column tray.

    Reflected DWSIM API for a rigorous ``DistillationColumn``::

        ConnectFeed(ISimulationObject feed, Int32 stagenumber)
        SetStreamFeedStage(MaterialStream stream, Int32 stageIndex)

    The second argument of ``ConnectFeed`` is the TRAY NUMBER (not a port
    index), and ``stageIndex`` is 0-based with 0 = top stage.  ``ConnectFeed``
    performs the connection itself, so no separate graphic ``ConnectObjects``
    call is needed for this stream; calling both makes DWSIM reject one of them.

    ``stage`` here is the 0-based stage index.

    **Both calls are required, in this order.**  Verified against this DWSIM
    install by reading back ``GetStreamFeedStageIndex``:

    * ``ConnectFeed`` alone registers the stream but leaves the tray unset -- the
      index reads back as ``-1``.  A column in that state has no feeds, so the
      solver fails its mass balance (``Failed to fulfill mass balance for
      <component>: Relative Error = ~1.0``).
    * ``SetStreamFeedStage`` alone raises ``NullReferenceException``, because the
      stream is not yet attached to the column.
    * ``ConnectFeed`` followed by ``SetStreamFeedStage`` binds the tray correctly.

    Order is therefore significant, and the two are not interchangeable; an
    earlier revision that treated them as alternatives silently produced
    columns with unbound feeds.
    """
    warnings: list[str] = []
    stream_object = _simulation_object(stream)

    connect = getattr(column, "ConnectFeed", None)
    if callable(connect):
        try:
            connect(stream_object, stage)
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(
                f"{label}: ConnectFeed failed ({type(exc).__name__}); "
                "the feed may not be bound to its designated stage"
            )
            return warnings
    else:
        warnings.append(f"{label}: column has no ConnectFeed; feed not connected")
        return warnings

    # Required even after ConnectFeed -- it is what actually pins the tray.
    setter = getattr(column, "SetStreamFeedStage", None)
    if callable(setter):
        try:
            setter(stream_object, stage)
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(
                f"{label}: SetStreamFeedStage failed ({type(exc).__name__}); "
                "the feed may not be bound to its designated stage"
            )
    else:
        warnings.append(
            f"{label}: column has no SetStreamFeedStage; the feed is connected but "
            "its tray is unset"
        )
    return warnings


def _set_column_specs(
    column: Any,
    *,
    condenser_spec: str,
    condenser_value: float,
    condenser_units: str = "mol/s",
    reboiler_spec: str,
    reboiler_value: float,
    reboiler_units: str = "mol/s",
) -> list[str]:
    """Set the two required column specifications (condenser + reboiler).

    A rigorous distillation column is under-determined without exactly two
    specifications.  DWSIM exposes::

        SetCondenserSpec(spectype: String, value: Double, units: String, compound: String)
        SetReboilerSpec(spectype: String, value: Double, units: String, compound: String)

    Typical ``spectype`` values include the product molar flow rate
    (``"Product_Molar_Flow_Rate"``), ``"Reflux_Ratio"`` and ``"Reboiler_Duty"``.
    The compound argument is only meaningful for component-specific specs and is
    passed as an empty string otherwise.
    """
    warnings: list[str] = []
    for label, method_name, spec_type, value, units in (
        ("condenser", "SetCondenserSpec", condenser_spec, condenser_value, condenser_units),
        ("reboiler", "SetReboilerSpec", reboiler_spec, reboiler_value, reboiler_units),
    ):
        method = getattr(column, method_name, None)
        if not callable(method):
            warnings.append(f"{label} spec: column has no {method_name}; specification not set")
            continue
        try:
            method(spec_type, float(value), units, "")
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(
                f"{label} spec ({spec_type}={value} {units}) could not be set: {type(exc).__name__}"
            )
    return warnings


#: DWSIM's simultaneous-correction (Naphtali-Sandholm) MESH solver name.
#:
#: CAUTION: ``SolvingMethodName`` is a plain string and its setter performs NO
#: validation -- any value is stored silently, and a wrong name only fails at
#: solve time with "Unable to find column solver with name '...'".  Worse,
#: ``CalculateFlowsheet2`` suppresses that failure entirely, so it surfaces only
#: via ``CalculateFlowsheet4`` or by opening the file in the GUI.
#:
#: Verified by solving in this install: the names that RESOLVE are
#:   "Wang-Henke Bubble-Point (BP) Solver"
#:   "Modified Wang-Henke Bubble-Point (MBP) Solver"
#: while "Sum-Rates (SR) Method" and the "Simultaneous Correction ..." strings
#: (which appear in the assembly only as *descriptions*) do NOT resolve.  The
#: default is therefore left on a name known to resolve.
_SC_SOLVER_NAME = "Modified Wang-Henke Bubble-Point (MBP) Solver"

#: Solver names verified to resolve in this DWSIM install.
RESOLVABLE_SOLVER_NAMES = (
    "Wang-Henke Bubble-Point (BP) Solver",
    "Modified Wang-Henke Bubble-Point (MBP) Solver",
)


def _set_column_solver(
    column: Any,
    *,
    solving_method: str,
    max_iterations: int,
    internal_tolerance: float = 1e-4,
    external_tolerance: float = 1e-4,
) -> list[str]:
    """Configure the rigorous-column solver so the exported tower actually solves.

    DWSIM's default cold start is ``Wang-Henke (Bubble Point)`` with 100
    iterations and no initial estimates; on a tall column (tens of stages) that
    routinely fails with "Solver reached the maximum number of iterations
    without converging".  The simultaneous-correction method
    (``Naphtali-Sandholm``) is far less sensitive to the starting point, so it is
    used by default, together with a larger iteration cap and looser loop
    tolerances.
    """
    warnings: list[str] = []
    for attribute, value in (
        ("SolvingMethodName", solving_method),
        ("MaxIterations", int(max_iterations)),
        ("InternalLoopTolerance", float(internal_tolerance)),
        ("ExternalLoopTolerance", float(external_tolerance)),
    ):
        target = getattr(column, attribute, None)
        if target is None:
            warnings.append(f"column solver setting {attribute} is not exposed by this DWSIM build")
            continue
        try:
            if callable(target):
                target(value)
            else:
                setattr(column, attribute, value)
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(f"column solver setting {attribute} could not be set: {type(exc).__name__}")
    return warnings


def _set_column_initial_estimates(
    column: Any,
    *,
    stages: int,
    top_temperature_K: float,
    bottom_temperature_K: float,
    liquid_flow_mol_s: float,
    vapor_flow_mol_s: float,
) -> list[str]:
    """Seed per-stage temperature and flow estimates with a linear profile.

    A rigorous bubble-point solver needs a starting temperature/flow profile; on
    a tall column an empty profile is the usual cause of non-convergence.  The
    temperature estimate is linearly interpolated from the condenser to the
    reboiler, which is close enough for the solver to correct.
    """
    warnings: list[str] = []
    if stages < 1:
        return warnings
    temps = [
        float(top_temperature_K)
        + (float(bottom_temperature_K) - float(top_temperature_K)) * (i / max(stages - 1, 1))
        for i in range(stages)
    ]
    flows = [float(liquid_flow_mol_s)] * stages
    vapors = [float(vapor_flow_mol_s)] * stages

    for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                 "UseVaporFlowEstimates"):
        try:
            setattr(column, flag, True)
        except Exception:  # pragma: no cover
            warnings.append(f"initial-estimate flag {flag} could not be enabled")

    from System import Array, Double  # type: ignore[import-not-found]

    for label, method_name, values in (
        ("temperature", "SetInitialTemperatureEstimates", temps),
        ("liquid-flow", "SetInitialLiquidMolarFlowEstimates", flows),
        ("vapor-flow", "SetInitialVaporMolarFlowEstimates", vapors),
    ):
        method = getattr(column, method_name, None)
        if not callable(method):
            warnings.append(f"initial {label} estimates: column has no {method_name}")
            continue
        try:
            method(Array[Double](values))
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(f"initial {label} estimates could not be set: {type(exc).__name__}")
    return warnings


def _apply_stage_count(column: Any, stages: int, warnings: list[str]) -> None:
    """Set a rigorous column's stage count, correctly resizing its stage list.

    **Use this instead of assigning ``NumberOfStages``.**  DWSIM keeps the stage
    count twice: the ``NumberOfStages`` property and the ``Stages`` list.  Assigning
    the property (``col.NumberOfStages = n``, ``col.set_NumberOfStages(n)`` or
    ``NumberStages``) updates the count but leaves ``Stages`` at its
    construction-time length of 12.  DWSIM then indexes that stale list by the new
    count and aborts with::

        ArgumentOutOfRangeException: 索引超出范围 (index out of range)

    raised from ``Column.GetSolverInputData``.  Only ``SetNumberOfStages(n)``
    rebuilds the list.

    A shared helper exists because this defect is silent and systemic: it made
    *every* column exported by this module unsolvable -- including a textbook
    ethanol/water 10-stage column -- while the error message never mentions the
    stage count, so it reads like a property-package or flowsheet fault.
    """
    setter = getattr(column, "SetNumberOfStages", None)
    if callable(setter):
        try:
            setter(stages)
        except Exception as exc:  # noqa: BLE001 - real DWSIM assemblies
            warnings.append(
                f"SetNumberOfStages failed ({type(exc).__name__}); "
                "the column may be left with a stale stage list"
            )
    else:
        warnings.append(
            "column has no SetNumberOfStages; falling back to the NumberOfStages "
            "property, which does not resize Stages and may break solving"
        )
        try:
            column.NumberOfStages = stages
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"NumberOfStages could not be set: {type(exc).__name__}")

    stage_list = getattr(column, "Stages", None)
    count = getattr(stage_list, "Count", None)
    if count is not None and int(count) != int(stages):
        warnings.append(
            f"stage list holds {int(count)} entries but {stages} stages were "
            "requested; the rigorous solver may fail with an index error"
        )


def _set_column_design(column: Any, design: ExtractiveColumnDesign) -> list[str]:
    """Apply the computed design numbers to the DWSIM column defensively.

    DWSIM column objects expose a variety of property names across versions.  We
    mirror the tolerant probing style of ``_save_flowsheet`` and record any
    parameter that could not be set so callers can act on the warning.
    """
    warnings: list[str] = []
    # The stage count goes through the dedicated helper: the plain property
    # spellings do NOT resize ``Stages`` and must not be attempted here.
    _apply_stage_count(column, design.theoretical_stages, warnings)
    attempts: tuple[tuple[str, float | int], ...] = (
        ("SetRefluxRatio", design.reflux_ratio),
        ("set_RefluxRatio", design.reflux_ratio),
        ("RefluxRatio", design.reflux_ratio),
        ("SetFeedStage", design.feed_stage),
        ("FeedStage", design.feed_stage),
        ("EntrainerStage", design.entrainer_stage),
    )
    for attribute, value in attempts:
        target = getattr(column, attribute, None)
        try:
            if callable(target):
                target(value)
            elif hasattr(column, attribute):
                setattr(column, attribute, value)
        except Exception as exc:  # pragma: no cover - real DWSIM assemblies
            warnings.append(f"column parameter {attribute} could not be set: {type(exc).__name__}")
    return warnings


def _set_binary_column_design(column: Any, stages: int, reflux: float, feed_stage: int) -> list[str]:
    """Apply computed binary-column numbers to a DWSIM column defensively."""
    warnings: list[str] = []
    _apply_stage_count(column, stages, warnings)
    attempts: tuple[tuple[str, float | int], ...] = (
        ("SetRefluxRatio", reflux),
        ("set_RefluxRatio", reflux),
        ("RefluxRatio", reflux),
        ("SetFeedStage", feed_stage),
        ("FeedStage", feed_stage),
    )
    for attribute, value in attempts:
        target = getattr(column, attribute, None)
        try:
            if callable(target):
                target(value)
            elif hasattr(column, attribute):
                setattr(column, attribute, value)
        except Exception as exc:  # noqa: BLE001 - real DWSIM assemblies
            warnings.append(f"column parameter {attribute} could not be set: {type(exc).__name__}")
    return warnings


def export_dwsim_binary_column(
    components: list[str],
    feed_composition: list[float],
    feed_flow_mol_s: float,
    feed_temperature_K: float,
    operating_pressure_kPa: float,
    stages: int,
    minimum_stages: float,
    reflux_ratio: float,
    minimum_reflux_ratio: float,
    feed_stage: int,
    condenser_temperature_K: float,
    reboiler_temperature_K: float,
    destination: Path,
    *,
    distillate_flow_mol_s: float | None = None,
    bottoms_flow_mol_s: float | None = None,
    pressure_drop_kPa: float | None = None,
    solving_method: str = "Naphtali-Sandholm",
    max_iterations: int = 500,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a DWSIM plain binary-distillation column flowsheet (no extractant).

    Builds a rigorous (or shortcut) column with a single feed and two product
    streams (distillate + bottoms); no extractant feed is added.  All stage/reflux
    numbers come from the deterministic binary short-cut design.  Provided as a
    standalone export so it does not depend on the ethanol-water extractive path.

    ``pressure_drop_kPa`` is an explicit request parameter: the column's total
    pressure drop is written as ``ColumnPressureDrop`` (Pa) and the feed is bound
    to its designated tray through ``ConnectFeed`` + ``SetStreamFeedStage``.  When
    ``pressure_drop_kPa`` is ``None`` DWSIM's own default is left in place.

    A rigorous column needs exactly TWO specifications.  The condenser spec is
    always the reflux ratio; the reboiler (bottom) spec is set to the bottoms
    molar flow when ``bottoms_flow_mol_s`` is given (falling back to the
    distillate flow when only that is available).  Without a reboiler spec the
    column is under-determined and will not solve.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    if len(components) != 2 or len(feed_composition) != 2:
        raise ValueError("binary column export requires two components and two feed fractions")
    if feed_stage < 1 or feed_stage >= stages:
        raise ValueError("feed_stage must lie between the top stage (1) and the stage above the reboiler")

    # Reboiler (bottom) specification: prefer the bottoms product flow; fall back
    # to the distillate flow so a valid second specification always exists.
    bottom_spec_value = bottoms_flow_mol_s if bottoms_flow_mol_s is not None else distillate_flow_mol_s
    if bottom_spec_value is not None and bottom_spec_value <= 0:
        raise ValueError("the bottom specification flow must be positive")

    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in components:
            flowsheet.AddCompound(_dwsim_compound_name(name))
        _add_property_package(flowsheet, "UNIQUAC")

        column_type = _column_object_type(object_type)
        column = flowsheet.AddObject(column_type, 250, 0, "Binary Distillation Column")

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, -60, "Feed")
        distillate = flowsheet.AddObject(object_type.MaterialStream, 500, -60, "Distillate")
        bottoms = flowsheet.AddObject(object_type.MaterialStream, 500, 60, "Bottoms")

        feed_stream = _simulation_object(feed)
        feed_stream.SetTemperature(float(feed_temperature_K))
        feed_stream.SetPressure(float(operating_pressure_kPa) * 1000.0)
        feed_stream.SetMolarFlow(float(feed_flow_mol_s))
        feed_stream.SetOverallComposition(_composition_argument([float(feed_composition[0]), float(feed_composition[1])]))

        column_object = _simulation_object(column)
        design_warnings = _set_binary_column_design(
            column_object,
            stages,
            reflux_ratio,
            feed_stage,
        )
        # Request-supplied pressure drop (written as ColumnPressureDrop in Pa).
        design_warnings.extend(_set_column_pressure_drop(column_object, pressure_drop_kPa))

        # Two column specifications are mandatory for a rigorous column: reflux
        # ratio at the condenser and a product flow at the reboiler.
        if bottom_spec_value is not None:
            design_warnings.extend(
                _set_column_specs(
                    column_object,
                    condenser_spec="Reflux_Ratio",
                    condenser_value=float(reflux_ratio),
                    condenser_units="",
                    reboiler_spec="Product_Molar_Flow_Rate",
                    reboiler_value=float(bottom_spec_value),
                    reboiler_units="mol/s",
                )
            )

        # Solver configuration + a starting temperature/flow profile.  Without
        # these a tall column cold-starts on Wang-Henke with no estimates and
        # hits the iteration cap ("Solver reached the maximum number of
        # iterations without converging").
        design_warnings.extend(
            _set_column_solver(
                column_object,
                solving_method=solving_method,
                max_iterations=max_iterations,
            )
        )
        design_warnings.extend(
            _set_column_initial_estimates(
                column_object,
                stages=stages,
                top_temperature_K=float(condenser_temperature_K),
                bottom_temperature_K=float(reboiler_temperature_K),
                liquid_flow_mol_s=float(reflux_ratio) * 0.5 + 0.5,
                vapor_flow_mol_s=float(reflux_ratio) * 0.5 + 1.0,
            )
        )

        # Bind the feed to its tray.  ConnectFeed performs the connection itself,
        # so this stream is deliberately excluded from the graphic connections below.
        design_warnings.extend(_register_feed_on_stage(column_object, feed, feed_stage - 1, "feed"))

        connection_attempts: tuple[tuple[Any, Any, int, int, str], ...] = (
            (column.GraphicObject, distillate.GraphicObject, 0, 0, "column->distillate"),
            (column.GraphicObject, bottoms.GraphicObject, 1, 0, "column->bottoms"),
        )
        for from_obj, to_obj, fidx, tidx, label in connection_attempts:
            try:
                flowsheet.ConnectObjects(from_obj, to_obj, fidx, tidx)
            except Exception as exc:  # noqa: BLE001 - surface, don't abort
                design_warnings.append(f"DWSIM rejected the {label} graphic connection ({type(exc).__name__})")

        _save_flowsheet_via_temp(automation, flowsheet, destination)
        # Surface any parameter the installed DWSIM build refused to accept; the
        # file is still written so the user can inspect and correct it in the GUI.
        for warning in design_warnings:
            print(f"[WARN] binary column: {warning}")
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the binary-distillation flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination


def export_dwsim_extractive_column(
    design: ExtractiveColumnDesign,
    destination: Path,
    *,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a DWSIM extractive-distillation column flowsheet.

    This is a **separate** export path from :func:`export_dwsim_flowsheet`,
    which produces a single TP-flash vessel.  The extractive column recovers
    high-purity ethanol overhead from an ethanol/water feed by injecting a
    high-boiling entrainer near the top; water and the entrainer leave the
    bottom.  All stage counts, reflux ratios, temperatures and pressures come
    from the deterministic :mod:`thermo_engine.column_design` model and are
    applied verbatim to the DWSIM column.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    spec = design.spec
    components = ["ethanol", "water", spec.entrainer]
    property_package = _COLUMN_PROPERTY_PACKAGES[spec.property_package]

    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in components:
            flowsheet.AddCompound(_dwsim_compound_name(name))
        _add_property_package(flowsheet, property_package)

        column_type = _column_object_type(object_type)
        column = flowsheet.AddObject(column_type, 250, 0, "Extractive Distillation Column")

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, -80, "Ethanol Water Feed")
        entrainer = flowsheet.AddObject(object_type.MaterialStream, 0, 80, "Entrainer")
        distillate = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Ethanol Product")
        bottoms = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Water Entrainer Bottoms")

        # Feed: ethanol/water binary over the feed components.
        feed_stream = _simulation_object(feed)
        feed_stream.SetTemperature(spec.feed_temperature_K)
        feed_stream.SetPressure(spec.feed_pressure_kPa * 1000.0)
        feed_stream.SetMolarFlow(spec.feed_flow_mol_s)
        feed_comp = list(spec.feed_composition) + [0.0]
        feed_stream.SetOverallComposition(_composition_argument(feed_comp))

        # Entrainer: pure high-boiling component, molar flow = ratio * feed.
        entrainer_stream = _simulation_object(entrainer)
        entrainer_stream.SetTemperature(design.condenser_temperature_K)
        entrainer_stream.SetPressure(spec.operating_pressure_kPa * 1000.0)
        entrainer_stream.SetMolarFlow(spec.entrainer_ratio * spec.feed_flow_mol_s)
        entrainer_stream.SetOverallComposition(_composition_argument([0.0, 0.0, 1.0]))

        # Apply the computed stage/reflux numbers; surface any DWSIM gaps.
        design_warnings = _set_column_design(_simulation_object(column), design)
        if design_warnings:
            design = design.model_copy(update={"warnings": list(dict.fromkeys(design.warnings + design_warnings))})

        # Connect feed / entrainer into the column and take the two products.
        # DWSIM's graphic ConnectObject is strict about which (from-port, to-port)
        # pairs are legal for a given object pair.  A DistillationColumn connects
        # the second feed (entrainer) on a distinct inlet port (port 1), whereas a
        # ShortcutColumn exposes a single feed inlet only.  Attempt each connection
        # and, on failure, record a warning instead of aborting the whole export so
        # a usable flowsheet is still produced (mirrors ``_set_column_design``).
        entrainer_port = 1 if column_type == object_type.DistillationColumn else 0
        connection_attempts: tuple[tuple[Any, Any, int, int, str], ...] = (
            (feed.GraphicObject, column.GraphicObject, 0, 0, "feed->column"),
            (entrainer.GraphicObject, column.GraphicObject, 0, entrainer_port, "entrainer->column"),
            (column.GraphicObject, distillate.GraphicObject, 0, 0, "column->distillate"),
            (column.GraphicObject, bottoms.GraphicObject, 1, 0, "column->bottoms"),
        )
        connection_warnings: list[str] = []
        for from_obj, to_obj, fidx, tidx, label in connection_attempts:
            try:
                flowsheet.ConnectObjects(from_obj, to_obj, fidx, tidx)
            except Exception as exc:  # noqa: BLE001 - surface, don't abort
                connection_warnings.append(
                    f"DWSIM rejected the {label} graphic connection ({type(exc).__name__})"
                )
        if connection_warnings:
            design = design.model_copy(
                update={"warnings": list(dict.fromkeys(design.warnings + connection_warnings))}
            )

        _save_flowsheet_via_temp(automation, flowsheet, destination)
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the extractive-distillation flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination


def export_generic_extractive_column(
    *,
    light: str,
    heavy: str,
    entrainer: str,
    feed_composition: list[float],
    feed_flow_mol_s: float,
    feed_temperature_K: float,
    feed_pressure_kPa: float,
    stages: int,
    reflux_ratio: float,
    feed_stage: int,
    entrainer_stage: int,
    entrainer_ratio: float,
    condenser_temperature_K: float,
    reboiler_temperature_K: float,
    property_package: str = "NRTL",
    destination: Path,
    pressure_drop_kPa: float | None = None,
    distillate_purity_mole_fraction: float | None = None,
    recovery: float = 1.0,
    distillate_flow_mol_s: float | None = None,
    bottoms_flow_mol_s: float | None = None,
    factory: AutomationFactory | None = None,
    object_type: Any | None = None,
) -> Path:
    """Create a generic ternary extractive-distillation column flowsheet.

    This is the **non-ethanol** counterpart of :func:`export_dwsim_extractive_column`.
    The feed is an arbitrary ``light``/``heavy`` key pair and ``entrainer`` is a
    third high-boiling solvent (e.g. ethyl acetate / n-propyl acetate separated by
    dimethyl sulfoxide). All stage/reflux/temperature numbers come from the
    deterministic ternary short-cut design and are passed explicitly, so no
    ethanol/water hard-coding is required. The rigorous ``DistillationColumn`` is
    fed at two inlets (feed + entrainer) and has distillate/bottoms on two outlets.

    ``pressure_drop_kPa`` is an explicit request parameter written as the column's
    ``ColumnPressureDrop`` (Pa); when ``None`` DWSIM's own default is kept.  Both
    feeds are bound to their own trays: the hydrocarbon feed at ``feed_stage`` and
    the entrainer above it at ``entrainer_stage`` (stage indices are converted to
    DWSIM's 0-based numbering, where 0 is the top stage).

    Product specification and the recovery contract
    -----------------------------------------------
    A rigorous column needs two consistent specifications.  This function writes
    the condenser on ``Stream_Ratio`` (= ``reflux_ratio``) and the reboiler on
    ``Product_Molar_Flow_Rate`` (= bottoms flow).  The two are coupled by
    ``D + B = total_in``, so the bottoms flow must be derived with the **same**
    recovery the design used.

    ``distillate_flow_mol_s`` and ``bottoms_flow_mol_s`` are the authoritative
    values: when supplied (as the design functions produce them) they are used
    verbatim, keeping a single source of truth.  Otherwise they are derived from
    the light-key feed and ``recovery``::

        d_light = recovery * feed_composition[0] * feed_flow_mol_s
        D       = d_light / purity
        B       = total_in - D

    Passing only ``distillate_purity_mole_fraction`` previously assumed complete
    light-key recovery, so a design at 90% recovery was written with an
    over-large distillate (and a correspondingly short bottoms).  DWSIM then
    aborted with e.g.::

        Failed to fulfill mass balance for Water: Relative Error = 0.87

    because the distillate demanded more water than the feed contained.  The
    ``recovery`` parameter closes that gap; callers holding a full design should
    pass its D and B directly.
    """
    destination = destination.resolve()
    if destination.suffix.casefold() != ".dwxmz":
        raise ValueError("DWSIM exports must use the .dwxmz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if factory is None or object_type is None:
        factory, object_type = _automation_factory()

    components = [light, heavy, entrainer]
    feed_comp3 = [float(feed_composition[0]), float(feed_composition[1]), 0.0]

    if feed_stage < 1 or feed_stage >= stages:
        raise ValueError("feed_stage must lie between the top stage (1) and the stage above the reboiler")
    if entrainer_stage < 1 or entrainer_stage >= feed_stage:
        raise ValueError("entrainer_stage must be above feed_stage and at least 1")

    try:
        automation = factory()
        flowsheet = automation.CreateFlowsheet()
        for name in components:
            flowsheet.AddCompound(_dwsim_compound_name(name))
        _add_property_package(flowsheet, property_package)

        column_type = _column_object_type(object_type)
        column = flowsheet.AddObject(column_type, 250, 0, "Extractive Distillation Column")

        feed = flowsheet.AddObject(object_type.MaterialStream, 0, -80, "Feed")
        entrainer_stream = flowsheet.AddObject(object_type.MaterialStream, 0, 80, "Entrainer")
        distillate = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Distillate")
        bottoms = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bottoms")

        feed_obj = _simulation_object(feed)
        feed_obj.SetTemperature(float(feed_temperature_K))
        feed_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
        feed_obj.SetMolarFlow(float(feed_flow_mol_s))
        feed_obj.SetOverallComposition(_composition_argument(feed_comp3))

        ent_obj = _simulation_object(entrainer_stream)
        ent_obj.SetTemperature(float(condenser_temperature_K))
        ent_obj.SetPressure(float(feed_pressure_kPa) * 1000.0)
        ent_obj.SetMolarFlow(float(entrainer_ratio * feed_flow_mol_s))
        ent_obj.SetOverallComposition(_composition_argument([0.0, 0.0, 1.0]))

        # Apply design numbers defensively across DWSIM column attribute spellings.
        warnings: list[str] = []
        col = _simulation_object(column)

        # Stage count via the dedicated helper: only SetNumberOfStages resizes the
        # ``Stages`` list.  Setting the NumberOfStages property leaves the list at
        # its construction-time length of 12 and makes the column unsolvable with
        # an index-out-of-range error.  See _apply_stage_count.
        _apply_stage_count(col, stages, warnings)

        for attribute, value in (
            ("SetRefluxRatio", reflux_ratio),
            ("set_RefluxRatio", reflux_ratio),
            ("RefluxRatio", reflux_ratio),
            ("SetFeedStage", feed_stage),
            ("FeedStage", feed_stage),
            ("SetEntrainerStage", entrainer_stage),
            ("EntrainerStage", entrainer_stage),
        ):
            target = getattr(col, attribute, None)
            try:
                if callable(target):
                    target(value)
                elif hasattr(col, attribute):
                    setattr(col, attribute, value)
            except Exception as exc:  # noqa: BLE001 - real DWSIM assemblies
                warnings.append(f"column parameter {attribute} could not be set: {type(exc).__name__}")

        # Request-supplied pressure drop (written as ColumnPressureDrop in Pa).
        warnings.extend(_set_column_pressure_drop(col, pressure_drop_kPa))

        # Attach the two column specifications.  A rigorous DistillationColumn owns
        # two ColumnSpec slots (keys "C" and "R"): the condenser spec defaults to a
        # reflux Stream_Ratio (which ``RefluxRatio`` already feeds) while the reboiler
        # spec defaults to Product_Molar_Flow_Rate with a value of 0.  An unset/zero
        # product-flow spec removes a degree of freedom the solver needs, and DWSIM
        # then fails with
        #   "Failed to fulfill mass balance for <component>: Relative Error = ~1.0"
        # because it cannot reconcile the component flows.  Writing the bottoms
        # product flow closes the balance; the distillate then follows from the
        # reflux-ratio spec.
        total_in = float(feed_flow_mol_s) * (1.0 + float(entrainer_ratio))
        if distillate_flow_mol_s is None or bottoms_flow_mol_s is None:
            # Derive from the recovery, not from complete light-key recovery.
            d_light = (
                float(recovery)
                * float(feed_composition[0])
                * float(feed_flow_mol_s)
            )
            purity = float(distillate_purity_mole_fraction or 1.0)
            d_derived = d_light / purity if 0.0 < purity <= 1.0 else d_light
            d_total = (
                float(distillate_flow_mol_s)
                if distillate_flow_mol_s is not None
                else d_derived
            )
            b_total = (
                float(bottoms_flow_mol_s)
                if bottoms_flow_mol_s is not None
                else total_in - d_total
            )
        else:
            d_total = float(distillate_flow_mol_s)
            b_total = float(bottoms_flow_mol_s)

        # Both specs must close the overall balance; a mismatch means the caller
        # passed design values from a different feed basis, which DWSIM will
        # otherwise report as an obscure iterative mass-balance failure.
        if abs((d_total + b_total) - total_in) > 1e-6:
            raise ValueError(
                "distillate + bottoms flow must equal feed + entrainer "
                f"({d_total:.6f} + {b_total:.6f} != {total_in:.6f} mol/s)"
            )

        warnings.extend(
            _set_column_product_specs(
                col,
                distillate_flow_mol_s=d_total,
                bottoms_flow_mol_s=b_total,
            )
        )

        # Bind both feeds to their trays.  ConnectFeed performs each connection
        # itself, so these two streams are excluded from the graphic connections.
        warnings.extend(_register_feed_on_stage(col, feed, feed_stage - 1, "feed"))
        warnings.extend(_register_feed_on_stage(col, entrainer_stream, entrainer_stage - 1, "entrainer"))

        # Take the two products through the column's own product connectors.
        # ``ConnectDistillate`` / ``ConnectBottoms`` wire the outlet streams to the
        # condenser and reboiler draw-offs directly.  The generic graphic
        # ``ConnectObjects(column, stream, 0|1, 0)`` calls used previously returned
        # without raising yet left the products unwired, so a solve reported success
        # while both product streams stayed at zero flow and default composition.
        for arg, method_name, label in (
            (distillate, "ConnectDistillate", "column->distillate"),
            (bottoms, "ConnectBottoms", "column->bottoms"),
        ):
            method = getattr(col, method_name, None)
            if not callable(method):
                warnings.append(f"DWSIM column has no {method_name}; {label} left unconnected")
                continue
            try:
                method(_simulation_object(arg))
            except Exception as exc:  # noqa: BLE001 - surface, don't abort
                warnings.append(f"DWSIM rejected the {label} connection ({type(exc).__name__})")

        # Configure the solver and seed a starting profile.  Without this the
        # column runs on DWSIM's default (Wang-Henke, 100 iterations, no
        # estimates) and fails with "Solver reached the maximum number of
        # iterations without converging"; the product streams then stay at zero
        # flow and default composition.  ``CalculateFlowsheet2`` does not report
        # that failure, so only ``CalculateFlowsheet4`` (or opening the file in
        # the GUI) reveals it.
        warnings.extend(
            _set_column_solver(
                col,
                solving_method=_SC_SOLVER_NAME,
                max_iterations=1000,
            )
        )
        warnings.extend(
            _set_column_initial_estimates(
                col,
                stages=stages,
                top_temperature_K=float(condenser_temperature_K),
                bottom_temperature_K=float(reboiler_temperature_K),
                liquid_flow_mol_s=float(reflux_ratio) * float(feed_flow_mol_s) + float(feed_flow_mol_s),
                vapor_flow_mol_s=float(reflux_ratio) * float(feed_flow_mol_s)
                + float(feed_flow_mol_s)
                + float(entrainer_ratio) * float(feed_flow_mol_s) / 2.0,
            )
        )

        _save_flowsheet_via_temp(automation, flowsheet, destination)
        # Surface any parameter the installed DWSIM build refused to accept; the
        # file is still written so the user can inspect and correct it in the GUI.
        for warning in warnings:
            print(f"[WARN] extractive column: {warning}")
    except ThermoEquiError:
        raise
    except Exception as exc:  # pragma: no cover - depends on installed DWSIM assemblies
        raise _runtime_error(
            "DWSIM could not create the generic extractive-distillation flowsheet.",
            details={"exception": type(exc).__name__},
        ) from exc
    if not destination.is_file():
        raise _runtime_error("DWSIM did not create the requested flowsheet file.")
    return destination
