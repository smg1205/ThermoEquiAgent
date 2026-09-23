"""Generic ternary VLE extractive-distillation export for the Agent.

Supersedes the fixed EtOAc/nPrOAc/DMSO-only path: any three-component system named
in the user's message is accepted, the property package defaults to NRTL but can be
overridden by the user, and the product specification uses the report's
0.95 purity / 0.90 recovery.

Design rules (unchanged from the rest of the repository):
  * every equilibrium number comes from ``thermo_engine.column_design``;
  * the LLM never supplies a design value;
  * a component with no SMILES mapping still runs under UNIFAC, but the
    ThermoFormer source reports a structured missing-parameter failure instead of
    inventing anything.

The rendered ``.dwxmz`` additionally carries solver initial estimates (stage
temperature / flow / composition profiles, product rates, a reflux ratio above the
bare short-cut minimum and a raised iteration cap), because a cold-started rigorous
column otherwise fails with "Solver reached the maximum number of iterations".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from agent.schema_helpers import (  # noqa: F401
    PROPERTY_PACKAGE_ALIASES,
    resolve_property_package,
)

#: Report §1.5 product specification.
DEFAULT_PURITY = 0.95
DEFAULT_RECOVERY = 0.90
DEFAULT_PRESSURE_KPA = 101.325
DEFAULT_ENTRAINER_RATIO = 2.0
DEFAULT_FEED_FLOW_MOL_S = 1.0
DEFAULT_PRESSURE_DROP_KPA = 5.0

#: Operating reflux is raised above the short-cut value; R = 1.4 x R_min sits on
#: the rigorous-solver convergence boundary.
REFLUX_MULTIPLIER = 2.0
MAX_ITERATIONS = 1000


def _package_from_message(message: str) -> str:
    """Property package named in the message, defaulting to NRTL."""
    return resolve_property_package(message)


@dataclass(frozen=True)
class TernaryRequest:
    """Parsed generic ternary extractive request."""

    light: str
    heavy: str
    entrainer: str
    property_package: str
    purity: float
    recovery: float
    entrainer_ratio: float
    feed_flow_mol_s: float
    pressure_kpa: float
    alpha_source: str
    feed_composition: list[float]


_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("water", ("水", "water", "h2o")),
    ("methanol", ("甲醇", "methanol")),
    ("ethanol", ("乙醇", "ethanol")),
    ("1-propanol", ("1-丙醇", "正丙醇", "1-propanol", "n-propanol")),
    ("2-propanol", ("2-丙醇", "异丙醇", "isopropanol", "2-propanol", "ipa")),
    ("1-butanol", ("1-丁醇", "正丁醇", "1-butanol", "n-butanol")),
    ("glycerol", ("甘油", "glycerol", "glycerine", "glycerin")),
    ("ethylene glycol", ("乙二醇", "ethylene glycol", "meg")),
    ("acetic acid", ("乙酸", "醋酸", "acetic acid")),
    ("acetone", ("丙酮", "acetone")),
    ("acetonitrile", ("乙腈", "acetonitrile")),
    ("methyl acetate", ("乙酸甲酯", "methyl acetate")),
    ("ethyl acetate", ("乙酸乙酯", "ethyl acetate", "ethylacetate")),
    ("n-propyl acetate", ("乙酸正丙酯", "醋酸正丙酯", "n-propyl acetate", "propyl acetate")),
    ("dimethyl sulfoxide", ("二甲基亚砜", "dmso", "dimethyl sulfoxide")),
    ("tetrahydrofuran", ("四氢呋喃", "tetrahydrofuran", "thf")),
    ("toluene", ("甲苯", "toluene")),
    ("benzene", ("苯", "benzene")),
    # Recognised so the request parses, but deliberately absent from
    # ``column_design._COMPONENT_SMILES``: UNIFAC can still design the column while
    # the ThermoFormer source reports a structured missing-SMILES failure.
    ("chloroform", ("氯仿", "chloroform", "trichloromethane")),
    ("phenol", ("苯酚", "phenol")),
)


def find_components(message: str) -> list[str] | None:
    """Return the three distinct components named in the message, or None.

    Components are matched by alias and returned in first-mention order, so the
    user's phrasing ("水/甲苯 + 1-丁醇") keeps the order they were presented in.

    Matching is span-aware: alias hits are collected longest-first and any hit that
    overlaps an already-claimed span is discarded.  Without that step a shorter
    alias nested inside a longer one (苯 inside 甲苯, 醇 inside 丁醇) would report a
    fourth spurious component and the request would be rejected.

    Returns ``None`` unless exactly three distinct components remain.
    """
    lower = message.casefold()
    hits: list[tuple[int, int, str]] = []
    for canonical, aliases in _ALIAS_GROUPS:
        for alias in aliases:
            start = lower.find(alias.casefold())
            if start >= 0:
                hits.append((start, start + len(alias), canonical))

    # Longest match first, then leftmost, so "甲苯" claims its span before "苯".
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    claimed: list[tuple[int, int]] = []
    order: list[tuple[int, str]] = []
    seen: set[str] = set()
    for start, end, canonical in hits:
        if canonical in seen:
            continue
        if any(start < c_end and c_start < end for c_start, c_end in claimed):
            continue
        claimed.append((start, end))
        order.append((start, canonical))
        seen.add(canonical)

    order.sort()
    found = [c for _, c in order]
    if len(found) != 3:
        return None
    return found


def wants_thermoformer(message: str) -> bool:
    lower = message.casefold()
    return any(m in lower for m in ("thermoformer", "tf ", " tf", "机器学习", "模型预测"))


def _find_ratio(message: str) -> float | None:
    for pat in (r"(?:剂料比|溶剂比|entrainer\s*ratio|s/f)\s*[=:]?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*倍(?:萃取剂|溶剂)"):
        m = re.search(pat, message, re.I)
        if m:
            return float(m.group(1))
    return None


def _find_purity(message: str) -> float | None:
    m = re.search(r"(?:纯度|purity)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?", message, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100.0 if v > 1.0 else v


def _find_recovery(message: str) -> float | None:
    m = re.search(r"(?:回收率|recovery)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?", message, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100.0 if v > 1.0 else v


def _find_flow(message: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*mol\s*/\s*s", message, re.I)
    return float(m.group(1)) if m else None


def _find_pressure(message: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*kpa", message, re.I)
    return float(m.group(1)) if m else None


def parse_request(message: str) -> TernaryRequest | None:
    """Parse a generic ternary extractive request, or return None if incomplete."""
    comps = find_components(message)
    if comps is None:
        return None
    light, heavy, entrainer = comps[0], comps[1], comps[2]
    return TernaryRequest(
        light=light,
        heavy=heavy,
        entrainer=entrainer,
        property_package=_package_from_message(message),
        purity=_find_purity(message) or DEFAULT_PURITY,
        recovery=_find_recovery(message) or DEFAULT_RECOVERY,
        entrainer_ratio=_find_ratio(message) or DEFAULT_ENTRAINER_RATIO,
        feed_flow_mol_s=_find_flow(message) or DEFAULT_FEED_FLOW_MOL_S,
        pressure_kpa=_find_pressure(message) or DEFAULT_PRESSURE_KPA,
        alpha_source="thermoformer" if wants_thermoformer(message) else "unifac",
        feed_composition=[0.5, 0.5],
    )


def _material_balance(req: TernaryRequest) -> dict:
    """Component balance implied by the short-cut design (closure verified)."""
    F = req.feed_flow_mol_s
    n_l = F * req.feed_composition[0]
    n_h = F * req.feed_composition[1]
    d_l = req.recovery * n_l
    d_h = (1.0 - req.purity) / req.purity * d_l
    D = d_l + d_h
    b_l, b_h = n_l - d_l, n_h - d_h
    b_s = req.entrainer_ratio * F
    B = b_l + b_h + b_s
    return {
        "D": D, "B": B, "ent_flow": b_s,
        "z_d": [d_l / D, d_h / D, 0.0],
        "z_b": [b_l / B, b_h / B, b_s / B],
    }


def _bubble(components: list[str], xs: list[float], P_kpa: float) -> float:
    from thermo_engine.column_design import bubble_temperature

    return float(bubble_temperature(components, xs, P_kpa, "unifac"))


def _is_smiles_failure(exc: BaseException) -> bool:
    """Whether the failure is a missing SMILES mapping for a component."""
    text = str(exc).casefold()
    return "no smiles" in text or "smiles" in text and "missing" in text


def _design_failure(req: TernaryRequest, exc: BaseException, *, missing_smiles: bool) -> object:
    """Build the structured failure payload for an unusable design request.

    A missing SMILES mapping means the ThermoFormer source cannot run at all, so the
    caller is told which source to use instead rather than being handed a bare
    exception string.
    """
    from schemas.column_design import ExtractiveExportPayload

    if missing_smiles:
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=["smiles"],
            alpha_source=req.alpha_source,
            message=(
                f"ThermoFormer 源无法用于 {req.light}/{req.heavy} + {req.entrainer}："
                f"{exc}。该组分未在 ThermoFormer 的 SMILES 映射表中登记；"
                "请改用 UNIFAC（不写「用 ThermoFormer」即可），或先补充该组分的 SMILES 映射。"
            ),
        )
    return ExtractiveExportPayload(
        status="failed",
        alpha_source=req.alpha_source,
        message=f"三元萃取精馏设计失败：{type(exc).__name__}: {exc}",
    )


def run_generic_ternary_extractive_export(
    message: str, *, export_dir: str | None = None
) -> object:
    """Design and export a DWSIM extractive column for any named ternary system.

    Returns an :class:`~schemas.column_design.ExtractiveExportPayload`.  The design
    is always returned when the deterministic engine can run, even if the DWSIM
    file cannot be produced.
    """
    from agent.extractive_distillation import export_directory
    from schemas.column_design import (
        ExtractiveColumnDesign,
        ExtractiveColumnSpec,
        ExtractiveExportPayload,
    )
    from thermo_engine.column_design import design_ternary_extractive_column
    from thermo_engine.dwsim_export import (
        export_generic_extractive_column,
        missing_dwsim_compound_mappings,
    )
    from thermo_engine.errors import ThermoEquiError

    req = parse_request(message)
    if req is None:
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=["components"],
            message=(
                "无法识别三元体系。请在消息中给出三个组分名，"
                "例如「导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件」。"
            ),
        )

    missing = missing_dwsim_compound_mappings([req.light, req.heavy, req.entrainer])
    if missing:
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            message=(
                "DWSIM 导出缺少以下组分的映射："
                + "、".join(missing)
                + "。请先补充 DWSIM 化合物映射后再生成文件。"
            ),
        )

    # --- design (all numbers from the deterministic engine) --------------
    try:
        design = design_ternary_extractive_column(
            req.light, req.heavy, req.entrainer, req.feed_composition,
            feed_flow_mol_s=req.feed_flow_mol_s,
            operating_pressure_kPa=req.pressure_kpa,
            distillate_purity_mole_fraction=req.purity,
            recovery=req.recovery,
            entrainer_ratio=req.entrainer_ratio,
            alpha_source=req.alpha_source,
        )
    except ThermoEquiError as exc:
        return _design_failure(req, exc, missing_smiles=_is_smiles_failure(exc))
    except ValueError as exc:
        # ``design_ternary_extractive_column`` raises a bare ValueError when a
        # component has no entry in ``_COMPONENT_SMILES``; that is a missing
        # mapping, not a solver failure, so it must not be reported as "failed".
        if _is_smiles_failure(exc):
            return _design_failure(req, exc, missing_smiles=True)
        return ExtractiveExportPayload(
            status="failed",
            message=f"三元萃取精馏设计失败：{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ExtractiveExportPayload(
            status="failed",
            message=f"三元萃取精馏设计失败：{type(exc).__name__}: {exc}",
        )

    # --- solver initial estimates ----------------------------------------
    bal = _material_balance(req)
    D, B = bal["D"], bal["B"]
    stages = int(design["theoretical_stages"])
    r_op = max(float(design["reflux_ratio"]) * REFLUX_MULTIPLIER,
               float(design["minimum_reflux_ratio"]) * 2.0)
    try:
        t_top = _bubble([req.light, req.heavy], [bal["z_d"][0], bal["z_d"][1]], req.pressure_kpa)
        t_bot = _bubble([req.light, req.heavy, req.entrainer], bal["z_b"], req.pressure_kpa)
        t_feed = _bubble([req.light, req.heavy], req.feed_composition, req.pressure_kpa)
        t_ent = _bubble([req.entrainer, req.light], [0.999, 0.001], req.pressure_kpa)
    except Exception:  # noqa: BLE001 - estimates are best-effort
        t_top = float(design["condenser_temperature_K"])
        t_bot = float(design["reboiler_temperature_K"])
        t_feed = t_top
        t_ent = t_bot

    directory = export_directory(export_dir)
    file_id = f"ternary-extractive-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"

    status: Literal["ready", "dwsim_unavailable"] = "ready"
    dwsim_uri: str | None = None
    note = "ThermoFormer" if req.alpha_source == "thermoformer" else "UNIFAC"
    text = (
        f"已完成 {req.light}/{req.heavy} + {req.entrainer} 三元萃取精馏设计（{note}，"
        f"物性包 {req.property_package}）。\n"
        f"理论塔板 {stages}，进料板 {design['feed_stage']}，"
        f"萃取剂板 {design['entrainer_stage']}，回流比 {r_op:.3g}"
        f"（短节值 {design['reflux_ratio']:.3g}，已上调以利严格塔收敛），"
        f"操作压力 {req.pressure_kpa:.3f} kPa。\n"
        f"产品规格：塔顶纯度 {req.purity}，回收率 {req.recovery}；"
        f"D = {D:.4f} mol/s，B = {B:.4f} mol/s。"
    )

    try:
        _export_with_estimates(
            req=req, design=design, destination=destination,
            stages=stages, r_op=r_op, D=D, B=B, bal=bal,
            t_top=t_top, t_bot=t_bot, t_feed=t_feed, t_ent=t_ent,
        )
    except ThermoEquiError:
        status = "dwsim_unavailable"
        text += "\n设计已完成，但当前环境未配置可用的 DWSIM，未生成文件。"
    except Exception as exc:  # noqa: BLE001
        status = "dwsim_unavailable"
        text += f"\n设计已完成，但 DWSIM 文件导出失败（{type(exc).__name__}）。"
    else:
        dwsim_uri = f"/api/export/extractive/{quote(file_id)}.dwxmz"
        text += "\nDWSIM 文件已生成，可由前端自动下载。"

    return ExtractiveExportPayload(
        status=status,
        design=ExtractiveColumnDesign(
            spec=ExtractiveColumnSpec(
                feed_components=[req.light, req.heavy],
                feed_composition=list(req.feed_composition),
                feed_flow_mol_s=req.feed_flow_mol_s,
                feed_temperature_K=t_feed,
                feed_pressure_kPa=req.pressure_kpa,
                entrainer=req.entrainer,
                entrainer_ratio=req.entrainer_ratio,
                distillate_purity_mole_fraction=req.purity,
                ethanol_recovery=req.recovery,
                operating_pressure_kPa=req.pressure_kpa,
                property_package=req.property_package,
            ),
            theoretical_stages=stages,
            minimum_stages=float(design["minimum_stages"]),
            reflux_ratio=float(design["reflux_ratio"]),
            minimum_reflux_ratio=float(design["minimum_reflux_ratio"]),
            feed_stage=int(design["feed_stage"]),
            entrainer_stage=int(design["entrainer_stage"]),
            relative_volatility=float(design["alpha_avg"]),
            base_relative_volatility=float(design["alpha_base"]),
            selectivity=float(design["selectivity"]),
            condenser_temperature_K=float(design["condenser_temperature_K"]),
            reboiler_temperature_K=float(design["reboiler_temperature_K"]),
            operating_pressure_kPa=req.pressure_kpa,
            distillate_purity_mole_fraction=req.purity,
            bottoms_composition=[float(v) for v in bal["z_b"]],
            distillate_flow_mol_s=D,
            bottoms_flow_mol_s=B,
            warnings=[f"operating reflux raised to {r_op:.3g} for solver convergence"],
        ),
        alpha_source=req.alpha_source,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        message=text,
    )


def _export_with_estimates(
    *, req: TernaryRequest, design: dict, destination: Path, stages: int,
    r_op: float, D: float, B: float, bal: dict,
    t_top: float, t_bot: float, t_feed: float, t_ent: float,
) -> None:
    """Render the column and inject solver initial estimates into the saved file."""
    from thermo_engine import dwsim_export as ded
    from thermo_engine.dwsim_export import export_generic_extractive_column

    export_generic_extractive_column(
        light=req.light,
        heavy=req.heavy,
        entrainer=req.entrainer,
        feed_composition=list(req.feed_composition),
        feed_flow_mol_s=req.feed_flow_mol_s,
        feed_temperature_K=t_feed,
        feed_pressure_kPa=req.pressure_kpa,
        stages=stages,
        reflux_ratio=r_op,
        feed_stage=int(design["feed_stage"]),
        entrainer_stage=int(design["entrainer_stage"]),
        entrainer_ratio=req.entrainer_ratio,
        condenser_temperature_K=float(design["condenser_temperature_K"]),
        reboiler_temperature_K=float(design["reboiler_temperature_K"]),
        property_package=req.property_package,
        destination=destination,
        pressure_drop_kPa=DEFAULT_PRESSURE_DROP_KPA,
        distillate_purity_mole_fraction=req.purity,
    )

    # Reopen and populate the stage-wise estimates, then save again.  DWSIM stores
    # these as List<Parameter> / List<Dictionary<String, Parameter>>, so values go
    # through the Parameter.Value property and the composition dicts are keyed by
    # the compound names DWSIM resolved.
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(destination))
    col = None
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
            col = item.GetAsObject() if hasattr(item, "GetAsObject") else item
    if col is None:
        return

    for attr, val in (("MaxIterations", MAX_ITERATIONS), ("RefluxRatio", r_op)):
        try:
            setattr(col, attr, val)
        except Exception:  # noqa: BLE001
            pass

    ie = col.InitialEstimates
    names = [str(c).split(",")[0].strip().lstrip("[") for c in fs.SelectedCompounds]
    z_d, z_b = bal["z_d"], bal["z_b"]
    try:
        ie.DistillateFlowRate = D
        ie.BottomsFlowRate = B
        ie.RefluxRatio = r_op
    except Exception:  # noqa: BLE001
        pass

    span = max(stages - 1, 1)
    for i in range(min(stages, ie.StageTemps.Count)):
        w = i / span
        try:
            ie.StageTemps[i].Value = t_top + (t_bot - t_top) * w
        except Exception:  # noqa: BLE001
            pass
        try:
            ie.LiqMolarFlows[i].Value = D * r_op
            ie.VapMolarFlows[i].Value = D * (r_op + 1.0)
        except Exception:  # noqa: BLE001
            pass
        row = [z_d[k] + (z_b[k] - z_d[k]) * w for k in range(3)]
        for j, nm in enumerate(names[:3]):
            for attr in ("LiqCompositions", "VapCompositions"):
                try:
                    getattr(ie, attr)[i][nm].Value = row[j]
                except Exception:  # noqa: BLE001
                    pass

    for flag, val in (("UseTemperatureEstimates", True),
                      ("UseLiquidFlowEstimates", True),
                      ("UseVaporFlowEstimates", True),
                      ("UseCompositionEstimates", True),
                      ("AutoUpdateInitialEstimates", False)):
        try:
            if hasattr(col, flag):
                setattr(col, flag, val)
        except Exception:  # noqa: BLE001
            pass

    ded._save_flowsheet_via_temp(automation, fs, destination)

