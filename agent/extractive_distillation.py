"""Extractive-distillation intent detection, parameter extraction and execution.

This module lets the chat flow recognise a request to *simulate* an
extractive-distillation column (ethanol/water -> high-purity ethanol by adding a
high-boiling entrainer near the top), pull the needed numbers out of the user's
free-text query, run the deterministic short-cut design in
:mod:`thermo_engine.column_design`, and try to render a DWSIM ``.dwxmz`` file.

It deliberately mirrors the repository rule that numeric design values must come
from the deterministic engine (``design_extractive_distillation_column``), never
from the LLM.  If DWSIM is not installed/configured, the deterministic design is
still returned under ``status=dwsim_unavailable`` so the user gets the stages,
reflux ratio and temperatures even though the file cannot yet be produced.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from schemas.column_design import (
    BinaryCaseSpec,
    BinaryDistillationPayload,
    BinaryDistillationResult,
    EntrainerOption,
    ExtractiveColumnSpec,
    ExtractiveExportPayload,
    LLEExportPayload,
    LLEExtractionSpec,
    TernaryLLESpec,
)
from schemas.domain import EvidenceStatement
from thermo_engine.column_design import (
    bubble_temperature,
    design_binary_distillation_column,
    design_extractive_distillation_column,
    design_generic_extractive_column,
    recommend_entrainer_for,
    recommend_extraction_entrainer,
)
from thermo_engine.dwsim_export import (
    export_dwsim_binary_column,
    export_dwsim_binary_lle_flowsheet,
    export_dwsim_extractive_column,
    export_dwsim_lle_extraction,
    export_dwsim_ternary_lle_flowsheet,
    export_generic_extractive_column,
    missing_dwsim_compound_mappings,
)
from thermo_engine.errors import ThermoEquiError

#: Relative (REST) prefix on which the generated files are served back.
DOWNLOAD_PREFIX = "/api/export/extractive"

#: Subdirectory under the workspace used as the default export store.
_DEFAULT_EXPORT_DIR = "data/exports/extractive"

#: Chinese/English aliases for the recognised entrainers.
_ENTRAINER_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("乙二醇", "ethylene glycol", "ethyleneglycol", "eg", "1,2-ethanediol"), "ethylene glycol"),
    (("甘油", "glycerol", "glycerine", "glycerin", "propane-1,2,3-triol", "1,2,3-propanetriol"), "glycerol"),
)

#: Feed-component aliases used when classifying a message as extractive and when
#: parsing the feed composition.
_ETHANOL_ALIASES = ("乙醇", "alcohol", "ethanol")
_WATER_ALIASES = ("水", "water", "h2o")

#: Keywords that mark a message as an extractive-distillation simulation request.
_EXTRACTIVE_REQUEST_MARKERS = (
    "萃取精馏",
    "萃取模拟",
    "萃取",
    "extractive distillation",
    "extractive-distillation",
)
_ACTIVE_VERB_MARKERS = (
    "萃取",
    "模拟",
    "设计",
    "分离",
    "simulate",
    "design",
    "recover",
    "提纯",
)

_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"

#: Markers that opt the extractive design into the ThermoFormer ML backend for
#: relative-volatility prediction.  Kept opt-in so UNIFAC stays the default.
_THERMOFORMER_MARKERS = (
    "thermoformer",
    "用thermoformer",
    "用 thermoformer",
    "thermo former",
    "tf后端",
)

#: Markers that ask for the entrainer short-list first (even when one is named),
#: e.g. 选萃取剂 / 推荐萃取剂 / 看候选.
_ENTRAINER_CHOICE_MARKERS = (
    "选萃取剂",
    "选择萃取剂",
    "推荐萃取剂",
    "看候选",
    "看看候选",
    "候选萃取剂",
    "有哪些萃取剂",
    "pick entrainer",
    "entrainer option",
)


def requests_thermoformer(message: str) -> bool:
    """Whether the extractive request asks to use the ThermoFormer backend."""
    lower = message.casefold()
    return any(marker in lower for marker in _THERMOFORMER_MARKERS)


def wants_entrainer_choices(message: str) -> bool:
    """Whether the request explicitly asks to see the candidate entrainer list."""
    lower = message.casefold()
    return any(marker in lower for marker in _ENTRAINER_CHOICE_MARKERS)


_EAC_NPAC_DMSO_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ethyl acetate", ("ethyl acetate", "ethylacetate", "乙酸乙酯")),
    ("n-propyl acetate", ("n-propyl acetate", "propyl acetate", "n propyl acetate", "乙酸正丙酯", "醋酸正丙酯")),
    ("dimethyl sulfoxide", ("dmso", "dimethyl sulfoxide", "dimethylsulfoxide", "二甲基亚砜")),
)


_ETHANOL_EAC_WATER_TIELINES_298K: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.8720, 0.1280), (0.0, 0.0142, 0.9858)),
    ((0.0575, 0.7634, 0.1791), (0.0280, 0.0189, 0.9531)),
    ((0.0996, 0.6409, 0.2595), (0.0479, 0.0218, 0.9303)),
    ((0.1311, 0.5270, 0.3419), (0.0623, 0.0265, 0.9112)),
    ((0.1519, 0.4125, 0.4356), (0.0784, 0.0334, 0.8882)),
    ((0.1609, 0.3689, 0.4702), (0.0859, 0.0400, 0.8741)),
    ((0.1557, 0.3056, 0.5387), (0.0923, 0.0438, 0.8639)),
)


def _mentions_all_alias_groups(message: str, groups: tuple[tuple[str, tuple[str, ...]], ...]) -> bool:
    lower = message.casefold()
    return all(any(alias.casefold() in lower for alias in aliases) for _, aliases in groups)


def is_eac_npac_dmso_vle_export_request(message: str) -> bool:
    """Fixed ternary VLE/extractive-distillation export for EtOAc/nPrOAc/DMSO."""
    lower = message.casefold()
    intent_markers = ("dwsim", "dwism", "export", "导出", "下载", "vle", "extractive", "萃取精馏")
    return _mentions_all_alias_groups(message, _EAC_NPAC_DMSO_ALIASES) and any(
        marker in lower for marker in intent_markers
    )


#: Markers that request a *plain* (non-extractive) binary distillation design,
#: e.g. "甲醇-水精馏", "精馏塔设计", "甲醇和水蒸馏".
_DISTILLATION_MARKERS = (
    "精馏塔",
    "精馏设计",
    "蒸馏塔",
    "蒸馏设计",
    "distillation",
    "distillation column",
)


def is_distillation_request(message: str) -> bool:
    """Whether the message asks for a plain distillation design (explicitly
    non-extractive; if it also mentions 萃取 it is handled as extractive)."""
    lower = message.casefold()
    if any(m in lower for m in _EXTRACTIVE_REQUEST_MARKERS):
        return False  # extractive requests take precedence
    if any(marker in lower for marker in _DISTILLATION_MARKERS):
        return True
    file_markers = ("dwsim", "dwism", "export", "导出", "下载", "vle")
    return any(marker in lower for marker in file_markers) and _find_distillation_components(message) is not None


#: Simple aliases for binary feed pairs supported by the design.
#: Each entry: (light_tokens, heavy_tokens, (light_comp, heavy_comp)).
#: Detection requires a light token AND a heavy token in the same message, so
#: "乙醇-甲苯" matches the toluene entry and not the ethanol-水 entry.
_DISTILLATION_BINARY_ALIASES: tuple[
    tuple[tuple[str, ...], tuple[str, ...], tuple[str, str]],
] = (
    (("甲醇", "methanol"), ("水", "water", "h2o"), ("methanol", "water")),
    (("乙醇", "ethanol"), ("甲苯", "toluene"), ("ethanol", "toluene")),
    (("乙醇", "ethanol"), ("水", "water", "h2o"), ("ethanol", "water")),
    # Straight-chain alkane reference system (NIST ThermoML 10.1016/j.fluid.2013.05.016).
    # DWSIM maps both names through ``thermo_engine.dwsim_export._COMPOUND_NAMES``;
    # the pair is a direct (non-extractive) binary distillation.
    (("正庚烷", "庚烷", "heptane"), ("正壬烷", "壬烷", "nonane"), ("heptane", "nonane")),
)

#: Markers that ask for a DWSIM *file* rather than only design numbers.
#: A bare "vle" is deliberately excluded here: it names the physics, not the
#: deliverable, and would otherwise fire on every VLE question.
_DWSIM_FILE_MARKERS = (
    "dwsim",
    "dwism",
    "dwxmz",
    "导出",
    "下载",
    "export",
    "download",
)

#: Default product specification for the report's binary VLE column (§1.5-1).
_BINARY_DEFAULT_PURITY = 0.995
_BINARY_DEFAULT_RECOVERY = 0.98
_BINARY_DEFAULT_PRESSURE_KPA = 101.325
#: Column pressure drop (kPa) written into the rendered DWSIM column.  The
#: archived ``report/dwsim`` binary columns all record 5000 Pa; keeping the same
#: value makes an agent-generated file structurally identical to those.
_BINARY_DEFAULT_PRESSURE_DROP_KPA = 5.0

#: Fraction of the feed taken as the bottoms (reboiler) product.  The requested
#: operating point splits the feed evenly, so ``bottoms = feed / 2`` and the
#: distillate takes the remainder.  This is the SECOND rigorous-column
#: specification and it is what actually makes the exported tower solvable --
#: without a product-flow spec DWSIM leaves both product streams at 0.
_BINARY_BOTTOMS_FEED_FRACTION = 0.5


def wants_dwsim_file(message: str) -> bool:
    """Whether the message asks for a DWSIM ``.dwxmz`` file to be produced."""
    lower = message.casefold()
    return any(marker in lower for marker in _DWSIM_FILE_MARKERS)


def _find_distillation_components(message: str) -> list[str] | None:
    lower = message.casefold()
    if _mentions_ipa(message) and _mentions_water(message):
        return ["isopropanol", "water"]
    for light_tokens, heavy_tokens, (c0, c1) in _DISTILLATION_BINARY_ALIASES:
        has_light = any(a in lower for a in light_tokens)
        has_heavy = any(a in lower for a in heavy_tokens)
        if has_light and has_heavy:
            return [c0, c1]
    return None


def is_binary_vle_dwsim_request(message: str) -> bool:
    """Whether the message asks for a binary-VLE DWSIM export.

    This is the report §1.5-1 case: a two-component VLE system (canonically
    2-propanol / water, but any binary pair in
    :data:`_DISTILLATION_BINARY_ALIASES` qualifies) that the user wants rendered
    as a downloadable DWSIM file.  Both halves are mandatory -- a recognised
    binary feed *and* an explicit DWSIM/export/download marker -- so a plain
    "甲醇-水精馏塔设计" stays a design-only request.

    Extractive-distillation and LLE-extraction requests take precedence and are
    excluded here.  The extractive exclusion is checked on the raw markers as
    well as through :func:`is_extractive_distillation_request`, because that
    router additionally demands an active verb and would otherwise let a bare
    "extractive distillation" phrase fall through to this plain-binary path.
    """
    lower = message.casefold()
    if any(marker in lower for marker in _EXTRACTIVE_REQUEST_MARKERS):
        return False
    if is_lle_extraction_request(message):
        return False
    if not wants_dwsim_file(message):
        return False
    return _find_distillation_components(message) is not None


def _find_feed_mole_fractions(message: str) -> list[float] | None:
    """Best-effort split of a binary feed (e.g. 甲醇50%水50%).

    Returns ``None`` when the message carries no two mole fractions that sum to
    one, so an unspecified feed is reported as a missing parameter instead of
    being silently invented as 50/50.
    """
    candidates = sorted(
        [float(v) for v in re.findall(_NUMBER, message) if 0.0 <= float(v) <= 1.0]
        + [float(v) / 100.0 for v in re.findall(r"(\d+(?:\.\d+)?)\s*%", message) if 0.0 <= float(v) <= 100.0]
    )
    # Prefer a pair that sums to 1.
    for a in candidates:
        for b in candidates:
            if abs(a + b - 1.0) <= 1e-3:
                return [a, b]
    return None


def run_binary_distillation(message: str) -> BinaryDistillationPayload:
    """Parse -> design -> report for a plain binary-distillation request.

    Returns a :class:`~schemas.column_design.BinaryDistillationPayload` with a
    computed short-cut design (stages, reflux, temperatures) or a missing-parameter
    / failure status.
    """
    components = _find_distillation_components(message)
    if components is None:
        return BinaryDistillationPayload(
            status="missing_parameters",
            missing_parameters=["components"],
            message="未能识别待精馏的二元体系，请说明（例如：甲醇-水精馏塔）。",
        )
    fractions = _find_feed_mole_fractions(message)
    if fractions is None:
        return BinaryDistillationPayload(
            status="missing_parameters",
            missing_parameters=["feed_composition"],
            message="未能解析进料组成（如甲醇50%、水50%），请补充。",
        )
    feed_composition = [fractions[0], fractions[1]]
    missing_mappings = missing_dwsim_compound_mappings(components)
    if missing_mappings:
        return BinaryDistillationPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            message=(
                "已识别为二元 VLE/精馏 DWSIM 导出，但这些组分没有明确的 DWSIM 名称映射："
                + ", ".join(missing_mappings)
                + "。请先导入或补充对应 DWSIM compound mapping 后再生成文件。"
            ),
        )
    alpha_source = "thermoformer" if requests_thermoformer(message) else None
    feed_flow_mol_s = _find_feed_flow(message) or 1.0
    pressure_kPa = _find_pressure(message) or _BINARY_DEFAULT_PRESSURE_KPA
    purity = _find_purity(message) or _BINARY_DEFAULT_PURITY
    recovery = _find_recovery(message) or _BINARY_DEFAULT_RECOVERY
    # Report §1.5-1 evaluates the relative volatility at the feed bubble point, so
    # derive it with the *same* activity source as the design rather than letting
    # the engine silently pick one.
    try:
        feed_temperature_K = bubble_temperature(
            components,
            feed_composition,
            pressure_kPa,
            alpha_source=alpha_source,
        )
    except Exception as exc:  # pragma: no cover - depends on the activity source
        return BinaryDistillationPayload(
            status="failed",
            message=f"进料泡点温度未能求解：{type(exc).__name__}",
        )
    try:
        result = design_binary_distillation_column(
            components,
            feed_composition,
            feed_flow_mol_s=feed_flow_mol_s,
            feed_temperature_K=feed_temperature_K,
            operating_pressure_kPa=pressure_kPa,
            distillate_purity_mole_fraction=purity,
            recovery=recovery,
            alpha_source=alpha_source,
        )
    except Exception as exc:  # pragma: no cover - guarded by schema validation
        return BinaryDistillationPayload(
            status="failed",
            message=f"精馏塔设计未能完成：{type(exc).__name__}",
        )
    model_note = "（含 ThermoFormer 泡点预测，未做实验校准）" if alpha_source == "thermoformer" else ""

    # Product split for the exported tower: bottoms take half the feed and the
    # distillate the remainder.  A rigorous DWSIM column needs exactly two
    # specifications, and the second one must be a product flow -- with only the
    # reflux ratio set, DWSIM solves the column but leaves BOTH product streams
    # at zero molar flow.
    bottoms_flow_mol_s = result.feed_flow_mol_s * _BINARY_BOTTOMS_FEED_FRACTION
    distillate_flow_mol_s = result.feed_flow_mol_s - bottoms_flow_mol_s

    # Feed tray: the request places the feed on the tray immediately above the
    # reboiler, so the feed enters at the bottom of the column.  ``feed_stage`` is
    # 1-based over the same stage list DWSIM uses ([0]=Condenser, [1..N-2]=trays,
    # [N-1]=Reboiler), so the tray above the reboiler is ``stages - 1`` and the
    # exporter's ``feed_stage - 1`` lands on DWSIM index ``stages - 2``.
    feed_stage = max(2, result.theoretical_stages - 1)

    # Render a DWSIM .dwxmz for the computed binary column.  A failure here keeps
    # the design and reports ``dwsim_unavailable`` instead of discarding numbers.
    dwsim_uri: str | None = None
    file_id: str | None = None
    dwsim_note = ""
    try:
        directory = export_directory()
        file_id = f"binary-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        destination = directory / f"{file_id}.dwxmz"
        export_dwsim_binary_column(
            components=result.components,
            feed_composition=result.feed_composition,
            feed_flow_mol_s=result.feed_flow_mol_s,
            feed_temperature_K=feed_temperature_K,
            operating_pressure_kPa=result.operating_pressure_kPa,
            stages=result.theoretical_stages,
            minimum_stages=result.minimum_stages,
            reflux_ratio=result.reflux_ratio,
            minimum_reflux_ratio=result.minimum_reflux_ratio,
            feed_stage=feed_stage,
            condenser_temperature_K=result.condenser_temperature_K,
            reboiler_temperature_K=result.reboiler_temperature_K,
            # The two rigorous-column specifications: reflux ratio at the
            # condenser, bottoms molar flow at the reboiler.
            distillate_flow_mol_s=distillate_flow_mol_s,
            bottoms_flow_mol_s=bottoms_flow_mol_s,
            # Match the archived report/dwsim binary columns, which were all
            # exported with a 5 kPa (5000 Pa) column pressure drop.  Omitting it
            # leaves ColumnPressureDrop=NaN and diverges from those files.
            pressure_drop_kPa=_BINARY_DEFAULT_PRESSURE_DROP_KPA,
            destination=destination,
        )
        dwsim_uri = f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz"
    except Exception as exc:  # pragma: no cover - depends on local DWSIM
        file_id = None
        dwsim_uri = None
        dwsim_note = f"（DWSIM 文件未生成：{type(exc).__name__}，仅返回设计数值）"

    payload = BinaryDistillationPayload(
        status="ready" if dwsim_uri is not None else "dwsim_unavailable",
        result=BinaryDistillationResult.model_validate(
            {
                "components": result.components,
                "feed_composition": result.feed_composition,
                "feed_flow_mol_s": result.feed_flow_mol_s,
                "operating_pressure_kPa": result.operating_pressure_kPa,
                "distillate_purity_mole_fraction": result.distillate_purity_mole_fraction,
                "relative_volatility": result.relative_volatility,
                "minimum_stages": result.minimum_stages,
                "minimum_reflux_ratio": result.minimum_reflux_ratio,
                "reflux_ratio": result.reflux_ratio,
                "theoretical_stages": result.theoretical_stages,
                "feed_stage": feed_stage,
                "condenser_temperature_K": result.condenser_temperature_K,
                "reboiler_temperature_K": result.reboiler_temperature_K,
                "distillate_flow_mol_s": distillate_flow_mol_s,
                "bottoms_flow_mol_s": bottoms_flow_mol_s,
                "alpha_source": result.alpha_source,
                "assumptions": result.assumptions,
            }
        ),
        alpha_source=alpha_source,
        feed_temperature_K=feed_temperature_K,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        message=(
            f"已生成二元精馏塔设计（{components[0]}-{components[1]}），"
            f"进料 x={feed_composition[0]:.3f}/{feed_composition[1]:.3f}，"
            f"F={result.feed_flow_mol_s:.3f} mol/s，塔顶纯度 {purity:.3f}、回收率 {recovery:.2f}。\n"
            f"相对挥发度 α={result.relative_volatility}；理论塔板数 {result.theoretical_stages}"
            f"（最小 {result.minimum_stages}）；回流比 {result.reflux_ratio}"
            f"（最小 {result.minimum_reflux_ratio}）；\n"
            f"进料板 {feed_stage}（塔釜上一级，共 {result.theoretical_stages} 级）；"
            f"塔釜采出 {bottoms_flow_mol_s:.4f} mol/s（= 进料的一半）、"
            f"塔顶采出 {distillate_flow_mol_s:.4f} mol/s；\n"
            f"塔顶温度 {result.condenser_temperature_K:.2f} K、塔釜温度 {result.reboiler_temperature_K:.2f} K"
            f"、进料泡点 {feed_temperature_K:.2f} K"
            f"，操作压强 {result.operating_pressure_kPa:.3f} kPa。{model_note}"
            + (" DWSIM 文件已生成，可直接下载并在 DWSIM 中打开复核。" if dwsim_uri else f" {dwsim_note}")
        ),
    )
    return payload


def export_directory(override: str | None = None) -> Path:
    """Resolve the export store directory, creating it if necessary."""
    configured = override or os.getenv("EXTRACTIVE_EXPORT_DIR") or _DEFAULT_EXPORT_DIR
    directory = Path(configured).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_generic_ternary_extractive_request(message: str) -> bool:
    """Whether the message names any three components plus an extractive/DWSIM intent.

    This is the general counterpart of
    :func:`is_eac_npac_dmso_vle_export_request`: instead of matching one fixed
    system it accepts any three components that can be resolved by alias.  It is
    consulted only after the fixed-system and LLE routers decline, so previously
    supported phrasings keep their exact behaviour.
    """
    from agent.generic_ternary_export import find_components, parse_request

    if is_eac_npac_dmso_vle_export_request(message):
        return False
    lower = message.casefold()
    if not any(marker in lower for marker in _EXTRACTIVE_REQUEST_MARKERS):
        return False
    if not wants_dwsim_file(message):
        return False
    return find_components(message) is not None and parse_request(message) is not None


def is_extractive_distillation_request(message: str) -> bool:
    """Decide whether a user message asks for an extractive-distillation simulation."""
    if is_eac_npac_dmso_vle_export_request(message):
        return True
    if is_generic_ternary_extractive_request(message):
        return True
    lower = message.casefold()
    has_extractive_marker = any(marker in lower for marker in _EXTRACTIVE_REQUEST_MARKERS)
    if not has_extractive_marker:
        return False
    # At least one alcohol + water + a request verb, or an explicit 萃取词.
    has_ethanol = any(alias in lower for alias in _ETHANOL_ALIASES)
    has_water = any(alias in lower for alias in _WATER_ALIASES)
    has_verb = any(verb in lower for verb in _ACTIVE_VERB_MARKERS)
    return has_verb and (has_ethanol or has_water or "萃取" in lower)


def _find_entrainer(message: str) -> str | None:
    lower = message.casefold()
    for aliases, canonical in _ENTRAINER_ALIASES:
        if any(alias in lower for alias in aliases):
            return canonical
    return None


def _find_feed_composition(message: str) -> list[float] | None:
    """Best-effort ethanol/water feed composition from free text.

    Supported forms (mole fractions):
      * ``组成 [0.4, 0.6]``/``组成：0.4、0.6``
      * ``乙醇40% 水60%``/``乙醇0.4 水0.6``
      * ``x_ethanol=0.4``/``乙醇摩尔分数0.4``
      * two loose numbers between 0 and 1 that sum within tolerance.
    """
    lower = message.casefold()

    bracket = re.search(
        rf"(?:组成|composition|feed)[^0-9]{{0,6}}[：:为=]?\s*\[?\s*("
        rf"{_NUMBER}(?:\s*(?:,|，|、)\s*{_NUMBER})+)\s*\]?",
        lower,
    )
    if bracket:
        values = [float(v) for v in re.findall(_NUMBER, bracket.group(1))]
        if len(values) == 2 and abs(sum(values) - 1.0) <= 1e-6:
            return values

    # "乙醇 0.4 水 0.6" style: capture numbers literally adjacent to the names.
    eth_pair = re.search(
        rf"(?:乙醇|ethanol)[^0-9]*(?:摩尔分数|组成为?|为|＝|=)?[^0-9]*({_NUMBER})\s*(?:%|分之|/)?"
        rf"[^\d]{{0,24}}"
        rf"(?:水|water)[^0-9]*(?:摩尔分数|组成为?|为|＝|=)?[^0-9]*({_NUMBER})",
        lower,
    )
    if eth_pair:
        a = float(eth_pair.group(1)) / 100.0 if eth_pair.group(1) and "%" in message and "/" not in eth_pair.group(
            0
        ) and eth_pair.group(1) not in ("0", "1") and float(eth_pair.group(1)) > 1 else float(eth_pair.group(1))
        b = float(eth_pair.group(2))
        if a > 1:
            a = a / 100.0
        if b > 1:
            b = b / 100.0
        if abs(a + b - 1.0) <= 0.01:
            return [a, b]

    # Loose pair of mole fractions.
    numbers = [
        v
        for v in (
            float(n) for n in re.findall(_NUMBER, message)
        )
        if 0.0 <= v <= 1.0
    ]
    if len(numbers) >= 2 and abs(numbers[0] + numbers[1] - 1.0) <= 1e-6:
        return [numbers[0], numbers[1]]
    return None


def _find_feed_flow(message: str) -> float | None:
    m = re.search(rf"({_NUMBER})\s*mol/s\b", message, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(rf"({_NUMBER})\s*kmol/?h\b", message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0 / 3600.0  # kmol/h -> mol/s
    m = re.search(rf"流量\s*[为是]?\s*({_NUMBER})", message)
    if m:
        return float(m.group(1))
    return None


def _find_temperature(message: str) -> float | None:
    from thermo_engine.units import temperature_to_kelvin

    low = message.casefold()
    # ``℃`` is not an ASCII word character, so a trailing ``\b`` never matches
    # after it; accept either a word boundary or end-of-input instead.
    m = re.search(rf"({_NUMBER})\s*(?:℃|°c|c(?![a-z0-9]))(?:\b|$)", low)
    if m:
        return temperature_to_kelvin(float(m.group(1)), "C")
    m = re.search(rf"({_NUMBER})\s*(?:k\b|k[^a-z]|kelvin)", low)
    if m:
        return temperature_to_kelvin(float(m.group(1)), "K")
    return None


def _find_pressure(message: str) -> float | None:
    from thermo_engine.units import pressure_to_kpa

    low = message.casefold()
    if "常压" in low or "atmospheric" in low:
        return 101.325
    m = re.search(rf"({_NUMBER})\s*(kpa|mpa|bar|atm)\b", low)
    if not m:
        return None
    unit = m.group(2)
    return pressure_to_kpa(float(m.group(1)), unit if unit != "kpa" else "kPa")


def _find_purity(message: str) -> float | None:
    low = message.casefold()
    # Require an explicit target-purity phrase so feed compositions like
    # "水60%" are not misread as a purity. Accept 纯度/purity + a number (with
    # optional %).
    m = re.search(
        rf"(?:塔顶)?\s*(?:乙醇)?\s*纯度\s*(?:≥|>=|大于|约|为|=|≈)?\s*({_NUMBER})\s*(?:%|％)?",
        low,
    )
    if not m:
        m = re.search(rf"purity[^0-9]{{0,10}}(?:of|≥|>=|>|:|=)?\s*({_NUMBER})\s*(?:%|％)?", low)
    if not m:
        return None
    value = float(m.group(1))
    return value / 100.0 if value > 1 else value


def _find_recovery(message: str) -> float | None:
    m = re.search(rf"(?:塔顶回收率|乙醇回收率|recovery)[^0-9]{{0,8}}({_NUMBER})\s*%?", message, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    return value / 100.0 if value > 1 else value


def extract_extractive_params(message: str) -> dict[str, object]:
    """Extract the design inputs present in a free-text query.

    Values that are absent are simply omitted from the returned dict; the caller
    decides whether a required field is missing.
    """
    params: dict[str, object] = {}
    feed_composition = _find_feed_composition(message)
    if feed_composition is not None:
        params["feed_composition"] = feed_composition
    feed_flow = _find_feed_flow(message)
    if feed_flow is not None:
        params["feed_flow_mol_s"] = feed_flow
    temperature = _find_temperature(message)
    if temperature is not None:
        params["feed_temperature_K"] = temperature
    pressure = _find_pressure(message)
    if pressure is not None:
        params["feed_pressure_kPa"] = pressure
    entrainer = _find_entrainer(message)
    if entrainer is not None:
        params["entrainer"] = entrainer
    purity = _find_purity(message)
    if purity is not None:
        params["distillate_purity_mole_fraction"] = purity
    recovery = _find_recovery(message)
    if recovery is not None:
        params["ethanol_recovery"] = recovery
    entrainer_ratio = _find_entrainer_ratio(message)
    if entrainer_ratio is not None:
        params["entrainer_ratio"] = entrainer_ratio
    return params


def _find_entrainer_ratio(message: str) -> float | None:
    low = message.casefold()
    # Only an explicit ratio phrase pins a number; a bare "萃取剂" must not
    # swallow an unrelated nearby number such as a target purity.
    for pattern in (
        rf"(?:萃取剂|溶剂|entrainer|solvent|s)[^\d]{{0,10}}(?:与进料)?\s*比\s*[为是]?\s*({_NUMBER})",
        rf"(?:萃取剂|溶剂|entrainer|solvent)\s*:\s*({_NUMBER})",
        rf"(?:进料)?(?:萃取剂比|溶剂比|entrainer.?to.?feed|solvent.?to.?feed)\s*[为是]?\s*({_NUMBER})",
    ):
        m = re.search(pattern, low)
        if m:
            value = float(m.group(1))
            if value > 0:
                return value
    return None


#: Human-friendly labels used in the missing-parameter prompt.
_PARAMETER_LABELS: dict[str, str] = {
    "feed_composition": "进料组成（乙醇/水摩尔分数）",
    "feed_flow_mol_s": "进料流量（mol/s）",
    "feed_temperature_K": "进料温度",
    "feed_pressure_kPa": "进料/操作压强",
    "entrainer": "萃取剂（如乙二醇/甘油）",
    "distillate_purity_mole_fraction": "塔顶乙醇目标纯度",
}


def build_extractive_spec(params: dict[str, object]) -> tuple[ExtractiveColumnSpec | None, list[str]]:
    """Build a validated spec from extracted params, reporting missing fields.

    ``feed_composition`` and ``entrainer`` are mandatory; the others fall back to
    documented engineering defaults so a first design can always be attempted.
    """
    missing: list[str] = []
    feed_composition = params.get("feed_composition")
    if not isinstance(feed_composition, list) or len(feed_composition) != 2:
        missing.append("feed_composition")
        feed_composition = [0.40, 0.60]
    entrainer = params.get("entrainer")
    if not isinstance(entrainer, str) or not entrainer:
        missing.append("entrainer")
        entrainer = "ethylene glycol"
    values = {
        "feed_components": ["ethanol", "water"],
        "feed_composition": feed_composition,
        "feed_flow_mol_s": float(params.get("feed_flow_mol_s", 1.0)),
        "feed_temperature_K": float(params.get("feed_temperature_K", 298.15)),
        "feed_pressure_kPa": float(params.get("feed_pressure_kPa", 101.325)),
        "entrainer": entrainer,
        "entrainer_ratio": float(params.get("entrainer_ratio", 2.0)),
        "distillate_purity_mole_fraction": float(params.get("distillate_purity_mole_fraction", 0.995)),
        "ethanol_recovery": float(params.get("ethanol_recovery", 0.98)),
        "operating_pressure_kPa": float(params.get("feed_pressure_kPa", 101.325)),
    }
    if params.get("operating_pressure_kPa") is not None:
        values["operating_pressure_kPa"] = float(params["operating_pressure_kPa"])
    spec = ExtractiveColumnSpec(**values)
    return spec, missing


def offer_entrainer_choices(
    spec: ExtractiveColumnSpec,
    alpha_source: str | None,
) -> tuple[list[EntrainerOption], str]:
    """Score candidate entrainers with the local model and project them for display.

    Runs ``recommend_extraction_entrainer`` (local UNIFAC or, when
    ``alpha_source == "thermoformer"``, the ThermoFormer bubble-point VLE) over
    the candidate pool and returns the ``EntrainerOption`` picker plus a
    human-readable message asking the user to choose.
    """
    recommendation = recommend_extraction_entrainer(spec, alpha_source=alpha_source)
    options = [
        EntrainerOption(
            name=candidate.name,
            canon_name=candidate.name,
            selectivity=candidate.selectivity,
            relative_volatility=candidate.relative_volatility,
            recommended=candidate.name == recommendation.recommended,
            note=candidate.note,
        )
        for candidate in recommendation.candidates
    ]
    source_label = "ThermoFormer" if alpha_source == "thermoformer" else "UNIFAC"
    names = "、".join(c.name for c in recommendation.candidates)
    message = (
        "已用本地模型（{}）筛出候选萃取剂（按选择性/相对挥发度排序）：{}。\n"
        "请从候选中选择萃取剂后继续，例如回复“用乙二醇”或“用甘油”。"
    ).format(source_label, names)
    return options, message


def run_eac_npac_dmso_vle_export(message: str, *, export_dir: str | None = None) -> ExtractiveExportPayload:
    """Run the fixed ethyl acetate / n-propyl acetate / DMSO VLE export.

    This is an extractive-distillation column: ethyl acetate is the light key,
    n-propyl acetate is the heavy key, and DMSO is the high-boiling entrainer.
    UNIFAC is the default volatility source; a ThermoFormer marker opts into the
    ML bubble-point path.
    """
    light = "ethyl acetate"
    heavy = "n-propyl acetate"
    entrainer = "dimethyl sulfoxide"
    components = [light, heavy, entrainer]
    missing_mappings = missing_dwsim_compound_mappings(components)
    if missing_mappings:
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            message=(
                "DWSIM export needs explicit compound mappings for: "
                + ", ".join(missing_mappings)
                + ". Please import/add the corresponding DWSIM compound mapping before generating the file."
            ),
        )

    params = extract_extractive_params(message)
    feed_composition = _find_feed_mole_fractions(message) or [0.5, 0.5]
    pressure = float(params.get("feed_pressure_kPa", _find_pressure(message) or 101.325))
    spec = ExtractiveColumnSpec(
        feed_components=[light, heavy],
        feed_composition=feed_composition,
        feed_flow_mol_s=float(params.get("feed_flow_mol_s", _find_feed_flow(message) or 1.0)),
        feed_temperature_K=float(params.get("feed_temperature_K", _find_temperature(message) or 360.13)),
        feed_pressure_kPa=pressure,
        entrainer=entrainer,
        entrainer_ratio=float(params.get("entrainer_ratio", _find_entrainer_ratio(message) or 2.0)),
        distillate_purity_mole_fraction=float(params.get("distillate_purity_mole_fraction", _find_purity(message) or 0.995)),
        ethanol_recovery=float(params.get("ethanol_recovery", _find_recovery(message) or 0.98)),
        operating_pressure_kPa=pressure,
        property_package="NRTL",
    )
    alpha_source = "thermoformer" if requests_thermoformer(message) else None
    try:
        design = design_generic_extractive_column(spec, alpha_source=alpha_source)
    except Exception as exc:  # noqa: BLE001
        hint = "; remove the ThermoFormer marker to use default UNIFAC" if alpha_source == "thermoformer" else ""
        return ExtractiveExportPayload(status="failed", message=f"EtOAc/nPrOAc/DMSO VLE design failed: {type(exc).__name__}{hint}")

    directory = export_directory(export_dir)
    file_id = f"eac-npac-dmso-vle-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    status: Literal["ready", "dwsim_unavailable"] = "ready"
    dwsim_uri: str | None = None
    source_note = "ThermoFormer" if alpha_source == "thermoformer" else "UNIFAC"
    message_text = (
        f"已完成乙酸乙酯/乙酸正丙酯/DMSO 三元 VLE 萃取精馏设计（{source_note}）。\n"
        f"理论塔板 {design.theoretical_stages}，进料板 {design.feed_stage}，DMSO 进料板 {design.entrainer_stage}，"
        f"回流比 {design.reflux_ratio:.3g}，操作压力 {design.operating_pressure_kPa:.3f} kPa。"
    )
    try:
        export_generic_extractive_column(
            light=light,
            heavy=heavy,
            entrainer=entrainer,
            feed_composition=[float(spec.feed_composition[0]), float(spec.feed_composition[1])],
            feed_flow_mol_s=spec.feed_flow_mol_s,
            feed_temperature_K=spec.feed_temperature_K,
            feed_pressure_kPa=spec.feed_pressure_kPa,
            stages=design.theoretical_stages,
            reflux_ratio=design.reflux_ratio,
            feed_stage=design.feed_stage,
            entrainer_stage=design.entrainer_stage,
            entrainer_ratio=spec.entrainer_ratio,
            condenser_temperature_K=design.condenser_temperature_K,
            reboiler_temperature_K=design.reboiler_temperature_K,
            property_package="NRTL",
            destination=destination,
        )
    except ThermoEquiError:
        status = "dwsim_unavailable"
        message_text += "\n设计已完成，但当前环境未配置可用的 DWSIM，未生成文件。"
    except Exception:
        status = "dwsim_unavailable"
        message_text += "\n设计已完成，但 DWSIM 文件导出失败；请检查 DWSIM 安装、DWSIM_HOME 或 compound mapping。"
    else:
        dwsim_uri = f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz"
        message_text += "\nDWSIM 文件已生成，可由前端自动下载。"

    return ExtractiveExportPayload(
        status=status,
        design=design,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        message=message_text,
    )


def run_extractive_export(message: str, *, export_dir: str | None = None) -> ExtractiveExportPayload:
    """Parse -> design -> export pipeline for a single extractive-distillation request.

    Returns a :class:`~schemas.column_design.ExtractiveExportPayload` describing
    either a ready DWSIM download, a missing-parameter request, or a design that
    could not be rendered to DWSIM (e.g. DWSIM not installed) but whose computed
    stages/reflux/temperatures are still returned.
    """
    # Isopropanol/water feeds take an orthogonal, generic short-cut route that
    # shares this same typed payload and Web channel (never the ethanol model).
    if is_eac_npac_dmso_vle_export_request(message):
        return run_eac_npac_dmso_vle_export(message, export_dir=export_dir)
    if is_generic_ternary_extractive_request(message):
        from agent.generic_ternary_export import run_generic_ternary_extractive_export

        return run_generic_ternary_extractive_export(message, export_dir=export_dir)
    if is_ipa_extractive_distillation_request(message):
        return run_ipa_extractive_export(message, export_dir=export_dir)
    params = extract_extractive_params(message)
    spec, missing = build_extractive_spec(params)
    # A missing entrainer is expected whenever the user hasn't picked one yet;
    # that case is handled by the candidate picker instead of a hard error.  Only
    # other missing fields (feed composition, flow, temperature, pressure,
    # purity) should pause for clarification.
    non_entrainer_missing = [name for name in missing if name != "entrainer"]
    if non_entrainer_missing:
        labels = "、".join(_PARAMETER_LABELS.get(name, name) for name in non_entrainer_missing)
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=non_entrainer_missing,
            message=(
                "已识别为萃取精馏模拟，但还缺少：" + labels + "。"
                "例如：乙醇/水摩尔组成、进料温度/压强、塔顶乙醇目标纯度。请补充后再试。"
            ),
        )

    wants_tf = requests_thermoformer(message)
    alpha_source = "thermoformer" if wants_tf else None

    # Offer the local-model entrainer short-list first when the user either
    # explicitly asks for candidates or has not committed to an entrainer yet.
    explicit_entrainer = _find_entrainer(message)
    if explicit_entrainer is None or wants_entrainer_choices(message):
        try:
            options, prompt = offer_entrainer_choices(spec, alpha_source)
        except Exception as exc:  # noqa: BLE001 - model blocking: fall back to message
            options, prompt = [], (
                f"候选萃取剂打分暂时不可用（{type(exc).__name__}）。请直接指定萃取剂，例如"
                "“用乙二醇”或“用甘油”后重试。"
            )
        return ExtractiveExportPayload(
            status="awaiting_entrainer",
            entrainer_candidates=options,
            alpha_source=alpha_source,
            message=prompt,
        )

    try:
        design = design_extractive_distillation_column(spec, alpha_source=alpha_source)
    except Exception as exc:  # pragma: no cover - guarded by schema validation
        hint = (
            "；已请求 ThermoFormer 后端但其运行被阻断（缺少依赖/权重，或预测失败），可去掉 ThermoFormer 用默认 UNIFAC 重试"
            if wants_tf
            else ""
        )
        return ExtractiveExportPayload(
            status="failed",
            message=f"萃取精馏设计未能完成：{type(exc).__name__}{hint}",
        )

    directory = export_directory(export_dir)
    file_id = f"extractive-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    dwsim_uri: str | None = None
    status: Literal["ready", "dwsim_unavailable"] = "ready"
    source_note = (
        "（采用 ThermoFormer 预测泡点相对挥发度；结果为 ML 近似值，未经实验验证）"
        if alpha_source == "thermoformer"
        else "（活度系数采用 UNIFAC 温度依赖预测）"
    )
    message_text = (
        "已生成萃取精馏塔设计。\n"
        f"理论板数：{design.theoretical_stages}；回流比 {design.reflux_ratio}（最小 {design.minimum_reflux_ratio}）；\n"
        f"进料板 {design.feed_stage}、萃取剂入口板 {design.entrainer_stage}；\n"
        f"塔顶温度 {design.condenser_temperature_K:.2f} K、塔釜温度 {design.reboiler_temperature_K:.2f} K；压强 "
        f"{design.operating_pressure_kPa:.3f} kPa。{source_note}\n"
        "DWSIM 文件已生成（结构+进料塔板+回流比已就绪）。打开后请为塔顶全凝器指定冷凝器规格、"
        "为塔釜再沸器指定热负荷（duty），并接上冷凝器/再沸器能量流后再计算。"
    )
    try:
        export_dwsim_extractive_column(
            design,
            destination,
        )
    except ThermoEquiError:
        status = "dwsim_unavailable"
        dwsim_uri = None
        message_text = (
            f"已生成萃取精馏塔设计（理论板数 {design.theoretical_stages}、回流比 {design.reflux_ratio}、塔顶 "
            f"{design.condenser_temperature_K:.1f} K、塔釜 {design.reboiler_temperature_K:.1f} K、压强 "
            f"{design.operating_pressure_kPa:.1f} kPa）。{source_note}\n"
            "但当前环境未配置可用的 DWSIM，未能生成文件。请安装 DWSIM 并设置 DWSIM_HOME 后再导出。"
        )
    except Exception:
        status = "dwsim_unavailable"
        message_text = (
            f"已生成萃取精馏塔设计（理论板数 {design.theoretical_stages}、回流比 {design.reflux_ratio}、塔顶 "
            f"{design.condenser_temperature_K:.1f} K、塔釜 {design.reboiler_temperature_K:.1f} K、压强 "
            f"{design.operating_pressure_kPa:.1f} kPa）。{source_note}\n"
            "但 DWSIM 导出未成功。可先用返回的设计参数在 DWSIM 中手动搭建，或检查 DWSIM 安装。"
        )
    else:
        dwsim_uri = f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz"

    return ExtractiveExportPayload(
        status=status,
        design=design,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        message=message_text,
    )


# --------------------------------------------------------------------------- #
# Isopropanol(IPA)/water extractive-distillation path
#
# A parallel, orthogonal route to the ethanol/water path above.  It recovers a
# user-selected light key ``isopropanol`` high-purity overhead from a water
# heavy using a high-boiling third-component entrainer inside a *generic*
# short-cut extractive design.  All numbers come from the deterministic engine
# (``design_generic_extractive_column`` / ``recommend_entrainer_for``); no
# experimental value or binary parameter is fabricated here.
# --------------------------------------------------------------------------- #
#: Engine-side canonical IPA/spc feed names (light, heavy).
_IPA_LIGHT_COMPONENT = "isopropanol"
_IPA_HEAVY_COMPONENT = "water"

#: Chinese/English tokens that pin the *light* key to isopropanol.
_IPA_ALIASES: tuple[str, ...] = (
    "异丙醇",
    "2-丙醇",
    "2丙醇",
    "异丙基醇",
    "2-propanol",
    "isopropanol",
    "isopropyl alcohol",
    "propan-2-ol",
    "ipa",
)
#: Tokens that pin the *heavy* key to water (reused for clarity/safety).
_WATER_ALIASES_LOCAL: tuple[str, ...] = ("水", "water", "h2o")


def _mentions_ipa(message: str) -> bool:
    low = message.casefold().replace("-", "-")
    return any(alias in low for alias in _IPA_ALIASES)


def _mentions_water(message: str) -> bool:
    low = message.casefold()
    return any(alias in low for alias in _WATER_ALIASES_LOCAL)


def is_ipa_extractive_distillation_request(message: str) -> bool:
    """Whether the (already-extractive) request targets an IPA/water feed.

    Requires an IPA token AND a water token in the same message, plus an
    extractive marker.  Because the Web client always re-sends the user's
    original full request (plus the chosen entrainer) on the confirmation
    turn, keying this purely on message text stays correct end to end.
    """
    if not _mentions_ipa(message) or not _mentions_water(message):
        return False
    lower = message.casefold()
    return any(marker in lower for marker in _EXTRACTIVE_REQUEST_MARKERS)


def _ipa_feed_composition(message: str) -> list[float] | None:
    """Two-normalized IPA(auto)/water feed mole fractions from free text.

    Accept ``组成 0.8/0.2``/``异丙醇80%水20%``/``IPA 0.8 水 0.2`` and a loose
    two-number pair summing to one as a last resort.
    """
    low = message.casefold()
    pair = re.search(
        rf"((?:异丙醇|2-丙醇|2丙醇|ipa|isopropanol)[^0-9]*(?:摩尔分数|组成为?|为|=)?"
        rf"[^0-9]*({_NUMBER})\s*(?:%|％)?)[^\d]{{0,30}}"
        rf"(水|water)[^0-9]*(?:摩尔分数|组成为?|为|=)?[^0-9]*({_NUMBER})",
        low,
        re.IGNORECASE,
    )
    candidates: list[float] = []
    if pair:
        a = float(pair.group(2))
        b = float(pair.group(4))
        if pair.group(0).count("%") + pair.group(0).count("％") > 0 or a > 1 or b > 1:
            a, b = a / 100.0, b / 100.0
        if abs(a + b - 1.0) <= 0.02:
            candidates = [a, b]
    else:
        bracket = re.search(
            rf"(?:组成|composition|feed)[^0-9]{{0,6}}[：:为=]?\s*\[?\s*("
            rf"{_NUMBER}(?:\s*(?:,|，|、)\s*{_NUMBER})+)\s*\]?",
            low,
        )
        if bracket:
            vals = [float(v) for v in re.findall(_NUMBER, bracket.group(1))]
            if len(vals) == 2 and abs(sum(vals) - 1.0) <= 1e-6:
                candidates = vals
    if candidates:
        return [candidates[0], candidates[1]]
    # Loose two-number pair summing to one.
    nums = [float(n) for n in re.findall(_NUMBER, message) if 0.0 <= float(n) <= 1.0]
    if len(nums) >= 2 and abs(nums[0] + nums[1] - 1.0) <= 1e-6:
        return [nums[0], nums[1]]
    return None


#: Candidates offered/ranked for the IPA/water extractive path.  Only solvents
#: that are genuinely used as high-boiling extractive-distillation entrainers for
#: the isopropanol/water system (and that the ``thermo`` library can resolve, and
#: that DWSIM's compound dictionary knows) are tried.  Unresolvable ones are
#: skipped deterministically by the engine; the ranking uses real UNIFAC short-cut
#: relative volatilities, never invented values.
_IPA_ENTRAINER_CANDIDATES: tuple[str, ...] = (
    "ethylene glycol",
    "glycerol",
)

#: Human-friendly parameter labels (shared English engine names) for prompts.
_IPA_PARAMETER_LABELS: dict[str, str] = {
    "feed_composition": "进料组成（异丙醇/水摩尔分数）",
    "feed_flow_mol_s": "进料流量（mol/s）",
    "feed_pressure_kPa": "进料/操作压强",
    "entrainer": "萃取剂",
}


def _build_ipa_spec(
    params: dict[str, object],
) -> tuple[ExtractiveColumnSpec | None, list[str]]:
    """Build a validated IPA/water spec (an :class:`ExtractiveColumnSpec` whose
    ``feed_components`` is ``["isopropanol", "water"]`` and whose other fields
    are generic).  The typed model carries an ``ethanol_recovery``-named field
    that, for this path, is the light-key (IPA) recovery fraction.
    """
    missing: list[str] = []
    composition = params.get("feed_composition")
    if not isinstance(composition, list) or len(composition) != 2:
        missing.append("feed_composition")
        composition = [0.80, 0.20]
    entrainer = params.get("entrainer")
    if not isinstance(entrainer, str) or not entrainer:
        missing.append("entrainer")
        entrainer = ""
    feed_flow = float(params.get("feed_flow_mol_s", 1.0))
    temperature = float(params.get("feed_temperature_K", 298.15))
    pressure = float(params.get("feed_pressure_kPa", 101.325))
    purity = float(params.get("distillate_purity_mole_fraction", 0.995))
    recovery = float(params.get("ethanol_recovery", 0.98))
    ratio = float(params.get("entrainer_ratio", 2.0))
    try:
        spec = ExtractiveColumnSpec(
            feed_components=[_IPA_LIGHT_COMPONENT, _IPA_HEAVY_COMPONENT],
            feed_composition=[float(composition[0]), float(composition[1])],
            feed_flow_mol_s=feed_flow,
            feed_temperature_K=temperature,
            feed_pressure_kPa=pressure,
            entrainer=entrainer or "ethylene glycol",
            entrainer_ratio=ratio,
            distillate_purity_mole_fraction=purity,
            ethanol_recovery=recovery,
            operating_pressure_kPa=pressure,
            property_package="NRTL",
        )
    except Exception:  # noqa: BLE001 - surfacing a structured missing request
        spec = None
    return spec, missing


def run_ipa_extractive_export(message: str, *, export_dir: str | None = None) -> ExtractiveExportPayload:
    """Parse -> candidate-list -> design -> DWSIM export for an IPA/water feed.

    Mirrors ``run_extractive_export`` but for the generic ``isopropanol``/
    ``water`` key pair.  When no entrainer is committed yet (or the user asked
    to see the short-list), it returns an ``awaiting_entrainer`` payload whose
    ``entrainer_candidates`` the Web picker renders for confirmation.  Once an
    entrainer is confirmed (the Web re-sends the full original message plus the
    choice) it runs ``design_generic_extractive_column`` and, if a DWSIM
    automation is available, renders a ``.dwxmz`` for download.
    """
    params = extract_extractive_params(message)
    ipa_composition = _ipa_feed_composition(message)
    if ipa_composition is not None:
        params["feed_composition"] = ipa_composition
    spec, missing = _build_ipa_spec(params)
    if spec is None:
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=missing,
            message="识别为异丙醇-水萃取精馏，但请求参数无效，请补充进料组成与萃取剂后重试。",
        )
    non_entrainer_missing = [name for name in missing if name != "entrainer"]
    if non_entrainer_missing:
        labels = "、".join(_IPA_PARAMETER_LABELS.get(name, name) for name in non_entrainer_missing)
        return ExtractiveExportPayload(
            status="missing_parameters",
            missing_parameters=non_entrainer_missing,
            message=(
                "已识别为异丙醇(IPA)-水萃取精馏，但还缺少：" + labels + "。"
                "例如：异丙醇/水摩尔组成、进料温度/压强、塔顶异丙醇目标纯度。请补充后再试。"
            ),
        )

    wants_tf = requests_thermoformer(message)
    alpha_source = "thermoformer" if wants_tf else None

    feed_temperature_K = spec.feed_temperature_K
    explicit_entrainer = _find_entrainer(message)
    if explicit_entrainer is None or wants_entrainer_choices(message):
        # Offer deterministic UNIFAC-ranked candidates (never fabricated).
        try:
            rec = recommend_entrainer_for(
                _IPA_LIGHT_COMPONENT,
                _IPA_HEAVY_COMPONENT,
                [float(spec.feed_composition[0]), float(spec.feed_composition[1])],
                feed_temperature_K,
                candidates=_IPA_ENTRAINER_CANDIDATES,
                alpha_source=alpha_source,
            )
            options = [
                EntrainerOption(
                    name=c.name,
                    canon_name=c.name,
                    selectivity=c.selectivity,
                    relative_volatility=c.relative_volatility,
                    recommended=c.name == rec.recommended,
                    note=c.note,
                )
                for c in rec.candidates
            ]
            _source_label = "ThermoFormer" if alpha_source == "thermoformer" else "UNIFAC"
            _names = "、".join(c.name for c in rec.candidates)
            prompt = (
                f"已用本地模型（{_source_label}）为异丙醇-水萃取筛出候选萃取剂（按相对挥发度排序）："
                f"{_names}。\n请从候选中选择萃取剂后继续，例如回复“用乙二醇”或“用甘油”。"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to instruct user
            options, prompt = [], (
                f"候选萃取剂打分暂时不可用（{type(exc).__name__}）。请直接指定萃取剂，例如"
                "“用乙二醇”或“用甘油”后重试。"
            )
        return ExtractiveExportPayload(
            status="awaiting_entrainer",
            entrainer_candidates=options,
            alpha_source=alpha_source,
            message=prompt,
        )

    # Entrainer committed -> run the generic short-cut design.
    try:
        design = design_generic_extractive_column(spec, alpha_source=alpha_source)
    except Exception as exc:  # noqa: BLE001 - guarded by schema/engine
        hint = (
            "；已请求 ThermoFormer 后端但其计算被阻断，可去掉 ThermoFormer 用默认 UNIFAC 重试"
            if wants_tf
            else ""
        )
        return ExtractiveExportPayload(
            status="failed",
            message=f"异丙醇-水萃取塔设计未能完成：{type(exc).__name__}{hint}",
        )

    directory = export_directory(export_dir)
    file_id = f"ipa-extractive-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    dwsim_uri: str | None = None
    status: Literal["ready", "dwsim_unavailable"] = "ready"
    source_note = (
        "（采用 ThermoFormer 预测泡点相对挥发度；结果为 ML 近似值，未经实验验证）"
        if alpha_source == "thermoformer"
        else "（活度系数采用 UNIFAC 温度依赖预测）"
    )
    message_text = (
        f"已生成异丙醇(IPA)-水萃取精馏塔设计（萃取剂 {spec.entrainer}）。\n"
        f"理论板数：{design.theoretical_stages}；回流比 {design.reflux_ratio}（最小 {design.minimum_reflux_ratio}）；\n"
        f"进料板 {design.feed_stage}、萃取剂入口板 {design.entrainer_stage}；\n"
        f"塔顶温度 {design.condenser_temperature_K:.2f} K、塔釜温度 {design.reboiler_temperature_K:.2f} K；压强 "
        f"{design.operating_pressure_kPa:.3f} kPa。{source_note}\n"
        "DWSIM 文件已生成（结构+进料塔板+回流比已就绪）。打开后请为塔顶全凝器指定冷凝器规格、"
        "为塔釜再沸器指定热负荷（duty），并接上冷凝器/再沸器能量流后再计算。"
    )
    try:
        from thermo_engine.column_design import bubble_temperature

        feed_temp = bubble_temperature(
            [_IPA_LIGHT_COMPONENT, _IPA_HEAVY_COMPONENT],
            [float(spec.feed_composition[0]), float(spec.feed_composition[1])],
            spec.operating_pressure_kPa,
            alpha_source if alpha_source == "thermoformer" else None,
        )
        export_generic_extractive_column(
            light=_IPA_LIGHT_COMPONENT,
            heavy=_IPA_HEAVY_COMPONENT,
            entrainer=spec.entrainer,
            feed_composition=[float(spec.feed_composition[0]), float(spec.feed_composition[1])],
            feed_flow_mol_s=spec.feed_flow_mol_s,
            feed_temperature_K=float(feed_temp),
            feed_pressure_kPa=spec.operating_pressure_kPa,
            stages=design.theoretical_stages,
            reflux_ratio=design.reflux_ratio,
            feed_stage=design.feed_stage,
            entrainer_stage=design.entrainer_stage,
            entrainer_ratio=spec.entrainer_ratio,
            condenser_temperature_K=design.condenser_temperature_K,
            reboiler_temperature_K=design.reboiler_temperature_K,
            destination=destination,
        )
    except ThermoEquiError:
        status = "dwsim_unavailable"
        dwsim_uri = None
        message_text = (
            f"已生成异丙醇(IPA)-水萃取精馏塔设计（理论板数 {design.theoretical_stages}、回流比 "
            f"{design.reflux_ratio}、塔顶 {design.condenser_temperature_K:.1f} K、塔釜 "
            f"{design.reboiler_temperature_K:.1f} K、压强 {design.operating_pressure_kPa:.1f} kPa）。{source_note}\n"
            "但当前环境未配置可用的 DWSIM，未能生成文件。请安装 DWSIM 并设置 DWSIM_HOME 后再导出。"
        )
    except Exception:  # noqa: BLE001 - DWSIM export failure
        status = "dwsim_unavailable"
        message_text = (
            f"已生成异丙醇(IPA)-水萃取精馏塔设计（理论板数 {design.theoretical_stages}、回流比 "
            f"{design.reflux_ratio}、塔顶 {design.condenser_temperature_K:.1f} K、塔釜 "
            f"{design.reboiler_temperature_K:.1f} K、压强 {design.operating_pressure_kPa:.1f} kPa）。{source_note}\n"
            "但 DWSIM 导出未成功。可先用返回的设计参数在 DWSIM 中手动搭建，或检查 DWSIM 安装。"
        )
    else:
        dwsim_uri = f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz"

    return ExtractiveExportPayload(
        status=status,
        design=design,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        message=message_text,
    )


#: Aliases for the supported liquid-liquid extraction ternary systems.  Each
#: entry maps the two feed solutes --- provided as (light-tokens, heavy-text)
#: pairs like in ``_DISTILLATION_BINARY_ALIASES`` --- together with the solvent
#: aliases, to the canonical component names.
_LLE_SYSTEMS: tuple[
    dict[str, object],
    ...,
] = (
    {
        "light_aliases": ("ethanol", "ethyl alcohol", "乙醇"),
        "heavy_aliases": ("ethyl acetate", "ethylacetate", "乙酸乙酯"),
        "solvent_aliases": ("water", "h2o", "水"),
        "solutes": ("ethanol", "ethyl acetate"),
        "solvent": "water",
    },
    {
        "light_aliases": ("propyl acetate", "乙酸正丙酯", "丙酸丙酯"),
        "heavy_aliases": ("ethyl acetate", "乙酸乙酯", "醋酸乙酯"),
        "solvent_aliases": ("dmso", "dimethyl sulfoxide", "二甲基亚砜", "dimethylsulfoxide", "dimesyl"),
        "solutes": ("n-propyl acetate", "ethyl acetate"),
        "solvent": "Dimethyl sulfoxide",
    },
)


def _find_lle_system(message: str) -> tuple[tuple[str, str], str] | None:
    """Find a supported liquid-liquid extraction ternary in free text.

    Both feed solutes and the extraction solvent must be mentioned.  Returns a
    ``(solutes, solvent)`` canonical tuple or ``None`` when the message does not
    map to a supported ternary extraction.
    """
    lower = message.casefold()
    for system in _LLE_SYSTEMS:
        has_light = any(alias in lower for alias in system["light_aliases"])
        has_heavy = any(alias in lower for alias in system["heavy_aliases"])
        has_solvent = any(alias in lower for alias in system["solvent_aliases"])
        if has_light and has_heavy and has_solvent:
            return (system["solutes"], system["solvent"])
    return None


def is_lle_extraction_request(message: str) -> bool:
    """Whether the message asks for a supported liquid-liquid extraction export.

    Keep this intentionally narrower than extractive distillation.  Overlapping
    ternary systems such as ethyl acetate / n-propyl acetate / DMSO must only
    route here when the user explicitly says LLE or liquid-liquid; generic words
    like DWSIM/export/extractive belong to the distillation/export routers.
    """
    lower = message.casefold()
    has_lle_marker = any(
        word in lower
        for word in (
            "lle",
            "liquid-liquid",
            "liquid liquid",
            "液液",
            "液-液",
            "液液萃取",
        )
    )
    if not has_lle_marker:
        return False
    return _find_lle_system(message) is not None


# --------------------------------------------------------------------------- #
# Binary (two-component) liquid-liquid extraction DWSIM path
#
# The report's third validation system is a *binary* LLE pair (water /
# 1-butanol: partially miscible, lower critical solution temperature) rendered
# as ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid`` with NRTL.
# ``report/dwsim/water_butanol_LLE_<T>K.dwxmz`` is the template.  This router
# generalises that template to any explicitly named binary pair.
# --------------------------------------------------------------------------- #

#: Markers that name the liquid-liquid physics (as opposed to VLE distillation).
_BINARY_LLE_MARKERS: tuple[str, ...] = (
    "lle",
    "liquid-liquid",
    "liquid liquid",
    "液液",
    "液-液",
    "分相",
    "互溶度",
    "部分互溶",
)

#: Default flash temperature for the binary LLE template, matching the lowest
#: temperature in ``report/dwsim`` (298.15 K / 25 degrees C).
_BINARY_LLE_DEFAULT_TEMPERATURE_K = 298.15
_BINARY_LLE_DEFAULT_PRESSURE_KPA = 101.325


def is_binary_lle_dwsim_request(message: str) -> bool:
    """Whether the message asks for a *binary* LLE DWSIM export.

    This is the report three-source system-3 case: a two-component partially
    miscible pair (canonically water / 1-butanol) that the user wants rendered as
    a downloadable DWSIM file following the ``report/dwsim`` template.

    Three conditions are mandatory so this router cannot steal work from the
    neighbouring flows:

    * exactly **two** resolvable components,
    * an explicit **LLE / liquid-liquid** marker -- a plain ``vle``, ``精馏`` or
      bubble-point pair stays with the binary-distillation router, and
    * an explicit **DWSIM / export / download** marker.

    Ternary solvent extraction (``is_lle_extraction_request``), the fixed
    EtOAc/nPrOAc/DMSO VLE export and extractive distillation all take precedence
    and are excluded here.
    """
    if is_eac_npac_dmso_vle_export_request(message):
        return False
    if is_lle_extraction_request(message):
        return False
    lower = message.casefold()
    # Extractive distillation needs an entrainer; those messages name a third
    # component or ask for a column, and belong to the extractive router.
    if is_extractive_distillation_request(message) and "液液" not in lower and "lle" not in lower:
        return False
    if not wants_dwsim_file(message):
        return False
    if not any(marker in lower for marker in _BINARY_LLE_MARKERS):
        return False
    # A three-component message belongs to the generic ternary router even when a
    # binary sub-pair inside it is resolvable ("acetonitrile toluene water"
    # resolves toluene + water).  Resolving exactly two components is what makes
    # this a *binary* case.
    if len(_ternary_lle_components(message)) >= 3:
        return False
    return len(_binary_lle_components(message)) == 2


def _binary_lle_components(message: str) -> list[str]:
    """The two canonical component names of a binary LLE request.

    Component identity is resolved independently of the LLM by
    ``thermo_engine.identity.resolve_literal_components``, which understands
    Chinese names, hyphenated binary pairs and list separators.  When the
    resolver finds nothing (e.g. a bare English name with no role evidence), a
    narrow alias table covers the canonical report pair so the template stays
    reachable.
    """
    from thermo_engine.identity import resolve_literal_components

    try:
        resolved = resolve_literal_components(message)
    except Exception:  # noqa: BLE001 - resolver is best-effort here
        resolved = []
    if len(resolved) == 2:
        return [component.name for _, component in resolved]
    if resolved:
        # One endpoint resolved: try to complete the pair from the alias table.
        names = [component.name for _, component in resolved]
        for aliases, canonical in _BINARY_LLE_ALIASES:
            if canonical in names:
                continue
            if any(alias in message.casefold() for alias in aliases):
                names.append(canonical)
                break
        if len(names) == 2:
            return names
        return names
    found: list[str] = []
    lower = message.casefold()
    for aliases, canonical in _BINARY_LLE_ALIASES:
        if any(alias in lower for alias in aliases) and canonical not in found:
            found.append(canonical)
    return found


#: Narrow alias table for the canonical partially-miscible pairs.  Only used to
#: *complete* an identity the resolver left half-resolved, never to invent one.
#: Values are the exact, case-sensitive DWSIM compound keys.
_BINARY_LLE_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("1-butanol", "n-butanol", "正丁醇", "butanol"), "1-butanol"),
    (("water", "h2o", "水"), "Water"),
)


def _find_binary_lle_temperature(message: str) -> float | None:
    """Flash temperature (K) for the binary LLE template, when stated."""
    return _find_temperature(message)


def run_binary_lle_export(message: str, *, export_dir: str | None = None) -> LLEExportPayload:
    """Parse -> template -> DWSIM export for a binary liquid-liquid pair.

    Mirrors the ``report/dwsim`` water/1-butanol assets for any explicitly named
    binary pair.  The file is a native ``Vessel`` two-liquid-phase flowsheet with
    the NRTL property package; all equilibrium numbers are solved by DWSIM, never
    by this project.
    """
    components = _binary_lle_components(message)
    if len(components) != 2:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["components"],
            message=(
                "已识别为二元液液（LLE）DWSIM 导出，但未能确定恰好两个组分。"
                "请明确写出这对部分互溶体系，例如“正丁醇-水二元液液萃取，导出 dwsim”。"
            ),
        )
    missing_mappings = missing_dwsim_compound_mappings(components)
    if missing_mappings:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            message=(
                "DWSIM export needs explicit compound mappings for: "
                + ", ".join(missing_mappings)
                + ". Please import/add the corresponding DWSIM compound mapping before generating the file."
            ),
        )
    fractions = _find_feed_mole_fractions(message)
    if fractions is None:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["feed_composition"],
            message=(
                f"未能解析 {components[0]} / {components[1]} 的进料摩尔组成。"
                "请在消息中给出两个相加为 1 的摩尔分数（例如 0.3/0.7）。"
            ),
        )
    temperature_K = _find_binary_lle_temperature(message)
    defaulted_temperature = temperature_K is None
    if temperature_K is None:
        temperature_K = _BINARY_LLE_DEFAULT_TEMPERATURE_K
    pressure_kPa = _find_pressure(message) or _BINARY_LLE_DEFAULT_PRESSURE_KPA
    feed_flow_mol_s = _find_feed_flow(message) or 1.0

    # A binary partially miscible pair has no third extraction solvent, so it is
    # described by BinaryCaseSpec (kind="lle", mode="two_liquid_vessel") rather
    # than the ternary LLEExtractionSpec.
    spec = BinaryCaseSpec(
        components=list(components),
        feed_composition=[float(fractions[0]), float(fractions[1])],
        kind="lle",
        mode="two_liquid_vessel",
        feed_flow_mol_s=feed_flow_mol_s,
        temperature_K=temperature_K,
        pressure_kPa=pressure_kPa,
        property_package="NRTL",
    )

    directory = export_directory(export_dir)
    file_id = f"binary-lle-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    split: dict[str, object] = {}
    try:
        export_dwsim_binary_lle_flowsheet(
            components=list(spec.components),
            feed_composition=list(spec.feed_composition),
            temperature_K=temperature_K,
            pressure_kPa=pressure_kPa,
            feed_flow_mol_s=feed_flow_mol_s,
            property_package="NRTL",
            destination=destination,
            split_out=split,
        )
    except ThermoEquiError as exc:
        return LLEExportPayload(
            status="dwsim_unavailable",
            binary_spec=spec,
            lle_kind="binary",
            message=(
                f"已解析二元液液体系 {components[0]} / {components[1]}，"
                f"但当前环境未配置可用的 DWSIM，未生成文件。原始错误：{exc.detail.message}"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - depends on the local DWSIM install
        return LLEExportPayload(
            status="failed",
            binary_spec=spec,
            lle_kind="binary",
            message=f"二元液液 DWSIM 导出失败：{type(exc).__name__}",
        )
    finally:
        # Never leave a zero-byte artefact behind when the render failed.
        if not destination.is_file() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)

    default_note = (
        f"（未在消息中给出温度，按模板默认 {_BINARY_LLE_DEFAULT_TEMPERATURE_K:.2f} K 生成）"
        if defaulted_temperature
        else ""
    )

    # Report DWSIM's own split.  A file that did not actually separate is called
    # out explicitly rather than being presented as a valid LLE render.
    split_note = ""
    if split.get("separated"):
        light_flow = split.get("light_flow_mol_s")
        heavy_flow = split.get("heavy_flow_mol_s")
        light_x = split.get("light_composition")
        heavy_x = split.get("heavy_composition")
        split_note = (
            f"\nDWSIM 解得两液相：Light_Liquid F={light_flow:.5f} mol/s x={light_x}；"
            f"Heavy_Liquid F={heavy_flow:.5f} mol/s x={heavy_x}。"
        )
    elif split:
        split_note = (
            f"\n注意：DWSIM 在该条件下未解出液液分相（Heavy_Liquid 流量 "
            f"{split.get('heavy_flow_mol_s')} mol/s），文件中的重相为空。"
            "请调整温度/组成后重试，或改用其他物性包。"
        )

    return LLEExportPayload(
        status="ready",
        binary_spec=spec,
        dwsim_file_uri=f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz",
        file_id=file_id,
        lle_kind="binary",
        message=(
            f"已按 report/dwsim 二元 LLE 模板生成 DWSIM 文件：{components[0]} / {components[1]}，"
            f"进料 z={spec.feed_composition[0]:.3f}/{spec.feed_composition[1]:.3f}，"
            f"F={spec.feed_flow_mol_s:.3f} mol/s，T={temperature_K:.2f} K，P={pressure_kPa:.3f} kPa。\n"
            f"流程为 Feed → Vessel_LLE → Vapor / Light_Liquid / Heavy_Liquid（NRTL 物性包），"
            f"Light_Liquid 为有机相、Heavy_Liquid 为水相。{default_note}"
            f"{split_note}\n"
            "液液平衡由 DWSIM 自带 NRTL 参数求解，本系统不生产平衡数；请在 DWSIM 中打开复核分相结果。"
        ),
    )


# --------------------------------------------------------------------------- #
# Generic ternary liquid-liquid DWSIM path
#
# Generalises the binary ``Vessel_LLE`` template to any three explicitly named
# components, so a user can type an arbitrary ternary LLE request instead of
# hitting the two-entry ``_LLE_SYSTEMS`` whitelist above.
#
# IMPORTANT: on DWSIM 9.0.5 the ternary split does not resolve through the
# Automation API (measured: all three FlashCalculationApproach kernels and every
# probed PreferredFlashAlgorithmTag return a single liquid phase; the
# FlashSettings that would enable an immiscible kernel are read-only).  The
# rendered file is structurally correct and fully solved, but its Heavy_Liquid
# phase is typically empty.  The chat answer says so explicitly rather than
# presenting the file as a working extraction.
# --------------------------------------------------------------------------- #

#: Markers that request a generic (non-whitelisted) ternary LLE export.
_TERNARY_LLE_MARKERS: tuple[str, ...] = (
    "三元",
    "ternary",
    "three-component",
    "three component",
)

#: Default flash temperature / pressure for the ternary template.
_TERNARY_LLE_DEFAULT_TEMPERATURE_K = 298.15
_TERNARY_LLE_DEFAULT_PRESSURE_KPA = 101.325


def _ternary_lle_components(message: str) -> list[str]:
    """Resolve exactly three component identities from a free-text request.

    Uses ``thermo_engine.identity.resolve_literal_components`` so identity comes
    from the chemistry itself (Chinese names, CAS numbers, list separators)
    rather than from a fixed system table.
    """
    from thermo_engine.identity import resolve_literal_components

    try:
        resolved = resolve_literal_components(message)
    except Exception:  # noqa: BLE001 - resolver is best-effort here
        return []
    return [component.name for _, component in resolved]


def is_generic_ternary_lle_request(message: str) -> bool:
    """Whether the message asks for a *generic* three-component LLE DWSIM export.

    Unlike :func:`is_lle_extraction_request`, this is not restricted to the
    ``_LLE_SYSTEMS`` whitelist: any three resolvable components qualify.  Three
    conditions are mandatory:

    * exactly **three** resolvable components (two components stay on the binary
      path; four or more are out of scope),
    * an explicit **LLE / liquid-liquid** marker, and
    * an explicit **DWSIM / export / download** marker.

    The whitelisted ternary systems are still handled by
    :func:`is_lle_extraction_request` and take precedence, so their existing
    behaviour (and tests) are untouched.
    """
    lower = message.casefold()
    if not wants_dwsim_file(message):
        return False
    if not any(marker in lower for marker in _BINARY_LLE_MARKERS):
        return False
    # The fixed extractive / ternary-VLE flows own their own routers.
    if is_eac_npac_dmso_vle_export_request(message):
        return False
    if is_extractive_distillation_request(message) and "液液" not in lower and "lle" not in lower:
        return False
    return len(_ternary_lle_components(message)) == 3


def run_generic_ternary_lle_export(message: str, *, export_dir: str | None = None) -> LLEExportPayload:
    """Parse -> generic ternary ``Vessel_LLE`` template -> DWSIM export.

    Any three named components are accepted.  Because DWSIM 9.0.5 cannot solve
    the ternary split through Automation (see the section comment above), the
    message always states the phase result DWSIM actually returned, and warns
    explicitly when the heavy phase came out empty.
    """
    components = _ternary_lle_components(message)
    if len(components) != 3:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["components"],
            lle_kind="ternary",
            message=(
                "已识别为三元液液（LLE）DWSIM 导出，但未能确定恰好三个组分。"
                "请明确写出这三个组分，例如“乙醇 乙酸乙酯 水 三元液液萃取，导出 dwsim”。"
            ),
        )
    missing_mappings = missing_dwsim_compound_mappings(components)
    if missing_mappings:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            lle_kind="ternary",
            message=(
                "DWSIM export needs explicit compound mappings for: "
                + ", ".join(missing_mappings)
                + ". Please import/add the corresponding DWSIM compound mapping before generating the file."
            ),
        )

    fractions = _find_lle_ternary_fractions(message, len(components))
    if fractions is None:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["feed_composition"],
            lle_kind="ternary",
            message=(
                "未能解析三组分进料摩尔组成。请给出三个相加为 1 的摩尔分数"
                "（例如“组成 0.129/0.188/0.683”）。"
            ),
        )

    temperature_K = _find_temperature(message)
    defaulted_temperature = temperature_K is None
    if temperature_K is None:
        temperature_K = _TERNARY_LLE_DEFAULT_TEMPERATURE_K
    pressure_kPa = _find_pressure(message) or _TERNARY_LLE_DEFAULT_PRESSURE_KPA
    feed_flow_mol_s = _find_feed_flow(message) or 1.0

    spec = TernaryLLESpec(
        components=list(components),
        feed_composition=[float(v) for v in fractions],
        temperature_K=temperature_K,
        pressure_kPa=pressure_kPa,
        feed_flow_mol_s=feed_flow_mol_s,
        property_package="NRTL",
    )

    directory = export_directory(export_dir)
    file_id = f"ternary-lle-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    split: dict[str, object] = {}
    try:
        export_dwsim_ternary_lle_flowsheet(
            components=list(components),
            feed_composition=[float(v) for v in fractions],
            temperature_K=temperature_K,
            pressure_kPa=pressure_kPa,
            feed_flow_mol_s=feed_flow_mol_s,
            property_package="NRTL",
            destination=destination,
            split_out=split,
        )
    except ThermoEquiError as exc:
        return LLEExportPayload(
            status="dwsim_unavailable",
            ternary_spec=spec,
            lle_kind="ternary",
            message=(
                f"已解析三元液液体系 {' / '.join(components)}，"
                f"但当前环境未配置可用的 DWSIM，未生成文件。原始错误：{exc.detail.message}"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - depends on the local DWSIM install
        return LLEExportPayload(
            status="failed",
            ternary_spec=spec,
            lle_kind="ternary",
            message=f"三元液液 DWSIM 导出失败：{type(exc).__name__}",
        )
    finally:
        if not destination.is_file() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)

    default_note = (
        f"（未在消息中给出温度，按模板默认 {_TERNARY_LLE_DEFAULT_TEMPERATURE_K:.2f} K 生成）"
        if defaulted_temperature
        else ""
    )

    if split.get("separated"):
        split_note = (
            f"\nDWSIM 解得两液相：Light_Liquid F={split.get('light_flow_mol_s'):.5f} mol/s "
            f"x={split.get('light_composition')}；"
            f"Heavy_Liquid F={split.get('heavy_flow_mol_s'):.5f} mol/s x={split.get('heavy_composition')}。"
        )
        warnings: list[str] = []
    else:
        split_note = (
            f"\n注意：DWSIM 在该条件下未解出液液分相（Heavy_Liquid 流量 "
            f"{split.get('heavy_flow_mol_s')} mol/s），文件中的重相为空。\n"
            "这是 DWSIM 9.0.5 的已知限制：三元液液分相无法通过 Automation 接口求解"
            "（三个闪蒸内核与全部 PreferredFlashAlgorithmTag 均返回单液相，"
            "ImmiscibleWaterOption 等设置在本机为只读）。\n"
            "文件结构正确且已求解，可在 DWSIM GUI 中手工把 Vessel_LLE 的闪蒸内核改为"
            "“Nested Loops (Immiscible)”后重算，即可得到两相。"
        )
        warnings = [
            "DWSIM 未解出三元液液分相（重相流量为 0）；文件已生成但需在 GUI 中选择不相溶闪蒸内核后重算。"
        ]

    return LLEExportPayload(
        status="ready",
        ternary_spec=spec,
        dwsim_file_uri=f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz",
        file_id=file_id,
        lle_kind="ternary",
        warnings=warnings,
        message=(
            f"已按三元 LLE 模板生成 DWSIM 文件：{' / '.join(components)}，"
            f"进料 z={'/'.join(f'{v:.3f}' for v in fractions)}，"
            f"F={feed_flow_mol_s:.3f} mol/s，T={temperature_K:.2f} K，P={pressure_kPa:.3f} kPa。\n"
            "流程为 Feed → Vessel_LLE → Vapor / Light_Liquid / Heavy_Liquid（NRTL 物性包），"
            f"Light_Liquid 为有机相、Heavy_Liquid 为水相。{default_note}"
            f"{split_note}\n"
            "液液平衡由 DWSIM 自带 NRTL 参数求解，本系统不生产平衡数。"
        ),
    )


def _find_lle_ternary_fractions(message: str, count: int) -> list[float] | None:
    """Three feed mole fractions from free text (bracket list or loose triplet)."""
    bracket = re.search(
        rf"(?:组成|composition|feed)[^0-9]{{0,6}}[：:为=]?\s*\[?\s*("
        rf"{_NUMBER}(?:\s*(?:,|，|、|/)\s*{_NUMBER})+)\s*\]?",
        message,
        re.IGNORECASE,
    )
    if bracket:
        values = [float(v) for v in re.findall(_NUMBER, bracket.group(1))]
        if len(values) == count and abs(sum(values) - 1.0) <= 1e-6:
            return values
    # A loose run of `count` fractions is ambiguous when temperatures/pressures
    # are also present, so only accept an explicit slash- or comma-separated run.
    run = re.search(
        rf"({_NUMBER})\s*(?:/|,|，|、)\s*({_NUMBER})\s*(?:/|,|，|、)\s*({_NUMBER})",
        message,
    )
    if run:
        values = [float(run.group(i)) for i in (1, 2, 3)]
        if all(0.0 <= v <= 1.0 for v in values) and abs(sum(values) - 1.0) <= 1e-6:
            return values
    return None


def _find_lle_solute_fractions(message: str) -> list[float] | None:
    """Best-effort two-solute feed mole fractions from the query."""
    candidates = sorted(
        [float(v) for v in re.findall(_NUMBER, message) if 0.0 <= float(v) <= 1.0]
        + [float(v) / 100.0 for v in re.findall(r"(\d+(?:\.\d+)?)\s*%", message) if 0.0 <= float(v) <= 100.0]
    )
    for a in candidates:
        for b in candidates:
            if abs(a + b - 1.0) <= 1e-3:
                return [a, b]
    return [0.5, 0.5]


def _find_lle_solvent_ratio(message: str) -> float | None:
    """Best-effort solvent : feed molar ratio from the query."""
    low = message.casefold()
    for pattern in (
        rf"(?:溶剂|萃取剂|solvent|s)[^\d]{{0,10}}(?:与进料)?\s*比\s*[为是]?\s*({_NUMBER})",
        rf"(?:溶剂|solvent)[^\d]{{0,6}}:?\s*({_NUMBER})",
    ):
        m = re.search(pattern, low)
        if m:
            value = float(m.group(1))
            if value > 0:
                return value
    return None


def _ethanol_eac_water_lle_products(
    *,
    feed_composition: list[float],
    feed_flow_mol_s: float,
    solvent_ratio: float,
) -> tuple[list[float], float, list[float], float, float]:
    """Estimate product streams from the nearest ethanol/ethyl acetate/water tie-line.

    Component order is ethanol, ethyl acetate, water.  The organic-rich endpoint
    is returned as raffinate; the water-rich endpoint is returned as extract.
    """
    total_flow = feed_flow_mol_s * (1.0 + solvent_ratio)
    overall = [
        feed_flow_mol_s * feed_composition[0] / total_flow,
        feed_flow_mol_s * feed_composition[1] / total_flow,
        feed_flow_mol_s * solvent_ratio / total_flow,
    ]
    best: tuple[float, list[float], float, list[float], float] | None = None
    for organic, aqueous in _ETHANOL_EAC_WATER_TIELINES_298K:
        direction = [organic[i] - aqueous[i] for i in range(3)]
        denom = sum(v * v for v in direction)
        if denom <= 0.0:
            continue
        organic_fraction = sum((overall[i] - aqueous[i]) * direction[i] for i in range(3)) / denom
        organic_fraction = max(0.0, min(1.0, organic_fraction))
        reconstructed = [organic_fraction * organic[i] + (1.0 - organic_fraction) * aqueous[i] for i in range(3)]
        residual = sum((overall[i] - reconstructed[i]) ** 2 for i in range(3)) ** 0.5
        if best is None or residual < best[0]:
            best = (residual, list(organic), organic_fraction, list(aqueous), 1.0 - organic_fraction)
    if best is None:
        raise ValueError("no ethanol/ethyl acetate/water LLE tie-line is available")
    residual, raffinate_x, raffinate_fraction, extract_x, extract_fraction = best
    return (
        raffinate_x,
        total_flow * raffinate_fraction,
        extract_x,
        total_flow * extract_fraction,
        residual,
    )


def run_lle_extraction_export(message: str, *, export_dir: str | None = None) -> LLEExportPayload:
    """Parse -> DWSIM export for a real liquid-liquid extraction tower request."""
    system = _find_lle_system(message)
    if system is None:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["system"],
            message="未识别到支持的三元 LLE 体系；当前固定示例为 ethanol + ethyl acetate + water。",
        )
    solutes, solvent = system
    missing_mappings = missing_dwsim_compound_mappings([*solutes, solvent])
    if missing_mappings:
        return LLEExportPayload(
            status="missing_parameters",
            missing_parameters=["dwsim_compound_mapping"],
            message=(
                "DWSIM export needs explicit compound mappings for: "
                + ", ".join(missing_mappings)
                + ". Please import/add the corresponding DWSIM compound mapping before generating the file."
            ),
        )
    fractions = _find_lle_solute_fractions(message) or [0.5, 0.5]

    spec = LLEExtractionSpec(
        feed_components=list(solutes),
        feed_composition=fractions,
        solvent=solvent,
        solvent_ratio=_find_lle_solvent_ratio(message) or 1.0,
        feed_flow_mol_s=_find_feed_flow(message) or 1.0,
        feed_temperature_K=_find_temperature(message) or 298.15,
        feed_pressure_kPa=_find_pressure(message) or 101.325,
        property_package="NRTL",
    )

    directory = export_directory(export_dir)
    file_id = f"lle-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    destination = directory / f"{file_id}.dwxmz"
    try:
        export_dwsim_lle_extraction(
            components=[*solutes, solvent],
            feed_composition=spec.feed_composition,
            feed_flow_mol_s=spec.feed_flow_mol_s,
            feed_temperature_K=spec.feed_temperature_K,
            feed_pressure_kPa=spec.feed_pressure_kPa,
            solvent=spec.solvent,
            solvent_ratio=spec.solvent_ratio,
            property_package="NRTL",
            destination=destination,
        )
    except ThermoEquiError as exc:
        return LLEExportPayload(
            status="failed",
            spec=spec,
            message=(
                "当前 DWSIM Automation 未暴露真实的液液萃取塔/液液萃取器对象，因此没有生成 DWSIM 文件。"
                "我不会再用 Vessel、Component Separator 或手动 tie-line 分流伪装成液液萃取塔。"
                f" 原始错误：{exc.detail.message}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return LLEExportPayload(
            status="failed",
            spec=spec,
            message=f"液液萃取塔 DWSIM 导出失败：{type(exc).__name__}",
        )

    dwsim_uri = f"{DOWNLOAD_PREFIX}/{quote(file_id)}.dwxmz"
    return LLEExportPayload(
        status="ready",
        spec=spec,
        dwsim_file_uri=dwsim_uri,
        file_id=file_id,
        lle_kind="ternary",
        message=(
            f"已生成真实 DWSIM 液液萃取塔文件：{solutes[0]} + {solutes[1]}，{solvent} 为溶剂，"
            f"溶剂比 {spec.solvent_ratio:.2f}。"
        ),
    )

def evidence_for_lle_extraction(payload: LLEExportPayload) -> list[EvidenceStatement]:
    """Compose the evidence statement for a liquid-liquid extraction export."""
    if payload.status == "ready":
        if payload.lle_kind == "binary":
            text = (
                "已按 report/dwsim 二元 LLE 模板渲染 DWSIM 流程（Feed → Vessel_LLE → "
                "Vapor / Light_Liquid / Heavy_Liquid，NRTL 物性包）；液液平衡由 DWSIM 自带 NRTL "
                "参数求解，本系统不计算 LLE 数值。"
            )
        else:
            text = (
                "已渲染液液萃取 DWSIM 流程；液液平衡由 DWSIM 自带 NRTL 参数求解，"
                "本系统不计算 LLE 数值。"
            )
    elif payload.status == "missing_parameters":
        text = "液液萃取缺少必要参数，尚未生成文件。"
    elif payload.status == "dwsim_unavailable":
        text = "设计参数适配完成，但当前环境未安装/配置 DWSIM，未生成文件。"
    else:
        text = "液液萃取导出未能完成。"
    return [
        EvidenceStatement(category="Calculation", text=text),
        EvidenceStatement(category="Warning", text="LLE 平衡数值请以 DWSIM 求解为准，本系统不生产平衡数。"),
    ]


def evidence_for_binary_distillation(payload: BinaryDistillationPayload) -> list[EvidenceStatement]:
    """Compose the evidence statements for a binary-VLE distillation design.

    Keeps the repository rule visible to the user: the stage/reflux numbers are
    deterministic short-cut values, while the equilibrium numbers inside the
    rendered DWSIM file are re-solved by DWSIM itself.
    """
    if payload.status == "ready":
        text = (
            "二元精馏塔由确定性短节法（Fenske/Underwood/Gilliland）设计，并已渲染为可下载的 DWSIM 文件；"
            "请在 DWSIM 中打开复核严格的塔剖面与冷凝器/再沸器负荷。"
        )
    elif payload.status == "dwsim_unavailable":
        text = "二元精馏塔设计已完成（理论板数/回流比/温度见结构化结果），但当前环境未安装或未配置 DWSIM，未生成文件。"
    elif payload.status == "missing_parameters":
        text = "二元 VLE 精馏设计缺少必要参数，尚未执行设计。"
    else:
        text = "二元 VLE 精馏设计未能完成。"
    statements = [EvidenceStatement(category="Calculation", text=text)]
    if payload.alpha_source == "thermoformer":
        statements.append(
            EvidenceStatement(
                category="Estimate",
                text="相对挥发度来自 ThermoFormer 泡点预测（ML 近似值，未做实验校准）。",
            )
        )
    return statements


def evidence_for_extractive(payload: ExtractiveExportPayload) -> list[EvidenceStatement]:
    """Compose the evidence statement attached to the chat response."""
    if payload.status == "awaiting_entrainer":
        model = "ThermoFormer" if payload.alpha_source == "thermoformer" else "UNIFAC"
        text = f"已用本地模型（{model}）筛出候选萃取剂，等待用户确认选择后再执行塔设计。"
    elif payload.status == "ready":
        if payload.design and "ThermoFormer" in " ".join(payload.design.assumptions):
            text = "萃取精馏塔由确定性短节模型设计，相对挥发度来自 ThermoFormer 泡点预测（ML 近似值，未实验验证），并渲染为可下载的 DWSIM 文件。"
        else:
            text = "萃取精馏塔将由确定性短节模型设计并渲染为可下载的 DWSIM 文件，请使用 DWSIM 打开查看。"
    elif payload.status == "missing_parameters":
        text = "萃取精馏模拟缺少必要参数，尚未执行设计。"
    elif payload.status == "dwsim_unavailable":
        text = "设计已完成，但当前环境未安装/配置 DWSIM，未生成文件。"
    else:
        text = "萃取精馏模拟未能完成。"
    return [EvidenceStatement(category="Calculation", text=text)]


__all__ = [
    "DOWNLOAD_PREFIX",
    "build_extractive_spec",
    "evidence_for_binary_distillation",
    "evidence_for_extractive",
    "evidence_for_lle_extraction",
    "export_directory",
    "extract_extractive_params",
    "is_binary_lle_dwsim_request",
    "is_binary_vle_dwsim_request",
    "is_distillation_request",
    "is_eac_npac_dmso_vle_export_request",
    "is_extractive_distillation_request",
    "is_generic_ternary_lle_request",
    "is_ipa_extractive_distillation_request",
    "is_lle_extraction_request",
    "offer_entrainer_choices",
    "requests_thermoformer",
    "run_binary_lle_export",
    "run_binary_distillation",
    "run_extractive_export",
    "run_eac_npac_dmso_vle_export",
    "run_generic_ternary_lle_export",
    "run_ipa_extractive_export",
    "run_lle_extraction_export",
    "wants_dwsim_file",
    "wants_entrainer_choices",
]
