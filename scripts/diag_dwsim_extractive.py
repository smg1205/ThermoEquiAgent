"""Diagnose why the extractive-distillation DWSIM export fails.

The chat agent wraps any DWSIM export error into a generic "DWSIM unavailable"
message, which hides the real cause.  This script invokes the extractive export
directly and prints the exact exception + its ``details`` so the underlying
problem becomes visible.

Run in the project's ``(thermo)`` environment:

    python scripts/diag_dwsim_extractive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# A previous ``pip install -e .`` registers an editable finder that can shadow
# (or redirect) the top-level ``apps`` / ``schemas`` / ``agent`` packages to a
# DIFFERENT source tree.  Insert this project's root at the very front of
# ``sys.path`` so we always import from THIS repo (same trick as run_server.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.column_design import ExtractiveColumnDesign, ExtractiveColumnSpec

# Build a spec matching the reported design (ethanol/water feed, entrainer).
# Adjust if your actual chat parameters differ; the design numbers shown in the
# chat message (21 stages / reflux 2.476 / 351.4 K / 415.1 K / 101.3 kPa) are
# only used here as a reproduction baseline.
spec = ExtractiveColumnSpec(
    feed_components=["ethanol", "water"],
    feed_composition=[0.15, 0.85],  # ethanol, water  (mole fraction)
    entrainer="ethylene glycol",
    feed_temperature_K=353.15,
    feed_pressure_kPa=101.325,
    operating_pressure_kPa=101.325,
    feed_flow_mol_s=1.0,
    entrainer_ratio=2.0,
    distillate_purity_mole_fraction=0.9,
    property_package="NRTL",
)

# Reuse the deterministic design model so we exercise the full path.
from thermo_engine.column_design import design_extractive_distillation_column

design = design_extractive_distillation_column(spec)
assert isinstance(design, ExtractiveColumnDesign)

destination = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive.dwxmz"
print(f"设计完成: 理论板 {design.theoretical_stages}, 回流比 {design.reflux_ratio:.3f}")
print(f"目标文件: {destination}")

# --- Probe which DWSIM compound names are actually recognized -------------
# The export list is hard-coded as ["ethanol", "water", spec.entrainer].  The
# real failure mode is that DWSIM's compound dictionary keyed by a *different*
# name -> AddCompound raises KeyNotFoundException.  Probe the available set.
import thermo_engine.dwsim_export as ded  # noqa: E402

_try_names = ["ethanol", "Ethanol", "water", "Water", "ethylene glycol", "Ethylene glycol", "EG"]
_probe_factory, _probe_object_type = ded._automation_factory()
_probe_automation = _probe_factory()
_probe_fs = _probe_automation.CreateFlowsheet()
for _n in _try_names:
    try:
        _probe_fs.AddCompound(_n)
        print(f"  [OK ] AddCompound({_n!r})")
    except Exception as _e:  # noqa: BLE001
        print(f"  [FAIL] AddCompound({_n!r}) -> {type(_e).__name__}: {_e}")
print("  (probe done; DWSIM flowsheet created without error)" if _probe_fs else "")

# --- Probe the DWSIM ObjectType enum for column members --------------------
# After the compound-name fix, the most likely next throw-point is
# ``_column_object_type`` which only looks for ShortcutColumn / DistillationColumn
# / Column.  Dump every ObjectType member so we can see the real column name(s).
print("\n=== ObjectType 枚举成员（找列对象类型）===")
_o_members = [m for m in dir(_probe_object_type) if not m.startswith("_")]
_col_candidates = [m for m in _o_members if "column" in m.lower() or "tower" in m.lower()]
print(f"  含 'column'/'tower' 的成员: {_col_candidates}")
if not _col_candidates:
    # Also show a sample of members so we at least see what IS available.
    print(f"  (未找到列对象成员; ObjectType 前 80 个成员: {_o_members[:80]})")

# --- Probe the column graphic object's connection topology ------------------
# The last failure ("Check the connections of the object") means a ShortcutColumn
# cannot express the two-feed (feed + entrainer at different stages) extractive
# topology.  We probe BOTH ShortcutColumn and DistillationColumn to see which one
# accepts the feed + entrainer + distillate + bottoms connections DWSIM needs.
print("\n=== 探测列对象图形端口 (拓扑) ===")

def _count_connections(fs: object) -> int:
    try:
        from thermo_engine.dwsim_export import _simulation_object
        fs_obj = _simulation_object(fs)
        conn = getattr(fs_obj, "Connections", None) or getattr(fs_obj, "GetConnections", None)
        if callable(conn):
            return len(conn())
        if conn is not None:
            return len(list(conn))
    except Exception:  # noqa: BLE001
        pass
    return -1


def _probe_column_ports() -> None:
    for _ctype_name in ("ShortcutColumn", "DistillationColumn"):
        _ct = getattr(_probe_object_type, _ctype_name, None)
        if _ct is None:
            print(f"\n  --- {_ctype_name}: (不可用) ---")
            continue
        print(f"\n  --- {_ctype_name} ---")
        try:
            _fs = _probe_automation.CreateFlowsheet()
            for _n in ("Ethanol", "Water", "Ethylene glycol"):
                _fs.AddCompound(_n)
            ded._add_property_package(_fs, "NRTL")
            _col = _fs.AddObject(_ct, 250, 0, "Probe Column")
            _feed = _fs.AddObject(_probe_object_type.MaterialStream, 0, -60, "Feed")
            _ent = _fs.AddObject(_probe_object_type.MaterialStream, 0, 60, "Entrainer")
            _dist = _fs.AddObject(_probe_object_type.MaterialStream, 500, -60, "Distillate")
            _bot = _fs.AddObject(_probe_object_type.MaterialStream, 500, 60, "Bottoms")

            # Try connecting feed and entrainer into the column with each (fidx, tidx).
            for _label, _obj in (("feed", _feed), ("entrainer", _ent)):
                for _t in (0, 1, 2):
                    try:
                        _fs.ConnectObjects(_obj.GraphicObject, _col.GraphicObject, 0, _t)
                        print(f"    {_label}->column (0,{_t}) [OK]")
                    except Exception as e:  # noqa: BLE001
                        print(f"    {_label}->column (0,{_t}) [FAIL] {type(e).__name__}")
            # Try distillate / bottoms out of the column.
            for _label, _obj in (("distillate", _dist), ("bottoms", _bot)):
                for _f in (0, 1, 2):
                    try:
                        _fs.ConnectObjects(_col.GraphicObject, _obj.GraphicObject, _f, 0)
                        print(f"    column->{_label} ({_f},0) [OK]")
                    except Exception as e:  # noqa: BLE001
                        print(f"    column->{_label} ({_f},0) [FAIL] {type(e).__name__}")

            # List column-design parameter methods that DWSIM exposes.
            sim = ded._simulation_object(_col)
            methods = [
                a for a in dir(sim)
                if any(k in a.lower() for k in ("stage", "reflux", "feed", "conden", "reboil", "stage"))
            ]
            print(f"    设计相关成员: {methods}")

            # --- Probe solver-facing method signatures via .NET reflection -----
            # A rigorous DistillationColumn does not solve from graphic connections
            # alone: it needs internal feed/spec registration.  Dump the parameter
            # lists of the key methods so they can be called correctly.
            try:
                import System.Reflection as _ref  # noqa: E402
                t = sim.GetType()
                for _mname in ("ConnectFeed", "ConnectFeedMaterialStream", "SetStreamFeedStage",
                               "SetCondenserSpec", "SetReboilerSpec", "SetNumberOfStages",
                               "SetRefluxRatio", "AddStages", "ConnectCondenserDuty", "ConnectReboilerDuty"):
                    _mis = t.GetMethods()
                    _hits = [_mi for _mi in _mis if _mi.Name == _mname]
                    if not _hits:
                        print(f"      {_mname}: <not found>")
                        continue
                    for _mi in _hits:
                        _parms = [f"{p.Name}:{p.ParameterType.Name}" for p in _mi.GetParameters()]
                        print(f"      {_mname}({', '.join(_parms)}) -> {_mi.ReturnType.Name}")
            except Exception as e:  # noqa: BLE001
                print(f"      reflection probe failed: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"    probe failed: {type(e).__name__}: {e}")


def _step(label: str, fn: object) -> None:
    try:
        fn()
        print(f"  [OK ] {label}")
    except Exception as _e:  # noqa: BLE001
        print(f"  [FAIL] {label} -> {type(_e).__name__}: {_e}")
        import traceback as _tb
        _tb.print_exc()


_step("probe column graphic ports", _probe_column_ports)

# --- Directly reproduce the export with per-phase bisection -----------------
# Build the same objects the export builds, so we can isolate exactly which call
# throws (instead of a single catch-all).
print("\n=== 分步复现导出 (bisect) ===")

_spec = design.spec
_components = ["ethanol", "water", _spec.entrainer]
_pp = ded._COLUMN_PROPERTY_PACKAGES[_spec.property_package]

def _b1() -> None:
    global _bs_fs
    _bs_fs = _probe_automation.CreateFlowsheet()
    for _n in _components:
        _bs_fs.AddCompound(ded._dwsim_compound_name(_n))
    ded._add_property_package(_bs_fs, _pp)

def _b2() -> None:
    global _bs_column_type, _bs_column
    _bs_column_type = ded._column_object_type(_probe_object_type)
    print(f"        列对象类型 -> {_bs_column_type}")

def _b3() -> None:
    global _bs_column
    _bs_column = _bs_fs.AddObject(_bs_column_type, 250, 0, "Extractive Distillation Column")

def _b4() -> None:
    global _bs_feed, _bs_entrainer, _bs_di, _bs_bo
    _bs_feed = _bs_fs.AddObject(_probe_object_type.MaterialStream, 0, -80, "Ethanol Water Feed")
    _bs_entrainer = _bs_fs.AddObject(_probe_object_type.MaterialStream, 0, 80, "Entrainer")
    _bs_di = _bs_fs.AddObject(_probe_object_type.MaterialStream, 500, -80, "Ethanol Product")
    _bs_bo = _bs_fs.AddObject(_probe_object_type.MaterialStream, 500, 80, "Water Entrainer Bottoms")

def _b5() -> None:
    _feed_s = ded._simulation_object(_bs_feed)
    _feed_s.SetTemperature(_spec.feed_temperature_K)
    _feed_s.SetPressure(_spec.feed_pressure_kPa * 1000.0)
    _feed_s.SetMolarFlow(_spec.feed_flow_mol_s)
    _feed_s.SetOverallComposition(ded._composition_argument(list(_spec.feed_composition) + [0.0]))

def _b6() -> None:
    _ent_s = ded._simulation_object(_bs_entrainer)
    _ent_s.SetTemperature(design.condenser_temperature_K)
    _ent_s.SetPressure(_spec.operating_pressure_kPa * 1000.0)
    _ent_s.SetMolarFlow(_spec.entrainer_ratio * _spec.feed_flow_mol_s)
    _ent_s.SetOverallComposition(ded._composition_argument([0.0, 0.0, 1.0]))

def _b7() -> None:
    ded._set_column_design(ded._simulation_object(_bs_column), design)

def _b8() -> None:
    # Use the production-computed entrainer port (=1 for DistillationColumn).
    _ent_port = 1 if _bs_column_type == _probe_object_type.DistillationColumn else 0
    _bs_fs.ConnectObjects(_bs_feed.GraphicObject, _bs_column.GraphicObject, 0, 0)
    _bs_fs.ConnectObjects(_bs_entrainer.GraphicObject, _bs_column.GraphicObject, 0, _ent_port)
    _bs_fs.ConnectObjects(_bs_column.GraphicObject, _bs_di.GraphicObject, 0, 0)
    _bs_fs.ConnectObjects(_bs_column.GraphicObject, _bs_bo.GraphicObject, 1, 0)

def _b9() -> None:
    dest = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive.dwxmz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet(_probe_automation, _bs_fs, dest)
    print(f"        保存文件 -> {dest} ({dest.stat().st_size} bytes)")


def _b8b() -> None:
    """Best-effort solver-side configuration of the rigorous column.

    A DistillationColumn's Validate() requires its internal feed streams and
    condenser/reboiler specs to be registered, not just graphic connections.  The
    reflection probe shows the authoritative signatures: ``ConnectFeed(feed:
    ISimulationObject, stagenumber: Int32)`` and ``ConnectFeedMaterialStream(
    stream, portnumber)``.  This step passes the *internal simulation object*
    (not the GraphicObject) as DWSIM expects.
    """
    col = ded._simulation_object(_bs_column)
    feed_sim = ded._simulation_object(_bs_feed)
    ent_sim = ded._simulation_object(_bs_entrainer)
    out: list[str] = []

    def _try(name: str, *args: object) -> None:
        try:
            method = getattr(col, name, None)
            if callable(method):
                method(*args)
                out.append(f"{name} [ok]")
            else:
                out.append(f"{name} [missing]")
        except Exception as e:  # noqa: BLE001
            out.append(f"{name} -> {type(e).__name__}: {e}")

    _try("set_NumberOfStages", design.theoretical_stages)
    _try("set_RefluxRatio", design.reflux_ratio)
    # Register feeds on their stages using the internal simulation object.
    _try("ConnectFeed", feed_sim, design.feed_stage)
    _try("ConnectFeed", ent_sim, design.entrainer_stage)
    # Also try setting feed stages via the stage-index API (if it accepts it).
    _try("SetStreamFeedStage", _bs_feed, design.feed_stage - 1)
    _try("SetStreamFeedStage", _bs_entrainer, design.entrainer_stage - 1)
    # Condenser / reboiler specs: probe both (specType, value) and single-arg len.
    _try("SetCondenserSpec", 0, design.reflux_ratio)
    _try("SetReboilerSpec", 0, 0.0)
    print(f"        solver config: {'; '.join(out)}")


_step("CreateFlowsheet + AddCompound + add property package", _b1)
_step("resolve column object type", _b2)
_step("AddObject column", _b3)
_step("AddObject 4 material streams", _b4)
_step("configure feed stream", _b5)
_step("configure entrainer stream", _b6)
_step("apply design numbers to column", _b7)
_step("solver-side config (feed stages + specs)", _b8b)
_step("connect objects", _b8)
_step("_save_flowsheet", _b9)

# --- Definitive check: the REAL production exporter (what the web page uses) --
print("\n=== 调用真实 export_dwsim_extractive_column (网页走的路径) ===")
from thermo_engine.dwsim_export import export_dwsim_extractive_column  # noqa: E402

_real_dest = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive_real.dwxmz"
try:
    _out = export_dwsim_extractive_column(design, _real_dest)
    print(f"  [OK ] 真实导出成功 -> {_out} ({_out.stat().st_size} bytes)")
except Exception as _e:  # noqa: BLE001
    print(f"  [FAIL] 真实导出失败 -> {type(_e).__name__}: {_e}")
    import traceback as _tb
    _tb.print_exc()

# --- Investigate what the rigorous column considers connected/missing ---------
# DWSIM's Column.Validate() raises "One or more stream connections is missing".
# To locate the missing connection, enumerate the column's internal feed stream
# list, stage count, and connection objects after graphic+solver configuration.
print("\n=== 调查严格塔内部认为已连接的流 (Validate 之后缺失项) ===")


def _investigate_validate() -> None:
    try:
        _iv_fs = _probe_automation.CreateFlowsheet()
        for _n in ("Ethanol", "Water", "Ethylene glycol"):
            _iv_fs.AddCompound(_n)
        ded._add_property_package(_iv_fs, "NRTL")
        _iv_col = _iv_fs.AddObject(_probe_object_type.DistillationColumn, 250, 0, "Inv Column")
        _iv_feed = _iv_fs.AddObject(_probe_object_type.MaterialStream, 0, -60, "Feed")
        _iv_ent = _iv_fs.AddObject(_probe_object_type.MaterialStream, 0, 60, "Entrainer")
        _iv_dist = _iv_fs.AddObject(_probe_object_type.MaterialStream, 500, -60, "Distillate")
        _iv_bot = _iv_fs.AddObject(_probe_object_type.MaterialStream, 500, 60, "Bottoms")

        col = ded._simulation_object(_iv_col)
        feed_sim = ded._simulation_object(_iv_feed)
        ent_sim = ded._simulation_object(_iv_ent)

        # Set stages + register feeds (same as production), but do NOT connect
        # graphics first so we see solver-only expectations.
        try:
            col.SetNumberOfStages(design.theoretical_stages)
            col.set_RefluxRatio(design.reflux_ratio)
            col.ConnectFeed(feed_sim, design.feed_stage)
            col.ConnectFeed(ent_sim, design.entrainer_stage)
            col.SetStreamFeedStage(feed_sim, design.feed_stage)
            col.SetStreamFeedStage(ent_sim, design.entrainer_stage)
            print("  configured stages+reflux+feeds [ok]")
        except Exception as e:  # noqa: BLE001
            print(f"  config error: {type(e).__name__}: {e}")

        # Make the four graphic connections (as the production export does) so the
        # Validate/Calculate test mirrors a fully-connected, realistic column.
        try:
            _iv_fs.ConnectObjects(_iv_feed.GraphicObject, _iv_col.GraphicObject, 0, 0)
            _iv_fs.ConnectObjects(_iv_ent.GraphicObject, _iv_col.GraphicObject, 0, 1)
            _iv_fs.ConnectObjects(_iv_col.GraphicObject, _iv_dist.GraphicObject, 0, 0)
            _iv_fs.ConnectObjects(_iv_col.GraphicObject, _iv_bot.GraphicObject, 1, 0)
            print("  graphic connections (feed, entrainer, distillate, bottoms) [ok]")
        except Exception as e:  # noqa: BLE001
            print(f"  graphic connection error: {type(e).__name__}: {e}")

        # --- Variant A test: graphic connections ONLY (no solver ConnectFeed) ----
        # Hypothesis: registering feeds via ConnectFeed/SetStreamFeedStage may
        # conflict with the graphic connections, causing Validate's "connection
        # missing".  Test a column set up purely by graphic connections.
        print("  [Variant A] 纯图形连接（不调用 ConnectFeed/SetStreamFeedStage）:")
        try:
            _va_fs = _probe_automation.CreateFlowsheet()
            for _n in ("Ethanol", "Water", "Ethylene glycol"):
                _va_fs.AddCompound(_n)
            ded._add_property_package(_va_fs, "NRTL")
            _va_col = _va_fs.AddObject(_probe_object_type.DistillationColumn, 250, 0, "A Column")
            _va_feed = _va_fs.AddObject(_probe_object_type.MaterialStream, 0, -60, "Feed")
            _va_ent = _va_fs.AddObject(_probe_object_type.MaterialStream, 0, 60, "Entrainer")
            _va_dist = _va_fs.AddObject(_probe_object_type.MaterialStream, 500, -60, "Distillate")
            _va_bot = _va_fs.AddObject(_probe_object_type.MaterialStream, 500, 60, "Bottoms")
            va_col = ded._simulation_object(_va_col)
            va_col.SetNumberOfStages(design.theoretical_stages)
            va_col.set_RefluxRatio(design.reflux_ratio)
            va_col.SetCondenserSpec("Reflux Ratio", float(design.reflux_ratio), "", "")
            va_col.SetReboilerSpec("Reboiler Duty", 1.0, "W", "")
            # graphic-only connections:
            _va_fs.ConnectObjects(_va_feed.GraphicObject, _va_col.GraphicObject, 0, 0)
            _va_fs.ConnectObjects(_va_ent.GraphicObject, _va_col.GraphicObject, 0, 1)
            _va_fs.ConnectObjects(_va_col.GraphicObject, _va_dist.GraphicObject, 0, 0)
            _va_fs.ConnectObjects(_va_col.GraphicObject, _va_bot.GraphicObject, 1, 0)
            try:
                saved = saved2 = None
                va_col.Calculate(None)
                print("      Calculate (graphic-only) -> OK")
            except Exception as e:  # noqa: BLE001
                print(f"      Calculate (graphic-only) -> {type(e).__name__}: {e}")
            _va_out = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive_variantA.dwxmz"
            _va_out.parent.mkdir(parents=True, exist_ok=True)
            ded._save_flowsheet(_probe_automation, _va_fs, _va_out)
            print(f"      保存 Variant A 文件 -> {_va_out}")
        except Exception as e:  # noqa: BLE001
            print(f"      Variant A failed: {type(e).__name__}: {e}")

        # --- Variant B: ShortcutColumn + single combined feed + Calculate ------
        # A ShortcutColumn only accepts ONE feed, but it can run Fenske/Underwood/
        # Gilliland to completion (no "connections missing").  Merge feed + entrainer
        # into one stream and see if DWSIM calculates a result.
        print("  [Variant B] 简捷塔 ShortcutColumn + 合并进料:")
        try:
            _vb_fs = _probe_automation.CreateFlowsheet()
            for _n in ("Ethanol", "Water", "Ethylene glycol"):
                _vb_fs.AddCompound(_n)
            ded._add_property_package(_vb_fs, "NRTL")
            _vb_col = _vb_fs.AddObject(_probe_object_type.ShortcutColumn, 250, 0, "B Column")
            _vb_feed = _vb_fs.AddObject(_probe_object_type.MaterialStream, 0, 0, "Combined Feed")
            _vb_dist = _vb_fs.AddObject(_probe_object_type.MaterialStream, 500, -60, "Distillate")
            _vb_bot = _vb_fs.AddObject(_probe_object_type.MaterialStream, 500, 60, "Bottoms")

            # Combined feed: ethanol + water + entrainer, merged from the two feeds.
            # Feed flow = feed + entrainer, composition blended proportionally.
            feed_frac = _spec.feed_flow_mol_s / (_spec.feed_flow_mol_s + _spec.entrainer_ratio * _spec.feed_flow_mol_s)
            ent_frac = 1.0 - feed_frac
            comp = [
                _spec.feed_composition[0] * feed_frac,          # ethanol
                _spec.feed_composition[1] * feed_frac,          # water
                ent_frac,                                        # entrainer
            ]
            tot = sum(comp)
            comp = [c / tot for c in comp]
            vb_stream = ded._simulation_object(_vb_feed)
            vb_stream.SetTemperature(_spec.feed_temperature_K)
            vb_stream.SetPressure(_spec.operating_pressure_kPa * 1000.0)
            vb_stream.SetMolarFlow(_spec.feed_flow_mol_s * (1.0 + _spec.entrainer_ratio))
            vb_stream.SetOverallComposition(ded._composition_argument(comp))

            # ShortcutColumn uses specs (separation task), not explicit stages/reflux.
            vb_col = ded._simulation_object(_vb_col)
            try:
                vb_col.set_RefluxRatio(design.reflux_ratio)
            except Exception:  # noqa: BLE001
                pass
            try:
                vb_col.SetNumberOfStages(design.theoretical_stages)
            except Exception:  # noqa: BLE001
                pass

            _vb_fs.ConnectObjects(_vb_feed.GraphicObject, _vb_col.GraphicObject, 0, 0)
            _vb_fs.ConnectObjects(_vb_col.GraphicObject, _vb_dist.GraphicObject, 0, 0)
            _vb_fs.ConnectObjects(_vb_col.GraphicObject, _vb_bot.GraphicObject, 1, 0)
            try:
                vb_col.Calculate(None)
                print("      Calculate (ShortcutColumn, merged feed) -> OK")
            except Exception as e:  # noqa: BLE001
                print(f"      Calculate (ShortcutColumn, merged feed) -> {type(e).__name__}: {e}")
            _vb_out = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive_shortcut.dwxmz"
            _vb_out.parent.mkdir(parents=True, exist_ok=True)
            ded._save_flowsheet(_probe_automation, _vb_fs, _vb_out)
            print(f"      保存 ShortcutColumn 文件 -> {_vb_out}")
        except Exception as e:  # noqa: BLE001
            print(f"      Variant B failed: {type(e).__name__}: {e}")

        # Enumerate internal feed/connection lists.
        for _attr in ("FeedStreams", "ConnectedStreams", "MiscStreams", "StageData",
                      "FeedConnections", "AllStreams", "GetFlowsheet", "FeedStreamIndexes"):
            try:
                _v = getattr(col, _attr)
                if callable(_v):
                    print(f"  {_attr}() -> {_v() if True else ''}")
                else:
                    print(f"  {_attr} = {_v}")
            except Exception as e:  # noqa: BLE001
                print(f"  {_attr}: {type(e).__name__}: {e}")

        # Probe feed-stage index resolution (0-based vs 1-based) and stage counts.
        print("  探测进料塔板索引解析 (GetStreamFeedStageIndex / Stages / NumberOfStages):")
        for _name, _sim, _stage in (("feed", feed_sim, design.feed_stage),
                                    ("entrainer", ent_sim, design.entrainer_stage),
                                    ("feed-1", feed_sim, design.feed_stage - 1),
                                    ("entrainer-1", ent_sim, design.entrainer_stage - 1)):
            try:
                col.SetStreamFeedStage(_sim, _stage)
                got = col.GetStreamFeedStageIndex(_sim)
                print(f"    SetStreamFeedStage({_name}, {_stage}) -> GetStreamFeedStageIndex = {got!r}")
            except Exception as e:  # noqa: BLE001
                print(f"    SetStreamFeedStage({_name}, {_stage}) -> {type(e).__name__}: {e}")
        for _attr in ("NumberOfStages", "get_NumberOfStages", "Stages"):
            try:
                v = getattr(col, _attr)
                cnt = getattr(v, "Count", None)
                if callable(v):
                    print(f"    {_attr}() = {v()!r}"[:160] if cnt is None else f"    {_attr}() Count={v().Count}")
                elif cnt is not None:
                    print(f"    {_attr}.Count = {cnt}")
                else:
                    print(f"    {_attr} = {v!r}"[:160])
            except Exception as e:  # noqa: BLE001
                print(f"    {_attr}: {type(e).__name__}: {e}")

        # Try invoking the column's own Calculate to surface the precise Validate error.
        print("  直接调用 col.Calculate(None) 复现 Validate 报错:")
        try:
            col.Calculate(None)
            print("    Calculate returned without error")
        except Exception as e:  # noqa: BLE001
            print(f"    Calculate -> {type(e).__name__}: {e}")

        # Show the connection topology as DWSIM's solver sees it.
        go = _iv_col.GraphicObject
        for _attr in ("InputConnectors", "OutputConnectors", "SpecialConnectors", "EnergyConnector"):
            try:
                _lst = getattr(go, _attr)
                if hasattr(_lst, "Count"):
                    print(f"  column.GraphicObject.{_attr} count = {_lst.Count}")
            except Exception as e:  # noqa: BLE001
                print(f"  {_attr}: {type(e).__name__}: {e}")

        # Probe which condenser/reboiler spec-type strings DWSIM accepts.
        print("  试探 SetCondenserSpec / SetReboilerSpec 的可接受 spec 字符串:")
        for _st in ("Reflux Ratio", "RefluxRatio", "Reflux ratio", "Distillate Flow Rate",
                    "Distillate Rate", "Bottoms Flow Rate", "Condenser Duty", "Reboiler Duty",
                    "Boilup Ratio", "Specified by User", "Product Purity"):
            try:
                col.SetCondenserSpec(_st, float(design.reflux_ratio), "", "")
                print(f"    SetCondenserSpec('{_st}') [ok]")
            except Exception as e:  # noqa: BLE001
                print(f"    SetCondenserSpec('{_st}') -> {type(e).__name__}")
            try:
                col.SetReboilerSpec(_st, 1.0, "W", "")
                print(f"    SetReboilerSpec('{_st}') [ok]")
            except Exception as e:  # noqa: BLE001
                print(f"    SetReboilerSpec('{_st}') -> {type(e).__name__}")

        # Inspect condenser-related properties that Validate() likely requires.
        for _attr in ("CondenserType", "get_CondenserType", "RefluxedAbsorber", "ReboiledAbsorber"):
            try:
                v = getattr(col, _attr)
                print(f"    {_attr} = {v!r}"[:160])
            except Exception as e:  # noqa: BLE001
                print(f"    {_attr}: {type(e).__name__}")

        # DWSIM rigorous columns need an energy stream on the condenser and reboiler
        # DWSIM rigorous columns need an energy stream on the condenser and reboiler
        # or Validate() reports "stream connections missing".  Probe an EnergyStream
        # object and ConnectCondenserDuty / ConnectReboilerDuty, then re-Calculate.
        print("  尝试建立 EnergyStream（探测 flowsheet API）:")
        try:
            _en = getattr(_probe_object_type, "EnergyStream", None)
        except Exception:  # noqa: BLE001
            _en = None
        _energy_members = [m for m in dir(_probe_object_type) if not m.startswith("_")
                           and any(k in m.lower() for k in ("energy", "utility", "heat"))]
        print(f"    ObjectType 能量成员: {_energy_members}")

        # Enumerate Flowsheet2 methods that may create energy streams.
        try:
            _fs_meths = [m for m in dir(_iv_fs) if any(k in m.lower() for k in ("energy", "stream", "addobject"))]
            print(f"    Flowsheet 能量/流相关方法: {_fs_meths}")
        except Exception as e:  # noqa: BLE001
            print(f"    flowsheet method probe failed: {type(e).__name__}: {e}")

        # Try several creation strategies for the energy streams.
        def _mk_energy(tag: str):
            fallbacks = []
            # 1) dedicated AddEnergyStream-style methods
            for _mname in ("AddEnergyStream", "NewEnergyStream", "CreateEnergyStream", "AddUtilityStream"):
                try:
                    m = getattr(_iv_fs, _mname)
                    if callable(m):
                        return m(tag)
                except Exception:  # noqa: BLE001
                    fallbacks.append(_mname)
            # 2) AddObject with the ObjectType.EnergyStream
            try:
                return _iv_fs.AddObject(getattr(_probe_object_type, "EnergyStream"), 250, 0, tag)
            except Exception as e:  # noqa: BLE001
                fallbacks.append(f"AddObject{type(e).__name__}")
            raise RuntimeError("no energy-stream creation path: " + ",".join(fallbacks))

        try:
            _cond_eng = _mk_energy("Condenser Duty")
            _reb_eng = _mk_energy("Reboiler Duty")
            col.ConnectCondenserDuty(ded._simulation_object(_cond_eng))
            col.ConnectReboilerDuty(ded._simulation_object(_reb_eng))
            print("    ConnectCondenserDuty/ConnectReboilerDuty [ok]")
            col.SetCondenserSpec("Reflux Ratio", float(design.reflux_ratio), "", "")
            col.SetReboilerSpec("Reboiler Duty", 1.0, "W", "")
            try:
                col.Calculate(None)
                print("    Calculate after EnergyStream -> returned without error")
            except Exception as e:  # noqa: BLE001
                print(f"    Calculate after EnergyStream -> {type(e).__name__}: {e}")
            try:
                _ev = Path(__file__).resolve().parent.parent / ".pytest-tmp" / "diag_extractive_energy.dwxmz"
                _ev.parent.mkdir(parents=True, exist_ok=True)
                ded._save_flowsheet(_probe_automation, _iv_fs, _ev)
                print(f"    保存 EnergyStream 版文件 -> {_ev}")
            except Exception as e:  # noqa: BLE001
                print(f"    保存失败: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"    energy connect failed: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  investigation failed: {type(e).__name__}: {e}")


_step("investigate rigorous column connections", _investigate_validate)
