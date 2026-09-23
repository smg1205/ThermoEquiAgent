"""Generate DWSIM processing + .dwxmz files for the three ternary demonstration systems.

Three systems, matching lunwen/Agent整合ThermoFormer进度与三源验证报告v4.md §2:

  (1) 2-propanol / water                    -- binary VLE isobaric bubble points.
  (2) ethyl acetate / n-propyl acetate + DMSO -- ternary VLE isobaric bubble points
                                                  (extractive-distillation solvent).
  (3) ethanol / ethyl acetate / water       -- ternary LLE liquid-liquid extractor.

For (1) and (2) this script reproduces the DWSIM column of the three-source
(experiment / ThermoFormer / DWSIM) comparison tables: it reads the existing
`docs/prediction_*.csv` (experiment + ThermoFormer already present), finds the
DWSIM isobaric bubble temperature by bisection on vapor molar flow, saves one
`.dwxmz` TP-flash flowsheet per representative composition, and writes a
completed `docs/dwsim_*_bubble.csv` that carries the missing DWSIM columns.

For (3) it shells out to the existing rigorous liquid-liquid-extractor builder
``scripts/generate_lle_ethanol_eac_water_dwsim.py`` to produce the extractor
`.dwxmz` (NRTL + explicit ethanol/ethyl-acetate/water BIPs + VLLE flash +
Burningham-Otto initial estimates).

All equilibrium numbers are computed by DWSIM, never by this script.  The feed
compositions / temperature bounds below are operating inputs and solver bounds.

Run (real terminal, full-access so pythonnet can initialize):

    conda activate thermo
    python scripts/generate_ternary_dwsim_demonstration.py \
        --outdir lunwen/dwsim_demonstration \
        --write-files

Prerequsite: DWSIM_HOME set (default C:\\Users\\34861\\AppData\\Local\\DWSIM) and
``pythonnet`` importable in the active environment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thermo_engine import dwsim_export as ded  # noqa: E402

P_PA = 760.0 * 133.322  # 760 mmHg -> Pa


# ---------------------------------------------------------------------------
# (1) 2-propanol / water -- binary isobaric bubble points.
# ---------------------------------------------------------------------------
IPA_SYSTEM = {
    "name": "ipa_water",
    "input_csv": _REPO / "docs" / "prediction_ipa_water.csv",
    "output_csv": _REPO / "docs" / "dwsim_ipa_bubble.csv",
    # representative liquid mole fractions (x_IPA) from the report §2.1
    "xs": [0.1, 0.3, 0.5, 0.7, 0.9],
    "t_min_c": 60.0,
    "t_max_c": 105.0,
    "compounds": [
        ("IPA", ["Isopropanol", "2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]),
        ("water", ["Water"]),
    ],
}


# ---------------------------------------------------------------------------
# (2) ethyl acetate / n-propyl acetate + DMSO -- ternary isobaric bubble points.
# ---------------------------------------------------------------------------
DMSO_SYSTEM = {
    "name": "dmso",
    "input_csv": _REPO / "docs" / "prediction_ternary_dms.csv",
    "output_csv": _REPO / "docs" / "dwsim_dmso_bubble.csv",
    "t_min_c": 60.0,
    "t_max_c": 130.0,
    "compounds": [
        ("ethyl acetate", ["Ethyl acetate"]),
        ("n-propyl acetate", ["N-propyl acetate", "n-Propyl acetate", "Propyl acetate"]),
        ("DMSO", ["Dimethyl sulfoxide", "DMSO"]),
    ],
    # 6 representative compositions from the report §2.2 table 2.2-1 (exact rows).
    "x_key": ("x_etac", "x_npac", "x_dmso"),
}


# ---------------------------------------------------------------------------
# shared DWSIM build helpers.
# ---------------------------------------------------------------------------
class _Globals:
    automation = None
    object_type = None
    resolved: dict[str, str] = {}


_GLOBALS = _Globals()


def _resolve(fs, logical_name: str, candidates: list[str]) -> str:
    """Add the compound to THIS flowsheet and return the canonical name.

    Every fresh flowsheet needs its own AddCompound call; caching the resolved
    name across flowsheets without re-adding leaves later flowsheets empty and
    collapses every flash to a zero-vapor (nan-bubble) result.
    """
    for cand in candidates:
        try:
            fs.AddCompound(cand)
            _GLOBALS.resolved[logical_name] = cand
            return cand
        except Exception:
            continue
    raise RuntimeError(f"compound not resolved: {logical_name} ({candidates})")


def _build_flash(x: list[float], t_c: float) -> tuple:
    fs = _GLOBALS.automation.CreateFlowsheet()
    for logical, cands in _GLOBALS.system["compounds"]:
        _resolve(fs, logical, cands)
    ded._add_property_package(fs, _GLOBALS.property_package)
    feed = fs.AddObject(_GLOBALS.object_type.MaterialStream, 0, 0, "Feed")
    sep = fs.AddObject(_GLOBALS.object_type.Vessel, 450, 0, "Flash")
    vapor = fs.AddObject(_GLOBALS.object_type.MaterialStream, 750, -80, "Vapor")
    lig = fs.AddObject(_GLOBALS.object_type.MaterialStream, 750, 80, "Liquid")
    f0 = ded._simulation_object(feed)
    f0.SetTemperature(t_c + 273.15)
    f0.SetPressure(P_PA)
    f0.SetMolarFlow(1.0)
    f0.SetOverallComposition(ded._composition_argument(x))
    # Drive the vessel as an isothermal-isobaric (PT) flash at the feed's own T/P.
    # DWSIM's Vessel defaults FlashTemperature=298.15 K, so without this the flash
    # runs at the wrong temperature and the bubble bisection collapses to nan.
    sep_obj = ded._simulation_object(sep)
    try:
        sep_obj.FlashTemperature = t_c + 273.15
        sep_obj.FlashPressure = P_PA
    except Exception:
        pass
    fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)
    return fs, ded._simulation_object(vapor)


#: Vapor molar flow at or below this is DWSIM's exact single-phase answer.  DWSIM
#: returns a hard 0.0 for a sub-cooled liquid, so this only absorbs float noise.
_VF_ZERO_TOL = 1e-12

#: Upper bound for an "incipient vapor" split.  The saved file should show a
#: sliver of vapor, because that is the visual proof that the temperature sits on
#: the bubble curve: a strictly single-phase file (V/F == 0) is indistinguishable
#: from a flash sitting several kelvin below the bubble point.  This is a
#: *ceiling*, not a target -- see :func:`_bubble`.
_VF_INCIPIENT_MAX = 1e-4

#: Bisection stopping width on temperature (K).
_T_BISECT_TOL = 1e-9

#: Maximum bisection iterations (interval 45 K halved 48 times ~ 1.6e-13 K).
_T_BISECT_MAX_ITER = 48

#: Temperature step (K) used when walking up from the bubble point to find the
#: first representable vapor split.  Chosen from the measured V/F slope: for
#: heptane / nonane the split grows ~0.063 per K, so ~0.001 K already yields
#: V/F ~ 6e-5, comfortably inside the default 1e-4 ceiling.
_T_PROBE_STEP = 0.001

#: How far above the bubble point the walk is allowed to travel (K) before it
#: gives up and reports V/F = 0.
_T_PROBE_SPAN = 0.5


def _bubble(x: list[float], vf_max: float = _VF_INCIPIENT_MAX) -> tuple[float, list[float], float]:
    """Locate the bubble point of ``x`` and return a state with incipient vapor.

    Solves for the **hottest temperature whose flash has V/F == 0** (that is the
    bubble point proper), then reports the first positive vapor split above it --
    the smallest one DWSIM will actually represent.

    Why incipient vapor rather than V/F = 0 or a fixed target:

    * ``V/F == 0`` (strictly single phase) destroys the evidence you need.  A
      flash 3 K below the bubble point and one exactly on it both report 0, so the
      file cannot show that the temperature is on the curve.
    * A *fixed* target is not always reachable.  DWSIM's flash quantises the
      vapor split -- it snaps to 0 below a minimum detectable amount, then jumps.
      Measured for 2-propanol / water at 760 mmHg, the smallest representable
      positive V/F is roughly 3.5e-12 at x_IPA = 0.1 and 6.9e-10 at x = 0.9, but
      it explodes near the minimum-boiling azeotrope (x ~ 0.68): **6.9e-3 at
      x = 0.5 and 2.5e-1 at x = 0.7**.  Asking for 1e-4 there is impossible, and
      the flash simply floors to 0 again.
    * Therefore we ask for "positive but as small as possible, and never more
      than ``vf_max``", and record which of the two happened.

    Bisection tracks the boundary between zero and non-zero vapor: V/F rises with
    T, so we keep the hottest zero-vapor temperature in ``lo`` and pull ``hi``
    down whenever vapor appears.

    The returned temperature is the *probe* temperature -- the bubble point plus
    the few thousandths of a kelvin needed to make the split representable -- not
    the exact V/F = 0 bubble point.  That is deliberate: the caller names the file
    and re-runs the flash at this temperature, so the saved flowsheet really does
    carry the incipient vapor it advertises, instead of collapsing back to a
    single-phase answer.  When no split at or below ``vf_max`` is reachable the
    exact bubble point is returned instead, with ``vapor_fraction == 0.0``.

    Returns ``(T_C, y, vapor_fraction)``.  ``vapor_fraction`` is 0.0 when even the
    smallest split exceeds ``vf_max`` (only possible hard against an azeotrope).
    """
    lo = float(_GLOBALS.system["t_min_c"])
    hi = float(_GLOBALS.system["t_max_c"])

    vf_lo, y_lo = _flash_at(x, lo)
    if vf_lo != 0.0:
        raise RuntimeError(
            f"bubble point bracket invalid: T={lo} C is already two-phase "
            f"(V/F={vf_lo:.3e}); lower t_min_c for system "
            f"{_GLOBALS.system['name']!r}"
        )

    # Phase 1: find the bubble point -- the hottest T that is still single-phase.
    t_bubble, y_bubble = lo, y_lo
    for _ in range(_T_BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        vf, y_mid = _flash_at(x, mid)
        if vf > 0.0:
            hi = mid                       # vapor present -> at/above bubble T
        else:
            lo = mid                       # single phase -> at/below bubble T
            t_bubble, y_bubble = mid, y_mid
        if hi - lo < _T_BISECT_TOL:
            break

    # Phase 2: walk up from the converged bubble point to catch the incipient
    # vapor.  Phase 1 has already squeezed the bracket to _T_BISECT_TOL, so there
    # is nothing left to bisect *inside* it -- the first representable split lies
    # *above* the bubble point, and the only way to find it is to probe upward in
    # fixed temperature steps until DWSIM returns a positive V/F that is still at
    # or below vf_max.  Stepping (rather than bisecting) also side-steps the
    # non-monotonic float noise seen within ~0.005 K of the bubble point.
    vf_inc, y_inc = 0.0, y_bubble
    t_inc = t_bubble
    t_probe, t_limit = t_bubble, t_bubble + _T_PROBE_SPAN
    while t_probe < t_limit:
        t_probe += _T_PROBE_STEP
        vf, y_mid = _flash_at(x, t_probe)
        if vf == 0.0:
            continue                       # still sub-cooled: keep walking up
        if vf <= vf_max:
            vf_inc, y_inc, t_inc = vf, y_mid, t_probe   # accept: incipient split
            break
        # Above the ceiling.  Do NOT give up here: within ~0.005 K of the bubble
        # point DWSIM's split is not monotonic (float quantisation), so a slightly
        # higher temperature can still land back inside the window -- measured for
        # heptane / nonane at x = 0.837, V/F is 2.5e-4 at +0.001 K but 5.0e-5 at
        # +0.003 K.  Keep walking; only a full sweep with no hit means the window
        # is unreachable.
        continue

    if vf_inc > 0.0:
        return t_inc, y_inc, vf_inc
    # No representable split under vf_max: report the bubble point itself (V/F = 0)
    # and let the caller surface that as "jumped clean over the window".
    return t_bubble, y_bubble, 0.0


def _flash_at(x: list[float], t_c: float) -> tuple[float, list[float]]:
    """Flash ``x`` at ``t_c`` and return ``(vapor_molar_flow, vapor_composition)``.

    DWSIM populates the vapor stream's equilibrium composition even when its
    molar flow is zero, and that composition is the bubble-point ``y`` the
    three-source tables quote, so it is read from the vapor stream regardless of
    the flow.  Only if DWSIM returns nothing usable (a NaN or a short array) do
    we fall back to the feed composition -- the correct ``V/F -> 0`` limit.
    """
    fs, vapor = _build_flash(x, t_c)
    _GLOBALS.automation.CalculateFlowsheet2(fs)
    vf = float(vapor.GetMolarFlow())
    try:
        y = [float(c) for c in list(vapor.GetOverallComposition())]
    except Exception:
        y = []
    if len(y) != len(x) or any(v != v for v in y):  # wrong length or NaN
        y = [float(v) for v in x]
    return vf, y


def _bubble_with_file(
    x: list[float], tag: str, outdir: Path, vf_max: float = _VF_INCIPIENT_MAX
) -> tuple[float, list[float], str]:
    """Solve the bubble point, then save the *converged* flowsheet for it.

    The saved file re-runs the flash at the converged temperature and re-checks
    the vapor split, so a file is never written under a "bubble point" name while
    actually containing a strongly two-phase flash.
    """
    tb, y, vf = _bubble(x, vf_max)
    fname = f"{tag}_{len(y)}comp_bubble_{tb:.1f}C.dwxmz".replace(" ", "_").replace("nan_", "")
    fs, vapor = _build_flash(x, tb)
    _GLOBALS.automation.CalculateFlowsheet2(fs)
    vf_saved = float(vapor.GetMolarFlow())
    if vf_saved > max(vf_max, 1e-11):
        raise RuntimeError(
            f"refusing to save {fname}: the re-run flash splits V/F={vf_saved:.3e} "
            f"at T={tb} C, above the incipient ceiling {vf_max:.0e}"
        )
    ded._save_flowsheet_via_temp(_GLOBALS.automation, fs, outdir / fname)
    return tb, y, fname


# ---------------------------------------------------------------------------
# system (1): read the binary CSV columns and go.
# ---------------------------------------------------------------------------
def _proximity_report(x: list[float], t_c: float, vf: float) -> str:
    """Describe how close the solved point sits to the true V/F = 0 bubble curve.

    The saved file carries a small non-zero V/F so the GUI shows a vapor trace,
    which raises the obvious question: is that trace the bubble point, or is the
    temperature merely somewhere below it?  The file alone cannot tell you.  This
    probes the flash either side of the solved temperature and reports:

    * ``dT_incipient`` -- how far the temperature had to move off the exact
      bubble point to produce the vapor trace.  Small means the trace *is* the
      bubble point; large means the trace is a real (if small) two-phase split.
    * the next representable vapor split above zero, which is what reveals that
      DWSIM quantises V/F: near the azeotrope the flash jumps from 0 straight to
      a large value, so no small trace exists at all.
    """
    dt = 0.01  # K
    vf_lo, _ = _flash_at(x, t_c - dt)
    vf_hi, _ = _flash_at(x, t_c + dt)
    slope = (vf_hi - vf_lo) / (2.0 * dt)

    # Walk up from the bubble point to find the first *representable* split, to
    # expose the quantisation gap described above.
    t_probe = t_c
    first = float("nan")
    for _ in range(60):
        t_probe += 1e-4
        v, _y = _flash_at(x, t_probe)
        if v > 0.0:
            first = v
            break
    parts = [f"dV/F/dT={slope:.3e} /K"]
    if vf > 0.0:
        parts.append(f"trace={vf:.3e}")
    else:
        parts.append("trace=0 (no representable split below ceiling)")
    if first == first:
        parts.append(f"next-split={first:.3e}")
    return "  ".join(parts)


def run_ipa(outdir: Path, write_files: bool, property_package: str) -> list[dict]:
    system = IPA_SYSTEM
    _GLOBALS.system = system
    _GLOBALS.property_package = property_package
    rows = []
    for x in system["xs"]:
        comp = [x, 1.0 - x]
        if write_files:
            tb, y, fname = _bubble_with_file(comp, f"ipa_x{x:g}", outdir)
        else:
            tb, y, _vf = _bubble(comp)
            fname = ""
        _vf_solved, _ = _flash_at(comp, tb)
        rows.append({
            "x_ipa": round(x, 4),
            "T_dwsim_C": round(tb, 2),
            "VF": round(_vf_solved, 10),
            "y_ipa_dwsim": round(float(y[0]), 4) if len(y) else float("nan"),
            "file": fname,
        })
        print(
            f"[ipa] x={x}: T_dwsim={tb:.2f} C  V/F={_vf_solved:.4e}  "
            f"y_ipa={y[0] if y else float('nan'):.4f}  | {_proximity_report(comp, tb, _vf_solved)}"
        )
    _write_csv(system["output_csv"], rows)
    return rows


# ---------------------------------------------------------------------------
# system (2): read the ternary CSV, pick representatives, write file.
# ---------------------------------------------------------------------------
def run_dmso(outdir: Path, write_files: bool, property_package: str) -> list[dict]:
    system = DMSO_SYSTEM
    _GLOBALS.system = system
    _GLOBALS.property_package = property_package
    x_etac_k, x_npac_k, x_dmso_k = system["x_key"]
    with open(system["input_csv"], newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    n = len(all_rows)
    # 6 representatives matching the report's chosen points, evenly spread.
    idxs = sorted(set(int(round(i * (n - 1) / 5.0)) for i in range(6)))
    reps = [all_rows[i] for i in idxs]
    rows = []
    for e in reps:
        x = [float(e[x_etac_k]), float(e[x_npac_k]), float(e[x_dmso_k])]
        tag = f"dmso_etac{x[0]:.3f}".replace(".", "p")
        if write_files:
            tb, y, fname = _bubble_with_file(x, tag, outdir)
        else:
            tb, y, _vf = _bubble(x)
            fname = ""
        y0 = float(y[0]) if len(y) >= 1 else float("nan")
        y1 = float(y[1]) if len(y) >= 2 else float("nan")
        y2 = float(y[2]) if len(y) >= 3 else float("nan")
        _vf_solved, _ = _flash_at(x, tb)
        rows.append({
            "x_etac": round(x[0], 4),
            "x_npac": round(x[1], 4),
            "x_dmso": round(x[2], 4),
            "T_exp_C": e["T_exp_C"],
            "T_tf_C": e["T_tf_C"],
            "T_dwsim_C": round(tb, 2),
            "VF": round(_vf_solved, 8),
            "y_etac_dwsim": round(y0, 4),
            "y_npac_dwsim": round(y1, 4),
            "y_dmso_dwsim": round(y2, 4),
            "file": fname,
        })
        print(f"[dmso] x={[round(v,4) for v in x]}  T_exp={e['T_exp_C']}  "
              f"T_tf={e['T_tf_C']}  T_dwsim={tb:.2f}  V/F={_vf_solved:.3e}  "
              f"y=[{y0:.4f},{y1:.4f},{y2:.4f}]  | {_proximity_report(x, tb, _vf_solved)}")
    _write_csv(system["output_csv"], rows)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote: {path}")


# ---------------------------------------------------------------------------
# system (3): ethanol / ethyl acetate / water -- LLE extractor .dwxmz.
# ---------------------------------------------------------------------------
def run_lle(outdir: Path, stages: int, calc: bool) -> Path:
    import importlib.util

    lle_script = _REPO / "scripts" / "generate_lle_ethanol_eac_water_dwsim.py"
    spec = importlib.util.spec_from_file_location("_lle_gen", lle_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    destination = outdir / "ethanol_eac_water_lle_extractor_nrtl.dwxmz"
    # Reuse the builder's own main by monkey-arg style: easier to call directly.
    # We drive it via subprocess to honour its argparse + save logic exactly.
    import subprocess

    cmd = [
        sys.executable,
        str(lle_script),
        "--out",
        str(destination),
        "--stages",
        str(stages),
    ]
    if calc:
        cmd.append("--calc")
    print("Running LLE builder:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    return destination


def main() -> int:
    global _VF_INCIPIENT_MAX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_REPO / "lunwen" / "dwsim_demonstration",
        help="Directory where the .dwxmz files are written.",
    )
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Also save one .dwxmz flash flowsheet per representative point (case 1 & 2).",
    )
    parser.add_argument(
        "--vf-max",
        type=float,
        default=_VF_INCIPIENT_MAX,
        help=(
            "Ceiling on the vapor fraction the saved bubble-point file may show. "
            f"The default ({_VF_INCIPIENT_MAX:.0e}) keeps a visible trace of vapor "
            "so the GUI proves the temperature sits on the bubble curve. Note "
            "DWSIM quantises V/F: hard against an azeotrope the smallest "
            "representable split can exceed this ceiling, in which case the point "
            "is saved single-phase and reported as such."
        ),
    )
    parser.add_argument(
        "--property-package",
        default="NRTL",
        choices=("NRTL", "UNIQUAC", "Wilson"),
        help="Property package for the case-1/2 bubble-point flashes.",
    )
    parser.add_argument(
        "--skip-lle",
        action="store_true",
        help="Skip the ternary LLE extractor build (case 3).",
    )
    parser.add_argument(
        "--lle-stages",
        type=int,
        default=8,
        help="Number of stages for the case-3 liquid-liquid extractor.",
    )
    parser.add_argument(
        "--lle-calc",
        action="store_true",
        help="Run the DWSIM solve on the case-3 extractor before saving.",
    )
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # The incipient-vapor ceiling is module state because _bubble() is called from
    # the per-system runners; set it once here from the CLI.
    if args.vf_max <= 0:
        parser.error("--vf-max must be positive")
    _VF_INCIPIENT_MAX = float(args.vf_max)
    print(f"bubble-point incipient-vapor ceiling: V/F <= {_VF_INCIPIENT_MAX:.3e}")

    factory, object_type = ded._automation_factory()
    _GLOBALS.automation = factory()
    _GLOBALS.object_type = object_type

    print("=== (1) 2-propanol / water -- DWSIM bubble points ===")
    run_ipa(outdir, args.write_files, args.property_package)

    print("\n=== (2) ethyl acetate / n-propyl acetate + DMSO -- DWSIM bubble points ===")
    run_dmso(outdir, args.write_files, args.property_package)

    if not args.skip_lle:
        print("\n=== (3) ethanol / ethyl acetate / water -- DWSIM LLE extractor ===")
        dest = run_lle(outdir, args.lle_stages, args.lle_calc)
        print(f"LLE extractor wrote: {dest}")

    print("\nDone. Files written under:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
