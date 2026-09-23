"""Re-run the ThermoFormer column of the three-source tables for all three systems.

Systems (matching report/Agent整合ThermoFormer进度与三源验证报告v5.md §2):

  (1) 2-propanol / water                       -- binary VLE, isobaric bubble points
  (2) ethyl acetate / n-propyl acetate + DMSO  -- ternary VLE, isobaric bubble points
  (3) water / 1-butanol                        -- binary LLE coexistence endpoints

Every number is produced by the real ThermoFormer backend
(``thermo_engine.thermoformer_backend.ThermoFormerBackend``), which selects its
checkpoint through the bundled ``models/registry.json`` catalogue with
``THERMOFORMER_CHECKPOINT`` left empty (the registry seed_0 default).  This
script never computes an equilibrium value itself.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the registry default: THERMOFORMER_CHECKPOINT empty -> _select_checkpoint
os.environ.pop("THERMOFORMER_CHECKPOINT", None)
os.environ["THERMOFORMER_SRC"] = r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\src"
os.environ["THERMOFORMER_USE_CUDA"] = "0"  # torch in this env is CPU-only build

OUTDIR = ROOT / "report" / "dwsim"

SMILES = {
    "isopropanol": "CC(C)O",
    "water": "O",
    "ethyl acetate": "CCOC(=O)C",
    "n-propyl acetate": "CCCOC(=O)C",
    "dmso": "CS(=O)C",
    "1-butanol": "CCCCO",
}

P_ISO_KPA = 760.0 * 0.133322  # 760 mmHg in kPa

# §2.1 representative liquid mole fractions (x_IPA)
IPA_XS = [0.1, 0.3, 0.5, 0.7, 0.9]

# §2.2 six representative ternary compositions (x_etac, x_npac, x_dmso)
DMSO_XS = [
    (0.3735, 0.0244, 0.6021),
    (0.2195, 0.1790, 0.6015),
    (0.0595, 0.3336, 0.6069),
    (0.3540, 0.1384, 0.5076),
    (0.1819, 0.3137, 0.5044),
    (0.0323, 0.4649, 0.5028),
]

# §2.3 LLE temperatures
LLE_TEMPERATURES = [298.15, 313.15, 343.15, 353.15]


def _component(name: str, smiles: str):
    from schemas.domain import ComponentIdentity

    return ComponentIdentity(component_id=name, name=name, smiles=smiles)


def _manifest(components, *, equilibrium_type, calculation_type, conditions):
    from schemas.domain import TaskManifest

    return TaskManifest(
        task_id="three-source-rerun",
        calculation_type=calculation_type,
        equilibrium_type=equilibrium_type,
        components=components,
        conditions=conditions,
    )


def run_ipa():
    from schemas.domain import ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    rows = []
    for x in IPA_XS:
        req = _manifest(
            [_component("isopropanol", SMILES["isopropanol"]), _component("water", SMILES["water"])],
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            conditions=ThermodynamicConditions(pressure_kPa=P_ISO_KPA, liquid_composition=[x, 1.0 - x]),
        )
        res = backend.bubble_point(req)
        pt = res.points[0]
        rows.append({
            "x_ipa": x,
            "T_tf_C": round(float(pt.temperature_K) - 273.15, 3),
            "y_ipa_tf": round(float(pt.vapor_composition[0]), 4),
            "residual": f"{float(pt.equilibrium_residual):.3e}",
        })
        print(f"[ipa] x={x}: T_tf={rows[-1]['T_tf_C']:.2f} C  y_tf={rows[-1]['y_ipa_tf']:.4f}"
              f"  res={rows[-1]['residual']}")
    return rows


def run_dmso():
    from schemas.domain import ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    rows = []
    for xe, xn, xd in DMSO_XS:
        req = _manifest(
            [
                _component("ethyl acetate", SMILES["ethyl acetate"]),
                _component("n-propyl acetate", SMILES["n-propyl acetate"]),
                _component("dmso", SMILES["dmso"]),
            ],
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            conditions=ThermodynamicConditions(pressure_kPa=P_ISO_KPA, liquid_composition=[xe, xn, xd]),
        )
        res = backend.bubble_point(req)
        pt = res.points[0]
        y = [float(v) for v in pt.vapor_composition]
        rows.append({
            "x_etac": xe, "x_npac": xn, "x_dmso": xd,
            "T_tf_C": round(float(pt.temperature_K) - 273.15, 3),
            "y_etac_tf": round(y[0], 4),
            "y_npac_tf": round(y[1], 4),
            "y_dmso_tf": round(y[2], 4),
            "y_sum": round(sum(y), 6),
            "residual": f"{float(pt.equilibrium_residual):.3e}",
        })
        print(f"[dmso] x=({xe},{xn},{xd}): T_tf={rows[-1]['T_tf_C']:.2f} C"
              f"  y_tf=({y[0]:.4f},{y[1]:.4f},{y[2]:.4f})  res={rows[-1]['residual']}")
    return rows


def run_lle():
    from schemas.domain import ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    rows = []
    for T in LLE_TEMPERATURES:
        req = _manifest(
            [_component("1-butanol", SMILES["1-butanol"]), _component("water", SMILES["water"])],
            equilibrium_type="LLE",
            calculation_type="lle",
            conditions=ThermodynamicConditions(temperature_K=T, pressure_kPa=101.325),
        )
        res = backend.lle(req)
        phases = res.phases
        comps = [[float(v) for v in p.composition] for p in phases]
        # 1-butanol is index 0 -> organic (butanol-rich) = larger index-0 fraction
        org, aq = (comps[0], comps[1]) if comps[0][0] >= comps[1][0] else (comps[1], comps[0])
        rows.append({
            "T_K": T,
            "tf_x_butanol_organic": round(org[0], 4),
            "tf_x_water_organic": round(org[1], 4),
            "tf_x_butanol_aqueous": round(aq[0], 4),
            "tf_x_water_aqueous": round(aq[1], 4),
            "residual": f"{float(res.residual):.3e}",
        })
        print(f"[lle] T={T}: org={org[0]:.4f}  aq={aq[0]:.4f}  res={rows[-1]['residual']}")
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote: {path}")


def main() -> int:
    from thermo_engine.thermoformer_backend import resolve_settings

    settings = resolve_settings()
    print(f"THERMOFORMER_SRC = {settings.src_path}")
    print(f"THERMOFORMER_CHECKPOINT = {settings.checkpoint_path}  (None => registry default)")

    verb = sys.argv[1] if len(sys.argv) > 1 else "all"
    status = 0

    if verb in ("all", "ipa"):
        try:
            _write(OUTDIR / "tf_rerun_ipa_vle.csv", run_ipa())
        except Exception as exc:
            print(f"[ipa] FAILED {type(exc).__name__}: {exc}")
            status = 1

    if verb in ("all", "dmso"):
        try:
            _write(OUTDIR / "tf_rerun_dmso_vle.csv", run_dmso())
        except Exception as exc:
            print(f"[dmso] FAILED {type(exc).__name__}: {exc}")
            status = 1

    if verb in ("all", "lle"):
        try:
            _write(OUTDIR / "tf_rerun_water_butanol_lle.csv", run_lle())
        except Exception as exc:
            print(f"[lle] FAILED {type(exc).__name__}: {exc}")
            status = 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
