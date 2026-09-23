"""Regenerate the two column designs (report §1.5) with UNIFAC vs ThermoFormer.

Both cases use the *same* deterministic Fenske-Underwood-Gilliland short-cut; only
the thermodynamic source of the relative volatility and the bubble temperatures
differs:

    alpha_source="unifac"        -> temperature-dependent UNIFAC activity coefficients
    alpha_source="thermoformer"  -> ThermoFormer isobaric/isothermal bubble prediction

Case 1 (report §1.5, 2-propanol / water binary distillation)
    x_IPA = 0.5, 101.325 kPa, distillate purity 0.995, recovery 0.98, F = 1.0 mol/s.
    The report's §1.5 table is quoted at x_IPA = 0.3, so both are produced.

Case 2 (report §1.5, ethyl acetate / n-propyl acetate + DMSO extractive distillation)
    light = ethyl acetate, heavy = n-propyl acetate, entrainer = DMSO,
    key ratio 0.5:0.5, F = 1.0 mol/s, entrainer ratio 2.0, 101.325 kPa,
    purity 0.95 / recovery 0.90 (the ``export_flow_examples.py`` case-2 defaults,
    which are what report §1.5 tabulates).

Every number comes from ``thermo_engine.column_design``; this script computes no
equilibrium value itself.  Writes report/dwsim/column_design_dual_source.csv/.json.
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

# Registry default checkpoint (seed_0), CPU env -> no CUDA.
os.environ.pop("THERMOFORMER_CHECKPOINT", None)
os.environ["THERMOFORMER_SRC"] = r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\src"
os.environ["THERMOFORMER_USE_CUDA"] = "0"

from thermo_engine.column_design import (  # noqa: E402
    design_binary_distillation_column,
    design_ternary_extractive_column,
)

OUTDIR = ROOT / "report" / "dwsim"
P_KPA = 101.325
F_MOL_S = 1.0
SOURCES = ("unifac", "thermoformer")

# Case 1 product split (the binary-column defaults).
PURITY_C1 = 0.995
RECOVERY_C1 = 0.98
# Case 2 product split: the export_flow_examples.py case-2 defaults, which are the
# values report §1.5 tabulates (0.995/0.98 would give a different, larger column).
PURITY_C2 = 0.95
RECOVERY_C2 = 0.90

IPA_XS = (0.3, 0.5)
EAC_NPAC_DMSO = dict(light="ethyl acetate", heavy="n-propyl acetate", entrainer="dmso")


def case1(x_ipa: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in SOURCES:
        try:
            d = design_binary_distillation_column(
                ["isopropanol", "water"],
                [x_ipa, 1.0 - x_ipa],
                feed_flow_mol_s=F_MOL_S,
                feed_temperature_K=None,
                operating_pressure_kPa=P_KPA,
                distillate_purity_mole_fraction=PURITY_C1,
                recovery=RECOVERY_C1,
                alpha_source=src,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [case1 x={x_ipa} {src}] FAILED {type(exc).__name__}: {exc}")
            continue
        out[src] = {
            "alpha": d.relative_volatility,
            "N": d.theoretical_stages,
            "N_min": d.minimum_stages,
            "R": d.reflux_ratio,
            "R_min": d.minimum_reflux_ratio,
            "feed_stage": d.feed_stage,
            "T_cond_K": d.condenser_temperature_K,
            "T_reb_K": d.reboiler_temperature_K,
        }
        print(f"  [case1 x={x_ipa} {src}] alpha={d.relative_volatility:.4f} N={d.theoretical_stages} "
              f"Nmin={d.minimum_stages:.3f} R={d.reflux_ratio:.3f} Rmin={d.minimum_reflux_ratio:.3f} "
              f"feed={d.feed_stage} Tcond={d.condenser_temperature_K:.2f} Treb={d.reboiler_temperature_K:.2f}")
    return out


def case2() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in SOURCES:
        try:
            d = design_ternary_extractive_column(
                EAC_NPAC_DMSO["light"],
                EAC_NPAC_DMSO["heavy"],
                EAC_NPAC_DMSO["entrainer"],
                [0.5, 0.5],
                feed_flow_mol_s=F_MOL_S,
                operating_pressure_kPa=P_KPA,
                distillate_purity_mole_fraction=PURITY_C2,
                recovery=RECOVERY_C2,
                entrainer_ratio=2.0,
                alpha_source=src,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [case2 {src}] FAILED {type(exc).__name__}: {exc}")
            continue
        out[src] = {
            "alpha_base": d["alpha_base"],
            "alpha_ext": d["alpha_ext"],
            "selectivity": d["selectivity"],
            "alpha_avg": d["alpha_avg"],
            "N": d["theoretical_stages"],
            "N_min": d["minimum_stages"],
            "R": d["reflux_ratio"],
            "R_min": d["minimum_reflux_ratio"],
            "feed_stage": d["feed_stage"],
            "entrainer_stage": d["entrainer_stage"],
            "T_cond_K": d["condenser_temperature_K"],
            "T_reb_K": d["reboiler_temperature_K"],
            "T_feed_K": d["feed_temperature_K"],
        }
        print(f"  [case2 {src}] a_base={d['alpha_base']} a_ext={d['alpha_ext']} sel={d['selectivity']} "
              f"a_avg={d['alpha_avg']} N={d['theoretical_stages']} Nmin={d['minimum_stages']} "
              f"R={d['reflux_ratio']} Rmin={d['minimum_reflux_ratio']} "
              f"Tcond/Treb={d['condenser_temperature_K']}/{d['reboiler_temperature_K']}")
    return out


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote: {path}")


def main() -> int:
    print("=== case 1: 2-propanol / water, binary distillation ===")
    c1 = {f"x_ipa={x}": case1(x) for x in IPA_XS}

    print("\n=== case 2: ethyl acetate / n-propyl acetate + DMSO, extractive ===")
    c2 = case2()

    # Flat comparison table: one row per (case, quantity).
    rows = []
    for x in IPA_XS:
        key = f"x_ipa={x}"
        u, t = c1.get(key, {}).get("unifac"), c1.get(key, {}).get("thermoformer")
        for q, k in (
            ("alpha", "alpha"), ("N", "N"), ("N_min", "N_min"), ("R", "R"),
            ("R_min", "R_min"), ("feed_stage", "feed_stage"),
            ("T_cond_K", "T_cond_K"), ("T_reb_K", "T_reb_K"),
        ):
            rows.append({
                "case": "1_binary_ipa_water", "feed": key, "quantity": q,
                "unifac": u.get(k) if u else "",
                "thermoformer": t.get(k) if t else "",
            })
    u2, t2 = c2.get("unifac"), c2.get("thermoformer")
    for q in ("alpha_base", "alpha_ext", "selectivity", "alpha_avg", "N", "N_min",
              "R", "R_min", "feed_stage", "entrainer_stage", "T_cond_K", "T_reb_K", "T_feed_K"):
        rows.append({
            "case": "2_extractive_eac_npac_dmso", "feed": "0.5:0.5, S/F=2.0", "quantity": q,
            "unifac": u2.get(q) if u2 else "",
            "thermoformer": t2.get(q) if t2 else "",
        })

    _write_rows(OUTDIR / "column_design_dual_source.csv", rows)
    (OUTDIR / "column_design_dual_source.json").write_text(
        json.dumps({"case1_binary_ipa_water": c1, "case2_extractive_eac_npac_dmso": c2},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote: {OUTDIR / 'column_design_dual_source.json'}")
    return 0 if (t2 or c1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
