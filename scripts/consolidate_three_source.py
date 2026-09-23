"""Consolidate the re-run three-source tables for all three report systems.

Inputs (all freshly produced in this session, nothing copied by hand):

  system (1) 2-propanol / water, binary VLE
      exp    : docs/prediction_ipa_water.csv  (T_exp_C, y_ipa_exp)
      TF     : report/dwsim/tf_rerun_ipa_vle.csv          (seed_0, this run)
      TF old : docs/prediction_ipa_vle.csv T_tf_C column  (external, not reproducible)
      DWSIM  : docs/dwsim_ipa_bubble.csv                  (this run)

  system (2) ethyl acetate / n-propyl acetate + DMSO, ternary VLE
      exp    : docs/prediction_ternary_dms.csv (T_exp_C, y_*_exp)
      TF     : report/dwsim/tf_rerun_dmso_vle.csv         (seed_0, this run)
      TF old : docs/prediction_ternary_dms.csv T_tf_C column (external)
      DWSIM  : docs/dwsim_dmso_bubble.csv                 (this run)

  system (3) water / 1-butanol, binary LLE
      exp    : datasets/lle/binary_lle.csv, aggregated over replicate tie-lines
      TF     : report/dwsim/tf_rerun_water_butanol_lle.csv (seed_0, this run)
      DWSIM  : report/dwsim/water_butanol_dwsim.csv        (this run)

Writes report/dwsim/three_source_<system>.csv plus a combined JSON summary.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUTDIR = ROOT / "report" / "dwsim"
EXP_LLE = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\datasets\lle\binary_lle.csv")


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote: {path}")


def system1() -> list[dict]:
    """2-propanol / water, isobaric bubble point at 760 mmHg."""
    exp_rows = {round(float(r["x_ipa"]), 4): r for r in _rows(DOCS / "prediction_ipa_water.csv")}
    tf = {float(r["x_ipa"]): r for r in _rows(OUTDIR / "tf_rerun_ipa_vle.csv")}
    dw = {float(r["x_ipa"]): r for r in _rows(DOCS / "dwsim_ipa_bubble.csv")}

    out = []
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        # experimental x grid in the source CSV is not exactly 0.1/0.3/... ;
        # pick the closest experimental row so the comparison stays same-point.
        key = min(exp_rows, key=lambda k: abs(k - x))
        e = exp_rows[key]
        t, d = tf[x], dw[x]
        out.append({
            "x_ipa": x,
            "x_ipa_exp_source": key,
            "T_exp_C": float(e["T_exp_C"]),
            "y_ipa_exp": float(e["y_ipa_exp"]),
            "T_tf_C": float(t["T_tf_C"]),
            "y_ipa_tf": float(t["y_ipa_tf"]),
            "tf_residual_kpa": t["residual"],
            "T_dwsim_C": float(d["T_dwsim_C"]),
            "y_ipa_dwsim": float(d["y_ipa_dwsim"]),
            "dwsim_file": d["file"],
            "T_tf_old_external_C": float(e["T_tf_C"]),
            "y_ipa_tf_old_external": float(e["y_ipa_tf"]),
        })
    return out


def system2() -> list[dict]:
    """ethyl acetate / n-propyl acetate + DMSO, isobaric bubble point at 760 mmHg."""
    exp_rows = {
        (round(float(r["x_etac"]), 4), round(float(r["x_npac"]), 4), round(float(r["x_dmso"]), 4)): r
        for r in _rows(DOCS / "prediction_ternary_dms.csv")
    }
    tf = {
        (round(float(r["x_etac"]), 4), round(float(r["x_npac"]), 4), round(float(r["x_dmso"]), 4)): r
        for r in _rows(OUTDIR / "tf_rerun_dmso_vle.csv")
    }
    dw = {
        (round(float(r["x_etac"]), 4), round(float(r["x_npac"]), 4), round(float(r["x_dmso"]), 4)): r
        for r in _rows(DOCS / "dwsim_dmso_bubble.csv")
    }

    out = []
    for key in list(tf):
        e, t, d = exp_rows.get(key), tf[key], dw.get(key)
        if e is None or d is None:
            print(f"  [warn] system2 missing exp/dwsim for {key}")
            continue
        out.append({
            "x_etac": key[0], "x_npac": key[1], "x_dmso": key[2],
            "T_exp_C": float(e["T_exp_C"]),
            "y_etac_exp": float(e["y_etac_exp"]),
            "y_npac_exp": float(e["y_npac_exp"]),
            "T_tf_C": float(t["T_tf_C"]),
            "y_etac_tf": float(t["y_etac_tf"]),
            "y_npac_tf": float(t["y_npac_tf"]),
            "y_dmso_tf": float(t["y_dmso_tf"]),
            "tf_residual_kpa": t["residual"],
            "T_dwsim_C": float(d["T_dwsim_C"]),
            "y_etac_dwsim": float(d["y_etac_dwsim"]),
            "y_npac_dwsim": float(d["y_npac_dwsim"]),
            "y_dmso_dwsim": float(d["y_dmso_dwsim"]),
            "dwsim_file": d["file"],
            "T_tf_old_external_C": float(e["T_tf_C"]),
            "y_etac_tf_old_external": float(e["y_etac_tf"]),
            "y_npac_tf_old_external": float(e["y_npac_tf"]),
        })
    return out


def system3() -> list[dict]:
    """water / 1-butanol, binary LLE coexistence endpoints."""
    # experiment: mean of replicate tie-lines straight from the shipped dataset
    by_T: dict[float, list[tuple[float, float]]] = defaultdict(list)
    with open(EXP_LLE, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if {r["smiles1"], r["smiles2"]} != {"CCCCO", "O"}:
                continue
            b1 = r["smiles1"] == "CCCCO"
            a = float(r["x_alpha_1"] if b1 else r["x_alpha_2"])
            b = float(r["x_beta_1"] if b1 else r["x_beta_2"])
            org, aq = (a, b) if a > b else (b, a)
            by_T[round(float(r["temperature_k"]), 1)].append((org, aq))

    tf = {round(float(r["T_K"]), 2): r for r in _rows(OUTDIR / "tf_rerun_water_butanol_lle.csv")}
    dw = {round(float(r["T_K"]), 2): r for r in _rows(OUTDIR / "water_butanol_dwsim.csv")}

    out = []
    for T in (298.15, 313.15, 343.15, 353.15):
        vals = next((v for k, v in by_T.items() if abs(k - T) < 0.06), None)
        t, d = tf[round(T, 2)], dw[round(T, 2)]
        out.append({
            "T_K": T,
            "n_exp_tielines": len(vals) if vals else "",
            "exp_x_butanol_organic": round(sum(v[0] for v in vals) / len(vals), 4) if vals else "",
            "exp_x_butanol_aqueous": round(sum(v[1] for v in vals) / len(vals), 4) if vals else "",
            "tf_x_butanol_organic": float(t["tf_x_butanol_organic"]),
            "tf_x_butanol_aqueous": float(t["tf_x_butanol_aqueous"]),
            "tf_residual": t["residual"],
            "dwsim_x_butanol_organic": float(d["dwsim_x_butanol_organic"]),
            "dwsim_x_butanol_aqueous": float(d["dwsim_x_butanol_aqueous"]),
            "dwsim_file": d["file"],
        })
    return out


def main() -> None:
    s1, s2, s3 = system1(), system2(), system3()
    _write(OUTDIR / "three_source_system1_ipa_vle.csv", s1)
    _write(OUTDIR / "three_source_system2_dmso_vle.csv", s2)
    _write(OUTDIR / "three_source_system3_water_butanol_lle.csv", s3)
    _write(OUTDIR / "water_butanol_three_source.csv", s3)
    (OUTDIR / "three_source_summary.json").write_text(
        json.dumps(
            {"system1_ipa_vle": s1, "system2_dmso_vle": s2, "system3_water_butanol_lle": s3},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {OUTDIR / 'three_source_summary.json'}")


if __name__ == "__main__":
    main()
