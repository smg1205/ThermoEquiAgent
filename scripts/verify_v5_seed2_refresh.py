"""Verify the refreshed §1.5 / §2.2 numbers in v5 against the computed CSV outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

R = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
V5 = (R / "report" / "Agent整合ThermoFormer进度与三源验证报告v5.md").read_text(encoding="utf-8")

ok = True

print("=== §2.2 代表点 (seed_2) ===")
rows = list(csv.DictReader(open(R / "report/dwsim/three_source_system2_bwt_seed2.csv",
                                newline="", encoding="utf-8")))
for T in ["376.91", "379.13", "379.25", "368.81", "370.42", "384.81"]:
    r = next(x for x in rows if x["T_exp_K"] == T)
    x = f"{float(r['x_butanol']):.3f} / {float(r['x_water']):.3f} / {float(r['x_toluene']):.3f}"
    ye = f"{float(r['y_butanol_exp']):.3f} / {float(r['y_water_exp']):.3f} / {float(r['y_toluene_exp']):.3f}"
    yt = f"{float(r['y_butanol_tf']):.3f} / {float(r['y_water_tf']):.3f} / {float(r['y_toluene_tf']):.3f}"
    yd = f"{float(r['y_butanol_dwsim']):.3f} / {float(r['y_water_dwsim']):.3f} / {float(r['y_toluene_dwsim']):.3f}"
    Td = f"{float(r['T_dwsim_K']):.2f}"
    marks = [x in V5, ye in V5, yt in V5, yd in V5, Td in V5]
    print(f"  T={T}: x={marks[0]} yexp={marks[1]} yTF={marks[2]} yDW={marks[3]} Tdw={marks[4]}")
    if not all(marks):
        ok = False

print("\n=== §2.2 统计量 ===")
s = json.loads((R / "report/dwsim/three_source_system2_bwt_seed2.json").read_text(encoding="utf-8"))
checks = [
    ("TF mean", f"{s['tf_max_err'][0]:.4f}"), ("TF median", f"{s['tf_max_err'][1]:.4f}"),
    ("TF max", f"{s['tf_max_err'][2]:.4f}"),
    ("DW mean", f"{s['dwsim_max_err'][0]:.4f}"), ("DW median", f"{s['dwsim_max_err'][1]:.4f}"),
    ("DW max", f"{s['dwsim_max_err'][2]:.4f}"),
    ("TF butanol", f"{s['per_species_mae_tf']['1-butanol']:.4f}"),
    ("TF water", f"{s['per_species_mae_tf']['water']:.4f}"),
    ("TF toluene", f"{s['per_species_mae_tf']['toluene']:.4f}"),
    ("DW butanol", f"{s['per_species_mae_dwsim']['1-butanol']:.4f}"),
    ("DW water", f"{s['per_species_mae_dwsim']['water']:.4f}"),
    ("DW toluene", f"{s['per_species_mae_dwsim']['toluene']:.4f}"),
    ("dT mean", f"{s['dT_mean']:+.2f}"),
]
for label, v in checks:
    hit = v in V5
    print(f"  {label:<12} {v:<9} in v5: {hit}")
    if not hit:
        ok = False

print("\n=== §1.5 表 1.5-2 (seed_2) ===")
d = list(csv.DictReader(open(R / "report/dwsim/column_design_1p5_butanol_water_toluene_seed2.csv",
                             newline="", encoding="utf-8")))
ext = [r for r in d if r["case"] == "extractive"
       and r["light"] == "water" and r["heavy"] == "toluene" and r["entrainer"] == "1-butanol"]
for src in ("unifac", "thermoformer"):
    r = next(x for x in ext if x["alpha_source"] == src)
    for field, fmt in (("alpha_base", ".4f"), ("alpha_ext", ".4f"), ("selectivity", ".3f"),
                       ("N", ""), ("N_min", ".3f"), ("R", ".3f"), ("R_min", ".3f"),
                       ("feed_stage", ""), ("T_cond_K", ".2f"), ("T_reb_K", ".2f")):
        v = format(float(r[field]), fmt) if fmt else str(r[field])
        hit = v in V5
        if not hit:
            print(f"  MISSING [{src}] {field} = {v}")
            ok = False
    print(f"  [{src}] checked")

print("\n=== 是否残留 seed_0 用于三元体系 ===")
import re
for m in re.finditer(r".{50}seed_0.{50}", V5):
    seg = m.group(0).replace("\n", " ")
    if "ternary" in seg or "三元" in seg:
        print("  POSSIBLE STALE:", seg[:110])
print("done")

print("\nRESULT:", "ALL VERIFIED" if ok else "*** ISSUES ***")
