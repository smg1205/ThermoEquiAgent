"""Why does DWSIM diverge from experiment on 1-butanol/water/toluene while
ThermoFormer agrees?  Diagnose by component and by composition region.

Uses the seed_2 three-source table (43 experimental points at 101.30 kPa).
"""
from __future__ import annotations

import csv
from pathlib import Path

R = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
rows = list(csv.DictReader(open(R / "report/dwsim/three_source_system2_bwt_seed2.csv",
                                newline="", encoding="utf-8")))

NAMES = ("butanol", "water", "toluene")


def f(r, k):
    return float(r[k])


print(f"points: {len(rows)}\n")

# --- per-component error -------------------------------------------------
print("=== 分组分误差（43 点）===")
print(f"{'组分':<10} {'TF MAE':>8} {'DW MAE':>8} {'TF bias':>9} {'DW bias':>9}")
for nm in NAMES:
    te = [f(r, f"y_{nm}_tf") - f(r, f"y_{nm}_exp") for r in rows]
    de = [f(r, f"y_{nm}_dwsim") - f(r, f"y_{nm}_exp") for r in rows]
    print(f"{nm:<10} {sum(abs(v) for v in te)/len(te):>8.4f} "
          f"{sum(abs(v) for v in de)/len(de):>8.4f} "
          f"{sum(te)/len(te):>+9.4f} {sum(de)/len(de):>+9.4f}")

# --- error vs toluene content -------------------------------------------
print("\n=== 按 x_甲苯 分组（DWSIM 的误差是否集中在某段）===")
bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.65)]
print(f"{'x_甲苯 区间':<16} {'n':>3} {'TF MAE':>8} {'DW MAE':>8}  {'DW 水误差':>9} {'DW 甲苯误差':>11}")
for lo, hi in bins:
    sub = [r for r in rows if lo <= f(r, "x_toluene") < hi]
    if not sub:
        continue
    t = sum(max(abs(f(r, f"y_{k}_tf") - f(r, f"y_{k}_exp")) for k in NAMES) for r in sub) / len(sub)
    d = sum(max(abs(f(r, f"y_{k}_dwsim") - f(r, f"y_{k}_exp")) for k in NAMES) for r in sub) / len(sub)
    dw_water = sum(abs(f(r, "y_water_dwsim") - f(r, "y_water_exp")) for r in sub) / len(sub)
    dw_tol = sum(abs(f(r, "y_toluene_dwsim") - f(r, "y_toluene_exp")) for r in sub) / len(sub)
    print(f"{lo:.2f}-{hi:.2f}{'':<9} {len(sub):>3} {t:>8.4f} {d:>8.4f}  {dw_water:>9.4f} {dw_tol:>11.4f}")

# --- error vs water content ---------------------------------------------
print("\n=== 按 x_水 分组 ===")
print(f"{'x_水 区间':<14} {'n':>3} {'TF MAE':>8} {'DW MAE':>8}  {'DW 水误差':>9}")
for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.45)]:
    sub = [r for r in rows if lo <= f(r, "x_water") < hi]
    if not sub:
        continue
    t = sum(max(abs(f(r, f"y_{k}_tf") - f(r, f"y_{k}_exp")) for k in NAMES) for r in sub) / len(sub)
    d = sum(max(abs(f(r, f"y_{k}_dwsim") - f(r, f"y_{k}_exp")) for k in NAMES) for r in sub) / len(sub)
    dw_water = sum(abs(f(r, "y_water_dwsim") - f(r, "y_water_exp")) for r in sub) / len(sub)
    print(f"{lo:.2f}-{hi:.2f}{'':<7} {len(sub):>3} {t:>8.4f} {d:>8.4f}  {dw_water:>9.4f}")

# --- DWSIM bubble temperature error --------------------------------------
print("\n=== DWSIM 泡点温度误差 vs 组成 ===")
print(f"{'x_甲苯 区间':<14} {'n':>3} {'平均 dT(K)':>11} {'DW MAE':>8}")
for lo, hi in bins:
    sub = [r for r in rows if lo <= f(r, "x_toluene") < hi]
    if not sub:
        continue
    dt = [f(r, "T_dwsim_K") - f(r, "T_exp_K") for r in sub]
    d = sum(max(abs(f(r, f"y_{k}_dwsim") - f(r, f"y_{k}_exp")) for k in NAMES) for r in sub) / len(sub)
    print(f"{lo:.2f}-{hi:.2f}{'':<7} {len(sub):>3} {sum(dt)/len(dt):>+11.2f} {d:>8.4f}")

# --- worst points --------------------------------------------------------
print("\n=== DWSIM 最差的 6 个点 ===")
scored = sorted(rows, key=lambda r: -max(abs(f(r, f"y_{k}_dwsim") - f(r, f"y_{k}_exp")) for k in NAMES))
print(f"{'T_exp':>7} {'T_dw':>7} {'x丁/水/甲':>22} {'y水_exp':>8} {'y水_dw':>8} {'y甲_exp':>8} {'y甲_dw':>8}")
for r in scored[:6]:
    x = f"{f(r,'x_butanol'):.3f}/{f(r,'x_water'):.3f}/{f(r,'x_toluene'):.3f}"
    print(f"{f(r,'T_exp_K'):>7.1f} {f(r,'T_dwsim_K'):>7.1f} {x:>22} "
          f"{f(r,'y_water_exp'):>8.3f} {f(r,'y_water_dwsim'):>8.3f} "
          f"{f(r,'y_toluene_exp'):>8.3f} {f(r,'y_toluene_dwsim'):>8.3f}")

print("\n=== 相关性：DWSIM 水误差 与 各变量 ===")
import statistics
dw_w = [abs(f(r, "y_water_dwsim") - f(r, "y_water_exp")) for r in rows]
for var, vals in (("x_toluene", [f(r, "x_toluene") for r in rows]),
                  ("x_water", [f(r, "x_water") for r in rows]),
                  ("x_butanol", [f(r, "x_butanol") for r in rows]),
                  ("dT", [f(r, "T_dwsim_K") - f(r, "T_exp_K") for r in rows])):
    try:
        c = statistics.correlation(dw_w, vals)
        print(f"  corr(DW水误差, {var:<10}) = {c:+.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  corr({var}) failed: {exc}")
