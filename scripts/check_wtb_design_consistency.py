"""用 DWSIM 简捷塔独立复核 §1.5 的设计意图，并回答物理可行性问题。

要回答的问题：设计给定 18 级 / R=2.095 / 进料板 8 / 萃取剂板 2，
塔顶水纯度 0.95、水回收率 0.90 在严格模型下能否达到？

先做一个独立的合理性检查：设计出自 FUG 短节法，含两个关键假设
  alpha_base = 2.3501（ThermoFormer 预测）
  选择性 = 0.8193 < 1  ← 报告 §1.5 自述：1-丁醇并未增强水/甲苯挥发性
在 alpha_avg = 2.1273、进料 0.5/0.5、要求塔顶轻组分 0.95 的条件下，
用 FUG 复算所需级数与回流比，与设计值对照，即可判断设计是否自洽。

该检查纯属确定性算术，不依赖 DWSIM，可快速给出结论。
"""

from __future__ import annotations

import math


def fenske(x_d_lk: float, x_b_lk: float, alpha_avg: float) -> float:
    """Fenske 最小级数（轻重关键组分形式）。"""
    ratio = (x_d_lk / (1.0 - x_d_lk)) / (x_b_lk / (1.0 - x_b_lk))
    return math.log(ratio) / math.log(alpha_avg)


def underwood_binary(x_d_lk: float, z_f: float, alpha_avg: float) -> float:
    """Underwood 最小回流比（恒 alpha 二元形式）。"""
    r_min = (x_d_lk / z_f - alpha_avg * (1.0 - x_d_lk) / (1.0 - z_f)) / (alpha_avg - 1.0)
    return max(r_min, 0.05)


def gilliland_stages(n_min: float, r: float, r_min: float) -> int:
    """Gilliland 关联给出的理论级数。"""
    x = (r - r_min) / (r + 1.0)
    y = 0.75 * (1.0 - x**0.5668)
    return math.ceil((n_min + y) / (1.0 - y))


def main() -> int:
    # 物料衡算（与 thermo_engine.column_design 一致）
    z_f = 0.5
    recovery = 0.90
    purity = 0.95

    n_l = 1.0 * z_f              # 进料中轻组分（水）流量
    d_l = recovery * n_l         # 塔顶轻组分
    d_h = (1.0 - purity) / purity * d_l
    D = d_l + d_h
    x_dl = d_l / D

    b_l = n_l - d_l
    b_h = 1.0 * (1.0 - z_f) - d_h
    B = b_l + b_h + 2.0          # 含萃取剂 2.0 mol/s
    x_bl = b_l / B

    print("=== 物料衡算 ===")
    print(f"  进料 水/甲苯 = {z_f:.4f}/{1 - z_f:.4f} mol/s，萃取剂 2.0 mol/s")
    print(f"  塔顶 D = {D:.6f} mol/s，x_水(塔顶) = {x_dl:.4f}")
    print(f"  塔釜 B = {B:.6f} mol/s，x_水(塔釜) = {x_bl:.6f}")
    print(f"  D + B = {D + B:.6f} (应 = 3.0)")

    print("\n=== 用报告 §1.5 的 alpha 复算 FUG ===")
    for label, alpha_base, alpha_ext, n_report, r_report in (
        ("ThermoFormer (seed_2)", 2.3501, 1.9256, 18, 2.095),
        ("UNIFAC", 1.9237, 1.4364, 25, 3.665),
    ):
        alpha_avg = (alpha_base * alpha_ext) ** 0.5
        n_min = fenske(x_dl, x_bl, alpha_avg)
        r_min = underwood_binary(x_dl, z_f, alpha_avg)
        r_op = 1.4 * r_min
        n_calc = gilliland_stages(n_min, r_op, r_min)
        print(f"\n  {label}")
        print(f"    alpha_base={alpha_base}  alpha_ext={alpha_ext}  alpha_avg={alpha_avg:.4f}")
        print(f"    N_min={n_min:.3f}  R_min={r_min:.3f}  R_op={r_op:.3f}  N={n_calc}")
        print(f"    报告值: N={n_report}  R={r_report}  -> "
              f"{'一致' if n_calc == n_report else f'不一致 (算得 {n_calc})'}")

    print("\n=== 关键判据 ===")
    print(f"  选择性 = alpha_ext/alpha_base = {1.9256 / 2.3501:.4f} (< 1)")
    print("  选择性 < 1 意味着萃取剂并未提高关键对的相对挥发度，")
    print("  即该塔实为「带萃取剂进料的三元精馏塔」，萃取精馏的增强效应不成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
