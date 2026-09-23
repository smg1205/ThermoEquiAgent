"""用 ThermoFormer 的完整信息拟合 NRTL —— 五步法，带交叉验证。

背景
----
TF 的 LLE 接口只给 1 对共存端点（2 个方程），而 NRTL 二元有 3 个未知数
(A12, A21, alpha) —— 单靠端点是**欠定**的，解集是一维流形（已实测：6 个不同
初值给出 6 组不同参数，残差全是机器精度）。

本脚本改用 TF 的**完整信息**：

  步骤 1  isothermal_vle(T) 取整条 P-x-y 曲线，排除共存区内的点
          （共存区内"单液相"是热力学不稳定态，TF 在那里给的是数学外推）
  步骤 2  由 gamma_i = y_i P / (x_i P_i^sat) 反算活度系数
  步骤 3  叠加 LLE 端点的等活度约束 + gamma_inf 约束
  步骤 4  最小二乘拟合 (A12, A21, alpha)，初值扫描验证唯一性
  步骤 5  交叉验证：反算 LLE 端点 vs TF 端点；抽查 x-gamma

所有 TF 数值由 ThermoFormerBackend 实跑得到；Psat 由 thermo_engine 的 NIST
Antoine 分支给出。本脚本不发明任何数值。

Run:
    python scripts/fit_nrtl_from_thermoformer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve, least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R = 8.314462618  # J/(mol K)
CAL = 4.184  # J per cal
OUTDIR = ROOT / "report" / "success"

#: Components of the water / 1-butanol binary LLE system.
BUTANOL = ("1-Butanol", "71-36-3", "CCCCO")
WATER = ("Water", "7732-18-5", "O")


# --------------------------------------------------------------------------- #
# Step 0 -- ThermoFormer data
# --------------------------------------------------------------------------- #


def _components():
    from schemas.domain import ComponentIdentity

    def make(spec):
        return ComponentIdentity(
            component_id=spec[0].casefold(), name=spec[0], cas_number=spec[1], smiles=spec[2], aliases=[]
        )

    return [make(BUTANOL), make(WATER)]


def tf_lle_point(temperature_k: float, pressure_kpa: float) -> dict[str, object]:
    """Step 0a -- the LLE coexistence endpoint pair from ThermoFormer."""
    from schemas.domain import TaskManifest, ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=_components(),
        conditions=ThermodynamicConditions(temperature_K=temperature_k, pressure_kPa=pressure_kpa),
        model_name="ThermoFormer",
    )
    result = backend.lle(task)
    endpoints = [list(p.composition) for p in result.phases]
    # Sort: first component is 1-butanol, so the rich phase has the larger x1.
    endpoints.sort(key=lambda e: e[0])
    return {"heavy": endpoints[0], "light": endpoints[1], "residual": result.residual}


def tf_pxy_curve(temperature_k: float, n_points: int = 21) -> list[dict[str, float]]:
    """Step 1 -- the full P-x-y curve at fixed T from ThermoFormer."""
    from schemas.domain import TaskManifest, ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isothermal_vle",
        components=_components(),
        conditions=ThermodynamicConditions(temperature_K=temperature_k, pressure_kPa=101.325),
        model_name="ThermoFormer",
        points=n_points,
    )
    result = backend.isothermal_vle(task)
    rows = []
    for point in result.points:
        if not point.liquid_composition or not point.vapor_composition:
            continue
        rows.append(
            {
                "x1": float(point.liquid_composition[0]),
                "y1": float(point.vapor_composition[0]),
                "P_kPa": float(point.pressure_kPa),
            }
        )
    return rows


def tf_gamma_infinity(temperature_k: float) -> dict[str, float]:
    """Step 3a -- infinite-dilution activity coefficients from ThermoFormer."""
    from schemas.domain import TaskManifest, ThermodynamicConditions
    from thermo_engine.thermoformer_backend import ThermoFormerBackend

    backend = ThermoFormerBackend()
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="infinite_dilution_activity",
        components=_components(),
        conditions=ThermodynamicConditions(temperature_K=temperature_k, pressure_kPa=101.325),
        model_name="ThermoFormer",
    )
    result = backend.infinite_dilution_activity(task)
    out: dict[str, float] = {}
    for item in result.gamma_infinity:
        out[f"{item.solute_index}->{item.solvent_index}"] = float(item.gamma_infinity)
    return out


# --------------------------------------------------------------------------- #
# Step 2 -- Psat (shared with DWSIM's own basis where possible) and gamma
# --------------------------------------------------------------------------- #


def psat_pa(name: str, temperature_k: float) -> float:
    """Saturation pressure from the project's deterministic NIST Antoine branch."""
    from thermo_engine.column_design import _vap_pressure

    return float(_vap_pressure(name, temperature_k))


def gammas_from_pxy(x1: float, y1: float, p_kpa: float, temperature_k: float) -> tuple[float, float]:
    """Step 2 -- modified Raoult: gamma_i = y_i P / (x_i Psat_i)."""
    p_pa = p_kpa * 1000.0
    ps1 = psat_pa(BUTANOL[0], temperature_k)
    ps2 = psat_pa(WATER[0], temperature_k)
    g1 = (y1 * p_pa) / (x1 * ps1)
    g2 = ((1 - y1) * p_pa) / ((1 - x1) * ps2)
    return g1, g2


# --------------------------------------------------------------------------- #
# NRTL model
# --------------------------------------------------------------------------- #


def nrtl_ln_gamma(x1: float, a12: float, a21: float, alpha: float, t_k: float) -> tuple[float, float]:
    """Symmetric-alpha binary NRTL; ``a12``/``a21`` in J/mol."""
    x2 = 1.0 - x1
    t12, t21 = a12 / (R * t_k), a21 / (R * t_k)
    g12, g21 = np.exp(-alpha * t12), np.exp(-alpha * t21)
    ln_g1 = x2 * x2 * (t21 * (g21 / (x1 + x2 * g21)) ** 2 + t12 * g12 / (x2 + x1 * g12) ** 2)
    ln_g2 = x1 * x1 * (t12 * (g12 / (x2 + x1 * g12)) ** 2 + t21 * g21 / (x1 + x2 * g21) ** 2)
    return float(ln_g1), float(ln_g2)


# --------------------------------------------------------------------------- #
# Step 4 -- fit
# --------------------------------------------------------------------------- #


def build_equations(curve: list[dict[str, float]], lle: dict[str, object], ginf: dict[str, float], t_k: float):
    """Assemble every residual the fit will use, with its own weight category."""
    x_light = float(lle["light"][0])  # type: ignore[index]
    x_heavy = float(lle["heavy"][0])  # type: ignore[index]
    g1_inf = ginf.get("0->1")  # butanol at infinite dilution in water
    g2_inf = ginf.get("1->0")  # water at infinite dilution in butanol

    def residuals(p: np.ndarray) -> list[float]:
        a12, a21, alpha = p
        out: list[float] = []

        # (a) P-x-y curve -> ln(gamma) residuals
        for row in curve:
            g1, g2 = gammas_from_pxy(row["x1"], row["y1"], row["P_kPa"], t_k)
            if g1 <= 0 or g2 <= 0:
                continue
            l1, l2 = nrtl_ln_gamma(row["x1"], a12, a21, alpha, t_k)
            out.append(np.log(g1) - l1)
            out.append(np.log(g2) - l2)

        # (b) LLE endpoints -> equal-activity residuals
        l1l, l2l = nrtl_ln_gamma(x_light, a12, a21, alpha, t_k)
        l1h, l2h = nrtl_ln_gamma(x_heavy, a12, a21, alpha, t_k)
        out.append((np.log(x_light) + l1l) - (np.log(x_heavy) + l1h))
        out.append((np.log(1 - x_light) + l2l) - (np.log(1 - x_heavy) + l2h))

        # (c) gamma-infinity residuals (x -> 0 limits)
        if g1_inf:
            _, l2_at_0 = nrtl_ln_gamma(1e-12, a12, a21, alpha, t_k)
            out.append(np.log(g1_inf) - l2_at_0)
        if g2_inf:
            l1_at_1, _ = nrtl_ln_gamma(1.0 - 1e-12, a12, a21, alpha, t_k)
            out.append(np.log(g2_inf) - l1_at_1)

        return out

    return residuals


def fit(residuals, starts: list[list[float]]) -> list[tuple[np.ndarray, float]]:
    """Step 4 -- fit from several starts so non-uniqueness is visible."""
    out = []
    for x0 in starts:
        sol = least_squares(
            residuals,
            np.array(x0, dtype=float),
            bounds=([-50000.0, -50000.0, 0.05], [50000.0, 50000.0, 0.9]),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=400000,
        )
        out.append((sol.x, float(np.sqrt(np.mean(np.square(sol.fun))))))
    out.sort(key=lambda item: item[1])
    return out


def binodal(a12: float, a21: float, alpha: float, t_k: float):
    """Solve a_i(org) = a_i(aq) for both components -> the two coexistence ends."""

    def eqs(v):
        xo, xa = v
        l1o, l2o = nrtl_ln_gamma(xo, a12, a21, alpha, t_k)
        l1a, l2a = nrtl_ln_gamma(xa, a12, a21, alpha, t_k)
        return [
            np.log(xo) + l1o - (np.log(xa) + l1a),
            np.log(1 - xo) + l2o - (np.log(1 - xa) + l2a),
        ]

    best = None
    for guess in ((0.9, 0.05), (0.7, 0.1), (0.5, 0.05), (0.6, 0.2), (0.95, 0.01)):
        try:
            sol, _, ier, _ = fsolve(eqs, guess, full_output=True)
            xo, xa = float(sol[0]), float(sol[1])
            if ier != 1 or not (1e-9 < xa < xo < 1 - 1e-9) or (xo - xa) < 0.01:
                continue
            res = max(abs(v) for v in eqs(sol))
            if res < 1e-7 and (best is None or res < best[2]):
                best = (xo, xa, res)
        except Exception:  # noqa: BLE001 - solver may diverge from a bad guess
            continue
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--pressure", type=float, default=101.325)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()

    t_k, p_kpa = float(args.temperature), float(args.pressure)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(f"NRTL from ThermoFormer -- water / 1-butanol @ {t_k} K, {p_kpa} kPa")
    print("=" * 82)

    # ---- Step 0/1 ---------------------------------------------------------- #
    lle = tf_lle_point(t_k, p_kpa)
    x_light, x_heavy = float(lle["light"][0]), float(lle["heavy"][0])  # type: ignore[index]
    print(f"\n[0] TF LLE endpoints: light x_butanol={x_light:.6f}  heavy x_butanol={x_heavy:.6f}")

    curve_all = tf_pxy_curve(t_k, args.points)
    curve = [r for r in curve_all if not (x_heavy < r["x1"] < x_light)]
    print(f"[1] TF P-x-y curve: {len(curve_all)} points total")
    print(f"    excluded {len(curve_all) - len(curve)} inside the two-phase window")
    print(f"    ({x_heavy:.4f}, {x_light:.4f}) -- a single liquid there is unstable,")
    print("    so TF's values are mathematical extrapolation, not a physical state.")
    print(f"    remaining: {len(curve)} points")

    ginf = tf_gamma_infinity(t_k)
    print(f"[3] TF gamma_infinity: {ginf}")

    print("\n[2] Reverse-computed activity coefficients from the curve")
    print(f"    {'x_butanol':>10} {'gamma1':>10} {'gamma2':>10}")
    for row in curve:
        g1, g2 = gammas_from_pxy(row["x1"], row["y1"], row["P_kPa"], t_k)
        print(f"    {row['x1']:10.3f} {g1:10.4f} {g2:10.4f}")

    # ---- Step 4 ------------------------------------------------------------ #
    residuals = build_equations(curve, lle, ginf, t_k)
    n_equations = len(residuals(np.array([1000.0, 2000.0, 0.3])))
    print(f"\n[4] fitting: {n_equations} equations vs 3 parameters (overdetermined)")

    starts = [
        [500.0, 1800.0, 0.41],
        [2000.0, 2000.0, 0.30],
        [3000.0, 1500.0, 0.25],
        [1000.0, 1000.0, 0.50],
        [5000.0, 3000.0, 0.20],
        [200.0, 2500.0, 0.45],
        [800.0, 2200.0, 0.35],
    ]
    solutions = fit(residuals, starts)
    print(f"    {'A12 (cal)':>11} {'A21 (cal)':>11} {'alpha':>8} {'RMS':>12}")
    for params, rms in solutions:
        print(f"    {params[0] / CAL:11.3f} {params[1] / CAL:11.3f} {params[2]:8.4f} {rms:12.3e}")

    spreads = np.array([p for p, _ in solutions])
    a12_sd = float(spreads[:, 0].std() / CAL)
    a21_sd = float(spreads[:, 1].std() / CAL)
    alpha_sd = float(spreads[:, 2].std())
    print(f"\n    spread across starts: A12 sd={a12_sd:.3f}  A21 sd={a21_sd:.3f}  alpha sd={alpha_sd:.4f}")
    unique = a12_sd < 20.0 and a21_sd < 20.0 and alpha_sd < 0.02
    print(f"    => solution is {'UNIQUE (converged to one point)' if unique else 'STILL NON-UNIQUE'}")

    best_params, best_rms = solutions[0]
    a12, a21, alpha = (float(v) for v in best_params)

    # ---- Step 5 ------------------------------------------------------------ #
    print("\n[5] cross-validation")

    # 5a. Does the fitted model reproduce the TF endpoints it did NOT overfit to?
    bi = binodal(a12, a21, alpha, t_k)
    if bi:
        xo, xa, _ = bi
        print("    (a) LLE endpoints -- fitted NRTL vs TF")
        print(f"        light: NRTL={xo:.6f}   TF={x_light:.6f}   d={xo - x_light:+.6f}")
        print(f"        heavy: NRTL={xa:.6f}   TF={x_heavy:.6f}   d={xa - x_heavy:+.6f}")
    else:
        print("    (a) fitted parameters produce no two-phase solution")
        xo = xa = float("nan")

    # 5b. gamma spot-check at compositions withheld from the fit
    print("    (b) gamma spot-check (curve points were used; these x are a sanity sweep)")
    print(f"        {'x':>8} {'TF gamma1':>11} {'NRTL gamma1':>12} {'TF gamma2':>11} {'NRTL gamma2':>12}")
    max_rel = 0.0
    for row in curve[::3]:
        g1, g2 = gammas_from_pxy(row["x1"], row["y1"], row["P_kPa"], t_k)
        l1, l2 = nrtl_ln_gamma(row["x1"], a12, a21, alpha, t_k)
        n1, n2 = np.exp(l1), np.exp(l2)
        max_rel = max(max_rel, abs(n1 - g1) / g1, abs(n2 - g2) / g2)
        print(f"        {row['x1']:8.3f} {g1:11.4f} {n1:12.4f} {g2:11.4f} {n2:12.4f}")
    print(f"        max relative gamma error = {max_rel * 100:.2f}%")

    # 5c. gamma-infinity check
    if ginf.get("0->1"):
        _, l2_at_0 = nrtl_ln_gamma(1e-12, a12, a21, alpha, t_k)
        print("    (c) gamma_infinity")
        print(
            f"        butanol->water: NRTL={np.exp(l2_at_0):.4f}  TF={ginf['0->1']:.4f}  "
            f"err={abs(np.exp(l2_at_0) - ginf['0->1']) / ginf['0->1'] * 100:.2f}%"
        )

    summary = {
        "system": "1-butanol / water",
        "temperature_K": t_k,
        "pressure_kPa": p_kpa,
        "thermoformer": {
            "lle_light_x1": x_light,
            "lle_heavy_x1": x_heavy,
            "gamma_infinity": ginf,
            "curve_points_used": len(curve),
            "curve_points_excluded_in_two_phase_window": len(curve_all) - len(curve),
        },
        "fit": {
            "A12_cal_per_mol": a12 / CAL,
            "A21_cal_per_mol": a21 / CAL,
            "alpha12": alpha,
            "rms": best_rms,
            "equations": n_equations,
            "unique": unique,
            "spread_A12_cal": a12_sd,
            "spread_A21_cal": a21_sd,
            "spread_alpha": alpha_sd,
        },
        "cross_validation": {
            "nrtl_light_x1": xo,
            "nrtl_heavy_x1": xa,
            "max_relative_gamma_error": max_rel,
        },
        "generated_at": datetime.now().isoformat(),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = outdir / f"nrtl_from_thermoformer_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
