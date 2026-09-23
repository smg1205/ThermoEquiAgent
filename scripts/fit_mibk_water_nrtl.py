"""Regress NRTL binary parameters for MIBK(1)/Water(2) from experimental tie-lines.

Data: doi 10.1016/j.fluid.2016.11.005 (binary_lle.csv).  Three temperatures, each
with the two coexistence compositions (organic = MIBK-rich, aqueous = water-rich).

NRTL two-suffix-ish form used (standard NRTL with symmetric alpha):
    tau_ij = Aij / (R*T)   (Aij in J/mol, T in K, R = 8.314)
    G_ij = exp(-alpha * tau_ij)
    ln g_i = x_j^2 [ tau_ji (G_ji/(x_i + x_j G_ji))^2 + tau_ij G_ij/(x_j + x_i G_ij)^2 ]

We fit (A12, A21, alpha) by minimizing the equal-activity residual over the tie
lines:  for each phase p in {organic, aqueous}, activity_i^p = x_i^p * gamma_i^p;
LLE requires activity_i^organic = activity_i^aqueous for i = 1,2.

The parameters come ONLY from the experimental tie-lines (not invented).  They are
Kept in J/mol for the energy terms; alpha is dimensionless.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

R = 8.314462618  # J/(mol K)

# tie-lines: (T_K, x_MIBK_organic, x_MIBK_aqueous)
# from binary_lle.csv (deduplicated; aqueous x_water = 1 - x_MIBK_aqueous)
TIE_LINES = [
    (333.15, 0.8109, 0.0023),
    (343.15, 0.7410, 0.0022),
    (353.15, 0.6924, 0.0019),
]


def gamma_i(x1, A12, A21, alpha, T):
    """Return (ln gamma1, ln gamma2) for mole fraction (x1, x2=1-x1)."""
    x2 = 1.0 - x1
    tau12 = A12 / (R * T)
    tau21 = A21 / (R * T)
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)
    # ln gamma1
    ln_g2 = x1 * x1 * (
        tau12 * (G12 / (x2 + x1 * G12)) ** 2
        + tau21 * G21 / (x1 + x2 * G21) ** 2
    )
    ln_g1 = x2 * x2 * (
        tau21 * (G21 / (x1 + x2 * G21)) ** 2
        + tau12 * G12 / (x2 + x1 * G12) ** 2
    )
    return ln_g1, ln_g2


def residual(params):
    A12, A21, alpha = params
    res = []
    for T, xorg, xaq in TIE_LINES:
        # organic phase: x1 = x_MIBK_organic ; aqueous: x1 = x_MIBK_aqueous
        ln_g1_org, ln_g2_org = gamma_i(xorg, A12, A21, alpha, T)
        ln_g1_aq, ln_g2_aq = gamma_i(xaq, A12, A21, alpha, T)
        # equal activity: ln(x1^org g1^org) = ln(x1^aq g1^aq)
        lnA1_org = np.log(xorg) + ln_g1_org
        lnA1_aq = np.log(xaq) + ln_g1_aq
        lnA2_org = np.log(1 - xorg) + ln_g2_org
        lnA2_aq = np.log(1 - xaq) + ln_g2_aq
        res.append(lnA1_org - lnA1_aq)
        res.append(lnA2_org - lnA2_aq)
    return res


def main():
    # initial guess: A in J/mol, typical magnitude a few kJ/mol
    x0 = np.array([5000.0, 8000.0, 0.2])
    bounds = (
        [-50000.0, -50000.0, 0.05],
        [50000.0, 50000.0, 0.8],
    )
    sol = least_squares(residual, x0, bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=100000)

    A12, A21, alpha = sol.x
    print("fitted NRTL params:")
    print(f"  A12 (MIBK->Water, J/mol) = {A12:.3f}")
    print(f"  A21 (Water->MIBK, J/mol) = {A21:.3f}")
    print(f"  alpha = {alpha:.5f}")
    # convert to cal/mol for DWSIM comparison (1 cal = 4.184 J)
    print(f"  A12 = {A12/4.184:.3f} cal/mol, A21 = {A21/4.184:.3f} cal/mol")
    print(f"residual norm = {np.linalg.norm(sol.fun):.3e}")

    # verify by reconstructing the binodal at the three temperatures
    print("\nreconstruction check:")
    for T, xorg, xaq in TIE_LINES:
        l1o, l2o = gamma_i(xorg, A12, A21, alpha, T)
        l1a, l2a = gamma_i(xaq, A12, A21, alpha, T)
        a1o, a1a = np.log(xorg) + l1o, np.log(xaq) + l1a
        a2o, a2a = np.log(1 - xorg) + l2o, np.log(1 - xaq) + l2a
        print(f"  T={T}: d(ln a1)={a1o-a1a:+.4e}  d(ln a2)={a2o-a2a:+.4e}")

    # save
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "lunwen" / "dwsim_demonstration" / "mibk_water_nrtl_fitted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "NRTL",
        "components": ["Methyl isobutyl ketone", "Water"],
        "source": "regressed from doi 10.1016/j.fluid.2016.11.005 tie-lines (333.15/343.15/353.15 K)",
        "A12_J_per_mol": float(A12),
        "A21_J_per_mol": float(A21),
        "alpha12": float(alpha),
        "A12_cal_per_mol": float(A12 / 4.184),
        "A21_cal_per_mol": float(A21 / 4.184),
        "residual_norm": float(np.linalg.norm(sol.fun)),
    }, indent=2), encoding="utf-8")
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
