"""Derive the comparison metrics needed for the v3 case-three report."""
import json

d = json.load(open(r"E:\PythonProject\ThermoEqui-Agent-main-3\.thermoformer-cache\campaign_v3_tfea_output.json",
                   encoding="utf-8-sig"))

print("=== Isothermal ternary (equimolar) ===")
for r in d["isothermal_ternary_equimolar"]:
    ptf, pu = r["P_TF"], r["P_UNIFAC"]
    rel = (ptf - pu) / pu * 100.0
    print("T=%sK  P_TF=%.2f  P_U=%.2f  dP=%+.2f  rel%%=%.1f" % (r["T"], ptf, pu, ptf - pu, rel))

print("=== Isobaric ternary ===")
for r in d["isobaric_ternary"]:
    print("x=%s  T_TF=%.2f  T_U=%.2f  dT=%+.2f" % (r["x"], r["T_TF"], r["T_UNIFAC"], r["T_TF"] - r["T_UNIFAC"]))

print("=== binary isothermal (EA/n-PrOAc) ===")
for r in d["isothermal_binary_ea_nproac"]:
    rel = (r["P_TF"] - r["P_UNIFAC"]) / r["P_UNIFAC"] * 100.0
    print("T=%s  P_TF=%.2f  P_U=%.2f  rel%%=%.1f" % (r["T"], r["P_TF"], r["P_UNIFAC"], rel))

print("=== isothermal TF residual max: %.4f ===" % max(r["residual_kpa"] for r in d["isothermal_ternary_equimolar"]))
print("=== isobaric TF residual max: %.4f ===" % max(r["residual_kpa"] for r in d["isobaric_ternary"]))

# Selectivity discussion numbers
e_unifac = d["extractive_column"]["unifac"]
e_tf = d["extractive_column"]["thermoformer"]
b_unifac = d["binary_column"]["unifac"]
b_tf = d["binary_column"]["thermoformer"]
print("=== binary UNIFAC alpha=%.4f N=%.3f Nmin=%.3f R=%.3f Rmin=%.3f Tcond=%.2f Treb=%.2f" % (
    b_unifac["relative_volatility"], b_unifac["theoretical_stages"], b_unifac["minimum_stages"],
    b_unifac["reflux_ratio"], b_unifac["minimum_reflux_ratio"],
    b_unifac["condenser_temperature_K"], b_unifac["reboiler_temperature_K"]))
print("=== extractive UNIFAC alpha_base=%.4f alpha_ext=%.4f sel=%.4f alpha_avg=%.4f N=%.3f Nmin=%.3f R=%.3f Rmin=%.3f Tcond=%.2f Treb=%.2f" % (
    e_unifac["alpha_base"], e_unifac["alpha_ext"], e_unifac["selectivity"], e_unifac["alpha_avg"],
    e_unifac["theoretical_stages"], e_unifac["minimum_stages"], e_unifac["reflux_ratio"],
    e_unifac["minimum_reflux_ratio"], e_unifac["condenser_temperature_K"], e_unifac["reboiler_temperature_K"]))
print("=== extractive TF alpha_base=%.4f alpha_ext=%.4f sel=%.4f alpha_avg=%.4f N=%.3f Nmin=%.3f R=%.3f Rmin=%.3f Tcond=%.2f Treb=%.2f" % (
    e_tf["alpha_base"], e_tf["alpha_ext"], e_tf["selectivity"], e_tf["alpha_avg"],
    e_tf["theoretical_stages"], e_tf["minimum_stages"], e_tf["reflux_ratio"],
    e_tf["minimum_reflux_ratio"], e_tf["condenser_temperature_K"], e_tf["reboiler_temperature_K"]))
