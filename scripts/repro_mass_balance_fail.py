"""Reproduce the failing request and inspect the mass balance it generates.

Message: 用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim，纯度 0.99

"Failed to fulfill mass balance for Water: Relative Error = 0.99924" means the
water entering the column and the water leaving it differ by ~100%, i.e. essentially
all the water is unaccounted for.  This prints the balance the design implies.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402,F401  (must load before any DWSIM Automation call)

from agent import generic_ternary_export as gte  # noqa: E402

MSG = "用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim，纯度 0.99"

req = gte.parse_request(MSG)
print("parsed request:")
for f in ("light", "heavy", "entrainer", "property_package", "purity", "recovery",
          "entrainer_ratio", "feed_flow_mol_s", "pressure_kpa", "alpha_source",
          "feed_composition"):
    print(f"  {f}: {getattr(req, f)}")

bal = gte._material_balance(req)
print("\nmaterial balance implied by the design:")
print(f"  feed            water {req.feed_flow_mol_s*req.feed_composition[0]:.4f}  "
      f"toluene {req.feed_flow_mol_s*req.feed_composition[1]:.4f} mol/s")
print(f"  entrainer feed  1-butanol {bal['ent_flow']:.4f} mol/s")
print(f"  distillate D    {bal['D']:.4f} mol/s  z={[round(v,4) for v in bal['z_d']]}")
print(f"  bottoms    B    {bal['B']:.4f} mol/s  z={[round(v,4) for v in bal['z_b']]}")
print(f"  closure D+B = {bal['D']+bal['B']:.4f}  vs feed+ent = "
      f"{req.feed_flow_mol_s + bal['ent_flow']:.4f}")

print("\nper-component check (in vs out):")
names = ["water", "toluene", "1-butanol"]
F = req.feed_flow_mol_s
for i, nm in enumerate(names):
    cin = F * req.feed_composition[i] if i < 2 else 0.0
    if i == 2:
        cin += bal["ent_flow"]
    cout = bal["D"] * bal["z_d"][i] + bal["B"] * bal["z_b"][i]
    rel = abs(cout - cin) / cin if cin else 0.0
    print(f"  {nm:<12} in={cin:7.4f}  out={cout:7.4f}  rel.err={rel:.3e}")

print("\n=== run the export ===")
outdir = Path(tempfile.mkdtemp(prefix="repro-tf-"))
p = gte.run_generic_ternary_extractive_export(MSG, export_dir=str(outdir))
print("status:", p.status)
print("message:", p.message.replace("\n", "\n         "))
print("file:", p.file_id)
if p.file_id:
    f = outdir / f"{p.file_id}.dwxmz"
    print(f"produced: {f.exists()} ({f.stat().st_size if f.exists() else 0} bytes)")
