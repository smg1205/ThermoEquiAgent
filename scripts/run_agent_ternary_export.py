"""Run the Agent's ternary extractive export end-to-end and inspect the result."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from agent import extractive_distillation as ed  # noqa: E402

outdir = Path(tempfile.mkdtemp(prefix="agent-ternary-export-"))
print("export dir:", outdir)

msg = "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件"
print("message:", msg)

try:
    payload = ed.run_eac_npac_dmso_vle_export(msg, export_dir=str(outdir))
except Exception as exc:  # noqa: BLE001
    print("RAISED:", type(exc).__name__, str(exc)[:300])
    raise SystemExit(1)

print("status:", payload.status)
print("file_id:", payload.file_id)
print("dwsim_file_uri:", payload.dwsim_file_uri)
print("message:", payload.message)
d = payload.design
if d is not None:
    print("\n=== design ===")
    for k in ("light", "heavy", "entrainer", "alpha_source", "alpha_base", "alpha_ext",
              "selectivity", "alpha_avg", "theoretical_stages", "minimum_stages",
              "reflux_ratio", "minimum_reflux_ratio", "feed_stage", "entrainer_stage",
              "condenser_temperature_K", "reboiler_temperature_K",
              "operating_pressure_kPa"):
        print(f"  {k}: {getattr(d, k, '<n/a>')}")

files = list(outdir.glob("*.dwxmz"))
print(f"\n.dwxmz produced: {[f.name for f in files]}")
for f in files:
    print(f"  {f.name}  {f.stat().st_size} bytes")
