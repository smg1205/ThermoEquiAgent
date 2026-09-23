"""Compute the §1.5 case-2 design (1-butanol / water / toluene) from ThermoFormer.

This is the **ThermoFormer** arm of report table 1.5-2: the ternary short-cut
extractive design for the key pair water / toluene with 1-butanol as entrainer,
driven by the ``thermoformer`` alpha source on the ``seed_2`` checkpoint of
``vle_overall_ternary``.

Every number comes from :func:`thermo_engine.column_design.design_ternary_extractive_column`
-- nothing is hard-coded here.  The result is written to JSON so the DWSIM
exporter can consume it in a **separate process**: pythonnet (clr) and torch
cannot coexist, so design and export must not share an interpreter.

Output: ``report/dwsim/design_1p5_wtb_thermoformer_seed2.json``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.pop("THERMOFORMER_CHECKPOINT", None)
os.environ["THERMOFORMER_SRC"] = r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\src"
os.environ["THERMOFORMER_USE_CUDA"] = "0"

CHECKPOINT = Path(
    r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\models\vle\prediction"
    r"\vle_overall_ternary\seed_2\best_model.pt"
)

OUT = ROOT / "report" / "dwsim" / "design_1p5_wtb_thermoformer_seed2.json"

LIGHT, HEAVY, ENTRAINER = "water", "toluene", "1-butanol"
FEED = [0.5, 0.5]
P_KPA = 101.325
F_MOL_S = 1.0
PURITY = 0.95
RECOVERY = 0.90
ENTRAINER_RATIO = 2.0


def main() -> int:
    from thermo_engine import column_design as cd
    from thermo_engine.thermoformer_backend import (
        ThermoFormerBackend,
        ThermoFormerSettings,
        resolve_settings,
    )

    base = resolve_settings()
    cd._thermoformer_backend = ThermoFormerBackend(
        ThermoFormerSettings(
            src_path=base.src_path,
            checkpoint_path=CHECKPOINT,
            feature_cache_path=base.feature_cache_path,
            use_cuda=False,
        )
    )

    design = cd.design_ternary_extractive_column(
        LIGHT,
        HEAVY,
        ENTRAINER,
        FEED,
        feed_flow_mol_s=F_MOL_S,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        entrainer_ratio=ENTRAINER_RATIO,
        alpha_source="thermoformer",
    )

    record = dict(design)
    record["checkpoint"] = str(CHECKPOINT)
    record["feed_composition"] = list(FEED)
    record["feed_flow_mol_s"] = F_MOL_S
    record["entrainer_ratio"] = ENTRAINER_RATIO
    record["distillate_purity_target"] = PURITY
    record["recovery"] = RECOVERY

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    for key, value in record.items():
        print(f"{key:34} {value}")
    print(f"\nwrote: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
