"""Inspect the condenser/reboiler specification state of the generated columns."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

OUT = ROOT / "report" / "dwsim"
FILES = [
    "ipa_water_binary_column_x0p5_unifac.dwxmz",
    "ipa_water_binary_column_x0p5_tf.dwxmz",
]


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    for name in FILES:
        p = OUT / name
        fs = automation.LoadFlowsheet2(str(p))
        column = None
        for item in fs.SimulationObjects.Values:
            if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
                column = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        if column is None:
            continue
        print(f"\n=== {name} ===")
        print(f"  CondenserType   : {column.CondenserType}")
        print(f"  RefluxRatio     : {column.RefluxRatio}")
        print(f"  DistillateFlow  : {column.DistillateFlowRate}")
        print(f"  VaporFlowRate   : {column.VaporFlowRate}")
        print(f"  ReboilerDuty    : {column.ReboilerDuty}")
        print(f"  CondenserDuty   : {column.CondenserDuty}")
        # Specifications dictionary
        try:
            specs = column.Specs
            print(f"  Specs count     : {len(specs)}")
            for k in list(specs.Keys):
                s = specs[k]
                try:
                    print(f"    [{k}] {s.GetType().Name}: Name={getattr(s, 'Name', '?')} "
                          f"Value={getattr(s, 'SpecValue', getattr(s, 'Value', '?'))} "
                          f"Active={getattr(s, 'IsActive', '?')}")
                except Exception as e:
                    print(f"    [{k}] {type(e).__name__}")
        except Exception as e:
            print("  Specs read err:", type(e).__name__, e)


if __name__ == "__main__":
    main()
