"""Read back BOTH feed stages of the extractive column (feed + entrainer)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

OUT = ROOT / "report" / "dwsim"
NAME = "verify_extractive_eac_npac_dmso_dp.dwxmz"


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(OUT / NAME))

    column = None
    streams = {}
    for item in fs.SimulationObjects.Values:
        cls = item.GetType().Name
        if cls in ("DistillationColumn", "AbsorptionColumn"):
            column = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        elif cls == "MaterialStream":
            streams[str(item.GraphicObject.Tag)] = (
                item.GetAsObject() if hasattr(item, "GetAsObject") else item
            )

    print(f"=== {NAME} ===")
    print(f"  stages = {column.NumberOfStages}   dP = {column.ColumnPressureDrop} Pa")
    for tag in ("Feed", "Entrainer", "Distillate", "Bottoms"):
        st = streams.get(tag)
        if st is None:
            print(f"  {tag}: not found")
            continue
        try:
            idx = column.GetStreamFeedStageIndex(st)
            print(f"  {tag}: feed-stage index = {idx}"
                  + (f"  (1-based {idx + 1})" if idx is not None and idx >= 0 else ""))
        except Exception as e:
            print(f"  {tag}: GetStreamFeedStageIndex -> {type(e).__name__}")


if __name__ == "__main__":
    main()
