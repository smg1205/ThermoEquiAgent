"""Verify the generated IPA/water column: stages, pressure drop, feed stage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

FILE = ROOT / "report" / "dwsim" / "ipa_water_binary_column_x0p5.dwxmz"


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))

    print(f"=== {FILE.name} ===")
    # compounds
    try:
        comps = [str(k) for k in list(fs.SelectedCompounds.Keys)]
        print("  compounds     :", comps)
    except Exception as e:
        print("  compounds read err:", type(e).__name__)
    # property package
    for pp in fs.PropertyPackages.Values:
        print("  property pkg  :", pp.GetType().Name if hasattr(pp, "GetType") else pp)

    column = None
    streams = {}
    for item in fs.SimulationObjects.Values:
        cls = item.GetType().Name
        tag = str(item.GraphicObject.Tag)
        if cls in ("DistillationColumn", "AbsorptionColumn", "ShortcutColumn"):
            column = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        elif cls == "MaterialStream":
            streams[tag] = item.GetAsObject() if hasattr(item, "GetAsObject") else item

    if column is None:
        print("  column not found")
        return

    print(f"  column        : {column.GetType().Name}")
    print(f"  stages        : {column.NumberOfStages}")
    print(f"  pressure drop : {column.ColumnPressureDrop} Pa")
    print(f"  reflux ratio  : {column.RefluxRatio}")

    for tag in ("Feed", "Distillate", "Bottoms"):
        st = streams.get(tag)
        if st is None:
            print(f"  {tag}: missing")
            continue
        try:
            comp = [float(c) for c in st.GetOverallComposition()]
        except Exception:
            comp = None
        try:
            flow = float(st.GetMolarFlow())
            temp = float(st.GetTemperature())
            pres = float(st.GetPressure())
        except Exception:
            flow = temp = pres = None
        line = f"  {tag:10s}: flow={flow} mol/s  T={temp} K  P={pres} Pa  z={comp}"
        if tag == "Feed":
            try:
                line += f"  feed-stage={column.GetStreamFeedStageIndex(st)}"
            except Exception:
                pass
        print(line)


if __name__ == "__main__":
    main()
