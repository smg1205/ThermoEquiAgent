"""Verify both generated IPA/water columns (UNIFAC-source and TF-source)."""
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


def describe(automation, path: Path):
    fs = automation.LoadFlowsheet2(str(path))
    print(f"\n=== {path.name} ===")
    try:
        print("  compounds    :", [str(k) for k in list(fs.SelectedCompounds.Keys)])
    except Exception:
        pass
    for pp in fs.PropertyPackages.Values:
        try:
            print("  property pkg :", pp.GetType().Name)
        except Exception:
            pass

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
    print(f"  stages       : {column.NumberOfStages}")
    print(f"  pressure drop: {column.ColumnPressureDrop} Pa")
    print(f"  reflux ratio : {column.RefluxRatio}")
    for tag in ("Feed", "Distillate", "Bottoms"):
        st = streams.get(tag)
        if st is None:
            print(f"  {tag}: missing")
            continue
        try:
            comp = [round(float(c), 4) for c in st.GetOverallComposition()]
            flow = float(st.GetMolarFlow())
            temp = float(st.GetTemperature())
        except Exception:
            comp, flow, temp = None, None, None
        extra = ""
        if tag == "Feed":
            try:
                extra = f"  feed-stage={column.GetStreamFeedStageIndex(st)} (1-based {column.GetStreamFeedStageIndex(st) + 1})"
            except Exception:
                pass
        print(f"  {tag:10s}  : flow={flow} mol/s  T={temp:.2f} K  z={comp}{extra}")


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    for name in FILES:
        p = OUT / name
        if not p.is_file():
            print(f"MISSING: {name}")
            continue
        describe(automation, p)


if __name__ == "__main__":
    main()
