"""Verify the IPA/water x_IPA = 0.3 columns (both thermodynamic sources)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

OUT = ROOT / "report" / "dwsim"
FILES = [
    "ipa_water_binary_column_x0p3_unifac.dwxmz",
    "ipa_water_binary_column_x0p3_tf.dwxmz",
]


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    for name in FILES:
        p = OUT / name
        if not p.is_file():
            print(f"MISSING {name}")
            continue
        fs = automation.LoadFlowsheet2(str(p))
        print(f"\n=== {name} ===")
        try:
            print("  compounds     :", [str(k) for k in list(fs.SelectedCompounds.Keys)])
        except Exception:
            pass
        column, streams = None, {}
        for item in fs.SimulationObjects.Values:
            cls = item.GetType().Name
            tag = str(item.GraphicObject.Tag)
            if cls in ("DistillationColumn", "AbsorptionColumn", "ShortcutColumn"):
                column = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            elif cls == "MaterialStream":
                streams[tag] = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        if column is None:
            print("  column not found")
            continue
        print(f"  stages         : {column.NumberOfStages}")
        print(f"  pressure drop  : {column.ColumnPressureDrop} Pa")
        print(f"  reflux ratio   : {column.RefluxRatio}")
        print(f"  solver         : {column.SolvingMethodName}  (max iters {column.MaxIterations})")
        print(f"  estimate flags : T={column.UseTemperatureEstimates} "
              f"L={column.UseLiquidFlowEstimates} V={column.UseVaporFlowEstimates}")
        specs = column.Specs
        for k in list(specs.Keys):
            print(f"  spec[{k}]        : {specs[k].SpecValue} ({specs[k].SType})")
        for tag in ("Feed", "Distillate", "Bottoms"):
            st = streams.get(tag)
            if st is None:
                print(f"  {tag}: missing")
                continue
            comp = [round(float(c), 4) for c in st.GetOverallComposition()]
            flow = float(st.GetMolarFlow())
            temp = float(st.GetTemperature())
            extra = ""
            if tag == "Feed":
                try:
                    idx = column.GetStreamFeedStageIndex(st)
                    extra = f"  feed-stage={idx} (1-based {idx + 1})"
                except Exception:
                    pass
            print(f"  {tag:10s}     : flow={flow} mol/s  T={temp:.2f} K  z={comp}{extra}")


if __name__ == "__main__":
    main()
