"""Load the generated columns and read back the feed stage via GetStreamFeedStageIndex."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

FILES = {
    "verify_binary_ipa_water_dp.dwxmz": ("Feed", 19),
    "verify_extractive_eac_npac_dmso_dp.dwxmz": ("Feed", 9),
}
OUT = ROOT / "report" / "dwsim"


def main():
    factory, object_type = _automation_factory()
    automation = factory()

    for name, (feed_tag, expected_stage) in FILES.items():
        fs = automation.LoadFlowsheet2(str(OUT / name))
        print(f"\n=== {name} ===")
        column = None
        feed_stream = None
        for item in fs.SimulationObjects.Values:
            cls = item.GetType().Name
            tag = str(item.GraphicObject.Tag)
            if cls in ("DistillationColumn", "AbsorptionColumn", "ShortcutColumn"):
                column = item
            if tag == feed_tag:
                feed_stream = item
        if column is None or feed_stream is None:
            print(f"  column={column is not None} feed={feed_stream is not None}")
            continue
        # unwrap the automation interface where DWSIM 9 returns one
        column = column.GetAsObject() if hasattr(column, "GetAsObject") else column
        feed_stream = feed_stream.GetAsObject() if hasattr(feed_stream, "GetAsObject") else feed_stream

        print(f"  column={column.GetType().Name}  stages={column.NumberOfStages}")
        print(f"  ColumnPressureDrop(Pa) = {column.ColumnPressureDrop}")
        for meth in ("GetStreamFeedStageIndex",):
            m = getattr(column, meth, None)
            if callable(m):
                try:
                    idx = m(feed_stream)
                    print(f"  {meth}(Feed) = {idx}   (design feed_stage = {expected_stage}, 0-based {expected_stage - 1})")
                except Exception as e:
                    print(f"  {meth} failed: {type(e).__name__}: {str(e)[:80]}")
        # also try the string overload
        m2 = getattr(column, "SetStreamFeedStage", None)
        if callable(m2):
            try:
                got = column.GetStreamFeedStageIndex(feed_tag)
                print(f"  GetStreamFeedStageIndex('{feed_tag}') = {got}")
            except Exception as e:
                print(f"  string overload: {type(e).__name__}")


if __name__ == "__main__":
    main()
