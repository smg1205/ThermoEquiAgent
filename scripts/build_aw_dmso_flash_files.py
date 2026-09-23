"""生成 乙酸 / 水 / DMSO 三元 TP-flash 工程文件，供报告 §1.4 的 SI 截图使用。

流程结构与本仓库既有的三元泡点文件一致：
    Feed (MaterialStream) → Vessel (TP flash) → Vapor / Liquid

物性包用 **UNIFAC**（理由见 §2.2：该体系三对二元在 DWSIM 中均无 NRTL 内置参数，
NRTL 会走估算且 α 取默认 0.2，闪蒸失真）。

温度取实验数据集中对应的泡点温度（等压 13.33 kPa），
这样打开文件计算后，汽相组成可直接与实验值比对。

输出：report/dwsim/aw_dmso_x*/ 下按点命名的 .dwxmz
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\datasets\vle\ternary_vle_english.csv")
OUTDIR = ROOT / "report" / "dwsim"

DOI = "10.1016/j.fluid.2008.09.010"
COMPOUNDS = ["Acetic acid", "Water", "Dimethyl sulfoxide"]
PROPERTY_PACKAGE = "UNIFAC"

#: 选取的代表点：覆盖富水端到富 DMSO 端（按 x_DMSO 排序后取 5 个）
def load_points() -> list[dict]:
    points = []
    with open(DATASET, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["doi"] != DOI or row["component_3_original_name"] != "二甲基亚砜":
                continue
            points.append(
                {
                    "T_c": float(row["temperature_c"]),
                    "P_kpa": float(row["source_pressure_kpa"]),
                    "x1": float(row["x1"]),
                    "x2": float(row["x2"]),
                    "y1": float(row["y1"]),
                    "y2": float(row["y2"]),
                }
            )
    return points


def main() -> int:
    from thermo_engine.dwsim_export import _add_property_package, _automation_factory

    points = load_points()
    # 按 x_DMSO 升序，均匀取 5 个代表点
    points.sort(key=lambda p: 1.0 - p["x1"] - p["x2"])
    idxs = [0, len(points) // 4, len(points) // 2, 3 * len(points) // 4, len(points) - 1]
    chosen = [points[i] for i in sorted(set(idxs))]

    factory, object_type = _automation_factory()
    automation = factory()

    from System import Array, Double

    written = []
    for point in chosen:
        x3 = max(0.0, 1.0 - point["x1"] - point["x2"])
        P_pa = point["P_kpa"] * 1000.0

        flowsheet = automation.CreateFlowsheet()
        for name in COMPOUNDS:
            flowsheet.AddCompound(name)
        _add_property_package(flowsheet, PROPERTY_PACKAGE)

        feed_w = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        vessel_w = flowsheet.AddObject(object_type.Vessel, 250, 0, "Flash")
        vap_w = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Vapor")
        liq_w = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Liquid")

        feed = feed_w.GetAsObject()
        feed.SetTemperature(point["T_c"] + 273.15)
        feed.SetPressure(P_pa)
        feed.SetMolarFlow(1.0)
        feed.SetOverallComposition(Array[Double]([point["x1"], point["x2"], x3]))

        vessel = vessel_w.GetAsObject()
        # Vessel 没有 SetTemperature/SetPressure：闪蒸规格写在 FlashTemperature /
        # FlashPressure 上（与仓库既有 rebuild_2p2_seed2.py 的口径一致）。
        vessel.FlashTemperature = point["T_c"] + 273.15
        vessel.FlashPressure = P_pa

        for from_w, to_w, fidx, tidx in (
            (feed_w, vessel_w, 0, 0),
            (vessel_w, vap_w, 0, 0),
            (vessel_w, liq_w, 1, 0),
        ):
            try:
                flowsheet.ConnectObjects(
                    from_w.GraphicObject, to_w.GraphicObject, fidx, tidx
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  连接失败 {type(exc).__name__}")

        # 文件名用 3 位小数的组成（与三源 CSV 的 x 列同精度），
        # 形如 x0p316 = 0.316、x0p093 = 0.093，便于与 CSV 逐行对应。
        def digits(value: float) -> str:
            return f"{value:.3f}".split(".")[1].rstrip("0") or "0"

        name = (
            f"aw_dmso_x0p{digits(point['x1'])}_x0p{digits(point['x2'])}"
            f"_3comp_bubble_{point['T_c']:.1f}C.dwxmz"
        )
        dest = OUTDIR / name
        try:
            automation.SaveFlowsheet(flowsheet, str(dest), False)
            written.append(name)
            print(f"  写出 {name}  ({dest.stat().st_size} 字节)")
        except Exception as exc:  # noqa: BLE001
            print(f"  保存失败 {name}: {type(exc).__name__}: {str(exc)[:70]}")

    print(f"\n共生成 {len(written)} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
