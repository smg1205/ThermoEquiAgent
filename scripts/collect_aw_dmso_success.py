"""为 report/success/乙酸-水-DMSO/ 汇总三元体系的全部工程文件与数据。

仿照 report/success/正庚烷-正壬烷/ 的结构：
  - 三元 TP-flash 泡点文件（.dwxmz）
  - 三源对照 CSV（pressure_kPa, x_..., T_exp, T_tf, T_dwsim, y_exp, y_tf, y_dwsim, file）
  - 设计 JSON（体系、数据源、物性包、进料、产品目标、短节法设计、严格塔实算）

数据来源：
  report/dwsim/three_source_system2_aw_dmso.csv   三源逐点数据
  report/dwsim/design_aw_dmso_multi_seed.json     各源短节法设计
  data/exports/flow_examples/water_acetic_acid_dmso_extractive_thermoformer.dwxmz  严格塔文件
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

THREE_SOURCE = ROOT / "report" / "dwsim" / "three_source_system2_aw_dmso.csv"
DESIGN_JSON = ROOT / "report" / "dwsim" / "design_aw_dmso_multi_seed.json"
SUMMARY_JSON = ROOT / "report" / "dwsim" / "three_source_system2_aw_dmso.json"

FLASH_SRC = ROOT / "report" / "dwsim"
COLUMN_SRC = (
    ROOT / "data" / "exports" / "flow_examples"
    / "water_acetic_acid_dmso_extractive_thermoformer.dwxmz"
)

DEST = ROOT / "report" / "success" / "乙酸-水-DMSO"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    # ---- 1. 复制三元 TP-flash 文件 ----
    copied = []
    for src in sorted(FLASH_SRC.glob("aw_dmso_x*_3comp_bubble_*.dwxmz")):
        shutil.copy2(src, DEST / src.name)
        copied.append(src.name)
    print(f"复制 {len(copied)} 个 TP-flash 文件")

    # ---- 2. 复制严格塔文件 ----
    if COLUMN_SRC.is_file():
        shutil.copy2(COLUMN_SRC, DEST / COLUMN_SRC.name)
        print(f"复制严格塔文件: {COLUMN_SRC.name}")
    else:
        print(f"[警告] 严格塔文件不存在: {COLUMN_SRC}")

    # ---- 3. 生成三源对照 CSV ----
    rows = list(csv.DictReader(open(THREE_SOURCE, encoding="utf-8")))
    tp_map = {}
    for name in copied:
        # 文件名形如 aw_dmso_x0p316_x0p279_3comp_bubble_84.5C.dwxmz
        # 注意 "x0p93" 表示 0.093（去掉前缀 "0p" 后补 "0."），不能直接当 0.93 解析。
        parts = name.replace("aw_dmso_", "").split("_")
        digits1 = parts[0].replace("x0p", "")
        digits2 = parts[1].replace("x0p", "")
        x1 = float("0." + digits1)
        x2 = float("0." + digits2)
        tp_map[(digits1, digits2)] = name

    out_csv = DEST / "aw_dmso_three_source_bubble.csv"
    # TF 列取各实验温度下的**等温**泡点预测（输入 T、x，输出 y），故不产出独立的
    # 泡点温度；T_thermoformer_K 一列因此留空，与 DWSIM 的等压泡点（有 T）不同。
    fieldnames = [
        "pressure_kPa", "x_acetic_acid", "x_water",
        "T_experiment_K", "T_thermoformer_K", "T_dwsim_K",
        "y_acetic_acid_experiment", "y_water_experiment", "y_dmso_experiment",
        "y_acetic_acid_thermoformer", "y_water_thermoformer", "y_dmso_thermoformer",
        "y_acetic_acid_dwsim", "y_water_dwsim", "y_dmso_dwsim",
        "file",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # 把 x 值转成文件名里的数字串：0.3164 -> "316"，0.093 -> "93"
            def to_digits(value: str) -> str:
                text = f"{float(value):.3f}"          # "0.316" / "0.093"
                return text.split(".")[1].rstrip("0") or "0"

            d1 = to_digits(row["x_acetic"])
            d2 = to_digits(row["x_water"])
            writer.writerow({
                "pressure_kPa": 13.33,
                "x_acetic_acid": row["x_acetic"],
                "x_water": row["x_water"],
                "T_experiment_K": row["T_exp_K"],
                "T_thermoformer_K": "n/a",   # 等温预测，无泡点温度
                "T_dwsim_K": row["T_dwsim_K"],
                "y_acetic_acid_experiment": row["y_acetic_exp"],
                "y_water_experiment": row["y_water_exp"],
                "y_dmso_experiment": row["y_dmso_exp"],
                "y_acetic_acid_thermoformer": row["y_acetic_seed_2"],
                "y_water_thermoformer": row["y_water_seed_2"],
                "y_dmso_thermoformer": row["y_dmso_seed_2"],
                "y_acetic_acid_dwsim": row["y_acetic_dwsim"],
                "y_water_dwsim": row["y_water_dwsim"],
                "y_dmso_dwsim": row["y_dmso_dwsim"],
                "file": tp_map.get((d1, d2), ""),
            })
    print(f"写出三源 CSV: {out_csv.name} ({len(rows)} 行)")

    # ---- 4. 生成设计 JSON ----
    designs = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    seed2 = designs["designs"]["seed_2"]
    unifac = designs["designs"]["unifac"]

    payload = {
        "system": "acetic acid / water / dimethyl sulfoxide (DMSO)",
        "experimental_source": "NIST ThermoML DOI 10.1016/j.fluid.2008.09.010",
        "mode": "extractive distillation; DMSO (highest boiler) as entrainer",
        "pressure_kPa": 101.325,
        "normal_boiling_points_K": {
            "water": 319.55,
            "acetic_acid": 390.94,
            "dimethyl_sulfoxide": 463.7,
        },
        "key_pair": {
            "light": "water",
            "heavy": "acetic_acid",
            "entrainer": "dimethyl_sulfoxide",
        },
        "feed": {
            "flow_mol_s": 1.0,
            "water_mole_fraction": 0.5,
            "acetic_acid_mole_fraction": 0.5,
        },
        "entrainer": {"ratio": 2.0, "flow_mol_s": 2.0},
        "product_targets": {
            "distillate_water_mole_fraction": 0.95,
            "water_recovery_to_distillate": 0.90,
        },
        "material_balance_mol_s": {
            "distillate": 0.473684,
            "bottoms": 2.526316,
            "closure": "D + B = 3.0 = feed + entrainer",
        },
        "three_source_accuracy": {
            "note": "17 experimental points, isobaric 13.33 kPa",
            "temperature_note": (
                "ThermoFormer was run as an ISOTHERMAL bubble point (input T and x, "
                "output y), so it produces no bubble temperature; the DWSIM column is "
                "an isobaric bubble point and does. Hence T_thermoformer_K is 'n/a' "
                "in the CSV."
            ),
            "dwsim": summary["dwsim"],
            "thermoformer_seeds": summary["seeds"],
        },
        "property_package_note": (
            "DWSIM has NO built-in NRTL binary parameters for any of the three pairs "
            "(acetic acid/water, acetic acid/DMSO, water/DMSO); it estimates them with "
            "alpha defaulting to 0.2. For point-wise bubble comparison this distorts the "
            "flash (vapour appears at 300 K, dT = +108..+142 K), so UNIFAC is used there. "
            "The rigorous column file uses NRTL and converges."
        ),
        "shortcut_column_design": {
            "source": "seed_2 (ThermoFormer vle_overall_ternary)",
            "alpha_base": seed2["alpha_base"],
            "alpha_ext": seed2["alpha_ext"],
            "selectivity": seed2["selectivity"],
            "alpha_avg": seed2["alpha_avg"],
            "theoretical_stages": seed2["theoretical_stages"],
            "minimum_stages": seed2["minimum_stages"],
            "reflux_ratio": seed2["reflux_ratio"],
            "minimum_reflux_ratio": seed2["minimum_reflux_ratio"],
            "feed_stage": seed2["feed_stage"],
            "entrainer_stage": seed2["entrainer_stage"],
        },
        "shortcut_column_design_unifac": {
            "source": "unifac",
            "alpha_base": unifac["alpha_base"],
            "alpha_ext": unifac["alpha_ext"],
            "selectivity": unifac["selectivity"],
            "theoretical_stages": unifac["theoretical_stages"],
            "reflux_ratio": unifac["reflux_ratio"],
            "feed_stage": unifac["feed_stage"],
            "entrainer_stage": unifac["entrainer_stage"],
        },
        "dwsim_rigorous_column": {
            "file": "water_acetic_acid_dmso_extractive_thermoformer.dwxmz",
            "property_package": "NRTL",
            "solving_method": "Napthali-Sandholm (Simultaneous Correction)",
            "max_iterations": 100,
            "stages": 15,
            "reflux_ratio": 1.441,
            "pressure_drop_kPa": 5.0,
            "specifications": {
                "condenser": {"type": "Stream_Ratio", "value": 1.441},
                "reboiler": {"type": "Product_Molar_Flow_Rate", "value": 2.526316,
                             "unit": "mol/s"},
            },
            "feeds": {
                "key_pair": {"stage": 14, "flow_mol_s": 1.0, "temperature_K": 366.01,
                             "composition": [0.5, 0.5, 0.0]},
                "entrainer": {"stage": 3, "flow_mol_s": 2.0, "temperature_K": 354.10,
                              "composition": [0.0, 0.0, 1.0]},
            },
            "converged": True,
            "solve_error_count": 0,
            "results": {
                "distillate": {"temperature_K": 412.59, "flow_mol_s": 0.4737,
                               "composition": [0.5036, 0.0289, 0.4675]},
                "bottoms": {"temperature_K": 462.57, "flow_mol_s": 2.5263,
                            "composition": [0.1035, 0.1925, 0.7040]},
            },
            "notes": [
                "feeds moved down (7->14, 2->3) relative to the shortcut estimate to converge",
                "bottoms temperature 462.57 K matches the DMSO boiling point 463.7 K",
                "distillate water fraction 0.5036 falls short of the 0.95 shortcut target",
            ],
        },
        "bubble_point_files": copied,
    }

    out_json = DEST / "aw_dmso_extractive_design.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"写出设计 JSON: {out_json.name}")

    print(f"\n目录内容 ({DEST}):")
    for item in sorted(DEST.iterdir()):
        print(f"  {item.name}  ({item.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
