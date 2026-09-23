"""用 5 个权值分别预测 水/乙酸/DMSO 三元 VLE，并与实验数据对照。

体系：乙酸 / 水 / 二甲基亚砜（DMSO）
数据：datasets/vle/ternary_vle_english.csv，DOI 10.1016/j.fluid.2008.09.010
      等压 ~99.98 mmHg，17 个 DMSO 数据点（NIST ThermoML verified）

目的：比较 vle_overall_ternary 的 5 个 seed 权值（seed_0 ~ seed_4）在该体系上的
      预测表现，给出逐点误差与汇总统计，供选型参考。

做法：对每个实验点，用给定权值做等温泡点预测（输入 T、x，输出 y），
      再与实验 y 比较。与 thermo_engine.column_design 使用的
      ThermoFormer backend 口径一致。

注意：torch 与 pythonnet(clr) 不能共存，本脚本不导入 DWSIM。
"""

from __future__ import annotations

import csv
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

DATASET = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\datasets\vle\ternary_vle_english.csv")
CKPT_ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\models\vle\prediction\vle_overall_ternary")
SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]
DOI = "10.1016/j.fluid.2008.09.010"

# 体系组分：顺序必须与 dataset 的 component_1/2/3 一致
COMPONENTS = [
    {"name": "acetic acid", "smiles": "CC(=O)O"},
    {"name": "water", "smiles": "O"},
    {"name": "dimethyl sulfoxide", "smiles": "CS(C)=O"},
]
ORIGINAL_NAMES = ("乙酸", "水", "二甲基亚砜")


def load_points() -> list[dict]:
    points = []
    with open(DATASET, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["doi"] != DOI:
                continue
            if row["component_3_original_name"] != ORIGINAL_NAMES[2]:
                continue      # 排除同 DOI 下的 DMF 数据
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


def predict(backend, points: list[dict]) -> list[list[float]]:
    """对每个点做泡点预测，返回预测的 y 列表。"""
    from schemas.domain import (
        ComponentIdentity,
        TaskManifest,
        ThermodynamicConditions,
    )

    species = [
        ComponentIdentity(
            component_id=c["name"].replace(" ", "_"),
            name=c["name"],
            smiles=c["smiles"],
            aliases=[],
        )
        for c in COMPONENTS
    ]

    results = []
    for point in points:
        x3 = max(0.0, 1.0 - point["x1"] - point["x2"])
        request = TaskManifest(
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            components=species,
            conditions=ThermodynamicConditions(
                temperature_K=point["T_c"] + 273.15,
                liquid_composition=[point["x1"], point["x2"], x3],
            ),
            model_name="ThermoFormer",
        )
        result = backend.bubble_point(request)
        y = list(result.points[0].vapor_composition) if result.points else []
        results.append([float(v) for v in y] if len(y) >= 3 else [float("nan")] * 3)
    return results


def main() -> int:
    from thermo_engine.thermoformer_backend import (
        ThermoFormerBackend,
        ThermoFormerSettings,
        resolve_settings,
    )

    points = load_points()
    print(f"体系: 乙酸 / 水 / 二甲基亚砜")
    print(f"数据源: {DOI}")
    print(f"数据点: {len(points)} 个（等压 ~{points[0]['P_kpa']:.2f} kPa）")
    print(f"权值: {', '.join(SEEDS)}\n")

    base = resolve_settings()
    summary: dict[str, dict] = {}
    all_preds: dict[str, list[list[float]]] = {}

    for seed in SEEDS:
        ckpt = CKPT_ROOT / seed / "best_model.pt"
        if not ckpt.is_file():
            print(f"[{seed}] 权值缺失，跳过")
            continue
        backend = ThermoFormerBackend(
            ThermoFormerSettings(
                src_path=base.src_path,
                checkpoint_path=ckpt,
                feature_cache_path=base.feature_cache_path,
                use_cuda=False,
            )
        )
        preds = predict(backend, points)
        all_preds[seed] = preds

        # 逐组分 MAE 与总体 MAE
        errors = []
        per_comp = [[], [], []]
        for point, y in zip(points, preds, strict=True):
            exp = [point["y1"], point["y2"], 1.0 - point["y1"] - point["y2"]]
            for k in range(3):
                diff = abs(y[k] - exp[k])
                per_comp[k].append(diff)
                errors.append(diff)

        summary[seed] = {
            "mae_overall": sum(errors) / len(errors),
            "mae_acetic": sum(per_comp[0]) / len(per_comp[0]),
            "mae_water": sum(per_comp[1]) / len(per_comp[1]),
            "mae_dmso": sum(per_comp[2]) / len(per_comp[2]),
            "worst": max(errors),
        }
        s = summary[seed]
        print(f"[{seed}] MAE={s['mae_overall']:.4f}  "
              f"乙酸={s['mae_acetic']:.4f} 水={s['mae_water']:.4f} "
              f"DMSO={s['mae_dmso']:.4f}  最差={s['worst']:.4f}")

    # 汇总排序
    print("\n=== 按总体 MAE 排序 ===")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mae_overall"])
    for rank, (seed, s) in enumerate(ranked, 1):
        print(f"  {rank}. {seed:7} MAE={s['mae_overall']:.4f}")

    # 逐点对照表（以最佳权值展示）
    if ranked:
        best = ranked[0][0]
        print(f"\n=== 逐点对照（最佳权值 {best}）===")
        hdr = (f"{'T(℃)':>7} {'x乙酸':>7} {'x水':>7} {'x DMSO':>7} | "
               f"{'y乙酸_exp':>9} {'y乙酸_pred':>10} | {'y水_exp':>8} {'y水_pred':>9} | "
               f"{'yDMSO_exp':>9} {'yDMSO_pred':>10} {'max_err':>8}")
        print(hdr)
        print("-" * len(hdr))
        worst_rows = []
        for point, y in zip(points, all_preds[best], strict=True):
            x3 = 1.0 - point["x1"] - point["x2"]
            exp = [point["y1"], point["y2"], 1.0 - point["y1"] - point["y2"]]
            err = max(abs(y[k] - exp[k]) for k in range(3))
            worst_rows.append(err)
            print(f"{point['T_c']:>7.2f} {point['x1']:>7.4f} {point['x2']:>7.4f} "
                  f"{x3:>7.4f} | {exp[0]:>9.4f} {y[0]:>10.4f} | "
                  f"{exp[1]:>8.4f} {y[1]:>9.4f} | {exp[2]:>9.4f} {y[2]:>10.4f} "
                  f"{err:>8.4f}")

    out = ROOT / "report" / "dwsim" / "dmso_aw_seed_comparison.json"
    out.write_text(
        json.dumps(
            {
                "system": "acetic acid / water / dimethyl sulfoxide",
                "doi": DOI,
                "n_points": len(points),
                "pressure_kpa": points[0]["P_kpa"],
                "summary": summary,
                "points": points,
                "predictions": all_preds,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
