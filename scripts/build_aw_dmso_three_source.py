"""三源对比：乙酸 / 水 / 二甲基亚砜（DMSO），替换 v5 报告 §2.2 的三元 VLE 体系。

三源定义（与报告 §2.2 一致）：
  实验   NIST ThermoML 三元 VLE 数据集（DOI 10.1016/j.fluid.2008.09.010）
  TF     vle_overall_ternary 权值的等温泡点预测
  DWSIM  Automation API 等压泡点实算（UNIFAC 物性包，见 PROPERTY_PACKAGE 说明）

与原体系的差别：
  原 §2.2 为 1-丁醇 / 水 / 甲苯（43 点，选择性 0.75 < 1，萃取不成立）
  本体系为 乙酸 / 水 / DMSO（17 点，选择性 2.37 > 1，萃取成立）
  组分顺序按 dataset 的 component_1/2/3 = 乙酸 / 水 / DMSO

泡点求解方法：固定 P 与总组成，二分温度，以 ``PhaseIds`` 判定相态
（['Liquid'] -> 未到泡点；出现 'Vapor' -> 已越过泡点），取刚出现汽相的温度。
``PhaseIds`` 是 DWSIM 判定相态的权威依据，比按温度或组成猜相可靠。

本脚本**只做计算**，把结果写成 CSV/JSON；报告正文由人工据此更新。
计算流程与本仓库既有的
``scripts/run_water_butanol_dwsim.py`` / ``build_butanol_water_toluene_three_source.py``
保持一致：DWSIM 侧对汽相摩尔流量二分求泡点（等压），TF 侧直接取泡点预测。

注意：torch 与 pythonnet(clr) 不能共存，故 TF 与 DWSIM 两阶段分开、各自独立运行，
中间以 JSON 传递。本脚本用命令行参数选择阶段。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\datasets\vle\ternary_vle_english.csv")
CKPT_ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\models\vle\prediction\vle_overall_ternary")
OUTDIR = ROOT / "report" / "dwsim"

DOI = "10.1016/j.fluid.2008.09.010"
ORIGINAL_NAMES = ("乙酸", "水", "二甲基亚砜")

#: 与 dataset 的 component_1/2/3 顺序严格对应（乙酸 / 水 / DMSO）
COMPONENTS = [
    {"name": "acetic acid", "smiles": "CC(=O)O", "dwsim": "Acetic acid"},
    {"name": "water", "smiles": "O", "dwsim": "Water"},
    {"name": "dimethyl sulfoxide", "smiles": "CS(=O)C", "dwsim": "Dimethyl sulfoxide"},
]

P_KPA_DEFAULT = 13.33          # 该数据集为等压 ~13.33 kPa（99.98 mmHg）

#: DWSIM 物性包。**必须用 UNIFAC，不能用 NRTL。**
#:
#: 实测（本机 DWSIM）：乙酸/水、乙酸/DMSO、水/DMSO 这三对**都没有内置 NRTL
#: 二元交互参数**，DWSIM 会走 ``EstimateMissingInteractionParameters`` 估算，
#: 且 α 一律取默认值 0.2：
#:     Acetic acid/Water:              -189.10 / 728.84 / 0.2
#:     Acetic acid/Dimethyl sulfoxide: -1233.93 / -1295.15 / 0.2
#:     Water/Dimethyl sulfoxide:       -1999.94 / -499.97 / 0.2
#: 用这套伪参数算出的闪蒸完全失真：300 K（低于所有组分沸点）即出现汽相、
#: 液相组成剧烈跳动、泡点二分收敛到搜索上界，温度偏差达 +108 ~ +142 K。
#:
#: UNIFAC 是基团贡献法，不需要二元交互参数，实测这三对均不触发任何
#: "Estimated ... IP set" 提示，相态行为随温度单调合理，故改用它。
PROPERTY_PACKAGE = "UNIFAC"


def load_points() -> list[dict]:
    """读取实验点（仅 DMSO 组，排除同 DOI 下的 DMF 数据）。"""
    points: list[dict] = []
    with open(DATASET, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["doi"] != DOI:
                continue
            if row["component_3_original_name"] != ORIGINAL_NAMES[2]:
                continue
            points.append(
                {
                    "T_exp_K": float(row["temperature_c"]) + 273.15,
                    "P_kpa": float(row["source_pressure_kpa"]),
                    "x1": float(row["x1"]),
                    "x2": float(row["x2"]),
                    "y1_exp": float(row["y1"]),
                    "y2_exp": float(row["y2"]),
                }
            )
    return points


# ---------------------------------------------------------------- TF 阶段
def run_tf(seeds: list[str]) -> dict:
    os.environ.pop("THERMOFORMER_CHECKPOINT", None)
    os.environ["THERMOFORMER_SRC"] = r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\src"
    os.environ["THERMOFORMER_USE_CUDA"] = "0"

    from schemas.domain import (
        ComponentIdentity,
        TaskManifest,
        ThermodynamicConditions,
    )
    from thermo_engine.thermoformer_backend import (
        ThermoFormerBackend,
        ThermoFormerSettings,
        resolve_settings,
    )

    points = load_points()
    base = resolve_settings()

    species = [
        ComponentIdentity(
            component_id=c["name"].replace(" ", "_"),
            name=c["name"],
            smiles=c["smiles"],
            aliases=[],
        )
        for c in COMPONENTS
    ]

    out: dict[str, list[list[float]]] = {}
    for seed in seeds:
        ckpt = CKPT_ROOT / seed / "best_model.pt"
        if not ckpt.is_file():
            print(f"[TF {seed}] 权值缺失，跳过")
            continue
        backend = ThermoFormerBackend(
            ThermoFormerSettings(
                src_path=base.src_path,
                checkpoint_path=ckpt,
                feature_cache_path=base.feature_cache_path,
                use_cuda=False,
            )
        )
        preds = []
        for point in points:
            x3 = max(0.0, 1.0 - point["x1"] - point["x2"])
            request = TaskManifest(
                equilibrium_type="VLE",
                calculation_type="bubble_point",
                components=species,
                conditions=ThermodynamicConditions(
                    temperature_K=point["T_exp_K"],
                    liquid_composition=[point["x1"], point["x2"], x3],
                ),
                model_name="ThermoFormer",
            )
            result = backend.bubble_point(request)
            y = (
                [float(v) for v in result.points[0].vapor_composition]
                if result.points
                else [float("nan")] * 3
            )
            preds.append(y)
        out[seed] = preds
        print(f"[TF {seed}] 完成 {len(preds)} 点")

    return {"points": points, "tf": out}


# ------------------------------------------------------------- DWSIM 阶段
def run_dwsim() -> dict:
    """对每个实验点做等压泡点实算：二分汽相摩尔流量使泡点条件成立。

    做法与仓库既有三源脚本一致 —— 固定 T、P 与液相总组成，
    调汽相分率直到液相组成回到设定值（即泡点）。
    """
    from thermo_engine.dwsim_export import _add_property_package, _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    points = load_points()
    results: list[dict] = []

    for index, point in enumerate(points):
        x3 = max(0.0, 1.0 - point["x1"] - point["x2"])
        P_pa = point["P_kpa"] * 1000.0

        from System import Array, Double

        def flash_at(T_K: float) -> tuple[list[str], list[float] | None, list[float] | None]:
            """在 (T, P, z) 下闪蒸，返回 (存在的相, 汽相组成, 液相组成)。

            ``PhaseIds`` 是 DWSIM 判定相态的权威依据（如 ['Liquid']、
            ['Vapor','Liquid']、['Vapor']），比按温度猜相可靠得多。
            """
            flowsheet = automation.CreateFlowsheet()
            for c in COMPONENTS:
                flowsheet.AddCompound(c["dwsim"])
            _add_property_package(flowsheet, PROPERTY_PACKAGE)
            stream = flowsheet.AddObject(
                object_type.MaterialStream, 0, 0, "F"
            ).GetAsObject()
            stream.SetTemperature(T_K)
            stream.SetPressure(P_pa)
            stream.SetMolarFlow(1.0)
            stream.SetOverallComposition(
                Array[Double]([point["x1"], point["x2"], x3])
            )
            try:
                errors = automation.CalculateFlowsheet4(flowsheet)
            except Exception:  # noqa: BLE001
                return [], None, None
            if errors.Count:
                return [], None, None

            ids = [str(v) for v in stream.PhaseIds]
            vapor = liquid = None
            phases = stream.Phases
            for i in range(phases.Count):
                phase = phases[i]
                try:
                    name = str(phase.Name)
                    values = [getattr(cc, "MoleFraction") for cc in phase.Compounds.Values]
                except Exception:  # noqa: BLE001
                    continue
                if not values or any(v is None for v in values):
                    continue
                floats = [float(v) for v in values]
                if any(v != v for v in floats):
                    continue
                if name == "Vapor" and any(abs(v) > 1e-12 for v in floats):
                    vapor = floats
                elif name.startswith("Liquid") and name != "OverallLiquid" \
                        and any(abs(v) > 1e-12 for v in floats):
                    liquid = floats
            return ids, vapor, liquid

        # ---- 泡点温度：固定 P 与总组成，二分 T 使相态由全液相转为两相 ----
        # 判据用 PhaseIds：['Liquid'] 表示未到泡点，出现 'Vapor' 表示已越过。
        lo, hi = 200.0, 550.0
        ids_lo, _, _ = flash_at(lo)
        ids_hi, _, _ = flash_at(hi)

        T_dwsim = float("nan")
        y_dwsim: list[float] = [float("nan")] * 3

        if ids_lo and "Vapor" not in ids_lo and ids_hi and "Vapor" in ids_hi:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                ids_mid, _, _ = flash_at(mid)
                if not ids_mid:
                    break
                if "Vapor" in ids_mid:
                    hi = mid          # 已越过泡点
                else:
                    lo = mid          # 仍为全液相
                if hi - lo < 1e-3:
                    break
            T_dwsim = hi          # 取刚出现汽相的温度作为泡点

        # 在泡点稍上方取汽相组成作为 DWSIM 预测
        if T_dwsim == T_dwsim:
            _, vapor, _ = flash_at(T_dwsim)
            if vapor is not None:
                y_dwsim = vapor

        results.append({"T_dwsim_K": T_dwsim, "y_dwsim": y_dwsim})
        dT = T_dwsim - point["T_exp_K"]
        print(
            f"[DWSIM {PROPERTY_PACKAGE}] {index + 1}/{len(points)} "
            f"T_exp={point['T_exp_K']:.2f} T_dwsim={T_dwsim:.2f} "
            f"dT={dT:+.2f} K"
        )

    return {"points": points, "dwsim": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tf", "dwsim", "merge"], required=True)
    parser.add_argument("--seeds", default="seed_2")
    parser.add_argument("--tf-json", default=str(OUTDIR / "three_source_aw_dmso_tf.json"))
    parser.add_argument("--dwsim-json", default=str(OUTDIR / "three_source_aw_dmso_dwsim.json"))
    args = parser.parse_args()

    if args.stage == "tf":
        payload = run_tf([s.strip() for s in args.seeds.split(",") if s.strip()])
        Path(args.tf_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote: {args.tf_json}")
        return 0

    if args.stage == "dwsim":
        payload = run_dwsim()
        Path(args.dwsim_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote: {args.dwsim_json}")
        return 0

    # merge
    tf_payload = json.loads(Path(args.tf_json).read_text(encoding="utf-8"))
    dw_payload = json.loads(Path(args.dwsim_json).read_text(encoding="utf-8"))
    points = tf_payload["points"]
    dwsim_rows = dw_payload["dwsim"]

    seeds = list(tf_payload["tf"].keys())
    if not seeds:
        raise SystemExit("TF 结果为空")

    csv_path = OUTDIR / "three_source_system2_aw_dmso.csv"
    header = [
        "T_exp_K", "T_dwsim_K",
        "x_acetic", "x_water", "x_dmso",
        "y_acetic_exp", "y_water_exp", "y_dmso_exp",
    ]
    for seed in seeds:
        header += [f"y_acetic_{seed}", f"y_water_{seed}", f"y_dmso_{seed}"]
    header += ["y_acetic_dwsim", "y_water_dwsim", "y_dmso_dwsim"]

    rows = []
    for index, point in enumerate(points):
        x3 = max(0.0, 1.0 - point["x1"] - point["x2"])
        row = [
            round(point["T_exp_K"], 2),
            round(dwsim_rows[index]["T_dwsim_K"], 2),
            point["x1"], point["x2"], round(x3, 4),
            point["y1_exp"], point["y2_exp"], round(1 - point["y1_exp"] - point["y2_exp"], 4),
        ]
        for seed in seeds:
            row += [round(v, 4) for v in tf_payload["tf"][seed][index]]
        row += [round(v, 4) for v in dwsim_rows[index]["y_dwsim"]]
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote: {csv_path}")

    # 汇总统计
    summary: dict = {"n_points": len(points), "seeds": {}}
    for seed in seeds:
        errs = []
        per = [[], [], []]
        for index, point in enumerate(points):
            exp = [
                point["y1_exp"],
                point["y2_exp"],
                1 - point["y1_exp"] - point["y2_exp"],
            ]
            y = tf_payload["tf"][seed][index]
            for k in range(3):
                d = abs(y[k] - exp[k])
                per[k].append(d)
                errs.append(d)
        summary["seeds"][seed] = {
            "mae_overall": round(sum(errs) / len(errs), 4),
            "mae_acetic": round(sum(per[0]) / len(per[0]), 4),
            "mae_water": round(sum(per[1]) / len(per[1]), 4),
            "mae_dmso": round(sum(per[2]) / len(per[2]), 4),
        }

    dw_errs = []
    dw_per = [[], [], []]
    dT = []
    for index, point in enumerate(points):
        exp = [point["y1_exp"], point["y2_exp"], 1 - point["y1_exp"] - point["y2_exp"]]
        y = dwsim_rows[index]["y_dwsim"]
        for k in range(3):
            d = abs(y[k] - exp[k])
            dw_per[k].append(d)
            dw_errs.append(d)
        dT.append(dwsim_rows[index]["T_dwsim_K"] - point["T_exp_K"])
    summary["dwsim"] = {
        "mae_overall": round(sum(dw_errs) / len(dw_errs), 4),
        "mae_acetic": round(sum(dw_per[0]) / len(dw_per[0]), 4),
        "mae_water": round(sum(dw_per[1]) / len(dw_per[1]), 4),
        "mae_dmso": round(sum(dw_per[2]) / len(dw_per[2]), 4),
        "dT_mean": round(sum(dT) / len(dT), 2),
    }

    json_path = OUTDIR / "three_source_system2_aw_dmso.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote: {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
