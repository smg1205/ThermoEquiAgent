"""生成 水 / 乙酸 + DMSO 的 DWSIM 萃取精馏塔文件。

体系与本报告的对应关系
----------------------
关键对：水（轻）/ 乙酸（重），萃取剂：二甲基亚砜（DMSO）
常压沸点：水 319.55 K < 乙酸 390.94 K < DMSO 463.7 K
        => DMSO 沸点最高，满足「萃取剂全走塔釜」的短节法假设。
选择性 > 1（UNIFAC 6.4569、TF 各权值 1.62~2.37），即 DMSO 确实增强了
水/乙酸的相对挥发度 —— 这是真正意义上的萃取精馏
（对比原体系 1-丁醇/水/甲苯 的选择性 0.75/0.82 < 1，萃取不成立）。

设计来源
--------
``report/dwsim/design_aw_dmso_multi_seed.json``，由
``scripts/design_aw_dmso_multi_seed.py`` 经
``thermo_engine.column_design.design_ternary_extractive_column`` 算出。
用 ``--source`` 选择条目：``unifac``（默认）或 ``seed_0``..``seed_4``。

    UNIFAC  选择性 6.4569  N=11  R=0.451  进料板 5  塔顶 372.23 K  塔釜 459.70 K
    seed_2  选择性 2.3731  N=15  R=1.441  进料板 7  塔顶 354.10 K  塔釜 366.59 K

**默认使用 UNIFAC 源**：其塔釜温度 459.70 K 接近 DMSO 常压沸点 463.7 K，
物理上合理；而各 TF 权值给出的塔釜温度（355~392 K）低于乙酸沸点 390.94 K，
富 DMSO 的塔釜混合物不可能在该温度沸腾。TF 在本体系的三元预测 MAE 为
0.22~0.29（17 个实验点），精度不佳，故 TF 源仅作对照。

物料衡算（萃取剂假定全部随塔釜出料）
------------------------------------
    进料     水 0.500 + 乙酸 0.500 mol/s
    萃取剂   DMSO 2.000 mol/s
    塔顶     0.473684 mol/s：水 0.450000，乙酸 0.023684   -> x_水 = 0.95
    塔釜     2.526316 mol/s：水 0.050000，乙酸 0.476316，DMSO 2.000000
    闭合     D + B = 3.000000 = 进料 + 萃取剂

物性包：UNIFAC
    与 §2.2 三源对比的 DWSIM 列口径一致。**不能用 NRTL**：实测 DWSIM 对
    乙酸/水、乙酸/DMSO、水/DMSO 三对都没有内置 NRTL 二元交互参数，会走
    估算且 α 一律取默认 0.2，导致闪蒸完全失真（300 K 即出现汽相、
    泡点温度偏差 +108~+142 K）。UNIFAC 为基团贡献法，不需二元参数。

运行需完整进程权限（pythonnet 需 OpenProcess）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DESIGN_JSON = ROOT / "report" / "dwsim" / "design_aw_dmso_multi_seed.json"
OUTDIR = ROOT / "data" / "exports" / "flow_examples"
DEST = OUTDIR / "water_acetic_acid_dmso_extractive_thermoformer.dwxmz"

PROPERTY_PACKAGE = "UNIFAC"
DP_KPA = 5.0
DEFAULT_KEY = "unifac"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="导出 水/乙酸 + DMSO 萃取塔")
    parser.add_argument(
        "--source", default=DEFAULT_KEY,
        help="设计来源键：unifac 或 seed_0..seed_4（默认 unifac）",
    )
    parser.add_argument("--out", default=None, help="输出文件名（默认按来源命名）")
    args = parser.parse_args()

    if not DESIGN_JSON.is_file():
        raise SystemExit(
            f"设计文件不存在: {DESIGN_JSON}\n"
            "请先运行: python scripts/design_aw_dmso_multi_seed.py"
        )

    payload = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
    key = args.source
    if key not in payload["designs"]:
        raise SystemExit(
            f"未知来源 {key!r}；可选: {', '.join(payload['designs'])}"
        )
    design = payload["designs"][key]

    # 输出文件名按来源区分（UNIFAC 源与各权值互不覆盖）
    dest = (
        Path(args.out)
        if args.out
        else OUTDIR / f"water_acetic_acid_dmso_extractive_{key}.dwxmz"
    )

    light = design["light"]
    heavy = design["heavy"]
    entrainer = design["entrainer"]

    print(f"=== 设计值（来源 {key}，alpha_source={design['alpha_source']}）===")
    for field in (
        "alpha_base", "alpha_ext", "selectivity", "alpha_avg",
        "theoretical_stages", "minimum_stages",
        "reflux_ratio", "minimum_reflux_ratio",
        "feed_stage", "entrainer_stage",
        "condenser_temperature_K", "reboiler_temperature_K", "feed_temperature_K",
        "distillate_flow_mol_s", "bottoms_flow_mol_s",
    ):
        print(f"  {field:30} {design[field]}")

    print(f"\n  体系: {light} / {heavy} + 萃取剂 {entrainer}")
    print(f"  物性包: {PROPERTY_PACKAGE}")

    if design["selectivity"] <= 1.0:
        raise SystemExit(
            f"选择性 {design['selectivity']} <= 1，萃取精馏不成立，拒绝生成"
        )

    from thermo_engine.dwsim_export import export_generic_extractive_column

    OUTDIR.mkdir(parents=True, exist_ok=True)

    export_generic_extractive_column(
        light=light,
        heavy=heavy,
        entrainer=entrainer,
        feed_composition=[0.5, 0.5],
        feed_flow_mol_s=1.0,
        feed_temperature_K=float(design["feed_temperature_K"]),
        feed_pressure_kPa=float(design["operating_pressure_kPa"]),
        stages=int(design["theoretical_stages"]),
        reflux_ratio=float(design["reflux_ratio"]),
        feed_stage=int(design["feed_stage"]),
        entrainer_stage=int(design["entrainer_stage"]),
        entrainer_ratio=2.0,
        condenser_temperature_K=float(design["condenser_temperature_K"]),
        reboiler_temperature_K=float(design["reboiler_temperature_K"]),
        property_package=PROPERTY_PACKAGE,
        destination=dest,
        pressure_drop_kPa=DP_KPA,
        distillate_purity_mole_fraction=float(design["distillate_purity_mole_fraction"]),
        recovery=0.90,
        distillate_flow_mol_s=float(design["distillate_flow_mol_s"]),
        bottoms_flow_mol_s=float(design["bottoms_flow_mol_s"]),
    )

    if not dest.is_file():
        raise SystemExit("导出报告成功但文件未生成")

    print(f"\n已导出: {dest}")
    print(f"大小:   {dest.stat().st_size} 字节")
    print("\n=== DWSIM GUI 规格 ===")
    print(f"  物性包: {PROPERTY_PACKAGE}")
    print(f"  组分: {light} / {heavy} / {entrainer}")
    print(f"  {design['theoretical_stages']} 级，进料板 {design['feed_stage']}，"
          f"萃取剂板 {design['entrainer_stage']}，R = {design['reflux_ratio']}")
    print(f"  进料 1.0 mol/s (水:乙酸 = 0.5:0.5) @ {design['feed_temperature_K']} K")
    print(f"  萃取剂 DMSO 2.0 mol/s @ {design['condenser_temperature_K']} K")
    print(f"  冷凝器 {design['condenser_temperature_K']} K / "
          f"再沸器 {design['reboiler_temperature_K']} K")
    print(f"  塔顶 {design['distillate_flow_mol_s']:.6f} mol/s (x_水 = 0.95)")
    print(f"  塔釜 {design['bottoms_flow_mol_s']:.6f} mol/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
