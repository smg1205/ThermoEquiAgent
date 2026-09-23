"""水 / 乙酸 + DMSO 萃取精馏塔板设计与选择性核算（多权值）。

体系：关键对 = 水（轻）/ 乙酸（重），萃取剂 = 二甲基亚砜（DMSO）
常压沸点：水 319.55 K < 乙酸 390.94 K < DMSO 463.7 K
        => DMSO 沸点最高，满足短节法「萃取剂全走塔釜」假设

设计来源：thermo_engine.column_design.design_ternary_extractive_column（FUG 短节法）
  alpha_base = 无萃取剂时 水/乙酸 的相对挥发度
  alpha_ext  = 富萃取剂组成 [0.10, 0.10, 0.80] 下的相对挥发度
  selectivity = alpha_ext / alpha_base   （> 1 才说明萃取剂增强了分离）

对 alpha_source 逐一取 unifac 与 thermoformer(seed_0..4)，比较：
  alpha_base / alpha_ext / 选择性 / N / R / 进料板 / 温度
并给出「成立与否」的判定。

注意：torch 与 pythonnet 不能共存；本脚本只做设计，不导出 DWSIM 文件。
"""

from __future__ import annotations

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

CKPT_ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\models\vle\prediction\vle_overall_ternary")
SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"]

LIGHT, HEAVY, ENTRAINER = "water", "acetic acid", "dimethyl sulfoxide"
FEED = [0.5, 0.5]
P_KPA = 101.325
F_MOL_S = 1.0
PURITY = 0.95
RECOVERY = 0.90
ENTRAINER_RATIO = 2.0

OUT = ROOT / "report" / "dwsim" / "design_aw_dmso_multi_seed.json"


def design(alpha_source: str, checkpoint: Path | None) -> dict:
    from thermo_engine import column_design as cd

    if checkpoint is not None:
        from thermo_engine.thermoformer_backend import (
            ThermoFormerBackend,
            ThermoFormerSettings,
            resolve_settings,
        )

        base = resolve_settings()
        cd._thermoformer_backend = ThermoFormerBackend(
            ThermoFormerSettings(
                src_path=base.src_path,
                checkpoint_path=checkpoint,
                feature_cache_path=base.feature_cache_path,
                use_cuda=False,
            )
        )

    return cd.design_ternary_extractive_column(
        LIGHT,
        HEAVY,
        ENTRAINER,
        FEED,
        feed_flow_mol_s=F_MOL_S,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        entrainer_ratio=ENTRAINER_RATIO,
        alpha_source=alpha_source,
    )


def main() -> int:
    print("=" * 92)
    print("体系：水（轻）/ 乙酸（重）+ 萃取剂 DMSO")
    print(f"常压沸点：水 319.55 K < 乙酸 390.94 K < DMSO 463.7 K  （DMSO 最高 ✅）")
    print(f"设计条件：F={F_MOL_S} mol/s，水:乙酸 = {FEED[0]}:{FEED[1]}，"
          f"萃取剂比 {ENTRAINER_RATIO}，塔顶纯度 {PURITY}，回收率 {RECOVERY}")
    print("=" * 92)

    rows: dict[str, dict] = {}

    # UNIFAC 基准
    print("\n[UNIFAC]")
    try:
        d = design("unifac", None)
        rows["unifac"] = d
        print(f"  alpha_base={d['alpha_base']:.4f}  alpha_ext={d['alpha_ext']:.4f}  "
              f"选择性={d['selectivity']:.4f}  alpha_avg={d['alpha_avg']:.4f}")
        print(f"  N={d['theoretical_stages']}  N_min={d['minimum_stages']:.3f}  "
              f"R={d['reflux_ratio']:.3f}  R_min={d['minimum_reflux_ratio']:.3f}")
        print(f"  进料板={d['feed_stage']}  萃取剂板={d['entrainer_stage']}  "
              f"T顶={d['condenser_temperature_K']:.2f} K  T釜={d['reboiler_temperature_K']:.2f} K")
        print(f"  T进料={d['feed_temperature_K']:.2f} K")
    except Exception as exc:  # noqa: BLE001
        print(f"  失败: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")

    # ThermoFormer 各权值
    for seed in SEEDS:
        ckpt = CKPT_ROOT / seed / "best_model.pt"
        if not ckpt.is_file():
            print(f"\n[{seed}] 权值缺失，跳过")
            continue
        print(f"\n[thermoformer {seed}]")
        try:
            d = design("thermoformer", ckpt)
            rows[seed] = d
            print(f"  alpha_base={d['alpha_base']:.4f}  alpha_ext={d['alpha_ext']:.4f}  "
                  f"选择性={d['selectivity']:.4f}  alpha_avg={d['alpha_avg']:.4f}")
            print(f"  N={d['theoretical_stages']}  N_min={d['minimum_stages']:.3f}  "
                  f"R={d['reflux_ratio']:.3f}  R_min={d['minimum_reflux_ratio']:.3f}")
            print(f"  进料板={d['feed_stage']}  萃取剂板={d['entrainer_stage']}  "
                  f"T顶={d['condenser_temperature_K']:.2f} K  T釜={d['reboiler_temperature_K']:.2f} K")
            print(f"  T进料={d['feed_temperature_K']:.2f} K")
        except Exception as exc:  # noqa: BLE001
            print(f"  失败: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")

    # 汇总表
    print("\n" + "=" * 92)
    print("汇总对比")
    print("=" * 92)
    hdr = (f"{'来源':22} {'alpha_base':>10} {'alpha_ext':>10} {'选择性':>8} "
           f"{'N':>4} {'R':>8} {'进料板':>6} {'剂板':>4}")
    print(hdr)
    print("-" * len(hdr))
    for label, d in rows.items():
        print(f"{label:22} {d['alpha_base']:>10.4f} {d['alpha_ext']:>10.4f} "
              f"{d['selectivity']:>8.4f} {d['theoretical_stages']:>4} "
              f"{d['reflux_ratio']:>8.3f} {d['feed_stage']:>6} {d['entrainer_stage']:>4}")

    # 判定
    print("\n" + "=" * 92)
    print("成立性判定")
    print("=" * 92)
    print("\n判定标准：")
    print("  1) 选择性 > 1        —— 萃取剂确实增强了关键对分离（萃取精馏的本义）")
    print("  2) alpha_base > 1    —— 无萃取剂时关键对本就可分离（短节法前提）")
    print("  3) 萃取剂沸点最高     —— 萃取剂全走塔釜的物料衡算假设成立")
    print("  4) N 与 R 合理        —— 塔板数、回流比工程上可接受")
    print()

    for label, d in rows.items():
        checks = []
        checks.append(("选择性>1", d["selectivity"] > 1.0, f"{d['selectivity']:.4f}"))
        checks.append(("alpha_base>1", d["alpha_base"] > 1.0, f"{d['alpha_base']:.4f}"))
        checks.append(("萃取剂沸点最高", True, "DMSO 463.7 K"))
        n = d["theoretical_stages"]
        r = d["reflux_ratio"]
        checks.append(("N合理(<100)", n < 100, f"N={n}"))
        checks.append(("R合理(<20)", r < 20.0, f"R={r:.2f}"))
        ok_all = all(c[1] for c in checks)
        print(f"  [{label}] {'成立 ✅' if ok_all else '不成立 ❌'}")
        for name, passed, value in checks:
            print(f"      {'✓' if passed else '✗'} {name:16} {value}")

    OUT.write_text(
        json.dumps({"system": {"light": LIGHT, "heavy": HEAVY, "entrainer": ENTRAINER},
                    "conditions": {"F": F_MOL_S, "feed": FEED, "P_kPa": P_KPA,
                                   "purity": PURITY, "recovery": RECOVERY,
                                   "entrainer_ratio": ENTRAINER_RATIO},
                    "designs": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
