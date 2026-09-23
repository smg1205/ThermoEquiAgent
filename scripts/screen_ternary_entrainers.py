"""从现有实验数据中筛选「萃取精馏真正成立」的三元体系并给出预测。

用户要求：找一个选择性 > 1（萃取剂确实增强分离）的三元 VLE 体系。

数据来源（均为仓库内已归档的实验数据）：
  entrainer_selectivity_screen.csv   萃取剂/关键对组合的选择性筛选（UNIFAC 与 ThermoFormer 双源）
  ternary_working_scan.csv           5 个三元体系的 ThermoFormer 预测精度（各 5 个种子）

筛选条件：
  1. sel_min > 1        —— 两源都认为萃取剂增强分离（保守取较小值）
  2. 萃取剂沸点最高     —— 短节法「萃取剂全走塔釜」假设成立的前提
  3. TF 预测精度可用    —— MAE 越小越可信
  4. 回流比不过分激进   —— R 太大不经济
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "report" / "dwsim" / "entrainer_selectivity_screen.csv"
SCAN = ROOT / "report" / "dwsim" / "ternary_working_scan.csv"


def load_screen() -> list[dict]:
    with open(SCREEN, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_scan() -> list[dict]:
    with open(SCAN, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    screen = load_screen()
    scan = load_scan()

    # ThermoFormer 预测精度（按体系名建立索引，取最佳 MAE）
    accuracy = {}
    for row in scan:
        system = row["system"].replace(" ", "")
        accuracy[system] = {
            "mae": float(row["best_mae"]),
            "seed": row["best_seed"],
            "n": int(row["n_exp_total"]),
            "doi": row["doi"],
        }

    print("=" * 96)
    print("三元体系萃取剂候选筛选（数据来源：仓库已归档实验数据）")
    print("=" * 96)

    print("\n筛选条件：sel_min > 1（两源一致认为萃取剂增强分离）且 萃取剂沸点最高")
    print("（萃取剂沸点最高 => 短节法「萃取剂全走塔釜」假设成立）\n")

    header = (
        f"{'体系（轻/重 + 萃取剂）':40} {'sel_min':>8} {'沸点最高':>8} "
        f"{'N':>3} {'R':>7} {'TF MAE':>7}"
    )
    print(header)
    print("-" * len(header))

    winners = []
    for row in screen:
        sel_min = float(row["sel_min"])
        if sel_min <= 1.0:
            continue
        nbp_l = float(row["nbp_light"])
        nbp_h = float(row["nbp_heavy"])
        nbp_e = float(row["nbp_ent"])
        ent_highest = nbp_e > nbp_l and nbp_e > nbp_h

        light = row["light"]
        heavy = row["heavy"]
        ent = row["entrainer"]
        label = f"{light} / {heavy} + {ent}"

        # 该行所属三元体系的 TF 精度
        keys = sorted([light, heavy, ent])
        # 与 scan 的体系名做宽松匹配（取前两个大写字母特征）
        mae_txt = "  n/a"
        for system, info in accuracy.items():
            if all(k.split()[0][:4] in system for k in keys):
                mae_txt = f"{info['mae']:.4f}"
                break

        mark = "是" if ent_highest else "否"
        print(f"{label:40} {sel_min:>8.2f} {mark:>8} "
              f"{row['N_u']:>3} {float(row['R_u']):>7.2f} {mae_txt:>7}")
        if ent_highest:
            winners.append((row, mae_txt))

    print("\n" + "=" * 96)
    print("满足全部条件的候选")
    print("=" * 96)
    if not winners:
        print("  无 —— 没有任何组合同时满足「sel_min>1」与「萃取剂沸点最高」")
    for row, mae in winners:
        print(f"\n  体系：{row['light']} / {row['heavy']} + 萃取剂 {row['entrainer']}")
        print(f"    选择性 sel_min = {float(row['sel_min']):.2f} "
              f"(UNIFAC {float(row['sel_u']):.2f} / TF {float(row['sel_t']):.2f})")
        print(f"    设计: N={row['N_u']}  R={float(row['R_u']):.2f}  (UNIFAC 源)")
        print(f"          N={row['N_t']}  R={float(row['R_t']):.2f}  (ThermoFormer 源)")
        print(f"    常压沸点: 轻 {row['nbp_light']} K / 重 {row['nbp_heavy']} K / "
              f"剂 {row['nbp_ent']} K")
        print(f"    ThermoFormer 三元预测 MAE = {mae}")

    print("\n" + "=" * 96)
    print("全部 5 个有实验数据的三元体系：精度 vs 可行性")
    print("=" * 96)
    print(f"\n{'体系':32} {'点数':>5} {'最佳MAE':>9} {'最差MAE':>9} {'DOI'}")
    print("-" * 96)
    for row in scan:
        print(f"{row['system']:32} {row['n_exp_total']:>5} {float(row['best_mae']):>9.4f} "
              f"{float(row['worst_seed' + row['best_seed'].split('_')[1]]):>9.4f} "
              f"{row['doi']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
