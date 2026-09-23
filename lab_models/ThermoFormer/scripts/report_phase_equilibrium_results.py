"""Create CSV, Markdown and LaTeX tables for the VLE/LLE benchmark campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (
        float(np.mean(clean)) if clean else None,
        float(np.std(clean, ddof=1)) if len(clean) > 1 else None,
        len(clean),
    )


def vle_rows(root: Path):
    raw, grouped = [], defaultdict(list)
    for path in root.glob("vle/three/results/*/seed_*/metrics.json"):
        protocol = path.parent.parent.name.split(".on.", 1)[-1]
        seed = int(path.parent.name.split("_")[-1])
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("direction") not in ("isothermal", "isobaric"):
                continue
            subset = "all" if row.get("component_count") is None else str(row["component_count"])
            direction = row["direction"]
            pairs = (
                (("P" if direction == "isothermal" else "T"), "mae", row.get("pressure_mae_kpa" if direction == "isothermal" else "temperature_mae_k")),
                (("P" if direction == "isothermal" else "T"), "rmse", row.get("pressure_rmse_kpa" if direction == "isothermal" else "temperature_rmse_k")),
                (("P" if direction == "isothermal" else "T"), "r2", row.get("pressure_r2" if direction == "isothermal" else "temperature_r2")),
                ("y", "mae", row.get("y_mae")), ("y", "rmse", row.get("y_rmse")), ("y", "r2", row.get("y_r2")),
            )
            for target, metric, value in pairs:
                item = {"task": "VLE", "protocol": protocol, "subset": subset, "seed": seed, "direction": direction, "target": target, "metric": metric, "value": value}
                raw.append(item)
                grouped[(protocol, subset, direction, target, metric)].append(value)
    return raw, grouped


def lle_rows(root: Path):
    raw, grouped = [], defaultdict(list)
    for path in root.glob("lle/formal/**/seed_*/metrics.json"):
        protocol = path.parent.parent.name
        seed = int(path.parent.name.split("_")[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for subset, values in payload.items():
            thermo = values["thermodynamic"]
            for metric, value in thermo.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)) or value is None:
                    item = {"task": "LLE", "protocol": protocol, "subset": subset, "seed": seed, "direction": "", "target": "endpoints", "metric": metric, "value": value}
                    raw.append(item)
                    grouped[(protocol, subset, "", "endpoints", metric)].append(value)
    return raw, grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_v, groups_v = vle_rows(args.input)
    raw_l, groups_l = lle_rows(args.input)
    raw = raw_v + raw_l
    fields = ("task", "protocol", "subset", "seed", "direction", "target", "metric", "value")
    with (args.output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(raw)
    aggregate = []
    for key, values in sorted({**groups_v, **groups_l}.items()):
        mean, std, n = stats(values)
        aggregate.append(dict(zip(("protocol", "subset", "direction", "target", "metric"), key), mean=mean, sample_std=std, valid_seeds=n))
    with (args.output / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("protocol", "subset", "direction", "target", "metric", "mean", "sample_std", "valid_seeds"))
        writer.writeheader(); writer.writerows(aggregate)
    lines = ["# VLE/LLE benchmark results", "", "Values are mean ± sample standard deviation across seeds 0–4. Undefined R² values are omitted rather than replaced by zero.", "", "| Protocol | Subset | Direction | Target | Metric | Mean ± SD | n |", "|---|---|---|---|---|---:|---:|"]
    tex = [r"\begin{tabular}{lllllrr}", r"Protocol & Subset & Direction & Target & Metric & Mean $\pm$ SD & n \\", r"\hline"]
    for row in aggregate:
        mean = "NA" if row["mean"] is None else f'{row["mean"]:.6g}'
        sd = "NA" if row["sample_std"] is None else f'{row["sample_std"]:.6g}'
        lines.append(f'| {row["protocol"]} | {row["subset"]} | {row["direction"]} | {row["target"]} | {row["metric"]} | {mean} ± {sd} | {row["valid_seeds"]} |')
        escaped = row["protocol"].replace("_", r"\_")
        tex.append(f'{escaped} & {row["subset"]} & {row["direction"]} & {row["target"]} & {row["metric"]} & {mean} $\\pm$ {sd} & {row["valid_seeds"]} \\\\')
    tex.append(r"\end{tabular}")
    (args.output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "results.tex").write_text("\n".join(tex), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"rows": aggregate}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
