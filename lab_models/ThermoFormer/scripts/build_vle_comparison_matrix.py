"""Build the registered VLE comparison matrix from frozen artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / "experiments/vle/comparison/v1/machine_learning"
OUT_ROOT = ROOT / "experiments/vle/comparison/v1"
THERMOFORMER_TABLE = ROOT / "experiments/summary/VLE_results_table.csv"

FIELDS = (
    "model", "training_scope", "evaluation_scope", "direction", "component_count",
    "status", "valid_seeds", "seed_count", "state_metric", "state_mae_mean",
    "state_mae_sample_std", "state_rmse_mean", "state_rmse_sample_std",
    "state_r2_mean", "state_r2_sample_std", "y_mae_mean", "y_mae_sample_std",
    "y_rmse_mean", "y_rmse_sample_std", "y_r2_mean", "y_r2_sample_std",
    "coverage_mean", "coverage_sample_std", "solver_failure_rate_mean",
    "solver_failure_rate_sample_std", "source",
)

PUBLICATION_FIELDS = (
    "Evaluation protocol", "Model", "P (kPa), isothermal", "y, isothermal",
    "T (K), isobaric", "y, isobaric", "Coverage (%)",
)
PROTOCOLS = (
    "Matched binary→binary", "Matched ternary→ternary",
    "Matched joint→ternary", "Matched joint→joint",
    "External direct→ternary", "External direct→joint",
    "Local-temperature reference 298.15±0.5 K",
)
MODEL_ORDER = (
    "ThermoFormer", "HANNA adapted", "TeNNet-SAC adapted",
    "SMILES-RNN", "UALF-GNN", "Descriptor ANN", "SPT-NRTL adapted",
    "HANNA official", "TeNNet-SAC official", "SPT-NRTL official",
    "SolvGNN", "GDI-GNN", "GE-GNN",
)


def digest(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".tex", ".py"}:
        payload = payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_stat(value: str) -> tuple[str, str]:
    mean, std = value.split("±", 1)
    return mean.strip(), std.strip()


def thermoformer_rows() -> list[dict[str, str]]:
    mapping = {
        "Binary training -> binary test": ("binary", "binary", "2"),
        "Mixed training -> joint test": ("joint", "joint", "2+3"),
        "Mixed training -> binary test": ("joint", "binary", "2"),
        "Mixed training -> ternary test": ("joint", "ternary", "3"),
        "Ternary training -> ternary test": ("ternary", "ternary", "3"),
    }
    rows: list[dict[str, str]] = []
    for source in read_csv(THERMOFORMER_TABLE):
        if source["setting"] not in mapping:
            continue
        train, test, count = mapping[source["setting"]]
        for direction, state_key, y_key, state_metric in (
            ("isothermal", "isothermal_P", "isothermal_y", "P_kPa"),
            ("isobaric", "isobaric_T", "isobaric_y", "T_K"),
        ):
            state = [split_stat(item.strip()) for item in source[state_key].split("/")]
            y = [split_stat(item.strip()) for item in source[y_key].split("/")]
            rows.append({
                "model": "ThermoFormer", "training_scope": train,
                "evaluation_scope": test, "direction": direction,
                "component_count": count, "status": "evaluated",
                "valid_seeds": "5", "seed_count": "5", "state_metric": state_metric,
                "state_mae_mean": state[0][0], "state_mae_sample_std": state[0][1],
                "state_rmse_mean": state[1][0], "state_rmse_sample_std": state[1][1],
                "state_r2_mean": state[2][0], "state_r2_sample_std": state[2][1],
                "y_mae_mean": y[0][0], "y_mae_sample_std": y[0][1],
                "y_rmse_mean": y[1][0], "y_rmse_sample_std": y[1][1],
                "y_r2_mean": y[2][0], "y_r2_sample_std": y[2][1],
                "coverage_mean": "1.0", "coverage_sample_std": "0.0",
                "solver_failure_rate_mean": "0.0", "solver_failure_rate_sample_std": "0.0",
                "source": THERMOFORMER_TABLE.relative_to(ROOT).as_posix(),
            })
    return rows


def adapted_rows() -> tuple[list[dict[str, str]], list[Path]]:
    specifications = (
        ("hanna_binary_adapted.on.binary_train_binary_test", "HANNA adapted", "binary", "binary", {"2"}),
        ("tennet_sac_binary_adapted.on.binary_train_binary_test", "TeNNet-SAC adapted", "binary", "binary", {"2"}),
        ("hanna_ternary_adapted.on.ternary_train_ternary_test", "HANNA adapted", "ternary", "ternary", {"3"}),
        ("tennet_sac_ternary_adapted.on.ternary_train_ternary_test", "TeNNet-SAC adapted", "ternary", "ternary", {"3"}),
        ("hanna_joint_adapted.on.joint_train_joint_test", "HANNA adapted", "joint", "joint", {"3", "2+3"}),
        ("tennet_sac_joint_adapted.on.joint_train_joint_test", "TeNNet-SAC adapted", "joint", "joint", {"3", "2+3"}),
        ("spt_nrtl_adapted.on.joint_train_joint_test", "SPT-NRTL adapted", "joint", "joint", {"3", "2+3"}),
    )
    output, inputs = [], []
    for directory, model, train, test, counts in specifications:
        parent = ML_ROOT / directory
        path = parent / ("aggregate.csv" if (parent / "aggregate.csv").is_file() else "metrics_summary.csv")
        inputs.append(path)
        for item in read_csv(path):
            if item["component_count"] not in counts:
                continue
            direction = item["direction"]
            prefix = "pressure" if direction == "isothermal" else "temperature"
            suffix = "kpa" if direction == "isothermal" else "k"
            output.append({
                "model": model, "training_scope": train, "evaluation_scope": test,
                "direction": direction, "component_count": item["component_count"],
                "status": item["status"], "valid_seeds": item["evaluated_seed_count"],
                "seed_count": item["seed_count"], "state_metric": "P_kPa" if direction == "isothermal" else "T_K",
                "state_mae_mean": item[f"{prefix}_mae_{suffix}_mean"],
                "state_mae_sample_std": item[f"{prefix}_mae_{suffix}_sample_std"],
                "state_rmse_mean": item[f"{prefix}_rmse_{suffix}_mean"],
                "state_rmse_sample_std": item[f"{prefix}_rmse_{suffix}_sample_std"],
                "state_r2_mean": item[f"{prefix}_r2_mean"],
                "state_r2_sample_std": item[f"{prefix}_r2_sample_std"],
                "y_mae_mean": item["y_mae_mean"], "y_mae_sample_std": item["y_mae_sample_std"],
                "y_rmse_mean": item["y_rmse_mean"], "y_rmse_sample_std": item["y_rmse_sample_std"],
                "y_r2_mean": item["y_r2_mean"], "y_r2_sample_std": item["y_r2_sample_std"],
                "coverage_mean": item["valid_coverage_mean"],
                "coverage_sample_std": item["valid_coverage_sample_std"],
                "solver_failure_rate_mean": item["solver_failure_rate_mean"],
                "solver_failure_rate_sample_std": item["solver_failure_rate_sample_std"],
                "source": path.relative_to(ROOT).as_posix(),
            })
    return output, inputs


def additional_baseline_rows() -> tuple[list[dict[str, str]], list[Path]]:
    specifications = (
        ("smiles_rnn.on.binary_train_binary_test", "SMILES-RNN", "binary", "binary", {"2"}),
        ("descriptor_ann.on.binary_train_binary_test", "Descriptor ANN", "binary", "binary", {"2"}),
        ("gdi_gnn.on.binary_train_binary_test", "GDI-GNN", "binary", "binary", {"2"}),
        ("ge_gnn.on.binary_train_binary_test", "GE-GNN", "binary", "binary", {"2"}),
        ("solvgnn.on.joint_train_joint_test", "SolvGNN", "joint", "joint", {"2", "3", "2+3"}),
        ("hanna.on.official_pretrained_to_joint_test", "HANNA official", "external", "joint", {"3", "2+3"}),
        ("tennet_sac.on.external_fixed_to_joint_test", "TeNNet-SAC official", "external", "joint", {"3", "2+3"}),
        ("spt_nrtl.on.external_fixed_to_joint_test", "SPT-NRTL official", "external", "joint", {"3", "2+3"}),
    )
    output, inputs = [], []
    for directory, model, train, test, counts in specifications:
        path = ML_ROOT / directory / "metrics_summary.csv"
        inputs.append(path)
        for item in read_csv(path):
            if item["component_count"] not in counts:
                continue
            direction = item["direction"]
            prefix = "pressure" if direction == "isothermal" else "temperature"
            suffix = "kpa" if direction == "isothermal" else "k"
            output.append({
                "model": model, "training_scope": train, "evaluation_scope": test,
                "direction": direction, "component_count": item["component_count"],
                "status": item["status"], "valid_seeds": item["evaluated_seed_count"],
                "seed_count": item["seed_count"], "state_metric": "P_kPa" if direction == "isothermal" else "T_K",
                "state_mae_mean": item[f"{prefix}_mae_{suffix}_mean"],
                "state_mae_sample_std": item[f"{prefix}_mae_{suffix}_sample_std"],
                "state_rmse_mean": item[f"{prefix}_rmse_{suffix}_mean"],
                "state_rmse_sample_std": item[f"{prefix}_rmse_{suffix}_sample_std"],
                "state_r2_mean": item[f"{prefix}_r2_mean"], "state_r2_sample_std": item[f"{prefix}_r2_sample_std"],
                "y_mae_mean": item["y_mae_mean"], "y_mae_sample_std": item["y_mae_sample_std"],
                "y_rmse_mean": item["y_rmse_mean"], "y_rmse_sample_std": item["y_rmse_sample_std"],
                "y_r2_mean": item["y_r2_mean"], "y_r2_sample_std": item["y_r2_sample_std"],
                "coverage_mean": item["valid_coverage_mean"], "coverage_sample_std": item["valid_coverage_sample_std"],
                "solver_failure_rate_mean": item["solver_failure_rate_mean"],
                "solver_failure_rate_sample_std": item["solver_failure_rate_sample_std"],
                "source": path.relative_to(ROOT).as_posix(),
            })
    return output, inputs


def ualf_rows() -> tuple[list[dict[str, str]], Path]:
    path = ML_ROOT / "ualf_gnn.on.binary_train_binary_test/metrics_summary.csv"
    measured = next(row for row in read_csv(path) if row["direction"] == "isobaric" and row["component_count"] == "2")
    common = {"model": "UALF-GNN", "training_scope": "binary", "evaluation_scope": "binary", "component_count": "2", "seed_count": "5"}
    result = [{
        **common, "direction": "isobaric", "status": measured["status"],
        "valid_seeds": measured["evaluated_seed_count"], "state_metric": "T_K",
        "state_mae_mean": measured["temperature_mae_k_mean"], "state_mae_sample_std": measured["temperature_mae_k_sample_std"],
        "state_rmse_mean": measured["temperature_rmse_k_mean"], "state_rmse_sample_std": measured["temperature_rmse_k_sample_std"],
        "state_r2_mean": measured["temperature_r2_mean"], "state_r2_sample_std": measured["temperature_r2_sample_std"],
        "y_mae_mean": measured["y_mae_mean"], "y_mae_sample_std": measured["y_mae_sample_std"],
        "y_rmse_mean": measured["y_rmse_mean"], "y_rmse_sample_std": measured["y_rmse_sample_std"],
        "y_r2_mean": measured["y_r2_mean"], "y_r2_sample_std": measured["y_r2_sample_std"],
        "coverage_mean": measured["valid_coverage_mean"], "coverage_sample_std": measured["valid_coverage_sample_std"],
        "solver_failure_rate_mean": measured["solver_failure_rate_mean"], "solver_failure_rate_sample_std": measured["solver_failure_rate_sample_std"],
        "source": path.relative_to(ROOT).as_posix(),
    }, {
        **common, "direction": "isothermal", "status": "not_applicable_native_direction_isobaric_only",
        "valid_seeds": "0", "state_metric": "P_kPa", "source": "src/thermoformer/baselines/machine_learning/schema.py:96",
    }]
    return result, path


def fmt(mean: str, std: str) -> str:
    if mean in ("", None):
        return "N/A"
    return f"{float(mean):.4f} ± {float(std):.4f}" if std not in ("", None) else f"{float(mean):.4f}"


def publication_protocol(row: dict[str, str]) -> str | None:
    if row["model"] in {"HANNA official", "TeNNet-SAC official", "SPT-NRTL official"}:
        return PROTOCOLS[4] if row["component_count"] == "3" else PROTOCOLS[5]
    if row["model"] in {"SolvGNN", "GDI-GNN", "GE-GNN"}:
        if row["model"] == "SolvGNN" and row["component_count"] != "2+3":
            return None
        return PROTOCOLS[6]
    key = (row["training_scope"], row["evaluation_scope"], row["component_count"])
    if key == ("binary", "binary", "2"):
        return PROTOCOLS[0]
    if key == ("ternary", "ternary", "3"):
        return PROTOCOLS[1]
    if row["training_scope"] == "joint" and row["component_count"] == "3":
        return PROTOCOLS[2]
    if key == ("joint", "joint", "2+3"):
        return PROTOCOLS[3]
    return None


def metric_cell(row: dict[str, str] | None, prefix: str) -> str:
    if row is None:
        return "MAE: N/A\nRMSE: N/A\nR2: N/A"
    return "\n".join(
        (f"MAE: {fmt(row[f'{prefix}_mae_mean'], row[f'{prefix}_mae_sample_std'])}",
         f"RMSE: {fmt(row[f'{prefix}_rmse_mean'], row[f'{prefix}_rmse_sample_std'])}",
         f"R2: {fmt(row[f'{prefix}_r2_mean'], row[f'{prefix}_r2_sample_std'])}")
    )


def coverage_line(row: dict[str, str] | None, label: str) -> str:
    if row is None or not row["coverage_mean"]:
        return f"{label}: N/A"
    return f"{label}: {100 * float(row['coverage_mean']):.2f} ± {100 * float(row['coverage_sample_std']):.2f}%"


def publication_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        protocol = publication_protocol(row)
        if protocol is not None:
            indexed.setdefault((protocol, row["model"]), {})[row["direction"]] = row
    output = []
    for protocol in PROTOCOLS:
        for model in MODEL_ORDER:
            pair = indexed.get((protocol, model))
            if not pair:
                continue
            iso, isob = pair.get("isothermal"), pair.get("isobaric")
            output.append({
                "Evaluation protocol": protocol, "Model": model,
                "P (kPa), isothermal": metric_cell(iso, "state"),
                "y, isothermal": metric_cell(iso, "y"),
                "T (K), isobaric": metric_cell(isob, "state"),
                "y, isobaric": metric_cell(isob, "y"),
                "Coverage (%)": "\n".join((coverage_line(iso, "Iso"), coverage_line(isob, "Isob"))),
            })
    return output


def markdown_cell(value: str) -> str:
    return value.replace("\n", "<br>")


def tex_cell(value: str) -> str:
    lines = [line.replace("±", r"$\pm$").replace("%", r"\%").replace("R2:", r"$R^2$:") for line in value.splitlines()]
    return r"\shortstack[l]{" + r" \\ ".join(lines) + "}"


def main() -> None:
    rows = thermoformer_rows()
    adapted, adapted_inputs = adapted_rows()
    additional, additional_inputs = additional_baseline_rows()
    ualf, ualf_input = ualf_rows()
    rows.extend(ualf)
    rows.extend(adapted)
    rows.extend(additional)
    rows.sort(key=lambda r: (r["training_scope"], r["evaluation_scope"], r["model"], r["component_count"], r["direction"]))
    for row in rows:
        for field in FIELDS:
            row.setdefault(field, "")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "VLE_comparison_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    md = [
        "# VLE current-data paper comparison matrix", "",
        "All values are mean ± sample standard deviation across frozen seeds 0–4. Model selection uses validation only; test labels are used only for final evaluation.", "",
        "Joint adapted rows with component scopes `3` and `2+3` are two reports from the same per-seed joint checkpoint, not separately selected models.", "",
        "UALF-GNN isothermal P/y remains N/A: its registered native direction is isobaric only, and its executable input/target contract is `(P, x) -> (T, y)`.", "",
        "| Model | Train → test | Direction | Components | Status | Seeds | State MAE | State RMSE | State R² | y MAE | y RMSE | y R² | Coverage | Solver failure |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['model']} | {row['training_scope']} → {row['evaluation_scope']} | {row['direction']} | {row['component_count']} | {row['status']} | {row['valid_seeds']}/{row['seed_count']} | "
            f"{fmt(row['state_mae_mean'], row['state_mae_sample_std'])} | {fmt(row['state_rmse_mean'], row['state_rmse_sample_std'])} | {fmt(row['state_r2_mean'], row['state_r2_sample_std'])} | "
            f"{fmt(row['y_mae_mean'], row['y_mae_sample_std'])} | {fmt(row['y_rmse_mean'], row['y_rmse_sample_std'])} | {fmt(row['y_r2_mean'], row['y_r2_sample_std'])} | "
            f"{fmt(row['coverage_mean'], row['coverage_sample_std'])} | {fmt(row['solver_failure_rate_mean'], row['solver_failure_rate_sample_std'])} |"
        )
    md_path = OUT_ROOT / "VLE_comparison_matrix.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{tabular}{lllllrrrrr}", r"Model & Train--test & Dir. & $n_c$ & Seeds & State MAE & State RMSE & State $R^2$ & $y$ MAE & Coverage \\", r"\hline"]
    for row in rows:
        values = [
            row["model"].replace("_", r"\_"), f"{row['training_scope']}--{row['evaluation_scope']}",
            row["direction"], row["component_count"], f"{row['valid_seeds']}/{row['seed_count']}",
            fmt(row["state_mae_mean"], row["state_mae_sample_std"]),
            fmt(row["state_rmse_mean"], row["state_rmse_sample_std"]),
            fmt(row["state_r2_mean"], row["state_r2_sample_std"]),
            fmt(row["y_mae_mean"], row["y_mae_sample_std"]),
            fmt(row["coverage_mean"], row["coverage_sample_std"]),
        ]
        tex.append(" & ".join(values).replace("±", r"$\pm$") + r" \\")
    tex.extend((r"\hline", r"\multicolumn{10}{l}{UALF-GNN isothermal: N/A; registered native direction is isobaric only.} \\", r"\end{tabular}"))
    tex_path = OUT_ROOT / "VLE_comparison_matrix.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    compact_rows = publication_rows(rows)
    compact_csv_path = OUT_ROOT / "VLE_comparison_table.csv"
    with compact_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLICATION_FIELDS)
        writer.writeheader(); writer.writerows(compact_rows)

    compact_md = [
        "# VLE baseline comparison", "",
        "Each metric reports MAE, RMSE, and R² as mean ± sample SD across completed seed records. Coverage reports isothermal/isobaric test-record percentages.", "",
    ]
    for protocol in PROTOCOLS:
        compact_md.extend((f"## {protocol}", ""))
        group = [row for row in compact_rows if row["Evaluation protocol"] == protocol]
        if not group:
            compact_md.extend(("_No completed metric row in the source matrix._", ""))
            continue
        compact_md.extend((
            "| Model | P (kPa), isothermal | y, isothermal | T (K), isobaric | y, isobaric | Coverage (%) |",
            "|---|---|---|---|---|---|",
        ))
        for row in group:
            compact_md.append("| " + " | ".join((
                row["Model"], markdown_cell(row["P (kPa), isothermal"]),
                markdown_cell(row["y, isothermal"]), markdown_cell(row["T (K), isobaric"]),
                markdown_cell(row["y, isobaric"]), markdown_cell(row["Coverage (%)"]),
            )) + " |")
        compact_md.append("")
    compact_md_path = OUT_ROOT / "VLE_comparison_table.md"
    compact_md_path.write_text("\n".join(compact_md) + "\n", encoding="utf-8")

    compact_tex = [
        r"\begin{tabular}{llllll}",
        r"Model & P (kPa), isothermal & $y$, isothermal & T (K), isobaric & $y$, isobaric & Coverage (\%) \\",
        r"\hline",
    ]
    for protocol in PROTOCOLS:
        compact_tex.append(r"\multicolumn{6}{l}{\textbf{" + protocol.replace("→", r"$\to$").replace("±", r"$\pm$") + r"}} \\")
        group = [row for row in compact_rows if row["Evaluation protocol"] == protocol]
        if not group:
            compact_tex.append(r"\multicolumn{6}{l}{\emph{No completed metric row in the source matrix.}} \\")
        for row in group:
            compact_tex.append(" & ".join((
                row["Model"].replace("_", r"\_"), tex_cell(row["P (kPa), isothermal"]),
                tex_cell(row["y, isothermal"]), tex_cell(row["T (K), isobaric"]),
                tex_cell(row["y, isobaric"]), tex_cell(row["Coverage (%)"]),
            )) + r" \\")
        compact_tex.append(r"\hline")
    compact_tex.append(r"\end{tabular}")
    compact_tex_path = OUT_ROOT / "VLE_comparison_table.tex"
    compact_tex_path.write_text("\n".join(compact_tex) + "\n", encoding="utf-8")

    compact_manifest_path = OUT_ROOT / "VLE_comparison_table_manifest.json"
    compact_manifest = {
        "schema_version": 1, "status": "completed",
        "source": {csv_path.relative_to(ROOT).as_posix(): digest(csv_path)},
        "protocol_groups": list(PROTOCOLS),
        "empty_protocol_groups": [
            protocol for protocol in PROTOCOLS
            if not any(row["Evaluation protocol"] == protocol for row in compact_rows)
        ],
        "metric_line_order": ["MAE", "RMSE", "R2"],
        "coverage_line_order": ["Iso", "Isob"], "row_count": len(compact_rows),
        "outputs": {p.relative_to(ROOT).as_posix(): digest(p) for p in (
            compact_csv_path, compact_md_path, compact_tex_path,
        )},
    }
    compact_manifest_path.write_text(json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command_log = [
        "conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --baselines tennet_sac_joint_adapted --benchmark binary_train_binary_test --seeds 0 1 2 3 4 --epochs 100 --patience 15 --device cuda",
        "conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --baselines hanna_joint_adapted tennet_sac_joint_adapted --benchmark joint_train_joint_test --seeds 0 1 2 3 4 --epochs 100 --patience 15 --device cuda",
        "python scripts/build_vle_comparison_matrix.py",
    ]
    input_paths = [
        THERMOFORMER_TABLE, ualf_input, *adapted_inputs, *additional_inputs,
        ROOT / "scripts/build_vle_comparison_matrix.py",
        ROOT / "src/thermoformer/baselines/machine_learning/schema.py",
        ROOT / "src/thermoformer/baselines/machine_learning/smoke.py",
    ]
    new_experiments = (
        "tennet_sac_binary_adapted.on.binary_train_binary_test",
        "hanna_joint_adapted.on.joint_train_joint_test",
        "tennet_sac_joint_adapted.on.joint_train_joint_test",
    )
    seed_artifacts = {}
    for experiment in new_experiments:
        seed_artifacts[experiment] = {}
        for seed in range(5):
            manifest_path = ML_ROOT / experiment / f"seed_{seed}/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_path = ROOT / manifest["checkpoint"]["path"]
            if digest(checkpoint_path) != manifest["checkpoint"]["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
            seed_artifacts[experiment][str(seed)] = {
                "manifest": {
                    "path": manifest_path.relative_to(ROOT).as_posix(),
                    "sha256": digest(manifest_path),
                },
                "checkpoint": manifest["checkpoint"],
            }
    manifest = {
        "schema_version": 1, "status": "completed", "seeds": [0, 1, 2, 3, 4],
        "selection_partition": "validation", "evaluation_partition": "test",
        "commands": command_log,
        "ualf_isothermal_audit": {
            "status": "not_applicable_native_direction_isobaric_only",
            "capability_evidence": "src/thermoformer/baselines/machine_learning/schema.py:96-107",
            "executable_contract_evidence": "src/thermoformer/baselines/machine_learning/smoke.py:241-245",
        },
        "inputs": {p.relative_to(ROOT).as_posix(): digest(p) for p in input_paths},
        "seed_artifacts": seed_artifacts,
        "outputs": {p.relative_to(ROOT).as_posix(): digest(p) for p in (csv_path, md_path, tex_path)},
        "row_count": len(rows),
    }
    manifest_path = OUT_ROOT / "VLE_comparison_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

