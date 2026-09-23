"""Test-exposed diagnostic phase diagrams for thermodynamic-model comparisons."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from thermo import Chemical

from ..configuration import load_experiment_config
from ..data import load_vle_dataset, retain_pure_anchored_systems
from ..data.splitting import canonical_smiles, sample_id
from ..reporting.artifacts import artifact_sha256, atomic_write_json, atomic_write_text
from .thermodynamic_runner import (
    BaselineCampaignSettings,
    _git_commit,
    _require_clean_scientific_code,
    _require_committed_file,
)


THERMOFORMER_LABEL = "ThermoFormer"
MODEL_LABELS = {
    "thermoformer": THERMOFORMER_LABEL,
    "nrtl": "NRTL",
    "wilson": "Wilson",
    "uniquac": "UNIQUAC",
}
MODEL_COLORS = {
    "thermoformer": "#D55E00",
    "nrtl": "#0072B2",
    "wilson": "#009E73",
    "uniquac": "#CC79A7",
}


@dataclass(frozen=True)
class PhaseDiagramSettings:
    protocol_by_direction: dict[str, tuple[str, ...]]
    thermoformer_seeds: tuple[int, ...]
    baseline_seed: int
    minimum_points: int
    temperature_precision: int
    pressure_precision: int
    pressure_range_floor_kpa: float
    temperature_range_floor_k: float
    candidate_output: str
    plot_data_output: str
    figure_pdf: str
    figure_png: str
    figure_svg: str
    figure_tiff: str
    report_output: str
    manifest_output: str

    @classmethod
    def load(cls, path: Path) -> "PhaseDiagramSettings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "protocol_by_direction",
            "thermoformer_seeds",
            "baseline_seed",
            "minimum_points",
            "temperature_precision",
            "pressure_precision",
            "pressure_range_floor_kpa",
            "temperature_range_floor_k",
            "candidate_output",
            "plot_data_output",
            "figure_pdf",
            "figure_png",
            "figure_svg",
            "figure_tiff",
            "report_output",
            "manifest_output",
        }
        if set(payload) != required:
            raise ValueError("Phase-diagram settings must contain exactly the documented fields")
        settings = cls(
            protocol_by_direction={
                str(direction): tuple(str(value) for value in protocols)
                for direction, protocols in payload["protocol_by_direction"].items()
            },
            thermoformer_seeds=tuple(int(value) for value in payload["thermoformer_seeds"]),
            baseline_seed=int(payload["baseline_seed"]),
            minimum_points=int(payload["minimum_points"]),
            temperature_precision=int(payload["temperature_precision"]),
            pressure_precision=int(payload["pressure_precision"]),
            pressure_range_floor_kpa=float(payload["pressure_range_floor_kpa"]),
            temperature_range_floor_k=float(payload["temperature_range_floor_k"]),
            candidate_output=str(payload["candidate_output"]),
            plot_data_output=str(payload["plot_data_output"]),
            figure_pdf=str(payload["figure_pdf"]),
            figure_png=str(payload["figure_png"]),
            figure_svg=str(payload["figure_svg"]),
            figure_tiff=str(payload["figure_tiff"]),
            report_output=str(payload["report_output"]),
            manifest_output=str(payload["manifest_output"]),
        )
        if tuple(settings.protocol_by_direction) != ("isothermal", "isobaric"):
            raise ValueError("Phase diagrams must define isothermal and isobaric protocols")
        if settings.thermoformer_seeds != (0, 1, 2, 3, 4):
            raise ValueError("Phase diagrams require the formal ThermoFormer seeds 0--4")
        if settings.baseline_seed != 0 or settings.minimum_points < 5:
            raise ValueError("Phase diagrams require baseline seed 0 and at least five points")
        if settings.pressure_range_floor_kpa <= 0 or settings.temperature_range_floor_k <= 0:
            raise ValueError("Phase-diagram normalization floors must be positive")
        return settings


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _verify_artifact(project_root: Path, record: dict[str, Any], label: str) -> Path:
    path = project_root / str(record["path"])
    if not path.is_file() or artifact_sha256(path) != str(record["sha256"]):
        raise ValueError(f"{label} does not match its manifest: {path}")
    return path


def _load_formal_prediction(
    *,
    project_root: Path,
    manifest_path: Path,
    protocol: str,
    seed: int,
    model: str | None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_committed_file(project_root, manifest_path, "phase-diagram input manifest")
    expected_protocol = (
        protocol
        if model is not None
        else f"c1_three_view_vanilla_fugacity_finetune.on.{protocol}"
    )
    observed_protocol = manifest.get("protocol")
    valid_identity = (
        manifest.get("status") == "completed"
        and manifest.get("run_kind") == "formal"
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("evaluation_partition") == "test"
        and observed_protocol == expected_protocol
    )
    if model is not None:
        valid_identity = valid_identity and manifest.get("model") == model
    else:
        valid_identity = valid_identity and manifest.get("analysis_status") == "confirmatory"
        valid_identity = valid_identity and manifest.get("split_protocol") == protocol
    if not valid_identity:
        raise ValueError(f"Invalid formal prediction manifest: {manifest_path}")
    prediction_path = _verify_artifact(
        project_root, manifest["artifacts"]["predictions"], "Prediction artifact"
    )
    _require_committed_file(project_root, prediction_path, "phase-diagram prediction input")
    return pd.read_csv(prediction_path), [
        {
            "kind": "seed_manifest",
            "protocol": protocol,
            "model": model or "thermoformer",
            "seed": seed,
            "path": manifest_path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(manifest_path),
        },
        {
            "kind": "predictions",
            "protocol": protocol,
            "model": model or "thermoformer",
            "seed": seed,
            "path": prediction_path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(prediction_path),
        },
    ]


def _sample_metadata(project_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = project_root / "configs/vle/comparison/studies/thermodynamic_models/config.json"
    experiment = load_experiment_config(config_path)
    loaded = load_vle_dataset(
        project_root / experiment.data.root,
        source_filter=experiment.data.source_filter,
        failed_weight=experiment.data.failed_weight,
        max_pressure_kpa=experiment.data.max_pressure_kpa,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=experiment.data.minimum_pure_anchor_temperatures,
    )
    metadata = {sample_id(row): row for row in samples}
    observed_names: dict[str, str] = {}
    for row in samples:
        for smiles, name in zip(row.smiles, row.names):
            key = canonical_smiles(smiles)
            if name and key not in observed_names:
                observed_names[key] = str(name)
    names: dict[str, str] = {}
    for smiles, observed in observed_names.items():
        try:
            resolved = str(Chemical("smiles=" + smiles).name)
        except Exception:
            resolved = observed if observed.isascii() else smiles
        names[smiles] = resolved if resolved.isascii() else smiles
    return metadata, names


def _state_columns(direction: str) -> tuple[str, str, float, str]:
    if direction == "isothermal":
        return "target_pressure_kpa", "predicted_pressure_kpa", 1.0, "P"
    if direction == "isobaric":
        return "target_temperature_k", "predicted_temperature_k", 1.0, "T"
    raise ValueError(f"Unsupported phase-diagram direction: {direction}")


def _finite_prediction(frame: pd.DataFrame, state_prediction: str) -> pd.Series:
    converged = frame["converged"].astype(str).str.lower().eq("true")
    return (
        converged
        & pd.to_numeric(frame[state_prediction], errors="coerce").notna()
        & pd.to_numeric(frame["y_pred_1"], errors="coerce").notna()
    )


def _protocol_records(
    *,
    project_root: Path,
    protocol: str,
    direction: str,
    settings: PhaseDiagramSettings,
    baseline_settings: BaselineCampaignSettings,
    sample_metadata: dict[str, Any],
    molecule_names: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    state_target, state_prediction, _, _ = _state_columns(direction)
    thermo_root = project_root / baseline_settings.thermoformer_output_template.format(
        protocol=protocol
    )
    inputs: list[dict[str, object]] = []
    thermo_frames: list[pd.DataFrame] = []
    for seed in settings.thermoformer_seeds:
        frame, records = _load_formal_prediction(
            project_root=project_root,
            manifest_path=thermo_root / f"seed_{seed}" / "manifest.json",
            protocol=protocol,
            seed=seed,
            model=None,
        )
        inputs.extend(records)
        frame = frame[frame["direction"] == direction].copy()
        frame = frame[_finite_prediction(frame, state_prediction)]
        frame = frame[["sample_id", state_prediction, "y_pred_1"]].rename(
            columns={
                state_prediction: f"thermoformer_state_seed_{seed}",
                "y_pred_1": f"thermoformer_y_seed_{seed}",
            }
        )
        thermo_frames.append(frame)
    merged = thermo_frames[0]
    for frame in thermo_frames[1:]:
        merged = merged.merge(frame, on="sample_id", how="inner", validate="one_to_one")

    reference, _ = _load_formal_prediction(
        project_root=project_root,
        manifest_path=thermo_root / "seed_0" / "manifest.json",
        protocol=protocol,
        seed=0,
        model=None,
    )
    reference = reference[reference["direction"] == direction].copy()
    stable_columns = [
        "sample_id",
        "system_id",
        "component_count",
        "doi",
        state_target,
        "x_1",
        "y_true_1",
        "component_smiles_1",
        "component_smiles_2",
    ]
    merged = reference[stable_columns].merge(
        merged, on="sample_id", how="inner", validate="one_to_one"
    )
    state_seed_columns = [f"thermoformer_state_seed_{seed}" for seed in settings.thermoformer_seeds]
    y_seed_columns = [f"thermoformer_y_seed_{seed}" for seed in settings.thermoformer_seeds]
    merged["thermoformer_state_mean"] = merged[state_seed_columns].mean(axis=1)
    merged["thermoformer_state_std"] = merged[state_seed_columns].std(axis=1, ddof=1)
    merged["thermoformer_y_mean"] = merged[y_seed_columns].mean(axis=1)
    merged["thermoformer_y_std"] = merged[y_seed_columns].std(axis=1, ddof=1)

    baseline_root = project_root / baseline_settings.formal_output_root / protocol
    for model in baseline_settings.models:
        baseline, records = _load_formal_prediction(
            project_root=project_root,
            manifest_path=(
                baseline_root / f"seed_{settings.baseline_seed}" / model / "manifest.json"
            ),
            protocol=protocol,
            seed=settings.baseline_seed,
            model=model,
        )
        inputs.extend(records)
        baseline = baseline[baseline["direction"] == direction].copy()
        baseline = baseline[_finite_prediction(baseline, state_prediction)]
        baseline = baseline[["sample_id", state_prediction, "y_pred_1"]].rename(
            columns={
                state_prediction: f"{model}_state_mean",
                "y_pred_1": f"{model}_y_mean",
            }
        )
        merged = merged.merge(baseline, on="sample_id", how="inner", validate="one_to_one")

    merged = merged[merged["component_count"] == 2].copy()
    merged["protocol"] = protocol
    merged["direction"] = direction
    merged["condition"] = [
        round(
            float(sample_metadata[value].temperature_k), settings.temperature_precision
        )
        if direction == "isothermal"
        else round(
            float(sample_metadata[value].pressure_kpa), settings.pressure_precision
        )
        for value in merged["sample_id"]
    ]
    merged["component_1_name"] = [
        molecule_names.get(canonical_smiles(value), value)
        for value in merged["component_smiles_1"]
    ]
    merged["component_2_name"] = [
        molecule_names.get(canonical_smiles(value), value)
        for value in merged["component_smiles_2"]
    ]
    return merged, inputs


def candidate_metrics(
    frame: pd.DataFrame,
    *,
    direction: str,
    pressure_range_floor_kpa: float,
    temperature_range_floor_k: float,
) -> dict[str, float]:
    """Compute dimensionless curve scores and interpretable errors for one case."""

    state_target, _, _, _ = _state_columns(direction)
    target = frame[state_target].to_numpy(dtype=float)
    y_true = frame["y_true_1"].to_numpy(dtype=float)
    floor = pressure_range_floor_kpa if direction == "isothermal" else temperature_range_floor_k
    state_scale = max(float(np.ptp(target)), floor)
    output: dict[str, float] = {"state_scale": state_scale}
    for model in MODEL_LABELS:
        state = frame[f"{model}_state_mean"].to_numpy(dtype=float)
        y = frame[f"{model}_y_mean"].to_numpy(dtype=float)
        state_error = state - target
        y_error = y - y_true
        output[f"{model}_state_mae"] = float(np.mean(np.abs(state_error)))
        output[f"{model}_state_rmse"] = float(np.sqrt(np.mean(state_error**2)))
        output[f"{model}_y_mae"] = float(np.mean(np.abs(y_error)))
        output[f"{model}_y_rmse"] = float(np.sqrt(np.mean(y_error**2)))
        output[f"{model}_curve_score"] = float(
            np.sqrt(np.mean((state_error / state_scale) ** 2 + y_error**2))
        )
    return output


SELECTION_LABELS = {
    "well_predicted": "Well-predicted",
    "representative": "Representative",
    "high_error": "High-error",
}


def select_representative_cases(
    candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Select two task minima, one median case, and one global error case."""

    if len(candidates) < 4:
        raise ValueError("At least four eligible binary cases are required")
    selected: list[dict[str, object]] = []
    for direction in ("isothermal", "isobaric"):
        eligible = sorted(
            (row for row in candidates if row["direction"] == direction),
            key=lambda row: (float(row["thermoformer_curve_score"]), str(row["case_id"])),
        )
        if not eligible:
            raise ValueError(f"At least one eligible {direction} case is required")
        best = dict(eligible[0])
        best["selection_category"] = "well_predicted"
        selected.append(best)

    worst = dict(
        max(
            candidates,
            key=lambda row: (float(row["thermoformer_curve_score"]), str(row["case_id"])),
        )
    )
    worst["selection_category"] = "high_error"
    selected_ids = {str(row["case_id"]) for row in (*selected, worst)}
    excluded_systems = {str(row["system_id"]) for row in (*selected, worst)}
    median_score = float(np.median([float(row["thermoformer_curve_score"]) for row in candidates]))
    representative_pool = [
        row
        for row in candidates
        if str(row["case_id"]) not in selected_ids
        and str(row["system_id"]) not in excluded_systems
    ]
    if not representative_pool:
        representative_pool = [
            row for row in candidates if str(row["case_id"]) not in selected_ids
        ]
    representative = dict(
        min(
            representative_pool,
            key=lambda row: (
                abs(float(row["thermoformer_curve_score"]) - median_score),
                str(row["case_id"]),
            ),
        )
    )
    representative["selection_category"] = "representative"
    return [selected[0], selected[1], representative, worst]


def select_extreme_cases(
    candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Backward-compatible alias for the representative four-case selection."""

    return select_representative_cases(candidates)


def build_case_tables(
    *, project_root: Path, settings: PhaseDiagramSettings
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Build the complete candidate ranking and the four selected case tables."""

    baseline_settings = BaselineCampaignSettings.load(
        project_root / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
    )
    sample_metadata, molecule_names = _sample_metadata(project_root)
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    for direction, protocols in settings.protocol_by_direction.items():
        for protocol in protocols:
            frame, records = _protocol_records(
                project_root=project_root,
                protocol=protocol,
                direction=direction,
                settings=settings,
                baseline_settings=baseline_settings,
                sample_metadata=sample_metadata,
                molecule_names=molecule_names,
            )
            frames.append(frame)
            inputs.extend(records)
    all_rows = pd.concat(frames, ignore_index=True)
    candidates: list[dict[str, object]] = []
    grouped = all_rows.groupby(["protocol", "direction", "system_id", "condition"], sort=True)
    for (protocol, direction, system, condition), frame in grouped:
        if len(frame) < settings.minimum_points or frame["x_1"].nunique() < settings.minimum_points:
            continue
        frame = frame.sort_values("x_1")
        metrics = candidate_metrics(
            frame,
            direction=str(direction),
            pressure_range_floor_kpa=settings.pressure_range_floor_kpa,
            temperature_range_floor_k=settings.temperature_range_floor_k,
        )
        case_id = f"{protocol}:{system}:{condition}"
        best_baseline = min(
            ("nrtl", "wilson", "uniquac"),
            key=lambda model: metrics[f"{model}_curve_score"],
        )
        candidate: dict[str, object] = {
            "case_id": case_id,
            "protocol": protocol,
            "direction": direction,
            "system_id": system,
            "condition": float(condition),
            "points": len(frame),
            "component_1": frame.iloc[0]["component_1_name"],
            "component_2": frame.iloc[0]["component_2_name"],
            "component_smiles_1": frame.iloc[0]["component_smiles_1"],
            "component_smiles_2": frame.iloc[0]["component_smiles_2"],
            "doi": frame.iloc[0]["doi"],
            "best_baseline": best_baseline,
            "thermoformer_minus_best_baseline_score": (
                metrics["thermoformer_curve_score"] - metrics[f"{best_baseline}_curve_score"]
            ),
            **metrics,
        }
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No fully covered phase-diagram candidates satisfy the selection rule")
    selected = select_representative_cases(candidates)
    selected_ids = {str(row["case_id"]): str(row["selection_category"]) for row in selected}
    plot_records: list[dict[str, object]] = []
    for row in all_rows.to_dict(orient="records"):
        case_id = f"{row['protocol']}:{row['system_id']}:{row['condition']}"
        if case_id not in selected_ids:
            continue
        output = dict(row)
        output["case_id"] = case_id
        output["selection_category"] = selected_ids[case_id]
        plot_records.append(output)
    unique_inputs = {
        (str(row["path"]), str(row["sha256"])): row for row in inputs
    }
    return candidates, selected, plot_records, list(unique_inputs.values())


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _atomic_figure(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_selected_cases(
    *, selected: Sequence[dict[str, object]], plot_records: Sequence[dict[str, object]], paths: Sequence[Path]
) -> None:
    """Render a full-width four-panel phase-diagram diagnostic figure."""

    _publication_style()
    frame = pd.DataFrame(plot_records)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.15), constrained_layout=False)
    axes_flat = axes.ravel()
    for panel_index, (ax, case) in enumerate(zip(axes_flat, selected)):
        subset = frame[frame["case_id"] == case["case_id"]].sort_values("x_1")
        direction = str(case["direction"])
        state_target, _, _, state_symbol = _state_columns(direction)
        ylabel = "Pressure, P (kPa)" if direction == "isothermal" else "Temperature, T (K)"
        for model in MODEL_LABELS:
            color = MODEL_COLORS[model]
            bubble_x = subset["x_1"].to_numpy(dtype=float)
            dew_x = subset[f"{model}_y_mean"].to_numpy(dtype=float)
            state = subset[f"{model}_state_mean"].to_numpy(dtype=float)
            bubble_order = np.argsort(bubble_x)
            dew_order = np.argsort(dew_x)
            ax.plot(bubble_x[bubble_order], state[bubble_order], color=color, lw=1.35)
            ax.plot(dew_x[dew_order], state[dew_order], color=color, lw=1.15, ls="--")
        thermo_state = subset["thermoformer_state_mean"].to_numpy(dtype=float)
        thermo_std = subset["thermoformer_state_std"].to_numpy(dtype=float)
        x_values = subset["x_1"].to_numpy(dtype=float)
        order = np.argsort(x_values)
        ax.fill_between(
            x_values[order],
            (thermo_state - thermo_std)[order],
            (thermo_state + thermo_std)[order],
            color=MODEL_COLORS["thermoformer"],
            alpha=0.13,
            linewidth=0,
        )
        target = subset[state_target].to_numpy(dtype=float)
        ax.scatter(
            subset["x_1"], target, s=17, color="#111111", edgecolor="white", linewidth=0.35, zorder=8
        )
        ax.scatter(
            subset["y_true_1"], target, s=17, facecolor="white", edgecolor="#111111", linewidth=0.8, zorder=8
        )
        category = SELECTION_LABELS[str(case["selection_category"])] + " case"
        condition = (
            f"T = {float(case['condition']):.2f} K"
            if direction == "isothermal"
            else f"P = {float(case['condition']):.2f} kPa"
        )
        ax.set_title(
            f"{category}: {case['component_1']} (1) + {case['component_2']} (2)\n{condition}",
            loc="left",
            pad=5,
        )
        ax.text(
            0.02,
            0.96,
            f"ThermoFormer {state_symbol} MAE = {float(case['thermoformer_state_mae']):.2f}"
            + (" kPa" if direction == "isothermal" else " K")
            + f"\ny MAE = {float(case['thermoformer_y_mae']):.3f}; n = {int(case['points'])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.set_xlabel(r"Mole fraction of component 1, $x_1$ or $y_1$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.025, 1.025)
        ax.tick_params(direction="in", top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.16,
            1.09,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=12.0,
            fontweight="bold",
            va="top",
        )

    method_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], lw=1.5, label=label)
        for model, label in MODEL_LABELS.items()
    ]
    branch_handles = [
        Line2D([0], [0], color="#555555", lw=1.3, ls="-", label="Bubble branch"),
        Line2D([0], [0], color="#555555", lw=1.3, ls="--", label="Dew branch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="white", label="Experiment (x)", markersize=4.5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#111111", label="Experiment (y)", markersize=4.5),
    ]
    fig.legend(
        handles=method_handles + branch_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        labelspacing=0.7,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.96, wspace=0.26, hspace=0.34)
    for path in paths:
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() == ".png":
            kwargs["dpi"] = 600
        if path.suffix.lower() in {".tif", ".tiff"}:
            kwargs["dpi"] = 600
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        _atomic_figure(fig, path, **kwargs)
    plt.close(fig)


def _report(
    selected: Sequence[dict[str, object]],
    figure_path: Path,
    project_root: Path,
    candidate_counts: dict[str, int],
) -> str:
    lines = [
        "# Test-exposed phase-diagram case study",
        "",
        "Status: completed_exploratory.",
        "",
        "Cases were selected after inspecting held-out test predictions. They diagnose low- and high-error behavior and are not preregistered or confirmatory evidence.",
        "",
        "## Selected cases",
        "",
        "| Category | Task | Protocol | System | Condition | Points | ThermoFormer state MAE | y MAE | Best classical model | ThermoFormer score − best classical score |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    protocol_labels = {
        "state_temperature_low_extrapolation": "Low-temperature extrapolation",
        "state_temperature_high_extrapolation": "High-temperature extrapolation",
        "state_pressure_low_extrapolation": "Low-pressure extrapolation",
        "state_pressure_high_extrapolation": "High-pressure extrapolation",
    }
    for row in selected:
        direction = str(row["direction"])
        category = SELECTION_LABELS[str(row["selection_category"])]
        task = "P-x-y" if direction == "isothermal" else "T-x-y"
        condition = (
            f"{float(row['condition']):.2f} K"
            if direction == "isothermal"
            else f"{float(row['condition']):.2f} kPa"
        )
        state_unit = "kPa" if direction == "isothermal" else "K"
        lines.append(
            f"| {category} | {task} | {protocol_labels[str(row['protocol'])]} | "
            f"{row['component_1']} + {row['component_2']} | {condition} | "
            f"{int(row['points'])} | {float(row['thermoformer_state_mae']):.3f} {state_unit} | "
            f"{float(row['thermoformer_y_mae']):.4f} | {MODEL_LABELS[str(row['best_baseline'])]} | "
            f"{float(row['thermoformer_minus_best_baseline_score']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "The curve score is the root mean square of state error normalized by the observed state range and the dimensionless vapor-composition error. Pressure and temperature ranges use 5 kPa and 5 K lower bounds, respectively. The figure selects the minimum-score case in each task, the distinct-system case nearest the complete candidate-set median, and one global maximum-score case.",
            "",
            f"The complete eligible set contains {candidate_counts['isothermal']} isothermal and {candidate_counts['isobaric']} isobaric curves. Exactly one selected panel is labeled high-error; the other three show well-predicted or median-representative behavior.",
            "",
            f"Figure: `{figure_path.relative_to(project_root).as_posix()}`.",
            "",
            "Solid lines are bubble branches plotted against liquid composition; dashed lines are dew branches plotted against predicted vapor composition. The ThermoFormer envelope is ±1 standard deviation over seeds 0--4. Classical models use the deterministic seed-0 fit.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase_diagram_study(
    *, project_root: Path, settings_path: Path, overwrite: bool = False
) -> tuple[Path, Path, Path]:
    """Select diagnostic cases, render the figure, and bind every input/output."""

    project_root = project_root.resolve()
    settings_path = settings_path.resolve()
    _require_clean_scientific_code(project_root)
    _require_committed_file(project_root, settings_path, "phase-diagram settings")
    generation_git_commit = _git_commit(project_root)
    settings = PhaseDiagramSettings.load(settings_path)
    outputs = [
        project_root / settings.candidate_output,
        project_root / settings.plot_data_output,
        project_root / settings.figure_pdf,
        project_root / settings.figure_png,
        project_root / settings.figure_svg,
        project_root / settings.figure_tiff,
        project_root / settings.report_output,
        project_root / settings.manifest_output,
    ]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Phase-diagram output exists; use --overwrite to replace it")
    candidates, selected, plot_records, inputs = build_case_tables(
        project_root=project_root, settings=settings
    )
    candidate_rows = []
    selected_by_id = {str(row["case_id"]): str(row["selection_category"]) for row in selected}
    for row in candidates:
        output = dict(row)
        output["selection_category"] = selected_by_id.get(str(row["case_id"]))
        candidate_rows.append(output)
    candidate_path = project_root / settings.candidate_output
    plot_data_path = project_root / settings.plot_data_output
    _atomic_csv(candidate_path, candidate_rows)
    _atomic_csv(plot_data_path, plot_records)
    figure_paths = [
        project_root / settings.figure_pdf,
        project_root / settings.figure_png,
        project_root / settings.figure_svg,
        project_root / settings.figure_tiff,
    ]
    plot_selected_cases(selected=selected, plot_records=plot_records, paths=figure_paths)
    report_path = project_root / settings.report_output
    candidate_counts = {
        direction: sum(1 for row in candidates if row["direction"] == direction)
        for direction in ("isothermal", "isobaric")
    }
    atomic_write_text(
        report_path,
        _report(selected, figure_paths[0], project_root, candidate_counts),
    )
    manifest_path = project_root / settings.manifest_output
    output_records = {
        path.stem + path.suffix.replace(".", "_"): {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(path),
        }
        for path in (*figure_paths, candidate_path, plot_data_path, report_path)
    }
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed_exploratory",
            "analysis_status": "test_exposed_exploratory",
            "experiment": "thermodynamic_phase_diagram_case_study",
            "generation_git_commit": generation_git_commit,
            "selection_partition": "test",
            "case_selection_used_test_labels": True,
            "selection_rule": "task minima, distinct-system candidate nearest global median, and one global maximum ThermoFormer normalized curve score",
            "selected_cases": selected,
            "settings": {
                "path": settings_path.relative_to(project_root).as_posix(),
                "sha256": artifact_sha256(settings_path),
            },
            "inputs": inputs,
            "outputs": output_records,
        },
    )
    return figure_paths[0], report_path, manifest_path
