"""Test-exposed ternary Gibbs-triangle diagnostics for VLE predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..data.splitting import canonical_smiles
from ..reporting.artifacts import artifact_sha256, atomic_write_json, atomic_write_text
from .phase_diagrams import (
    MODEL_COLORS,
    MODEL_LABELS,
    SELECTION_LABELS,
    _atomic_csv,
    _atomic_figure,
    _load_formal_prediction,
    _publication_style,
    _sample_metadata,
    _state_columns,
)
from .thermodynamic_runner import (
    BaselineCampaignSettings,
    _git_commit,
    _require_clean_scientific_code,
    _require_committed_file,
)


SQRT3_OVER_2 = float(np.sqrt(3.0) / 2.0)
REQUIRED_CLASSICAL_MODELS = ("nrtl", "wilson")
OPTIONAL_CLASSICAL_MODELS = ("uniquac",)
EXPERIMENT_VAPOR_ZORDER = 6
THERMOFORMER_ZORDER = 10
DISPLAY_NAME_OVERRIDES = {
    canonical_smiles("CC(=O)C(C)C"): "3-methyl-2-butanone",
    canonical_smiles("CC(=O)CC(C)C"): "4-methyl-2-pentanone",
    canonical_smiles("COC(C)(C)C"): "2-methoxy-2-methylpropane",
}


@dataclass(frozen=True)
class TernaryPhaseDiagramSettings:
    """Frozen selection, plotting, and artifact settings for ternary cases."""

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
    def load(cls, path: Path) -> "TernaryPhaseDiagramSettings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = set(cls.__dataclass_fields__)
        if set(payload) != required:
            raise ValueError("Ternary phase-diagram settings contain undocumented fields")
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
        if tuple(settings.protocol_by_direction) != ("isothermal",):
            raise ValueError("The available ternary campaign contains isothermal protocols only")
        if len(settings.protocol_by_direction["isothermal"]) != 2:
            raise ValueError("Ternary cases require the low- and high-temperature protocols")
        if settings.thermoformer_seeds != (0, 1, 2, 3, 4):
            raise ValueError("Ternary cases require ThermoFormer seeds 0--4")
        if settings.baseline_seed != 0 or settings.minimum_points < 5:
            raise ValueError("Ternary cases require baseline seed 0 and at least five points")
        return settings


def barycentric_to_cartesian(compositions: np.ndarray) -> np.ndarray:
    """Map normalized three-component compositions to an equilateral triangle."""

    values = np.asarray(compositions, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("Ternary compositions must have exactly three components")
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(totals <= 0):
        raise ValueError("Ternary compositions must be finite and have a positive sum")
    normalized = values / totals
    return np.stack(
        (normalized[..., 1] + 0.5 * normalized[..., 2], SQRT3_OVER_2 * normalized[..., 2]),
        axis=-1,
    )


def ternary_model_marker_style(model: str) -> dict[str, object]:
    """Return hollow marker styling with ThermoFormer on the top layer."""

    if model not in MODEL_LABELS:
        raise ValueError(f"Unknown ternary phase-diagram model: {model}")
    return {
        "facecolor": "none",
        "edgecolor": MODEL_COLORS[model],
        "linewidth": 1.25 if model == "thermoformer" else 0.75,
        "size": 30 if model == "thermoformer" else 24,
        "zorder": THERMOFORMER_ZORDER if model == "thermoformer" else 5,
    }


def ternary_candidate_metrics(
    frame: pd.DataFrame,
    *,
    direction: str,
    pressure_range_floor_kpa: float,
    temperature_range_floor_k: float,
) -> dict[str, float]:
    """Score a fixed-condition ternary case using state and all three y errors."""

    state_target, _, _, _ = _state_columns(direction)
    target = frame[state_target].to_numpy(dtype=float)
    y_true = frame[["y_true_1", "y_true_2", "y_true_3"]].to_numpy(dtype=float)
    floor = pressure_range_floor_kpa if direction == "isothermal" else temperature_range_floor_k
    state_scale = max(float(np.ptp(target)), floor)
    output: dict[str, float] = {"state_scale": state_scale}
    for model in MODEL_LABELS:
        state_column = f"{model}_state_mean"
        y_columns = [f"{model}_y_mean_{component}" for component in (1, 2, 3)]
        if state_column not in frame or any(column not in frame for column in y_columns):
            continue
        state = frame[state_column].to_numpy(dtype=float)
        y = frame[y_columns].to_numpy(dtype=float)
        valid = np.isfinite(state) & np.all(np.isfinite(y), axis=1)
        output[f"{model}_coverage"] = float(np.mean(valid))
        if not np.all(valid):
            continue
        state_error = state - target
        y_error = y - y_true
        output[f"{model}_state_mae"] = float(np.mean(np.abs(state_error)))
        output[f"{model}_state_rmse"] = float(np.sqrt(np.mean(state_error**2)))
        output[f"{model}_y_mae"] = float(np.mean(np.abs(y_error)))
        output[f"{model}_y_rmse"] = float(np.sqrt(np.mean(y_error**2)))
        output[f"{model}_curve_score"] = float(
            np.sqrt(np.mean((state_error / state_scale) ** 2) + np.mean(y_error**2))
        )
    return output


def select_ternary_representatives(
    candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Select two protocol minima, one median case, and one global error case."""

    if len(candidates) < 4:
        raise ValueError("At least four eligible ternary cases are required")
    selected: list[dict[str, object]] = []
    protocols = (
        "state_temperature_high_extrapolation",
        "state_temperature_low_extrapolation",
    )
    for protocol in protocols:
        eligible = sorted(
            (row for row in candidates if row["protocol"] == protocol),
            key=lambda row: (float(row["thermoformer_curve_score"]), str(row["case_id"])),
        )
        if not eligible:
            raise ValueError(f"At least one eligible ternary case is required for {protocol}")
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


def select_ternary_extremes(candidates: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Backward-compatible alias for the representative four-case selection."""

    return select_ternary_representatives(candidates)


def _valid_ternary_predictions(frame: pd.DataFrame, state_column: str) -> pd.Series:
    converged = frame["converged"].astype(str).str.lower().eq("true")
    columns = [state_column, "y_pred_1", "y_pred_2", "y_pred_3"]
    return converged & frame[columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)


def _protocol_records(
    *,
    project_root: Path,
    protocol: str,
    direction: str,
    settings: TernaryPhaseDiagramSettings,
    baseline_settings: BaselineCampaignSettings,
    sample_metadata: dict[str, Any],
    molecule_names: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    state_target, state_prediction, _, _ = _state_columns(direction)
    thermo_root = project_root / baseline_settings.thermoformer_output_template.format(protocol=protocol)
    inputs: list[dict[str, object]] = []
    seed_frames: list[pd.DataFrame] = []
    for seed in settings.thermoformer_seeds:
        frame, records = _load_formal_prediction(
            project_root=project_root,
            manifest_path=thermo_root / f"seed_{seed}" / "manifest.json",
            protocol=protocol,
            seed=seed,
            model=None,
        )
        inputs.extend(records)
        frame = frame[(frame["direction"] == direction) & (frame["component_count"] == 3)].copy()
        frame = frame[_valid_ternary_predictions(frame, state_prediction)]
        columns = ["sample_id", state_prediction, "y_pred_1", "y_pred_2", "y_pred_3"]
        renamed = {state_prediction: f"thermoformer_state_seed_{seed}"}
        renamed.update({f"y_pred_{i}": f"thermoformer_y_seed_{seed}_{i}" for i in (1, 2, 3)})
        seed_frames.append(frame[columns].rename(columns=renamed))
    merged = seed_frames[0]
    for frame in seed_frames[1:]:
        merged = merged.merge(frame, on="sample_id", how="inner", validate="one_to_one")

    reference, _ = _load_formal_prediction(
        project_root=project_root,
        manifest_path=thermo_root / "seed_0" / "manifest.json",
        protocol=protocol,
        seed=0,
        model=None,
    )
    reference = reference[(reference["direction"] == direction) & (reference["component_count"] == 3)]
    stable = [
        "sample_id", "system_id", "component_count", "doi", state_target,
        "component_smiles_1", "component_smiles_2", "component_smiles_3",
        "x_1", "x_2", "x_3", "y_true_1", "y_true_2", "y_true_3",
    ]
    merged = reference[stable].merge(merged, on="sample_id", how="inner", validate="one_to_one")
    state_columns = [f"thermoformer_state_seed_{seed}" for seed in settings.thermoformer_seeds]
    merged["thermoformer_state_mean"] = merged[state_columns].mean(axis=1)
    merged["thermoformer_state_std"] = merged[state_columns].std(axis=1, ddof=1)
    for component in (1, 2, 3):
        y_columns = [f"thermoformer_y_seed_{seed}_{component}" for seed in settings.thermoformer_seeds]
        merged[f"thermoformer_y_mean_{component}"] = merged[y_columns].mean(axis=1)
        merged[f"thermoformer_y_std_{component}"] = merged[y_columns].std(axis=1, ddof=1)

    baseline_root = project_root / baseline_settings.formal_output_root / protocol
    for model in baseline_settings.models:
        baseline, records = _load_formal_prediction(
            project_root=project_root,
            manifest_path=baseline_root / f"seed_{settings.baseline_seed}" / model / "manifest.json",
            protocol=protocol,
            seed=settings.baseline_seed,
            model=model,
        )
        inputs.extend(records)
        baseline = baseline[(baseline["direction"] == direction) & (baseline["component_count"] == 3)].copy()
        baseline = baseline[_valid_ternary_predictions(baseline, state_prediction)]
        columns = ["sample_id", state_prediction, "y_pred_1", "y_pred_2", "y_pred_3"]
        renamed = {state_prediction: f"{model}_state_mean"}
        renamed.update({f"y_pred_{i}": f"{model}_y_mean_{i}" for i in (1, 2, 3)})
        how = "inner" if model in REQUIRED_CLASSICAL_MODELS else "left"
        merged = merged.merge(
            baseline[columns].rename(columns=renamed), on="sample_id", how=how, validate="one_to_one"
        )

    merged["protocol"] = protocol
    merged["direction"] = direction
    merged["condition"] = [
        round(float(sample_metadata[value].temperature_k), settings.temperature_precision)
        if direction == "isothermal"
        else round(float(sample_metadata[value].pressure_kpa), settings.pressure_precision)
        for value in merged["sample_id"]
    ]
    for component in (1, 2, 3):
        merged[f"component_{component}_name"] = [
            DISPLAY_NAME_OVERRIDES.get(
                canonical_smiles(str(value)),
                molecule_names.get(canonical_smiles(str(value)), str(value)),
            )
            for value in merged[f"component_smiles_{component}"]
        ]
    return merged, inputs


def build_ternary_case_tables(
    *, project_root: Path, settings: TernaryPhaseDiagramSettings
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build all eligible ternary candidates and four selected plotting tables."""

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
        unique_x = frame[["x_1", "x_2", "x_3"]].drop_duplicates()
        if len(frame) < settings.minimum_points or len(unique_x) < settings.minimum_points:
            continue
        metrics = ternary_candidate_metrics(
            frame,
            direction=str(direction),
            pressure_range_floor_kpa=settings.pressure_range_floor_kpa,
            temperature_range_floor_k=settings.temperature_range_floor_k,
        )
        if any(f"{model}_curve_score" not in metrics for model in ("thermoformer", *REQUIRED_CLASSICAL_MODELS)):
            continue
        available_baselines = [
            model for model in ("nrtl", "wilson", "uniquac") if f"{model}_curve_score" in metrics
        ]
        best_baseline = min(available_baselines, key=lambda model: metrics[f"{model}_curve_score"])
        case_id = f"{protocol}:{system}:{condition}"
        first = frame.iloc[0]
        candidates.append(
            {
                "case_id": case_id,
                "protocol": protocol,
                "direction": direction,
                "system_id": system,
                "condition": float(condition),
                "points": len(frame),
                "component_1": first["component_1_name"],
                "component_2": first["component_2_name"],
                "component_3": first["component_3_name"],
                "component_smiles_1": first["component_smiles_1"],
                "component_smiles_2": first["component_smiles_2"],
                "component_smiles_3": first["component_smiles_3"],
                "doi": first["doi"],
                "best_baseline": best_baseline,
                "uniquac_available": "uniquac_curve_score" in metrics,
                "thermoformer_minus_best_baseline_score": (
                    metrics["thermoformer_curve_score"] - metrics[f"{best_baseline}_curve_score"]
                ),
                **metrics,
            }
        )
    if not candidates:
        raise ValueError("No ternary cases satisfy the coverage and composition-point requirements")
    selected = select_ternary_representatives(candidates)
    selected_ids = {str(row["case_id"]): str(row["selection_category"]) for row in selected}
    plot_records: list[dict[str, object]] = []
    for row in all_rows.to_dict(orient="records"):
        case_id = f"{row['protocol']}:{row['system_id']}:{row['condition']}"
        if case_id in selected_ids:
            output = dict(row)
            output["case_id"] = case_id
            output["selection_category"] = selected_ids[case_id]
            plot_records.append(output)
    unique_inputs = {(str(row["path"]), str(row["sha256"])): row for row in inputs}
    return candidates, selected, plot_records, list(unique_inputs.values())


def _draw_triangle(ax: plt.Axes, labels: Sequence[str]) -> None:
    vertices = barycentric_to_cartesian(np.eye(3))
    boundary = np.vstack((vertices, vertices[0]))
    ax.plot(boundary[:, 0], boundary[:, 1], color="#333333", lw=0.9, zorder=1)
    for fraction in (0.2, 0.4, 0.6, 0.8):
        for fixed_component in range(3):
            endpoints = []
            for free_component in range(3):
                if free_component == fixed_component:
                    continue
                composition = np.zeros(3)
                composition[fixed_component] = fraction
                composition[free_component] = 1.0 - fraction
                endpoints.append(barycentric_to_cartesian(composition)[0:2])
            endpoints = np.asarray(endpoints)
            ax.plot(endpoints[:, 0], endpoints[:, 1], color="#D9D9D9", lw=0.45, zorder=0)
    ax.text(-0.035, -0.045, f"{labels[0]} (1)", ha="left", va="top", fontsize=8.0)
    ax.text(1.035, -0.045, f"{labels[1]} (2)", ha="right", va="top", fontsize=8.0)
    ax.text(0.5, SQRT3_OVER_2 + 0.045, f"{labels[2]} (3)", ha="center", va="bottom", fontsize=8.0)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.09, SQRT3_OVER_2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_ternary_cases(
    *, selected: Sequence[dict[str, object]], plot_records: Sequence[dict[str, object]], paths: Sequence[Path]
) -> None:
    """Render four publication-grade Gibbs triangles with tie-line diagnostics."""

    _publication_style()
    frame = pd.DataFrame(plot_records)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.75))
    marker_by_model = {"thermoformer": "o", "nrtl": "s", "wilson": "D", "uniquac": "P"}
    available_models = [
        model
        for model in MODEL_LABELS
        if all(f"{model}_y_mean_{component}" in frame for component in (1, 2, 3))
        and frame[[f"{model}_y_mean_{component}" for component in (1, 2, 3)]].notna().any().any()
    ]
    for panel_index, (ax, case) in enumerate(zip(axes.ravel(), selected)):
        subset = frame[frame["case_id"] == case["case_id"]].copy()
        labels = [str(case[f"component_{component}"]) for component in (1, 2, 3)]
        _draw_triangle(ax, labels)
        x = subset[["x_1", "x_2", "x_3"]].to_numpy(dtype=float)
        y_true = subset[["y_true_1", "y_true_2", "y_true_3"]].to_numpy(dtype=float)
        x_xy = barycentric_to_cartesian(x)
        y_xy = barycentric_to_cartesian(y_true)
        for start, end in zip(x_xy, y_xy):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#B8B8B8", lw=0.5, alpha=0.65, zorder=1)
        ax.scatter(x_xy[:, 0], x_xy[:, 1], marker="^", s=22, facecolor="#B8B8B8", edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(
            y_xy[:, 0],
            y_xy[:, 1],
            marker="o",
            s=22,
            facecolor="#111111",
            edgecolor="white",
            linewidth=0.3,
            zorder=EXPERIMENT_VAPOR_ZORDER,
        )
        model_plot_order = [model for model in available_models if model != "thermoformer"]
        if "thermoformer" in available_models:
            model_plot_order.append("thermoformer")
        for model in model_plot_order:
            columns = [f"{model}_y_mean_{component}" for component in (1, 2, 3)]
            if any(column not in subset for column in columns):
                continue
            values = subset[columns].to_numpy(dtype=float)
            valid = np.all(np.isfinite(values), axis=1)
            if not np.any(valid):
                continue
            coordinates = barycentric_to_cartesian(values[valid])
            style = ternary_model_marker_style(model)
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                marker=marker_by_model[model],
                s=style["size"],
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                alpha=0.96,
                zorder=style["zorder"],
            )
        direction = str(case["direction"])
        state_symbol = "P" if direction == "isothermal" else "T"
        state_unit = "kPa" if direction == "isothermal" else "K"
        condition = (
            f"T = {float(case['condition']):.2f} K"
            if direction == "isothermal"
            else f"P = {float(case['condition']):.2f} kPa"
        )
        category = SELECTION_LABELS[str(case["selection_category"])]
        protocol_label = (
            "Low-temperature extrapolation"
            if case["protocol"] == "state_temperature_low_extrapolation"
            else "High-temperature extrapolation"
        )
        ax.set_title(
            f"{protocol_label} | {category}\n"
            f"ternary {('P-x-y' if direction == 'isothermal' else 'T-x-y')}; {condition}",
            loc="left",
            pad=8,
        )
        ax.text(
            0.02, 0.94,
            f"ThermoFormer {state_symbol} MAE = {float(case['thermoformer_state_mae']):.2f} {state_unit}\n"
            f"all-component y MAE = {float(case['thermoformer_y_mae']):.3f}; n = {int(case['points'])}",
            transform=ax.transAxes, va="top", fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        ax.text(-0.06, 1.03, chr(ord("a") + panel_index), transform=ax.transAxes, fontsize=12.0, fontweight="bold", va="top")

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#B8B8B8", markeredgecolor="white", label="Liquid composition x", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="white", label="Experimental vapor y", markersize=5),
    ]
    for model in available_models:
        label = MODEL_LABELS[model]
        handles.append(
            Line2D([0], [0], marker=marker_by_model[model], color="none",
                   markerfacecolor="none",
                   markeredgecolor=MODEL_COLORS[model], label=label, markersize=5)
        )
    handles.append(Line2D([0], [0], color="#B8B8B8", lw=0.7, label="Experimental tie line"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        labelspacing=0.7,
    )
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.14, top=0.97, wspace=0.10, hspace=0.18)
    for path in paths:
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() == ".png":
            kwargs["dpi"] = 600
        if path.suffix.lower() in {".tif", ".tiff"}:
            kwargs["dpi"] = 600
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        _atomic_figure(fig, path, **kwargs)
    plt.close(fig)


def _report(selected: Sequence[dict[str, object]], figure_path: Path, project_root: Path, counts: dict[str, int]) -> str:
    lines = [
        "# Test-exposed ternary phase-diagram case study", "", "Status: completed_exploratory.", "",
        "Four ternary Gibbs-triangle cases were selected after inspecting held-out test predictions. The selection is diagnostic and is not confirmatory evidence.", "",
        "| Category | Task | System | Condition | n | ThermoFormer state MAE | all-component y MAE | Best available classical model | ThermoFormer score - best classical score | UNIQUAC available |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in selected:
        direction = str(row["direction"])
        condition = f"{float(row['condition']):.2f} K" if direction == "isothermal" else f"{float(row['condition']):.2f} kPa"
        unit = "kPa" if direction == "isothermal" else "K"
        system = " + ".join(str(row[f"component_{i}"]) for i in (1, 2, 3))
        lines.append(
            f"| {SELECTION_LABELS[str(row['selection_category'])]} | "
            f"{'P-x-y' if direction == 'isothermal' else 'T-x-y'} | {system} | {condition} | {int(row['points'])} | "
            f"{float(row['thermoformer_state_mae']):.3f} {unit} | {float(row['thermoformer_y_mae']):.4f} | "
            f"{MODEL_LABELS[str(row['best_baseline'])]} | "
            f"{float(row['thermoformer_minus_best_baseline_score']):+.4f} | "
            f"{'yes' if row['uniquac_available'] else 'no'} |"
        )
    better_count = sum(
        float(row["thermoformer_minus_best_baseline_score"]) < 0 for row in selected
    )
    lines.extend([
        "", "Eligible fixed-condition ternary P-x-y candidates: "
        f"{counts['state_temperature_low_extrapolation']} low-temperature-extrapolation curves and "
        f"{counts['state_temperature_high_extrapolation']} high-temperature-extrapolation curves.", "",
        "The registered pressure-interpolation/extrapolation test partitions contain no ternary T-x-y samples. Accordingly, no ternary T-x-y panel is fabricated or inferred in this figure.", "",
        "The score combines state error normalized by the observed state range with the mean squared error over all three vapor-composition components. NRTL and Wilson coverage is required. UNIQUAC is shown only when its fitted parameters and solver output are available; missing predictions are never replaced by ideal-mixture values.", "",
        f"ThermoFormer has a lower normalized curve score than the best available classical model in {better_count}/{len(selected)} selected cases. The figure selects the minimum-score case in each temperature protocol, the distinct-system case nearest the complete six-case median, and one global maximum-score case. Exactly one panel is labeled high-error.", "",
        "Gray triangles denote liquid compositions, black circles denote experimental vapor compositions, and gray segments are experimental tie lines. Colored symbols denote model vapor compositions.", "",
        f"Figure: `{figure_path.relative_to(project_root).as_posix()}`.", "",
    ])
    return "\n".join(lines)


def run_ternary_phase_diagram_study(
    *, project_root: Path, settings_path: Path, overwrite: bool = False
) -> tuple[Path, Path, Path]:
    """Select ternary cases, render Gibbs triangles, and bind all artifacts."""

    project_root = project_root.resolve()
    settings_path = settings_path.resolve()
    _require_clean_scientific_code(project_root)
    _require_committed_file(project_root, settings_path, "ternary phase-diagram settings")
    generation_git_commit = _git_commit(project_root)
    settings = TernaryPhaseDiagramSettings.load(settings_path)
    outputs = [project_root / getattr(settings, field) for field in (
        "candidate_output", "plot_data_output", "figure_pdf", "figure_png", "figure_svg",
        "figure_tiff", "report_output", "manifest_output",
    )]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Ternary phase-diagram output exists; use --overwrite to replace it")
    candidates, selected, plot_records, inputs = build_ternary_case_tables(project_root=project_root, settings=settings)
    selected_by_id = {str(row["case_id"]): str(row["selection_category"]) for row in selected}
    candidate_rows = [{**row, "selection_category": selected_by_id.get(str(row["case_id"]))} for row in candidates]
    candidate_path = project_root / settings.candidate_output
    plot_data_path = project_root / settings.plot_data_output
    _atomic_csv(candidate_path, candidate_rows)
    _atomic_csv(plot_data_path, plot_records)
    figure_paths = [project_root / getattr(settings, field) for field in ("figure_pdf", "figure_png", "figure_svg", "figure_tiff")]
    plot_ternary_cases(selected=selected, plot_records=plot_records, paths=figure_paths)
    counts = {
        protocol: sum(row["protocol"] == protocol for row in candidates)
        for protocol in settings.protocol_by_direction["isothermal"]
    }
    report_path = project_root / settings.report_output
    atomic_write_text(report_path, _report(selected, figure_paths[0], project_root, counts))
    manifest_path = project_root / settings.manifest_output
    output_paths = (*figure_paths, candidate_path, plot_data_path, report_path)
    atomic_write_json(manifest_path, {
        "schema_version": 1,
        "status": "completed_exploratory",
        "analysis_status": "test_exposed_exploratory",
        "experiment": "ternary_thermodynamic_phase_diagram_case_study",
        "generation_git_commit": generation_git_commit,
        "selection_partition": "test",
        "case_selection_used_test_labels": True,
        "selection_rule": "protocol minima, distinct-system candidate nearest global median, and one global maximum ThermoFormer ternary normalized curve score",
        "required_classical_models": list(REQUIRED_CLASSICAL_MODELS),
        "optional_classical_models": list(OPTIONAL_CLASSICAL_MODELS),
        "selected_cases": selected,
        "settings": {"path": settings_path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(settings_path)},
        "inputs": inputs,
        "outputs": {
            path.stem + path.suffix.replace(".", "_"): {
                "path": path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(path)
            }
            for path in output_paths
        },
    })
    return figure_paths[0], report_path, manifest_path
