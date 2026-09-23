"""Exploratory phase-diagram cases from the joint binary--ternary test split."""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from thermo import Chemical
from thermo.unifac import UFSG, UNIFAC_group_assignment_DDBST

from ..configuration import load_experiment_config
from ..data import discover_vle_workbooks
from ..data.splitting import canonical_smiles, dataset_digest, load_split_assignment, system_id
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
from .ternary_phase_diagrams import barycentric_to_cartesian
from .external_vapor_pressure import validated_external_vapor_pressures
from .phasepy_adapter import (
    fit_phasepy_activity_model,
    phasepy_version,
    predict_phasepy_activity_model,
)
from .thermodynamic_runner import (
    BaselineCampaignSettings,
    _git_commit,
    _require_clean_scientific_code,
    _require_committed_file,
)


PROTOCOL = "overall_binary_ternary"
THERMOFORMER_COLOR = MODEL_COLORS["thermoformer"]
CLASSICAL_MODELS = ("nrtl", "wilson", "uniquac")
CLASSICAL_MARKERS = {"nrtl": "s", "wilson": "D", "uniquac": "P"}
DISPLAY_NAME_OVERRIDES = {
    "CC(=O)C(C)C": "3-methyl-2-butanone",
    "CS(C)=O": "dimethyl sulfoxide",
    "CC(=O)OCC(C)C": "isobutyl acetate",
    "CCCOC(C)=O": "propyl acetate",
    "c1ccccc1": "benzene",
    "c1ccsc1": "thiophene",
    "NC1CCCCC1": "cyclohexylamine",
    "Nc1ccccc1": "aniline",
}


@dataclass(frozen=True)
class OverallPhaseDiagramSettings:
    seed: int
    minimum_points: int
    pressure_range_floor_kpa: float
    temperature_range_floor_k: float
    binary_candidate_output: str
    binary_plot_data_output: str
    binary_figure_stem: str
    ternary_candidate_output: str
    ternary_plot_data_output: str
    ternary_figure_stem: str
    combined_figure_stem: str
    classical_fit_audit_output: str
    report_output: str
    manifest_output: str

    @classmethod
    def load(cls, path: Path) -> "OverallPhaseDiagramSettings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {field for field in cls.__dataclass_fields__}
        if set(payload) != required:
            raise ValueError("Overall phase-diagram config has unexpected fields")
        settings = cls(**payload)
        if settings.seed != 0:
            raise ValueError("The exploratory joint-test case study is frozen to seed 0")
        if settings.minimum_points < 5:
            raise ValueError("At least five composition points are required per case")
        if settings.pressure_range_floor_kpa <= 0 or settings.temperature_range_floor_k <= 0:
            raise ValueError("Curve-score normalization floors must be positive")
        return settings


def overall_candidate_metrics(
    frame: pd.DataFrame,
    *,
    direction: str,
    component_count: int,
    pressure_range_floor_kpa: float,
    temperature_range_floor_k: float,
) -> dict[str, float]:
    """Compute state and all-component composition errors for one fixed-condition curve."""

    state_target, state_prediction, _, _ = _state_columns(direction)
    target = frame[state_target].to_numpy(dtype=float)
    prediction = frame[state_prediction].to_numpy(dtype=float)
    components = range(1, component_count + 1)
    y_true = frame[[f"y_true_{index}" for index in components]].to_numpy(dtype=float)
    y_prediction = frame[[f"y_pred_{index}" for index in components]].to_numpy(dtype=float)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(y_prediction)):
        raise ValueError("A phase-diagram candidate contains non-finite predictions")
    floor = pressure_range_floor_kpa if direction == "isothermal" else temperature_range_floor_k
    state_scale = max(float(np.ptp(target)), floor)
    state_error = prediction - target
    y_error = y_prediction - y_true
    return {
        "state_scale": state_scale,
        "thermoformer_state_mae": float(np.mean(np.abs(state_error))),
        "thermoformer_state_rmse": float(np.sqrt(np.mean(state_error**2))),
        "thermoformer_y_mae": float(np.mean(np.abs(y_error))),
        "thermoformer_y_rmse": float(np.sqrt(np.mean(y_error**2))),
        "thermoformer_curve_score": float(
            np.sqrt(np.mean((state_error / state_scale) ** 2) + np.mean(y_error**2))
        ),
    }


def select_six_representative_cases(
    candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Select two task minima per direction, one median case, and one maximum."""

    if len(candidates) < 6:
        raise ValueError("At least six eligible cases are required")
    worst = dict(
        max(
            candidates,
            key=lambda row: (float(row["thermoformer_curve_score"]), str(row["case_id"])),
        )
    )
    worst["selection_category"] = "high_error"
    best_by_direction: dict[str, list[dict[str, object]]] = {}
    for direction in ("isothermal", "isobaric"):
        eligible = sorted(
            (
                row
                for row in candidates
                if row["direction"] == direction and row["case_id"] != worst["case_id"]
            ),
            key=lambda row: (float(row["thermoformer_curve_score"]), str(row["case_id"])),
        )
        if len(eligible) < 2:
            raise ValueError(f"At least two non-maximum {direction} cases are required")
        best_by_direction[direction] = []
        for row in eligible[:2]:
            selected = dict(row)
            selected["selection_category"] = "well_predicted"
            best_by_direction[direction].append(selected)

    reserved_ids = {
        str(row["case_id"])
        for row in (
            *best_by_direction["isothermal"],
            *best_by_direction["isobaric"],
            worst,
        )
    }
    median_score = float(
        np.median([float(row["thermoformer_curve_score"]) for row in candidates])
    )
    representative_pool = [
        row for row in candidates if str(row["case_id"]) not in reserved_ids
    ]
    if not representative_pool:
        raise ValueError("No unreserved median-representative case is available")
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
    return [
        best_by_direction["isothermal"][0],
        best_by_direction["isobaric"][0],
        representative,
        best_by_direction["isothermal"][1],
        best_by_direction["isobaric"][1],
        worst,
    ]


def _display_name(smiles: str, names: dict[str, str]) -> str:
    canonical = canonical_smiles(smiles)
    return DISPLAY_NAME_OVERRIDES.get(canonical, names.get(canonical, smiles))


def build_overall_case_tables(
    *, project_root: Path, settings: OverallPhaseDiagramSettings
) -> tuple[
    dict[int, list[dict[str, object]]],
    dict[int, list[dict[str, object]]],
    dict[int, list[dict[str, object]]],
    list[dict[str, object]],
]:
    """Build binary and ternary candidates from the formal seed-0 joint test split."""

    campaign = BaselineCampaignSettings.load(
        project_root / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
    )
    prediction_root = project_root / campaign.thermoformer_output_template.format(protocol=PROTOCOL)
    predictions, inputs = _load_formal_prediction(
        project_root=project_root,
        manifest_path=prediction_root / f"seed_{settings.seed}" / "manifest.json",
        protocol=PROTOCOL,
        seed=settings.seed,
        model=None,
    )
    metadata, molecule_names = _sample_metadata(project_root)
    predictions = predictions.copy()
    predictions["condition"] = [
        round(float(metadata[sample].temperature_k), 2)
        if direction == "isothermal"
        else round(float(metadata[sample].pressure_kpa), 2)
        for sample, direction in zip(predictions["sample_id"], predictions["direction"])
    ]
    converged = predictions["converged"].astype(str).str.lower().eq("true")
    nonphysical = predictions["nonphysical"].astype(str).str.lower().eq("true")
    predictions = predictions[converged & ~nonphysical].copy()

    candidates_by_count: dict[int, list[dict[str, object]]] = {}
    selected_by_count: dict[int, list[dict[str, object]]] = {}
    plot_rows_by_count: dict[int, list[dict[str, object]]] = {}
    for component_count in (2, 3):
        candidates: list[dict[str, object]] = []
        subset = predictions[predictions["component_count"] == component_count]
        grouped = subset.groupby(["direction", "system_id", "condition"], sort=True)
        for (direction, system_id, condition), frame in grouped:
            unique_x = frame[[f"x_{index}" for index in range(1, component_count + 1)]].drop_duplicates()
            if len(frame) < settings.minimum_points or len(unique_x) < settings.minimum_points:
                continue
            metrics = overall_candidate_metrics(
                frame,
                direction=str(direction),
                component_count=component_count,
                pressure_range_floor_kpa=settings.pressure_range_floor_kpa,
                temperature_range_floor_k=settings.temperature_range_floor_k,
            )
            first = frame.iloc[0]
            case_id = f"{component_count}c:{direction}:{system_id}:{float(condition):.2f}"
            candidate: dict[str, object] = {
                "case_id": case_id,
                "protocol": PROTOCOL,
                "seed": settings.seed,
                "direction": direction,
                "component_count": component_count,
                "system_id": system_id,
                "condition": float(condition),
                "points": len(frame),
                "doi": first["doi"],
                **metrics,
            }
            for index in range(1, component_count + 1):
                smiles = str(first[f"component_smiles_{index}"])
                candidate[f"component_smiles_{index}"] = smiles
                candidate[f"component_{index}"] = _display_name(smiles, molecule_names)
            candidates.append(candidate)
        if len(candidates) < 4:
            raise ValueError(f"Fewer than four eligible {component_count}-component cases")
        selected = select_six_representative_cases(candidates)
        category_by_id = {
            str(row["case_id"]): str(row["selection_category"]) for row in selected
        }
        plot_rows: list[dict[str, object]] = []
        for row in subset.to_dict(orient="records"):
            case_id = (
                f"{component_count}c:{row['direction']}:{row['system_id']}:"
                f"{float(row['condition']):.2f}"
            )
            if case_id in category_by_id:
                output = dict(row)
                output["case_id"] = case_id
                output["selection_category"] = category_by_id[case_id]
                plot_rows.append(output)
        candidates_by_count[component_count] = candidates
        selected_by_count[component_count] = selected
        plot_rows_by_count[component_count] = plot_rows
    return candidates_by_count, selected_by_count, plot_rows_by_count, inputs


def fit_case_specific_classical_models(
    *,
    project_root: Path,
    settings: OverallPhaseDiagramSettings,
    selected_by_count: dict[int, list[dict[str, object]]],
    plot_rows_by_count: dict[int, list[dict[str, object]]],
) -> tuple[dict[int, list[dict[str, object]]], dict[str, object], list[dict[str, object]]]:
    """Fit post-hoc system-specific NRTL, Wilson and UNIQUAC models.

    Interaction parameters use every curated dataset row from each displayed chemical system.
    Pure-component vapor pressures use validated external correlations with explicit
    units, validity ranges and positive dPsat/dT. The resulting curves are deliberately
    in-sample diagnostics, not unknown-system predictive baselines.
    """

    metadata, _ = _sample_metadata(project_root)
    full_dataset = tuple(metadata.values())
    split_path = project_root / 'datasets/splits/vle' / PROTOCOL / f"seed_{settings.seed}.json"
    _require_committed_file(project_root, split_path, "overall split")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_ids = set().union(
        *(set(values) for values in split_payload["partitions"].values())
    )
    dataset = tuple(sample for identity, sample in metadata.items() if identity in split_ids)
    split = load_split_assignment(split_path, dataset)
    if split.protocol != PROTOCOL or split.seed != settings.seed:
        raise ValueError("Overall split identity does not match the case-study config")
    campaign_path = project_root / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
    campaign = BaselineCampaignSettings.load(campaign_path)
    config_path = project_root / "configs/vle/comparison/studies/thermodynamic_models/config.json"
    selected_systems = {
        str(row["system_id"])
        for count in (2, 3)
        for row in selected_by_count[count]
    }
    displayed_names: dict[str, str] = {}
    for count in (2, 3):
        for row in selected_by_count[count]:
            for index in range(1, count + 1):
                displayed_names[canonical_smiles(str(row[f"component_smiles_{index}"]))] = str(
                    row[f"component_{index}"]
                )
    uniquac_sizes: dict[str, tuple[float, float]] = {}
    uniquac_size_audit: list[dict[str, object]] = []
    for smiles, name in sorted(displayed_names.items()):
        try:
            chemical = Chemical(name)
            groups = UNIFAC_group_assignment_DDBST(chemical.CAS, "UNIFAC")
            r_value = float(sum(UFSG[group].R * count for group, count in groups.items()))
            q_value = float(sum(UFSG[group].Q * count for group, count in groups.items()))
            if not groups or not math.isfinite(r_value) or not math.isfinite(q_value) or min(r_value, q_value) <= 0.0:
                raise ValueError("UNIFAC group assignment is unavailable")
            uniquac_sizes[smiles] = (r_value, q_value)
            uniquac_size_audit.append(
                {"smiles": smiles, "name": name, "cas": chemical.CAS, "r": r_value, "q": q_value, "source": "thermo DDBST UNIFAC subgroup assignment"}
            )
        except (KeyError, TypeError, ValueError) as error:
            uniquac_size_audit.append(
                {"smiles": smiles, "name": name, "r": None, "q": None, "source": "unavailable", "error": str(error).splitlines()[0][:120]}
            )
    fit_by_system: dict[str, list[Any]] = {key: [] for key in selected_systems}
    for row in full_dataset:
        key = system_id(row)
        if key in fit_by_system:
            fit_by_system[key].append(row)
    test_by_system: dict[str, list[Any]] = {key: [] for key in selected_systems}
    for row in split.test:
        key = system_id(row)
        if key in test_by_system:
            test_by_system[key].append(row)

    system_temperature_bounds: dict[str, tuple[float, float]] = {}
    required_ranges: dict[str, tuple[float, float]] = {}
    for key, fit_rows in fit_by_system.items():
        displayed_sample_ids = {
            str(row["sample_id"])
            for count in (2, 3)
            for row in plot_rows_by_count[count]
            if str(row["system_id"]) == key
        }
        displayed_temperatures = np.asarray(
            [metadata[identity].temperature_k for identity in displayed_sample_ids], dtype=float
        )
        solver_bounds = (
            max(150.0, float(displayed_temperatures.min()) - 20.0),
            min(1500.0, float(displayed_temperatures.max()) + 20.0),
        )
        system_temperature_bounds[key] = solver_bounds
        fit_temperatures = np.asarray([row.temperature_k for row in fit_rows], dtype=float)
        required = (
            min(float(fit_temperatures.min()), solver_bounds[0]),
            max(float(fit_temperatures.max()), solver_bounds[1]),
        )
        components = {canonical_smiles(value) for row in fit_rows for value in row.smiles}
        for component in components:
            existing = required_ranges.get(component)
            required_ranges[component] = (
                min(required[0], existing[0]) if existing else required[0],
                max(required[1], existing[1]) if existing else required[1],
            )
    vapor_pressure, vapor_audit = validated_external_vapor_pressures(
        displayed_names, required_ranges
    )

    prediction_maps: dict[str, dict[tuple[str, str], dict[str, object]]] = {
        model: {} for model in campaign.models
    }
    fit_records: list[dict[str, object]] = []
    for key in sorted(selected_systems):
        fit_rows = fit_by_system[key]
        prediction_rows = test_by_system[key]
        if not fit_rows or not prediction_rows:
            raise ValueError(f"Selected system is absent from the overall test split: {key}")
        components = tuple(sorted(canonical_smiles(value) for value in fit_rows[0].smiles))
        solver_bounds = system_temperature_bounds[key]
        missing_vapor_pressure = [value for value in components if value not in vapor_pressure]
        for model in campaign.models:
            if missing_vapor_pressure:
                fit_records.append(
                    {
                        "system_id": key,
                        "component_count": fit_rows[0].component_count,
                        "components": list(components),
                        "model": model,
                        "fit_partition": "all curated dataset rows belonging to this exact chemical system",
                        "training_rows": len(fit_rows),
                        "registered_overall_test_rows": len(prediction_rows),
                        "additional_full_dataset_rows": len(fit_rows) - len(prediction_rows),
                        "success": False,
                        "unavailable_reason": "missing_valid_external_vapor_pressure",
                        "missing_components": missing_vapor_pressure,
                        "isobaric_branch_temperature_bounds_k": list(solver_bounds),
                        "predictions": 0,
                        "converged_predictions": 0,
                    }
                )
                continue
            fitted = fit_phasepy_activity_model(
                model,
                fit_rows,
                vapor_pressure,
                prediction_samples=prediction_rows,
                temperature_bounds_k=solver_bounds,
                maximum_evaluations=campaign.maximum_optimizer_iterations,
                nrtl_alpha=campaign.nrtl_alpha,
                uniquac_sizes=uniquac_sizes,
            )
            predictions = (
                predict_phasepy_activity_model(
                    prediction_rows,
                    fitted,
                    vapor_pressure,
                    temperature_bounds_k=solver_bounds,
                )
                if fitted.success
                else []
            )
            prediction_maps[model].update(
                {
                    (str(record["sample_id"]), str(record["direction"])): record
                    for record in predictions
                }
            )
            converged = sum(bool(record["converged"]) for record in predictions)
            fit_records.append(
                {
                    "system_id": key,
                    "component_count": fit_rows[0].component_count,
                    "components": list(components),
                    "model": model,
                    "fit_partition": "all curated dataset rows belonging to this exact chemical system",
                    "training_rows": len(fit_rows),
                    "registered_overall_test_rows": len(prediction_rows),
                    "additional_full_dataset_rows": len(fit_rows) - len(prediction_rows),
                    "success": bool(fitted.success),
                    "backend": "phasepy",
                    "phasepy_version": fitted.phasepy_version,
                    "phasepy_equilibrium_solver": (
                        "bubblePy" if all(row.experiment_mode == "isothermal" for row in prediction_rows)
                        else "bubbleTy_or_bubblePy_by_experiment_mode"
                    ),
                    "initial_direct_vle_loss": fitted.initial_training_loss,
                    "final_direct_vle_loss": fitted.final_training_loss,
                    "optimizer_status": fitted.optimizer_status,
                    "optimizer_message": fitted.optimizer_message,
                    "isobaric_branch_temperature_bounds_k": list(solver_bounds),
                    "vapor_pressure_methods": {
                        component: vapor_pressure[component].method for component in components
                    },
                    "predictions": len(predictions),
                    "converged_predictions": converged,
                    "ordered_pair_parameters_k": [
                        {"from": left, "to": right, "value": value}
                        for left_index, left in enumerate(components)
                        for right_index, right in enumerate(components)
                        if left_index != right_index
                        for value in [float(fitted.interaction_k[left_index, right_index])]
                    ],
                }
            )

    augmented: dict[int, list[dict[str, object]]] = {}
    for count in (2, 3):
        augmented[count] = []
        for source in plot_rows_by_count[count]:
            row = dict(source)
            identity = (str(row["sample_id"]), str(row["direction"]))
            for model in campaign.models:
                prediction = prediction_maps[model].get(identity)
                row[f"{model}_converged"] = bool(prediction and prediction["converged"])
                row[f"{model}_predicted_pressure_kpa"] = (
                    prediction.get("predicted_pressure_kpa") if prediction else None
                )
                row[f"{model}_predicted_temperature_k"] = (
                    prediction.get("predicted_temperature_k") if prediction else None
                )
                for index in range(1, count + 1):
                    row[f"{model}_y_pred_{index}"] = (
                        prediction.get(f"y_pred_{index}") if prediction else None
                    )
            augmented[count].append(row)

    for count in (2, 3):
        frame = pd.DataFrame(augmented[count])
        for case in selected_by_count[count]:
            subset = frame[frame["case_id"] == case["case_id"]]
            direction = str(case["direction"])
            target_name = "target_pressure_kpa" if direction == "isothermal" else "target_temperature_k"
            prediction_name = "predicted_pressure_kpa" if direction == "isothermal" else "predicted_temperature_k"
            target = pd.to_numeric(subset[target_name], errors="coerce").to_numpy(dtype=float)
            y_true = subset[[f"y_true_{index}" for index in range(1, count + 1)]].to_numpy(dtype=float)
            for model in campaign.models:
                state = pd.to_numeric(subset[f"{model}_{prediction_name}"], errors="coerce").to_numpy(dtype=float)
                y_prediction = subset[[f"{model}_y_pred_{index}" for index in range(1, count + 1)]].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)
                valid_state = np.isfinite(target) & np.isfinite(state)
                valid_y = np.isfinite(y_true) & np.isfinite(y_prediction)
                case[f"{model}_state_mae"] = (
                    float(np.mean(np.abs(state[valid_state] - target[valid_state])))
                    if np.any(valid_state) else None
                )
                case[f"{model}_y_mae"] = (
                    float(np.mean(np.abs(y_prediction[valid_y] - y_true[valid_y])))
                    if np.any(valid_y) else None
                )
                case[f"{model}_valid_rows"] = int(valid_state.sum())

    audit: dict[str, object] = {
        "analysis_status": "test_exposed_descriptive_in_sample_fit",
        "fit_scope": "one parameter set per displayed chemical system using all matching curated dataset rows across prior split partitions",
        "full_dataset_rows": len(full_dataset),
        "full_dataset_sha256": dataset_digest(full_dataset),
        "pure_vapor_pressure_scope": "validated external correlations only; no mixture-endpoint fit or out-of-range extrapolation",
        "parameterization": "Phasepy g_ij/T or a_ij/T; NRTL alpha fixed at 0.3",
        "thermodynamic_backend": "phasepy",
        "phasepy_version": phasepy_version(),
        "phasepy_activity_models": list(campaign.models),
        "phasepy_equilibrium_solvers": ["bubblePy", "bubbleTy"],
        "phasepy_vapor_model": "ideal_gas virial-gamma; validated external Psat callback",
        "models": list(campaign.models),
        "isobaric_branch_rule": "system test-temperature range expanded by 20 K on both sides; label-exposed diagnostic root selection",
        "pure_vapor_pressure_audit": vapor_audit,
        "uniquac_size_parameters": uniquac_size_audit,
        "fits": fit_records,
    }
    experiment = load_experiment_config(config_path)
    workbooks = discover_vle_workbooks(project_root / experiment.data.root, experiment.data.source_filter)
    provenance = [
        {"kind": "split", "path": split_path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(split_path)},
        {"kind": "baseline_settings", "path": campaign_path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(campaign_path)},
        {"kind": "baseline_config", "path": config_path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(config_path)},
        *[
            {"kind": "dataset_workbook", "path": path.relative_to(project_root).as_posix(), "sha256": artifact_sha256(path)}
            for path in workbooks
        ],
    ]
    return augmented, audit, provenance


def _figure_paths(project_root: Path, stem: str) -> list[Path]:
    return [project_root / f"{stem}.{suffix}" for suffix in ("pdf", "png", "svg", "tiff")]


def _plot_classical_binary(ax: plt.Axes, subset: pd.DataFrame, direction: str) -> None:
    state_name = "predicted_pressure_kpa" if direction == "isothermal" else "predicted_temperature_k"
    for model in CLASSICAL_MODELS:
        state = pd.to_numeric(subset[f"{model}_{state_name}"], errors="coerce").to_numpy()
        y_prediction = pd.to_numeric(subset[f"{model}_y_pred_1"], errors="coerce").to_numpy()
        x = pd.to_numeric(subset["x_1"], errors="coerce").to_numpy()
        valid = np.isfinite(state) & np.isfinite(y_prediction) & np.isfinite(x)
        if valid.sum() < 2:
            continue
        if direction == "isobaric":
            ax.scatter(x[valid], state[valid], marker=CLASSICAL_MARKERS[model], s=16,
                       facecolor="none", edgecolor=MODEL_COLORS[model], linewidth=0.75,
                       alpha=0.82, zorder=4)
            ax.scatter(y_prediction[valid], state[valid], marker=CLASSICAL_MARKERS[model], s=12,
                       facecolor="none", edgecolor=MODEL_COLORS[model], linewidth=0.65,
                       alpha=0.48, zorder=4)
        else:
            ax.plot(x[valid][np.argsort(x[valid])], state[valid][np.argsort(x[valid])],
                    color=MODEL_COLORS[model], lw=1.05, alpha=0.90, zorder=4)
            ax.plot(y_prediction[valid][np.argsort(y_prediction[valid])],
                    state[valid][np.argsort(y_prediction[valid])], color=MODEL_COLORS[model],
                    lw=1.0, ls=":", alpha=0.90, zorder=4)


def _plot_classical_ternary(ax: plt.Axes, subset: pd.DataFrame) -> bool:
    plotted = False
    for model in CLASSICAL_MODELS:
        values = subset[[f"{model}_y_pred_{index}" for index in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        if not np.any(valid):
            continue
        coordinates = barycentric_to_cartesian(values[valid])
        ax.scatter(coordinates[:, 0], coordinates[:, 1], marker=CLASSICAL_MARKERS[model],
                   s=19, facecolor="none", edgecolor=MODEL_COLORS[model], linewidth=0.75,
                   alpha=0.82, zorder=8)
        plotted = True
    return plotted


def plot_binary_overall_cases(
    *, selected: Sequence[dict[str, object]], plot_rows: Sequence[dict[str, object]], paths: Sequence[Path]
) -> None:
    """Render six binary phase diagrams from the joint model's seed-0 test split."""

    _publication_style()
    frame = pd.DataFrame(plot_rows)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.45))
    for panel, (ax, case) in enumerate(zip(axes.ravel(), selected)):
        subset = frame[frame["case_id"] == case["case_id"]].copy()
        direction = str(case["direction"])
        state_target, state_prediction, _, symbol = _state_columns(direction)
        bubble_order = np.argsort(subset["x_1"].to_numpy(dtype=float))
        dew_order = np.argsort(subset["y_pred_1"].to_numpy(dtype=float))
        state = subset[state_prediction].to_numpy(dtype=float)
        ax.plot(
            subset["x_1"].to_numpy(dtype=float)[bubble_order],
            state[bubble_order],
            color=THERMOFORMER_COLOR,
            lw=1.8,
            zorder=5,
        )
        ax.plot(
            subset["y_pred_1"].to_numpy(dtype=float)[dew_order],
            state[dew_order],
            color=THERMOFORMER_COLOR,
            lw=1.6,
            ls="--",
            zorder=5,
        )
        _plot_classical_binary(ax, subset, direction)
        target = subset[state_target].to_numpy(dtype=float)
        ax.scatter(subset["x_1"], target, s=24, color="#111111", edgecolor="white", linewidth=0.4, zorder=8)
        ax.scatter(subset["y_true_1"], target, s=24, facecolor="white", edgecolor="#111111", linewidth=0.9, zorder=8)
        condition = (
            f"T = {float(case['condition']):.2f} K"
            if direction == "isothermal"
            else f"P = {float(case['condition']):.2f} kPa"
        )
        system_label = f"{case['component_1']} (1) + {case['component_2']} (2)"
        wrapped_system = textwrap.fill(system_label, width=31, break_long_words=False)
        ax.set_title(
            f"{SELECTION_LABELS[str(case['selection_category'])]} case\n"
            f"{wrapped_system}\n{condition}",
            loc="left",
            pad=6,
        )
        unit = "kPa" if direction == "isothermal" else "K"
        ax.text(
            0.02,
            0.96,
            f"ThermoFormer {symbol} MAE = {float(case['thermoformer_state_mae']):.2f} {unit}\n"
            f"all-component y MAE = {float(case['thermoformer_y_mae']):.3f}; n = {int(case['points'])}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.4,
            zorder=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.5},
        )
        ax.set_xlabel(r"Mole fraction of component 1, $x_1$ or $y_1$")
        ax.set_ylabel("Pressure, P (kPa)" if direction == "isothermal" else "Temperature, T (K)")
        ax.set_xlim(-0.025, 1.025)
        ax.tick_params(direction="in")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.18, 1.09, chr(ord("a") + panel), transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    handles = [
        *[Line2D([0], [0], color=MODEL_COLORS[model], lw=1.7 if model == "thermoformer" else 1.1,
                 label=MODEL_LABELS[model]) for model in ("thermoformer", *CLASSICAL_MODELS)],
        Line2D([0], [0], color="#444444", lw=1.4, ls="-", label="Bubble branch"),
        Line2D([0], [0], color="#444444", lw=1.2, ls=":", label="Dew branch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="white", label="Experiment (x)", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#111111", label="Experiment (y)", markersize=5),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, frameon=False, columnspacing=1.25)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.965, wspace=0.32, hspace=0.48)
    for path in paths:
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        if path.suffix.lower() == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        _atomic_figure(fig, path, **kwargs)
    plt.close(fig)


def plot_ternary_overall_cases(
    *, selected: Sequence[dict[str, object]], plot_rows: Sequence[dict[str, object]], paths: Sequence[Path]
) -> None:
    """Render six ternary Gibbs triangles with ThermoFormer drawn on the top layer."""

    _publication_style()
    frame = pd.DataFrame(plot_rows)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.65))
    for panel, (ax, case) in enumerate(zip(axes.ravel(), selected)):
        subset = frame[frame["case_id"] == case["case_id"]]
        vertices = barycentric_to_cartesian(np.eye(3))
        boundary = np.vstack((vertices, vertices[0]))
        ax.plot(boundary[:, 0], boundary[:, 1], color="#333333", lw=1.0, zorder=1)
        for fraction in (0.2, 0.4, 0.6, 0.8):
            for fixed in range(3):
                endpoints = []
                for free in range(3):
                    if free == fixed:
                        continue
                    composition = np.zeros(3)
                    composition[fixed] = fraction
                    composition[free] = 1.0 - fraction
                    endpoints.append(barycentric_to_cartesian(composition))
                endpoints = np.asarray(endpoints)
                ax.plot(endpoints[:, 0], endpoints[:, 1], color="#D9D9D9", lw=0.5, zorder=0)
        labels = [str(case[f"component_{index}"]) for index in (1, 2, 3)]
        ax.text(-0.035, -0.045, f"{labels[0]} (1)", ha="left", va="top", fontsize=10.5)
        ax.text(1.035, -0.045, f"{labels[1]} (2)", ha="right", va="top", fontsize=10.5)
        ax.text(0.5, np.sqrt(3) / 2 + 0.045, f"{labels[2]} (3)", ha="center", va="bottom", fontsize=10.5)
        x = subset[["x_1", "x_2", "x_3"]].to_numpy(dtype=float)
        y_true = subset[["y_true_1", "y_true_2", "y_true_3"]].to_numpy(dtype=float)
        y_pred = subset[["y_pred_1", "y_pred_2", "y_pred_3"]].to_numpy(dtype=float)
        x_xy, y_true_xy, y_pred_xy = map(barycentric_to_cartesian, (x, y_true, y_pred))
        for start, end in zip(x_xy, y_true_xy):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#B8B8B8", lw=0.55, alpha=0.65, zorder=1)
        ax.scatter(x_xy[:, 0], x_xy[:, 1], marker="^", s=24, facecolor="#B8B8B8", edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(y_true_xy[:, 0], y_true_xy[:, 1], marker="o", s=24, facecolor="#111111", edgecolor="white", linewidth=0.3, zorder=7)
        classical_available = _plot_classical_ternary(ax, subset)
        ax.scatter(y_pred_xy[:, 0], y_pred_xy[:, 1], marker="o", s=34, facecolor="none", edgecolor=THERMOFORMER_COLOR, linewidth=1.35, zorder=10)
        if not classical_available:
            ax.text(
                0.98,
                0.62,
                "Classical models: N/A\n(validated Psat unavailable)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.8,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
                zorder=12,
            )
        direction = str(case["direction"])
        condition = (
            f"T = {float(case['condition']):.2f} K"
            if direction == "isothermal"
            else f"P = {float(case['condition']):.2f} kPa"
        )
        ax.set_title(
            f"{SELECTION_LABELS[str(case['selection_category'])]} ternary "
            f"{'P-x-y' if direction == 'isothermal' else 'T-x-y'}; {condition}",
            loc="left",
            pad=8,
            fontsize=10.5,
        )
        symbol, unit = ("P", "kPa") if direction == "isothermal" else ("T", "K")
        ax.text(
            0.02,
            0.94,
            f"ThermoFormer {symbol} MAE = {float(case['thermoformer_state_mae']):.2f} {unit}\n"
            f"all-component y MAE = {float(case['thermoformer_y_mae']):.3f}; n = {int(case['points'])}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.09, np.sqrt(3) / 2 + 0.10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-0.18, 1.09, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#B8B8B8", markeredgecolor="white", label="Liquid composition x", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="white", label="Experimental vapor y", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=THERMOFORMER_COLOR, label="ThermoFormer vapor y", markersize=6),
        *[Line2D([0], [0], marker=CLASSICAL_MARKERS[model], color="none", markerfacecolor="none",
                 markeredgecolor=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} vapor y", markersize=5)
          for model in CLASSICAL_MODELS],
        Line2D([0], [0], color="#B8B8B8", lw=0.8, label="Experimental tie line"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.13, top=0.97, wspace=0.08, hspace=0.20)
    for path in paths:
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        if path.suffix.lower() == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        _atomic_figure(fig, path, **kwargs)
    plt.close(fig)


def plot_combined_overall_cases(
    *,
    binary_selected: Sequence[dict[str, object]],
    binary_plot_rows: Sequence[dict[str, object]],
    ternary_selected: Sequence[dict[str, object]],
    ternary_plot_rows: Sequence[dict[str, object]],
    paths: Sequence[Path],
) -> None:
    """Render a compact Nature-style 4-by-3 binary/ternary case plate."""

    _publication_style()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    panel_label_size = 12.0
    title_size = 10.4
    metric_size = 8.8
    binary_frame = pd.DataFrame(binary_plot_rows)
    ternary_frame = pd.DataFrame(ternary_plot_rows)
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 9.35))

    for panel, (ax, case) in enumerate(zip(axes[:2].ravel(), binary_selected)):
        subset = binary_frame[binary_frame["case_id"] == case["case_id"]].copy()
        direction = str(case["direction"])
        state_target, state_prediction, _, symbol = _state_columns(direction)
        state = subset[state_prediction].to_numpy(dtype=float)
        bubble_order = np.argsort(subset["x_1"].to_numpy(dtype=float))
        dew_order = np.argsort(subset["y_pred_1"].to_numpy(dtype=float))
        ax.plot(subset["x_1"].to_numpy(dtype=float)[bubble_order], state[bubble_order],
                color=THERMOFORMER_COLOR, lw=1.75, zorder=6)
        ax.plot(subset["y_pred_1"].to_numpy(dtype=float)[dew_order], state[dew_order],
                color=THERMOFORMER_COLOR, lw=1.55, ls="--", zorder=6)
        _plot_classical_binary(ax, subset, direction)
        target = subset[state_target].to_numpy(dtype=float)
        ax.scatter(subset["x_1"], target, s=22, color="#202020", edgecolor="white",
                   linewidth=0.35, zorder=9)
        ax.scatter(subset["y_true_1"], target, s=22, facecolor="white", edgecolor="#202020",
                   linewidth=0.9, zorder=9)
        condition = (f"T = {float(case['condition']):.2f} K" if direction == "isothermal"
                     else f"P = {float(case['condition']):.2f} kPa")
        system = f"{case['component_1']} + {case['component_2']}"
        ax.set_title(
            f"{SELECTION_LABELS[str(case['selection_category'])]} · "
            f"{'P–x–y' if direction == 'isothermal' else 'T–x–y'}\n"
            f"{textwrap.fill(system, width=27, break_long_words=False)}\n{condition}",
            loc="left", pad=4, fontsize=title_size,
        )
        unit = "kPa" if direction == "isothermal" else "K"
        ax.text(0.03, 0.95,
                f"{symbol} MAE {float(case['thermoformer_state_mae']):.2f} {unit}\n"
                f"y MAE {float(case['thermoformer_y_mae']):.3f} · n={int(case['points'])}",
                transform=ax.transAxes, va="top", fontsize=metric_size,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.0},
                zorder=12)
        ax.set_xlabel(r"Mole fraction, $x_1$ or $y_1$")
        ax.set_ylabel("P (kPa)" if direction == "isothermal" else "T (K)")
        ax.set_xlim(-0.025, 1.025)
        ax.tick_params(direction="in")
        ax.spines[["top", "right"]].set_visible(False)

    for offset, (ax, case) in enumerate(zip(axes[2:].ravel(), ternary_selected)):
        panel = offset + 6
        subset = ternary_frame[ternary_frame["case_id"] == case["case_id"]]
        vertices = barycentric_to_cartesian(np.eye(3))
        boundary = np.vstack((vertices, vertices[0]))
        ax.plot(boundary[:, 0], boundary[:, 1], color="#303030", lw=0.9, zorder=1)
        for fraction in (0.2, 0.4, 0.6, 0.8):
            for fixed in range(3):
                endpoints = []
                for free in range(3):
                    if free == fixed:
                        continue
                    composition = np.zeros(3)
                    composition[fixed] = fraction
                    composition[free] = 1.0 - fraction
                    endpoints.append(barycentric_to_cartesian(composition))
                endpoints = np.asarray(endpoints)
                ax.plot(endpoints[:, 0], endpoints[:, 1], color="#DEDEDE", lw=0.45, zorder=0)
        labels = [str(case[f"component_{index}"]) for index in (1, 2, 3)]
        ax.text(-0.015, -0.025, "1", ha="left", va="top", fontsize=9.5, fontweight="bold")
        ax.text(1.015, -0.025, "2", ha="right", va="top", fontsize=9.5, fontweight="bold")
        ax.text(0.5, np.sqrt(3) / 2 + 0.025, "3", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        x = subset[["x_1", "x_2", "x_3"]].to_numpy(dtype=float)
        y_true = subset[["y_true_1", "y_true_2", "y_true_3"]].to_numpy(dtype=float)
        y_pred = subset[["y_pred_1", "y_pred_2", "y_pred_3"]].to_numpy(dtype=float)
        x_xy, y_true_xy, y_pred_xy = map(barycentric_to_cartesian, (x, y_true, y_pred))
        for start, end in zip(x_xy, y_true_xy):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#B9B9B9",
                    lw=0.5, alpha=0.62, zorder=1)
        ax.scatter(x_xy[:, 0], x_xy[:, 1], marker="^", s=20, facecolor="#B9B9B9",
                   edgecolor="white", linewidth=0.25, zorder=3)
        ax.scatter(y_true_xy[:, 0], y_true_xy[:, 1], marker="o", s=21, facecolor="#202020",
                   edgecolor="white", linewidth=0.25, zorder=7)
        classical_available = _plot_classical_ternary(ax, subset)
        ax.scatter(y_pred_xy[:, 0], y_pred_xy[:, 1], marker="o", s=31, facecolor="none",
                   edgecolor=THERMOFORMER_COLOR, linewidth=1.25, zorder=12)
        if not classical_available:
            ax.text(
                0.98,
                0.62,
                "Classical: N/A\n(validated Psat unavailable)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
                zorder=14,
            )
        direction = str(case["direction"])
        condition = (f"T = {float(case['condition']):.2f} K" if direction == "isothermal"
                     else f"P = {float(case['condition']):.2f} kPa")
        ax.set_title(
            f"{SELECTION_LABELS[str(case['selection_category'])]} · "
            f"{'P–x–y' if direction == 'isothermal' else 'T–x–y'}\n{condition}",
            loc="left", pad=5, fontsize=title_size,
        )
        symbol, unit = ("P", "kPa") if direction == "isothermal" else ("T", "K")
        ax.text(0.02, 0.93,
                f"{symbol} MAE {float(case['thermoformer_state_mae']):.2f} {unit}\n"
                f"y MAE {float(case['thermoformer_y_mae']):.3f} · n={int(case['points'])}",
                transform=ax.transAxes, va="top", fontsize=metric_size,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.0},
                zorder=14)
        ax.set_xlim(-0.075, 1.075)
        component_key = "\n".join(f"{index} {label}" for index, label in enumerate(labels, start=1))
        ax.text(0.5, -0.11, component_key, ha="center", va="top",
                fontsize=title_size, linespacing=1.02)
        ax.set_ylim(-0.16, np.sqrt(3) / 2 + 0.08)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.text(0.012, 0.745, "Binary mixtures", fontsize=10.5, fontweight="bold",
             rotation=90, ha="center", va="center")
    fig.text(0.012, 0.305, "Ternary mixtures", fontsize=10.5, fontweight="bold",
             rotation=90, ha="center", va="center")
    handles = [
        Line2D([0], [0], color=THERMOFORMER_COLOR, marker="o", markerfacecolor="none",
               markeredgecolor=THERMOFORMER_COLOR, lw=1.7, label="ThermoFormer"),
        *[Line2D([0], [0], color=MODEL_COLORS[model], marker=CLASSICAL_MARKERS[model],
                 markerfacecolor="none", markeredgecolor=MODEL_COLORS[model], lw=1.1,
                 label=MODEL_LABELS[model]) for model in CLASSICAL_MODELS],
        Line2D([0], [0], color="#444444", lw=1.2, ls=":", label="Dew branch (binary)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#202020", label="Experimental liquid / vapor", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#202020", label="Experimental vapor (binary)", markersize=5),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#B9B9B9", label="Liquid x (ternary)", markersize=5),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4,
               frameon=False, columnspacing=1.15, handletextpad=0.55, labelspacing=0.5)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.105, top=0.985,
                        wspace=0.42, hspace=0.68)
    # Anchor panel labels to the subplot grid cells rather than the rendered
    # axes boxes. Equal-aspect ternary axes are horizontally inset by
    # Matplotlib, whereas the binary Cartesian axes fill their cells.
    for panel, ax in enumerate(axes.ravel()):
        cell = ax.get_subplotspec().get_position(fig)
        fig.text(
            cell.x0 - 0.045,
            cell.y1 + 0.006,
            chr(ord("a") + panel),
            fontsize=panel_label_size,
            fontweight="bold",
            ha="right",
            va="top",
        )
    for path in paths:
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        if path.suffix.lower() == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        _atomic_figure(fig, path, **kwargs)
    plt.close(fig)


def _report(
    selected_by_count: dict[int, list[dict[str, object]]],
    candidate_counts: dict[int, int],
    binary_figure: Path,
    ternary_figure: Path,
    combined_figure: Path,
    classical_audit_path: Path,
    classical_audit: dict[str, object],
    project_root: Path,
) -> str:
    lines = [
        "# Joint binary--ternary model phase-diagram cases",
        "",
        "Status: completed_exploratory.",
        "",
        "The cases come from the formal `overall_binary_ternary` seed-0 test partition. The five formal seeds use different system-disjoint test assignments, so predictions are not combined across seeds at the case level. Case selection inspected test errors and is diagnostic rather than confirmatory.",
        "",
        "| Test subset | Category | Task | System | Condition | n | State MAE | all-component y MAE |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for component_count in (2, 3):
        for row in selected_by_count[component_count]:
            direction = str(row["direction"])
            system = " + ".join(str(row[f"component_{index}"]) for index in range(1, component_count + 1))
            condition = (
                f"{float(row['condition']):.2f} K"
                if direction == "isothermal"
                else f"{float(row['condition']):.2f} kPa"
            )
            unit = "kPa" if direction == "isothermal" else "K"
            lines.append(
                f"| {component_count}-component test | {SELECTION_LABELS[str(row['selection_category'])]} | "
                f"{'P-x-y' if direction == 'isothermal' else 'T-x-y'} | {system} | {condition} | "
                f"{int(row['points'])} | {float(row['thermoformer_state_mae']):.3f} {unit} | "
                f"{float(row['thermoformer_y_mae']):.4f} |"
            )
    system_fit_rows = [
        row for row in classical_audit["fits"] if row["model"] == CLASSICAL_MODELS[0]
    ]
    additional_rows = sum(int(row["additional_full_dataset_rows"]) for row in system_fit_rows)
    vapor_audit = classical_audit["pure_vapor_pressure_audit"]
    covered_entries = [
        row for row in vapor_audit["entries"].values() if row["status"] == "covered"
    ]
    method_counts: dict[str, int] = {}
    for row in covered_entries:
        method = str(row["correlation_method"])
        method_counts[method] = method_counts.get(method, 0) + 1
    method_summary = ", ".join(
        f"{method}: {count}" for method, count in sorted(method_counts.items())
    )
    lines.extend(
        [
            "",
            "## Post-hoc fitted classical-model comparison",
            "",
            "NRTL, Wilson, and UNIQUAC are evaluated with the open-source Phasepy backend. Interaction parameters are fitted separately for each displayed chemical system using every matching experimental row in the complete curated dataset, irrespective of its previous train/validation/test assignment; Phasepy `bubblePy` and `bubbleTy` then calculate the displayed bubble states. Pure-component vapor pressures use external DIPPR, Antoine, Wagner, VDI, or HEOS correlations from the `thermo` property database, injected through Phasepy's Psat callback. Each selected correlation must cover every fit and solver temperature, reports native Pa and output kPa units, and satisfy dPsat/dT > 0 throughout the required range. Components without a valid correlation are marked unavailable rather than extrapolated. UNIQUAC r and q use DDBST UNIFAC subgroup assignments resolved by compound identity. These are fully data-exposed, in-sample descriptive fits—not predictive baselines for unseen mixtures.",
            "Classical-model isothermal outputs are connected as fitted branches. Classical-model isobaric outputs are shown as discrete hollow points because independent bubble-temperature solves can select different mathematical roots; no post-hoc smoothing is applied.",
            "",
            f"Validated external vapor-pressure coverage: {vapor_audit['covered_components']}/{vapor_audit['components']} displayed components ({float(vapor_audit['component_coverage']):.1%}). Selected methods: {method_summary or 'none'}.",
            f"Full-dataset lookup found {additional_rows} additional rows beyond the registered overall test rows across the {len(system_fit_rows)} unique displayed systems. Because the overall split is system-disjoint, zero additional rows means all same-system rows were already in its test partition; the present recalculation changes only the pure-property source and the refitted classical-model parameters.",
            "",
            "Each cell reports state MAE / all-component vapor-composition MAE.",
            "",
            "| Subset | Task and system | ThermoFormer | NRTL fitted | Wilson fitted | UNIQUAC fitted |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for component_count in (2, 3):
        for row in selected_by_count[component_count]:
            direction = str(row["direction"])
            unit = "kPa" if direction == "isothermal" else "K"
            system = " + ".join(str(row[f"component_{index}"]) for index in range(1, component_count + 1))
            cells = [
                f"{float(row['thermoformer_state_mae']):.3f} {unit} / {float(row['thermoformer_y_mae']):.4f}"
            ]
            for model in CLASSICAL_MODELS:
                state = row.get(f"{model}_state_mae")
                y_mae = row.get(f"{model}_y_mae")
                cells.append(
                    f"{float(state):.3f} {unit} / {float(y_mae):.4f}"
                    if state is not None and y_mae is not None else "N/A"
                )
            lines.append(
                f"| {component_count}-component | {'P-x-y' if direction == 'isothermal' else 'T-x-y'}; {system} | "
                + " | ".join(cells)
                + " |"
            )
    comparison_rows = [
        row for component_count in (2, 3) for row in selected_by_count[component_count]
    ]
    lines.extend(["", "Descriptive win counts (lower error):"])
    for model in CLASSICAL_MODELS:
        state_rows = [
            row for row in comparison_rows if row.get(f"{model}_state_mae") is not None
        ]
        composition_rows = [
            row for row in comparison_rows if row.get(f"{model}_y_mae") is not None
        ]
        state_wins = sum(
            float(row["thermoformer_state_mae"]) < float(row[f"{model}_state_mae"])
            for row in state_rows
        )
        composition_wins = sum(
            float(row["thermoformer_y_mae"]) < float(row[f"{model}_y_mae"])
            for row in composition_rows
        )
        lines.append(
            f"- ThermoFormer has lower state MAE than {MODEL_LABELS[model]} in {state_wins}/{len(state_rows)} covered cases and lower vapor-composition MAE in {composition_wins}/{len(composition_rows)} covered cases."
        )
    lines.extend(
        [
            "",
            "These counts describe the selected, data-exposed cases only. They are not a statistical model ranking, and the classical models had access to all available dataset rows for the displayed systems during parameter fitting.",
        ]
    )
    lines.extend(
        [
            "",
            f"Eligible fixed-condition curves: {candidate_counts[2]} binary and {candidate_counts[3]} ternary.",
            "",
            "Within each test subset, the figure contains the two lowest-error isothermal cases, the two lowest-error isobaric cases, one remaining case nearest the candidate-score median, and one global maximum-error case. Thus each figure contains exactly one high-error panel.",
            "",
            "The normalized score combines state RMSE divided by the observed state range with all-component vapor-composition RMSE. Pressure and temperature ranges use lower bounds of 5 kPa and 5 K.",
            "",
            f"Binary figure: `{binary_figure.relative_to(project_root).as_posix()}`.",
            "",
            f"Ternary figure: `{ternary_figure.relative_to(project_root).as_posix()}`.",
            "",
            f"Combined figure: `{combined_figure.relative_to(project_root).as_posix()}`.",
            "",
            f"Classical fit parameters and optimizer audit: `{classical_audit_path.relative_to(project_root).as_posix()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_overall_phase_diagram_study(
    *, project_root: Path, config_path: Path, overwrite: bool = False
) -> tuple[Path, Path, Path]:
    """Generate binary and ternary exploratory cases with complete provenance."""

    project_root = project_root.resolve()
    config_path = config_path.resolve()
    _require_clean_scientific_code(project_root)
    _require_committed_file(project_root, config_path, "overall phase-diagram config")
    settings = OverallPhaseDiagramSettings.load(config_path)
    binary_figures = _figure_paths(project_root, settings.binary_figure_stem)
    ternary_figures = _figure_paths(project_root, settings.ternary_figure_stem)
    combined_figures = _figure_paths(project_root, settings.combined_figure_stem)
    binary_candidate_path = project_root / settings.binary_candidate_output
    binary_plot_path = project_root / settings.binary_plot_data_output
    ternary_candidate_path = project_root / settings.ternary_candidate_output
    ternary_plot_path = project_root / settings.ternary_plot_data_output
    classical_audit_path = project_root / settings.classical_fit_audit_output
    report_path = project_root / settings.report_output
    manifest_path = project_root / settings.manifest_output
    outputs = [
        *binary_figures,
        *ternary_figures,
        *combined_figures,
        binary_candidate_path,
        binary_plot_path,
        ternary_candidate_path,
        ternary_plot_path,
        classical_audit_path,
        report_path,
        manifest_path,
    ]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Overall phase-diagram output exists; use --overwrite")
    candidates, selected, plot_rows, inputs = build_overall_case_tables(
        project_root=project_root, settings=settings
    )
    plot_rows, classical_audit, classical_inputs = fit_case_specific_classical_models(
        project_root=project_root,
        settings=settings,
        selected_by_count=selected,
        plot_rows_by_count=plot_rows,
    )
    inputs.extend(classical_inputs)
    plot_binary_overall_cases(selected=selected[2], plot_rows=plot_rows[2], paths=binary_figures)
    plot_ternary_overall_cases(selected=selected[3], plot_rows=plot_rows[3], paths=ternary_figures)
    plot_combined_overall_cases(
        binary_selected=selected[2],
        binary_plot_rows=plot_rows[2],
        ternary_selected=selected[3],
        ternary_plot_rows=plot_rows[3],
        paths=combined_figures,
    )
    _atomic_csv(binary_candidate_path, candidates[2])
    _atomic_csv(binary_plot_path, plot_rows[2])
    _atomic_csv(ternary_candidate_path, candidates[3])
    _atomic_csv(ternary_plot_path, plot_rows[3])
    atomic_write_json(classical_audit_path, classical_audit)
    atomic_write_text(
        report_path,
        _report(
            selected,
            {key: len(value) for key, value in candidates.items()},
            binary_figures[0],
            ternary_figures[0],
            combined_figures[0],
            classical_audit_path,
            classical_audit,
            project_root,
        ),
    )
    output_records = {
        path.stem + path.suffix.replace(".", "_"): {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": artifact_sha256(path),
        }
        for path in outputs
        if path != manifest_path
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_exploratory",
        "analysis_status": "test_exposed_exploratory",
        "experiment": "overall_binary_ternary_phase_diagram_cases",
        "protocol": PROTOCOL,
        "seed": settings.seed,
        "selection_partition": "test",
        "case_selection_used_test_labels": True,
        "generation_git_commit": _git_commit(project_root),
        "selection_rule": "per component count: two direction minima, remaining median, global maximum",
        "selected_cases": {str(key): value for key, value in selected.items()},
        "inputs": [
            {
                "kind": "config",
                "path": config_path.relative_to(project_root).as_posix(),
                "sha256": artifact_sha256(config_path),
            },
            *inputs,
        ],
        "outputs": output_records,
    }
    atomic_write_json(manifest_path, manifest)
    return binary_figures[0], ternary_figures[0], report_path
