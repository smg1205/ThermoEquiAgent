"""Five-seed summaries for cumulative ThermoFormer training-stage ablations."""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Mapping, Sequence


STAGE_ORDER = ("stage0", "stage1", "stage2", "stage3")
_DIRECTIONS = ("isothermal", "isobaric")
_FIELDS = {
    "isothermal": {
        "state_mae": "pressure_mae_kpa",
        "state_rmse": "pressure_rmse_kpa",
        "state_r2": "pressure_r2",
        "y_mae": "y_mae",
        "y_rmse": "y_rmse",
        "y_r2": "y_r2",
        "valid_coverage": "valid_coverage",
        "solver_failure_rate": "solver_failure_rate",
        "nonphysical_rate": "nonphysical_rate",
    },
    "isobaric": {
        "state_mae": "temperature_mae_k",
        "state_rmse": "temperature_rmse_k",
        "state_r2": "temperature_r2",
        "y_mae": "y_mae",
        "y_rmse": "y_rmse",
        "y_r2": "y_r2",
        "valid_coverage": "valid_coverage",
        "solver_failure_rate": "solver_failure_rate",
        "nonphysical_rate": "nonphysical_rate",
    },
}


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def _summary(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise RuntimeError("A stage metric has no contributing seeds")
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _stages_in_order(stage_metrics: Mapping[str, object]) -> tuple[str, ...]:
    unknown = set(stage_metrics).difference(STAGE_ORDER)
    if unknown:
        raise RuntimeError("Unknown cumulative-stage labels: " + ", ".join(sorted(unknown)))
    return tuple(stage for stage in STAGE_ORDER if stage in stage_metrics)


def build_stage_summary(
    stage_metrics: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Return mean and sample standard deviation for every frozen stage snapshot.

    Input is indexed by stage then seed. Each seed payload must contain the
    two directional metric rows produced by the VLE evaluator. An optional
    ``physics_residuals.teacher_forced_fugacity`` value is aggregated once per
    stage and repeated in both directional rows for a compact result table.
    """

    result: list[dict[str, object]] = []
    for stage in _stages_in_order(stage_metrics):
        by_seed = stage_metrics[stage]
        seed_ids = tuple(sorted(by_seed))
        if not seed_ids:
            raise RuntimeError(f"{stage} has no seed metrics")
        direction_values = {
            direction: {field: [] for field in _FIELDS[direction]}
            for direction in _DIRECTIONS
        }
        fugacity_values: list[float] = []
        for seed in seed_ids:
            payload = by_seed[seed]
            for direction in _DIRECTIONS:
                row = payload.get(direction)
                if not isinstance(row, Mapping):
                    raise RuntimeError(f"{stage} seed {seed} lacks {direction} metrics")
                for output_name, source_name in _FIELDS[direction].items():
                    direction_values[direction][output_name].append(
                        _finite(row.get(source_name), f"{stage}.{seed}.{source_name}")
                    )
            residuals = payload.get("physics_residuals")
            if isinstance(residuals, Mapping) and "teacher_forced_fugacity" in residuals:
                fugacity_values.append(
                    _finite(
                        residuals["teacher_forced_fugacity"],
                        f"{stage}.{seed}.teacher_forced_fugacity",
                    )
                )
        fugacity_mean, fugacity_std = (
            _summary(fugacity_values) if fugacity_values else (None, None)
        )
        for direction in _DIRECTIONS:
            row: dict[str, object] = {
                "stage": stage,
                "direction": direction,
                "available_seeds": len(seed_ids),
                "seed_ids": ";".join(str(seed) for seed in seed_ids),
                "teacher_forced_fugacity_mean": fugacity_mean,
                "teacher_forced_fugacity_std": fugacity_std,
            }
            for field, values in direction_values[direction].items():
                mean, std = _summary(values)
                row[field + "_mean"] = mean
                row[field + "_std"] = std
            result.append(row)
    return result


def paired_stage_deltas(
    stage_metrics: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Quantify all adjacent-stage prediction and diagnostic changes."""

    ordered = _stages_in_order(stage_metrics)
    result: list[dict[str, object]] = []
    for previous, current in zip(ordered, ordered[1:]):
        previous_seeds = set(stage_metrics[previous])
        current_seeds = set(stage_metrics[current])
        if previous_seeds != current_seeds:
            raise RuntimeError(f"{previous} and {current} do not share identical seed sets")
        for direction in _DIRECTIONS:
            for metric, source_name in _FIELDS[direction].items():
                higher_is_better = metric in {"state_r2", "y_r2", "valid_coverage"}
                deltas: list[float] = []
                relative_percent: list[float] = []
                for seed in sorted(previous_seeds):
                    baseline = _finite(
                        stage_metrics[previous][seed][direction].get(source_name),
                        f"{previous}.{seed}.{source_name}",
                    )
                    updated = _finite(
                        stage_metrics[current][seed][direction].get(source_name),
                        f"{current}.{seed}.{source_name}",
                    )
                    delta = updated - baseline
                    deltas.append(delta)
                    if baseline != 0.0:
                        relative_percent.append(100.0 * delta / baseline)
                mean_delta, std_delta = _summary(deltas)
                relative_mean, relative_std = (
                    _summary(relative_percent) if relative_percent else (None, None)
                )
                result.append(
                    {
                        "from_stage": previous,
                        "to_stage": current,
                        "direction": direction,
                        "metric": metric,
                        "available_seeds": len(deltas),
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "mean_relative_percent": relative_mean,
                        "std_relative_percent": relative_std,
                        "improved_seed_count": sum((delta > 0.0) if higher_is_better else (delta < 0.0) for delta in deltas),
                    }
                )
    return result
