"""Frozen overall-predictive-performance task matrix for ML VLE baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from ...data.loading import VLESample, load_vle_dataset, retain_pure_anchored_systems
from ...data.splitting import DatasetPartitions, load_split_assignment, sample_id, system_id
from .schema import BASELINE_CAPABILITIES, Direction


UNIFIED_METRIC_KEYS = (
    "pressure_mae_kpa",
    "pressure_rmse_kpa",
    "pressure_r2",
    "temperature_mae_k",
    "temperature_rmse_k",
    "temperature_r2",
    "y_mae",
    "y_rmse",
    "y_r2",
    "valid_coverage",
    "nonphysical_rate",
    "solver_failure_rate",
)


@dataclass(frozen=True)
class OverallBenchmark:
    key: str
    split_protocol: Literal["vle_overall_binary", "vle_overall_ternary", "vle_overall_binary_ternary"]
    train_component_counts: tuple[int, ...]
    test_component_counts: tuple[int, ...]


OVERALL_BENCHMARKS = (
    OverallBenchmark("binary_train_binary_test", "vle_overall_binary", (2,), (2,)),
    OverallBenchmark("ternary_train_ternary_test", "vle_overall_ternary", (3,), (3,)),
    OverallBenchmark("joint_train_joint_test", "vle_overall_binary_ternary", (2, 3), (2, 3)),
)

HANNA_OFFICIAL_PRETRAINED_BENCHMARK = OverallBenchmark(
    "official_pretrained_to_joint_test",
    "vle_overall_binary_ternary",
    (2, 3),
    (2, 3),
)

EXTERNAL_FIXED_TO_JOINT_TEST = OverallBenchmark(
    "external_fixed_to_joint_test",
    "vle_overall_binary_ternary",
    (2, 3),
    (2, 3),
)

_BENCHMARK_BY_COMPONENT_COUNTS = {
    (2,): OVERALL_BENCHMARKS[0],
    (3,): OVERALL_BENCHMARKS[1],
    (2, 3): OVERALL_BENCHMARKS[2],
}


def benchmark_for_baseline(baseline: str) -> OverallBenchmark:
    """Return the sole registered benchmark allowed by native cardinality support."""
    if baseline == "hanna":
        return HANNA_OFFICIAL_PRETRAINED_BENCHMARK
    if baseline in {"tennet_sac", "spt_nrtl"}:
        return EXTERNAL_FIXED_TO_JOINT_TEST
    if baseline == "spt_nrtl_adapted":
        return _BENCHMARK_BY_COMPONENT_COUNTS[(2, 3)]
    try:
        counts = BASELINE_CAPABILITIES[baseline].native_component_counts
    except KeyError as error:
        raise ValueError(f"Unknown machine-learning baseline: {baseline}") from error
    try:
        return _BENCHMARK_BY_COMPONENT_COUNTS[counts]
    except KeyError as error:
        raise ValueError(
            f"No paper benchmark is registered for native component counts {counts}"
        ) from error


@dataclass(frozen=True)
class BenchmarkAssignment:
    benchmark: OverallBenchmark
    seed: int
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    train_system_ids: tuple[str, ...]
    validation_system_ids: tuple[str, ...]
    test_system_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["benchmark"] = asdict(self.benchmark)
        return payload


def load_registered_vle_dataset(project_root: Path) -> tuple[VLESample, ...]:
    """Load exactly the two registered VLE workbooks used by the benchmarks."""
    loaded = load_vle_dataset(
        project_root / "datasets" / "vle",
        source_filter="_vle_",
        failed_weight=0.0,
        max_pressure_kpa=500.0,
    )
    return tuple(retain_pure_anchored_systems(loaded.samples, minimum_temperatures=2))


def _filter_partition(
    split: DatasetPartitions,
    benchmark: OverallBenchmark,
) -> tuple[tuple[VLESample, ...], tuple[VLESample, ...], tuple[VLESample, ...]]:
    train = tuple(row for row in split.train if row.component_count in benchmark.train_component_counts)
    validation = tuple(
        row for row in split.validation if row.component_count in benchmark.test_component_counts
    )
    test = tuple(row for row in split.test if row.component_count in benchmark.test_component_counts)
    if not train or not validation or not test:
        raise ValueError(f"Registered benchmark '{benchmark.key}' has an empty partition")
    return train, validation, test


def registered_assignment(
    project_root: Path,
    benchmark: OverallBenchmark,
    seed: int,
    dataset: Sequence[VLESample] | None = None,
) -> BenchmarkAssignment:
    if seed not in (0, 1, 2, 3, 4):
        raise ValueError("Benchmark seeds are frozen to 0--4")
    samples = tuple(dataset) if dataset is not None else load_registered_vle_dataset(project_root)
    split_path = project_root / 'datasets/splits/vle' / benchmark.split_protocol / f"seed_{seed}.json"
    split = load_split_assignment(split_path, samples)
    if split.protocol != benchmark.split_protocol or split.seed != seed:
        raise ValueError("Registered split identity does not match the benchmark request")
    train, validation, test = _filter_partition(split, benchmark)
    partitions = [set(map(sample_id, rows)) for rows in (train, validation, test)]
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise RuntimeError("Baseline benchmark partitions overlap")
    return BenchmarkAssignment(
        benchmark=benchmark,
        seed=seed,
        train_sample_ids=tuple(map(sample_id, train)),
        validation_sample_ids=tuple(map(sample_id, validation)),
        test_sample_ids=tuple(map(sample_id, test)),
        train_system_ids=tuple(sorted({system_id(row) for row in train})),
        validation_system_ids=tuple(sorted({system_id(row) for row in validation})),
        test_system_ids=tuple(sorted({system_id(row) for row in test})),
    )


def native_evaluation_cells() -> list[dict[str, object]]:
    """Return the explicit model/task support matrix; unsupported cells stay N/A."""
    rows: list[dict[str, object]] = []
    for capability in BASELINE_CAPABILITIES.values():
        benchmark = benchmark_for_baseline(capability.key)
        for direction in ("isothermal", "isobaric"):
            unsupported_counts = [
                count
                for count in benchmark.test_component_counts
                if capability.support_reason(count, direction) is not None
            ]
            reason = (
                f"native model does not support {direction} prediction"
                if direction not in capability.native_directions
                else (
                    f"native model does not support component counts {unsupported_counts}"
                    if unsupported_counts else None
                )
            )
            rows.append(
                {
                    "baseline": capability.key,
                    "benchmark": benchmark.key,
                    "direction": direction,
                    "status": "supported" if reason is None else "not_applicable",
                    "reason": reason or "",
                }
            )
    return rows

