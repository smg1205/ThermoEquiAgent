"""Strict multi-seed aggregation for completed paper experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .artifacts import artifact_sha256, portable_artifact_path, resolve_artifact_path


IDENTITY_FIELDS = ("scope", "direction", "component_count", "subgroup")
PROVENANCE_FIELDS = (
    "protocol",
    "dataset_sha256",
    "git_commit",
    "resolved_config_sha256",
    "feature_cache_sha256",
    "feature_definition_sha256",
    "pure_property_catalog_sha256",
    "environment_sha256",
)
REQUIRED_ARTIFACTS = (
    "checkpoint",
    "history",
    "training_curves",
    "predictions",
    "metrics",
    "resolved_config",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _file_digest(path: Path) -> str:
    return artifact_sha256(path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_aggregate_state() -> tuple[str, bool, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    artifact_prefixes = ("results/", "figures/", "runs/", "checkpoints/", "cache/")
    code_paths: list[str] = []
    for line in status.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.startswith(artifact_prefixes):
            code_paths.append(path)
    return commit, bool(code_paths), code_paths


def _validate_manifest_artifacts(manifest: dict[str, Any], manifest_path: Path) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Completed manifest is missing artifact provenance: {manifest_path}")
    for name in REQUIRED_ARTIFACTS:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Completed manifest is missing artifact '{name}'")
        path_value = entry.get("path")
        expected_digest = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_digest, str):
            raise ValueError(f"Malformed artifact provenance for '{name}'")
        path = resolve_artifact_path(path_value)
        if not path.is_file() or _file_digest(path) != expected_digest:
            raise ValueError(f"Artifact '{name}' is missing or has changed: {path}")


def _provenance_value(manifest: dict[str, Any], field: str) -> Any:
    if field == "pure_property_catalog_sha256":
        return manifest.get(field) or "none"
    if field in {"feature_cache_sha256", "feature_definition_sha256"} and not manifest.get(field):
        # Historical Uni-Mol runs predate split-specific RDKit scaling; their
        # all-molecule feature subset digest is also their feature definition.
        return manifest.get("feature_subset_sha256")
    return manifest.get(field)


def _stage_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write an empty aggregate table: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _write_csv_pair(
    first_path: Path,
    first_rows: Sequence[dict[str, Any]],
    second_path: Path,
    second_rows: Sequence[dict[str, Any]],
) -> None:
    """Replace both aggregate tables transactionally, restoring old files on error."""
    targets = (first_path, second_path)
    staged_list: list[Path] = []
    try:
        staged_list.append(_stage_csv(first_path, first_rows))
        staged_list.append(_stage_csv(second_path, second_rows))
    except Exception:
        for path in staged_list:
            if path.exists():
                path.unlink()
        raise
    staged = tuple(staged_list)
    backups = tuple(
        path.with_name(f".{path.name}.{os.getpid()}.bak") for path in targets
    )
    existed = tuple(path.exists() for path in targets)
    try:
        for path, backup, had_original in zip(targets, backups, existed):
            if had_original:
                os.replace(path, backup)
        for temporary, path in zip(staged, targets):
            os.replace(temporary, path)
    except Exception:
        for path, backup, had_original in zip(targets, backups, existed):
            if had_original and backup.exists():
                if path.exists():
                    path.unlink()
                os.replace(backup, path)
            elif not had_original and path.exists():
                path.unlink()
        raise
    finally:
        for path in (*staged, *backups):
            if path.exists():
                path.unlink()


def aggregate_protocol_results(
    protocol_result_dir: Path,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    aggregate_kind: Literal["formal", "diagnostic"] = "formal",
    compatible_training_git_commits: Sequence[str] | None = None,
    mixed_commit_justification: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate completed seeds without inventing unavailable subgroup metrics.

    Formal headline scopes (global and component cardinality) must exist for every
    seed.  Finer subgroups may be absent when a held-out split contains no samples
    in that stratum; their summaries explicitly record the contributing seed IDs.
    """
    expected = tuple(expected_seeds)
    if not expected:
        raise ValueError("Expected seeds must be non-empty")
    if len(expected) != len(set(expected)):
        raise ValueError("Expected seeds must be unique")
    if aggregate_kind not in {"formal", "diagnostic"}:
        raise ValueError("aggregate_kind must be formal or diagnostic")
    if aggregate_kind == "formal" and set(expected) != {0, 1, 2, 3, 4}:
        raise ValueError("Formal aggregation requires exactly seeds {0,1,2,3,4}")
    missing: list[int] = []
    by_seed: dict[int, list[dict[str, Any]]] = {}
    manifests: dict[int, dict[str, Any]] = {}
    for seed in expected:
        seed_dir = protocol_result_dir / f"seed_{seed}"
        manifest_path = seed_dir / "manifest.json"
        metrics_path = seed_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            missing.append(seed)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or int(manifest.get("seed", -1)) != seed:
            missing.append(seed)
            continue
        _validate_manifest_artifacts(manifest, manifest_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, list) or not all(isinstance(row, dict) for row in metrics):
            raise ValueError(f"Malformed metrics file: {metrics_path}")
        if not metrics:
            raise ValueError(f"Metrics file is empty: {metrics_path}")
        manifests[seed] = manifest
        by_seed[seed] = metrics
    if missing:
        raise ValueError(f"Protocol is missing completed seeds: {missing}")

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in IDENTITY_FIELDS)

    common_provenance_fields = tuple(
        field for field in PROVENANCE_FIELDS if field != "git_commit"
    )
    reference_provenance = {
        field: _provenance_value(manifests[expected[0]], field)
        for field in common_provenance_fields
    }
    if any(value in (None, "") for value in reference_provenance.values()):
        raise ValueError("Completed manifest is missing required provenance")
    for seed, manifest in manifests.items():
        provenance = {
            field: _provenance_value(manifest, field)
            for field in common_provenance_fields
        }
        if provenance != reference_provenance:
            raise ValueError(f"Manifest provenance differs for seed {seed}")

    training_git_commits_by_seed = {
        seed: _provenance_value(manifest, "git_commit")
        for seed, manifest in manifests.items()
    }
    if any(value in (None, "") for value in training_git_commits_by_seed.values()):
        raise ValueError("Completed manifest is missing required git_commit provenance")
    training_git_commits = sorted(set(training_git_commits_by_seed.values()))
    if len(training_git_commits) > 1:
        if compatible_training_git_commits is None:
            raise ValueError("Manifest provenance differs for training git_commit")
        approved = tuple(dict.fromkeys(compatible_training_git_commits))
        if (
            not approved
            or any(not isinstance(value, str) or not value for value in approved)
            or set(approved) != set(training_git_commits)
        ):
            raise ValueError(
                "Observed training git commits do not exactly match the approved set"
            )
        if (
            not isinstance(mixed_commit_justification, str)
            or not mixed_commit_justification.strip()
        ):
            raise ValueError("Mixed training commits require a non-empty justification")
        training_git_commit = "mixed-approved"
    else:
        training_git_commit = training_git_commits[0]
        mixed_commit_justification = None
    reference_provenance["git_commit"] = training_git_commit
    current_commit, code_dirty, dirty_code_paths = _git_aggregate_state()
    if aggregate_kind == "formal":
        if code_dirty:
            raise RuntimeError(
                "Formal aggregation requires committed code; dirty paths: "
                + ", ".join(dirty_code_paths[:10])
            )

    for seed, rows in by_seed.items():
        identities = [key(row) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate metric identity for seed {seed}")
    flattened: list[dict[str, Any]] = []
    indexed: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for seed, rows in by_seed.items():
        indexed[seed] = {key(row): row for row in rows}
        for row in rows:
            flattened.append({"seed": seed, **row})
    metric_keys = set().union(*(set(seed_rows) for seed_rows in indexed.values()))
    scope_availability: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for identity in sorted(metric_keys, key=lambda value: tuple(str(item) for item in value)):
        available_seeds = [seed for seed in expected if identity in indexed[seed]]
        if (
            aggregate_kind == "formal"
            and identity[0] in {"all", "cardinality"}
            and len(available_seeds) != len(expected)
        ):
            raise ValueError(
                f"Headline metric scope {identity} must be available for all seeds; "
                f"found {available_seeds}"
            )
        rows = [indexed[seed][identity] for seed in available_seeds]
        fields = sorted(set().union(*(set(row) for row in rows)) - set(IDENTITY_FIELDS))
        result: dict[str, Any] = dict(zip(IDENTITY_FIELDS, identity))
        result["seeds"] = len(expected)
        result["scope_available_seeds"] = len(available_seeds)
        result["scope_seed_ids"] = ";".join(str(seed) for seed in available_seeds)
        scope_availability.append(
            {
                **dict(zip(IDENTITY_FIELDS, identity)),
                "available_seeds": available_seeds,
            }
        )
        for field in fields:
            values = [row.get(field) for row in rows]
            invalid_values = [
                value
                for value in values
                if value is not None
                and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                )
            ]
            if invalid_values:
                raise ValueError(
                    f"Non-numeric metric '{field}' in scope {identity}: "
                    f"{invalid_values[:3]}"
                )
            numeric_by_seed = [
                (seed, float(row.get(field)))
                for seed, row in zip(available_seeds, rows)
                if isinstance(row.get(field), (int, float))
                and not isinstance(row.get(field), bool)
            ]
            numeric = [value for _, value in numeric_by_seed]
            if any(not math.isfinite(value) for value in numeric):
                raise ValueError(f"Non-finite metric '{field}' in scope {identity}")
            result[f"{field}_available_seeds"] = len(numeric)
            result[f"{field}_seed_ids"] = ";".join(
                str(seed) for seed, _ in numeric_by_seed
            )
            if not numeric:
                result[f"{field}_mean"] = None
                result[f"{field}_std"] = None
            else:
                array = np.asarray(numeric, dtype=float)
                result[f"{field}_mean"] = float(array.mean())
                result[f"{field}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary.append(result)
    # Both tables are fully validated before either existing artifact is replaced.
    for row in flattened:
        for field, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite metric '{field}' in seed table")
    prefix = "" if aggregate_kind == "formal" else "diagnostic_"
    by_seed_path = protocol_result_dir / f"{prefix}metrics_by_seed.csv"
    summary_path = protocol_result_dir / f"{prefix}metrics_summary.csv"
    manifest_name = (
        "aggregate_manifest.json"
        if aggregate_kind == "formal"
        else "diagnostic_aggregate_manifest.json"
    )
    aggregate_manifest_path = protocol_result_dir / manifest_name
    # This atomic pointer is the validity boundary. Even a process kill while
    # replacing the two CSVs leaves no completed aggregate manifest.
    _atomic_json(
        aggregate_manifest_path,
        {
            "status": "aggregating",
            "aggregate_kind": aggregate_kind,
            "protocol": reference_provenance["protocol"],
            "seeds": list(expected),
            "training_git_commit": training_git_commit,
            "training_git_commits": training_git_commits,
            "training_git_commits_by_seed": {
                str(seed): commit
                for seed, commit in training_git_commits_by_seed.items()
            },
            "mixed_commit_justification": mixed_commit_justification,
            "aggregation_git_commit": current_commit,
        },
    )
    _write_csv_pair(
        by_seed_path,
        flattened,
        summary_path,
        summary,
    )
    aggregate_manifest = {
        "status": "completed" if aggregate_kind == "formal" else "diagnostic",
        "aggregate_kind": aggregate_kind,
        "protocol": reference_provenance["protocol"],
        "seeds": list(expected),
        "training_git_commit": training_git_commit,
        "training_git_commits": training_git_commits,
        "training_git_commits_by_seed": {
            str(seed): commit
            for seed, commit in training_git_commits_by_seed.items()
        },
        "mixed_commit_justification": mixed_commit_justification,
        "aggregation_git_commit": current_commit,
        # Backward-compatible alias for the code that produced this aggregate.
        "git_commit": current_commit,
        "git_dirty": code_dirty,
        "dirty_code_paths": dirty_code_paths,
        "input_provenance": reference_provenance,
        "scope_availability": scope_availability,
        "input_manifest_sha256": {
            str(seed): _file_digest(protocol_result_dir / f"seed_{seed}" / "manifest.json")
            for seed in expected
        },
        "outputs": {
            "metrics_by_seed": {
                "path": portable_artifact_path(by_seed_path),
                "sha256": _file_digest(by_seed_path),
            },
            "metrics_summary": {
                "path": portable_artifact_path(summary_path),
                "sha256": _file_digest(summary_path),
            },
        },
        "artifact_path_schema": "project-relative-v1",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _atomic_json(aggregate_manifest_path, aggregate_manifest)
    return summary
