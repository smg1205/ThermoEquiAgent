"""Paper-facing dataset audit summaries for ThermoFormer."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem

from .loading import DatasetLoadResult, VLESample, pure_anchor_temperatures
from .splitting import canonical_smiles, component_id, dataset_digest, system_id


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "mean": float(array.mean()),
    }


def summarize_samples(samples: Sequence[VLESample]) -> dict[str, object]:
    systems: dict[tuple[str, ...], list[VLESample]] = defaultdict(list)
    component_systems: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    raw_to_canonical: dict[str, str] = {}
    invalid_smiles: list[str] = []
    for sample in samples:
        systems[sample.system_key].append(sample)
        for raw in sample.smiles:
            molecule = Chem.MolFromSmiles(raw)
            if molecule is None:
                invalid_smiles.append(raw)
            canonical = canonical_smiles(raw)
            raw_to_canonical[raw] = canonical
            component_systems[canonical].add(sample.system_key)
    canonical_to_raw: dict[str, set[str]] = defaultdict(set)
    for raw, canonical in raw_to_canonical.items():
        canonical_to_raw[canonical].add(raw)
    collisions = {
        canonical: sorted(raw_values)
        for canonical, raw_values in canonical_to_raw.items()
        if len(raw_values) > 1
    }
    anchors = pure_anchor_temperatures(samples)
    component_rows = Counter(sample.component_count for sample in samples)
    system_counts = Counter(len(key) for key in systems)
    return {
        "dataset_sha256": dataset_digest(samples),
        "rows": len(samples),
        "systems": len(systems),
        "components_raw": len(raw_to_canonical),
        "components_canonical": len(canonical_to_raw),
        "invalid_smiles": sorted(set(invalid_smiles)),
        "canonical_collisions": collisions,
        "rows_by_component_count": {str(key): value for key, value in sorted(component_rows.items())},
        "systems_by_component_count": {str(key): value for key, value in sorted(system_counts.items())},
        "rows_by_source": dict(sorted(Counter(Path(sample.source).name for sample in samples).items())),
        "rows_by_quality": dict(sorted(Counter(sample.quality_status for sample in samples).items())),
        "rows_by_mode": dict(sorted(Counter(sample.experiment_mode for sample in samples).items())),
        "mode_confidence": _distribution(sample.experiment_mode_confidence for sample in samples),
        "unique_nonempty_doi": len({sample.doi for sample in samples if sample.doi}),
        "rows_with_missing_doi": sum(not sample.doi for sample in samples),
        "temperature_k": _distribution(sample.temperature_k for sample in samples),
        "pressure_kpa": _distribution(sample.pressure_kpa for sample in samples),
        "liquid_mole_fraction": _distribution(
            value for sample in samples for value in sample.liquid_composition
        ),
        "vapor_mole_fraction": _distribution(
            value for sample in samples for value in sample.vapor_composition
        ),
        "points_per_system": _distribution(len(rows) for rows in systems.values()),
        "systems_per_component": _distribution(
            len(system_keys) for system_keys in component_systems.values()
        ),
        "pure_anchor_temperatures_per_component": _distribution(
            len(anchors.get(raw, set())) for raw in raw_to_canonical
        ),
        "components_with_at_least_two_pure_anchor_temperatures": sum(
            len(anchors.get(raw, set())) >= 2 for raw in raw_to_canonical
        ),
    }


def ternary_subsystem_rows(
    samples: Sequence[VLESample],
    binary_reference_samples: Sequence[VLESample] | None = None,
) -> list[dict[str, object]]:
    """Measure ternary pair coverage against an explicit binary reference set.

    Dataset-wide audits omit ``binary_reference_samples``. Experiment protocols
    must pass the actual training partition so validation/test-only binaries do
    not leak into the reported coverage class.
    """
    reference = samples if binary_reference_samples is None else binary_reference_samples
    binary_systems = {
        tuple(sorted(canonical_smiles(value) for value in sample.smiles))
        for sample in reference
        if sample.component_count == 2
    }
    ternary: dict[tuple[str, ...], list[VLESample]] = defaultdict(list)
    for sample in samples:
        if sample.component_count == 3:
            key = tuple(sorted(canonical_smiles(value) for value in sample.smiles))
            ternary[key].append(sample)
    rows: list[dict[str, object]] = []
    for components, states in sorted(ternary.items()):
        pairs = list(itertools.combinations(components, 2))
        covered = [pair in binary_systems for pair in pairs]
        rows.append(
            {
                "ternary_system_id": system_id(states[0]),
                "component_1": components[0],
                "component_2": components[1],
                "component_3": components[2],
                "component_id_1": component_id(components[0]),
                "component_id_2": component_id(components[1]),
                "component_id_3": component_id(components[2]),
                "points": len(states),
                "binary_subsystem_12": int(covered[0]),
                "binary_subsystem_13": int(covered[1]),
                "binary_subsystem_23": int(covered[2]),
                "covered_binary_subsystems": sum(covered),
                "coverage_class": f"{sum(covered)}/3",
            }
        )
    return rows


def write_ternary_subsystem_csv(path: Path, samples: Sequence[VLESample]) -> None:
    rows = ternary_subsystem_rows(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["ternary_system_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _range_text(summary: dict[str, float], digits: int = 4) -> str:
    return f"{summary['minimum']:.{digits}f}–{summary['maximum']:.{digits}f}"


def write_data_audit_report(
    path: Path,
    load_result: DatasetLoadResult,
    modeling_samples: Sequence[VLESample],
    per_workbook: dict[str, DatasetLoadResult],
) -> dict[str, object]:
    loaded = summarize_samples(load_result.samples)
    modeled = summarize_samples(modeling_samples)
    coverage = ternary_subsystem_rows(modeling_samples)
    coverage_counts = dict(sorted(Counter(row["coverage_class"] for row in coverage).items()))
    payload = {
        "loading": {
            "combined": {
                "raw_rows": load_result.audit.raw_rows,
                "accepted_before_deduplication": load_result.audit.accepted_before_deduplication,
                "duplicates_removed": load_result.audit.duplicates_removed,
                "loaded_samples": load_result.audit.loaded_samples,
                "rejected": load_result.audit.rejected,
            },
            "by_workbook": {
                name: {
                    "raw_rows": result.audit.raw_rows,
                    "loaded_samples": result.audit.loaded_samples,
                    "duplicates_removed": result.audit.duplicates_removed,
                    "rejected": result.audit.rejected,
                }
                for name, result in per_workbook.items()
            },
        },
        "loaded": loaded,
        "modeling": modeled,
        "ternary_binary_subsystem_coverage": coverage_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected = load_result.audit.rejected
    workbook_lines = "\n".join(
        f"| {name} | {result.audit.raw_rows:,} | {result.audit.loaded_samples:,} | "
        f"{result.audit.duplicates_removed:,} |"
        for name, result in per_workbook.items()
    )
    component_lines = "\n".join(
        f"| {count}-component | {modeled['rows_by_component_count'].get(str(count), 0):,} | "
        f"{modeled['systems_by_component_count'].get(str(count), 0):,} |"
        for count in (2, 3)
    )
    coverage_lines = "\n".join(
        f"| {label} | {count:,} |" for label, count in coverage_counts.items()
    )
    text = f"""# ThermoFormer data audit

This report is generated from the two English workbooks under `dataset/` using the same loader and default low-pressure/quality filters as training. The modeling set additionally applies the current requirement of at least two pure-endpoint temperatures per retained component. Counts below are observations, not inferred sample-size claims.

## Row-accounting ledger

| Workbook | Raw rows | Loaded after row filters/dedup | Duplicates removed |
|---|---:|---:|---:|
{workbook_lines}
| **Combined** | **{load_result.audit.raw_rows:,}** | **{load_result.audit.loaded_samples:,}** | **{load_result.audit.duplicates_removed:,}** |

Combined rejection ledger: failed quality `{rejected.get('failed_quality', 0):,}`, missing SMILES `{rejected.get('missing_smiles', 0):,}`, invalid state/composition `{rejected.get('invalid_temperature_or_pressure', 0) + rejected.get('invalid_composition', 0):,}`, pressure above 500 kPa `{rejected.get('pressure_above_limit', 0):,}`, invalid explicit mode `{rejected.get('invalid_experiment_mode', 0):,}`. The pure-anchor filter then removes `{load_result.audit.loaded_samples - len(modeling_samples):,}` rows, leaving **{len(modeling_samples):,}** modeling rows.

## Modeling-set coverage

| Cardinality | Rows | Unordered systems |
|---|---:|---:|
{component_lines}

- Canonical components: **{modeled['components_canonical']:,}**; invalid SMILES: **{len(modeled['invalid_smiles']):,}**; raw-to-canonical collisions: **{len(modeled['canonical_collisions']):,}**.
- Temperature range: **{_range_text(modeled['temperature_k'])} K**; pressure range: **{_range_text(modeled['pressure_kpa'])} kPa**.
- Liquid mole-fraction range: **{_range_text(modeled['liquid_mole_fraction'])}**; vapor mole-fraction range: **{_range_text(modeled['vapor_mole_fraction'])}**.
- Points/system: median **{modeled['points_per_system']['median']:.1f}**, IQR **{modeled['points_per_system']['q25']:.1f}–{modeled['points_per_system']['q75']:.1f}**, range **{modeled['points_per_system']['minimum']:.0f}–{modeled['points_per_system']['maximum']:.0f}**.
- Systems/component: median **{modeled['systems_per_component']['median']:.1f}**, IQR **{modeled['systems_per_component']['q25']:.1f}–{modeled['systems_per_component']['q75']:.1f}**, range **{modeled['systems_per_component']['minimum']:.0f}–{modeled['systems_per_component']['maximum']:.0f}**.
- Quality labels: `{json.dumps(modeled['rows_by_quality'], ensure_ascii=False)}`; modes: `{json.dumps(modeled['rows_by_mode'], ensure_ascii=False)}`.
- Non-empty unique DOI strings: **{modeled['unique_nonempty_doi']:,}**; rows with missing DOI: **{modeled['rows_with_missing_doi']:,}**.

## Ternary-to-binary subsystem coverage

Coverage is computed by canonical molecular identity. Each ternary system is assigned according to how many of its three binary subsystems occur anywhere in the retained modeling set.

| Binary subsystems present | Ternary systems |
|---|---:|
{coverage_lines}

The row-level mapping is saved as `experiments/data_quality/reports/ternary_binary_subsystem_coverage.csv` and is suitable for stratified binary-to-ternary evaluation.

## Leakage and identifiability audit

- Reversed component order is canonicalized for duplicate detection and stable IDs; A–B and B–A cannot cross a system-disjoint split.
- The legacy CLI generates grouped splits in memory but does not persist the exact row assignment. Paper experiments must use the versioned JSON artifacts under `splits/` and refuse dataset-digest mismatch.
- Pure endpoints are part of mixture workbooks rather than an independent pure-property source. Main random-system experiments protect reference systems on the training side. In unseen-component experiments, moving a test component's endpoint system into training would be leakage and is prohibited; learned `P_i^sat(T)` therefore becomes a genuine molecular extrapolation.
- No experimental uncertainty columns or repeat-level variance are available in the current workbooks. Training weights reflect quality flags, not calibrated measurement uncertainty.
- There are no quaternary rows. The supported and auditable scope is binary and ternary VLE only.

## Reproducibility record

- Loaded-set SHA-256: `{loaded['dataset_sha256']}`
- Modeling-set SHA-256: `{modeled['dataset_sha256']}`
- Stable component identities use RDKit canonical isomeric SMILES; split files store hashed state IDs plus the complete modeling-set digest.

Machine-readable audit details are saved in `experiments/data_quality/reports/data_audit.json`.
"""
    path.write_text(text, encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
