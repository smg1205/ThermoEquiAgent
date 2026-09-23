"""Interpretability utilities for interaction-specific multi-view gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import VLESample, VLETensorDataset, collate_vle


VIEW_NAMES = ("rdkit", "unimol", "functional_group")


def _composition_region(first: float, second: float) -> str:
    lower, upper = sorted((first, second))
    if lower < 0.1 or upper > 0.9:
        return "edge"
    if abs(first - second) < 0.2:
        return "balanced_pair"
    return "interior"


def collect_gate_records(
    model: torch.nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    functional_group_names: Sequence[str],
    functional_group_slice: slice,
    train_samples: Sequence[VLESample],
    batch_size: int,
    device: torch.device,
    protocol: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Collect one symmetric gate record per physical molecular pair."""

    dataset = VLETensorDataset(samples, feature_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_vle)
    train_components = {smile for sample in train_samples for smile in sample.smiles}
    train_systems = {sample.system_key for sample in train_samples}

    def family(smiles: str) -> str:
        counts = feature_map[smiles][functional_group_slice]
        if counts.size == 0 or float(np.max(counts)) <= 0.0:
            return "BASE/OTHER"
        return str(functional_group_names[int(np.argmax(counts))])

    records: list[dict[str, Any]] = []
    offset = 0
    model.to(device).eval()
    with torch.no_grad():
        for host_batch in loader:
            local_samples = samples[offset : offset + host_batch.x.shape[0]]
            offset += host_batch.x.shape[0]
            batch = host_batch.to(device)
            output = model(
                batch.molecules,
                batch.temperature_k,
                batch.pressure_kpa,
                batch.x,
                batch.mask,
                return_view_weights=True,
            )
            if output.view_weights is None:
                raise RuntimeError("Model did not return multi-view gate weights")
            weights = output.view_weights.detach().cpu()
            for row_index, sample in enumerate(local_samples):
                if any(smile not in train_components for smile in sample.smiles):
                    generalization = "unseen_component"
                elif sample.system_key not in train_systems:
                    generalization = "unseen_mixture"
                else:
                    generalization = "known_mixture"
                for first in range(sample.component_count):
                    for second in range(first + 1, sample.component_count):
                        pair_families = sorted(
                            (family(sample.smiles[first]), family(sample.smiles[second]))
                        )
                        record: dict[str, Any] = {
                            "protocol": protocol,
                            "seed": seed,
                            "component_count": sample.component_count,
                            "generalization": generalization,
                            "chemical_family_pair": " <-> ".join(pair_families),
                            "composition_region": _composition_region(
                                sample.liquid_composition[first],
                                sample.liquid_composition[second],
                            ),
                        }
                        for view_index, view_name in enumerate(VIEW_NAMES):
                            record[view_name] = float(
                                weights[row_index, first, second, view_index]
                            )
                        records.append(record)
    return records


def gate_statistics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate gate weights globally and by the requested scientific strata."""

    if not records:
        raise ValueError("Gate analysis requires at least one pair record")
    strata = (
        ("global", lambda row: "all"),
        ("cardinality", lambda row: "binary" if row["component_count"] == 2 else "ternary"),
        ("generalization", lambda row: str(row["generalization"])),
        ("chemical_family_pair", lambda row: str(row["chemical_family_pair"])),
        ("composition_region", lambda row: str(row["composition_region"])),
    )
    output: list[dict[str, Any]] = []
    for scope, key in strata:
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[(str(row["protocol"]), int(row["seed"]), key(row))].append(row)
        for (protocol, seed, subgroup), rows in sorted(grouped.items()):
            for view in VIEW_NAMES:
                values = np.asarray([float(row[view]) for row in rows], dtype=np.float64)
                output.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "scope": scope,
                        "subgroup": subgroup,
                        "view": view,
                        "mean_weight": float(values.mean()),
                        "std_weight": float(values.std()),
                        "pair_count": len(values),
                    }
                )
    return output
