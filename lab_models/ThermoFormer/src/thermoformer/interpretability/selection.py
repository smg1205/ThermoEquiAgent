"""Leakage-aware system selection for ThermoFormer interpretation."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from ..data import VLESample


def select_best_validation_seed(result_root: Path) -> int:
    """Select a completed seed solely by its stored validation objective."""
    candidates: list[tuple[float, int]] = []
    for manifest_path in sorted(result_root.glob("seed_*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        loss = float(payload["best_validation_loss"])
        seed = int(payload["seed"])
        if math.isfinite(loss):
            candidates.append((loss, seed))
    if not candidates:
        raise FileNotFoundError(f"No completed validation-selected seed below {result_root}")
    return min(candidates)[1]


def eligible_ternary_systems(
    all_samples: Sequence[VLESample],
    candidate_samples: Sequence[VLESample],
) -> tuple[tuple[str, ...], ...]:
    """Keep ternaries with experimental A-B, A-C, and B-C data present."""
    available_binary = {
        tuple(sorted(sample.smiles))
        for sample in all_samples
        if sample.component_count == 2
    }
    ternaries = {
        tuple(sorted(sample.smiles))
        for sample in candidate_samples
        if sample.component_count == 3
    }
    selected = []
    for system in sorted(ternaries):
        binary_subsystems = {
            tuple(sorted(pair)) for pair in itertools.combinations(system, 2)
        }
        if binary_subsystems <= available_binary:
            selected.append(system)
    return tuple(selected)


def grouped_systems(samples: Iterable[VLESample]) -> dict[tuple[str, ...], list[VLESample]]:
    grouped: dict[tuple[str, ...], list[VLESample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.system_key].append(sample)
    return dict(grouped)


def canonical_state(sample: VLESample) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    order = sorted(range(sample.component_count), key=lambda index: sample.smiles[index])
    return (
        tuple(sample.smiles[index] for index in order),
        tuple(sample.names[index] for index in order),
        tuple(sample.liquid_composition[index] for index in order),
        tuple(sample.vapor_composition[index] for index in order),
    )


def experimental_azeotrope_proxy(rows: Sequence[VLESample]) -> tuple[bool, float | None]:
    """Detect an internal y1-x1 sign change without using model predictions."""
    points = []
    for sample in rows:
        _, _, liquid, vapor = canonical_state(sample)
        if 0.02 < liquid[0] < 0.98:
            points.append((liquid[0], vapor[0] - liquid[0]))
    if len(points) < 3:
        return False, None
    points.sort()
    crossings = []
    for (left_x, left_delta), (right_x, right_delta) in zip(points, points[1:]):
        if left_delta == 0.0:
            crossings.append(left_x)
        elif left_delta * right_delta < 0.0:
            fraction = abs(left_delta) / (abs(left_delta) + abs(right_delta))
            crossings.append(left_x + fraction * (right_x - left_x))
    return (bool(crossings), crossings[0] if crossings else None)


def molecule_family(smiles: str) -> str:
    """Assign a compact chemistry label for stratified, non-mechanistic summaries."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "unparsed"
    if smiles == "O":
        return "water"
    atoms = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    if atoms <= {"C"}:
        if all(not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() == 1.0 for bond in molecule.GetBonds()):
            return "alkane/cycloalkane"
        return "aromatic/unsaturated hydrocarbon"
    if any(atom.GetSymbol() in {"F", "Cl", "Br", "I"} for atom in molecule.GetAtoms()):
        return "halogenated"
    if "S(=O)" in smiles or "S(=O)(=O)" in smiles:
        return "sulfoxide/sulfone"
    if "C(=O)O" in smiles or "C(=O)O" in Chem.MolToSmiles(molecule):
        return "carboxylic acid/ester"
    if any(atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0 for atom in molecule.GetAtoms()):
        return "alcohol/polyol"
    if any(atom.GetSymbol() == "N" for atom in molecule.GetAtoms()):
        return "nitrogen-containing"
    if any(atom.GetSymbol() == "O" for atom in molecule.GetAtoms()):
        return "ether/carbonyl"
    if any(atom.GetSymbol() == "S" for atom in molecule.GetAtoms()):
        return "sulfur-containing"
    return "other"


def system_family(smiles: Sequence[str]) -> str:
    return " + ".join(sorted(molecule_family(value) for value in smiles))
