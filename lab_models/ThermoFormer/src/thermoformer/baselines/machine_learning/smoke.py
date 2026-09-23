"""Leakage-safe, validation-only smoke runner for audited ML VLE baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ...data.loading import VLESample, load_vle_dataset, retain_pure_anchored_systems
from ...data.splitting import dataset_digest, load_split_assignment, sample_id, system_id
from ...reporting.artifacts import artifact_sha256
from .models import (
    DescriptorANN,
    ExternalModelAssets,
    GDIGNN,
    GEGNN,
    SMILESRNN,
    SolvGNN,
    UALFGNN,
    trainable_parameter_count,
)
from .schema import BASELINE_CAPABILITIES, BaselineCapability
from .protocols import OVERALL_BENCHMARKS, registered_assignment


SMOKE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SmokeConfig:
    seed: int = 0
    max_train_rows: int = 8
    max_validation_rows: int = 8
    epochs: int = 1
    learning_rate: float = 1e-3
    asset_root: str = "datasets/molecular_features/upstream_baselines"
    settings_sha256: str = ""


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return artifact_sha256(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _smiles_vector(samples: Sequence[VLESample]) -> Tensor:
    """Train-only character one-hot encoding plus the native P,x conditions."""
    characters = tuple(sorted({character for sample in samples for smiles in sample.smiles[:2] for character in smiles}))
    if not characters:
        raise ValueError("SMILES-RNN requires at least one character")
    character_index = {character: index for index, character in enumerate(characters)}
    maximum_length = max(len(smiles) for sample in samples for smiles in sample.smiles[:2])
    width_per_molecule = maximum_length * len(characters)
    values = torch.zeros(len(samples), 2 * width_per_molecule + 2)
    for row, sample in enumerate(samples):
        for component, smiles in enumerate(sample.smiles[:2]):
            for position, character in enumerate(smiles):
                offset = component * width_per_molecule + position * len(characters)
                values[row, offset + character_index[character]] = 1.0
        values[row, -2] = sample.pressure_kpa / 500.0
        values[row, -1] = sample.liquid_composition[0]
    return values


def _graph_tensors(
    samples: Sequence[VLESample],
    component_count: int = 2,
    atom_dim: int = 16,
    max_atoms: int = 32,
) -> tuple[Tensor, Tensor, Tensor]:
    from rdkit import Chem

    atom_types = (
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
        "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
        "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
        "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
    )

    def one_hot(value: object, choices: Sequence[object]) -> list[float]:
        return [float(value == choice) for choice in choices]

    def canonical_features(atom: Any) -> list[float]:
        from rdkit.Chem.rdchem import HybridizationType

        values = one_hot(atom.GetSymbol(), atom_types)
        values += one_hot(atom.GetDegree(), tuple(range(11)))
        values += one_hot(atom.GetValence(Chem.ValenceType.IMPLICIT), tuple(range(7)))
        values += [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        values += one_hot(
            atom.GetHybridization(),
            (
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ),
        )
        values += [float(atom.GetIsAromatic())]
        values += one_hot(atom.GetTotalNumHs(), tuple(range(5)))
        if len(values) != 74:
            raise RuntimeError("Canonical SolvGNN atom feature dimension changed")
        return values

    atoms = torch.zeros(len(samples), component_count, max_atoms, atom_dim)
    adjacency = torch.zeros(len(samples), component_count, max_atoms, max_atoms)
    mask = torch.zeros(len(samples), component_count, max_atoms)
    for row, sample in enumerate(samples):
        for component, smiles in enumerate(sample.smiles[:component_count]):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"RDKit could not parse SMILES: {smiles}")
            count = min(molecule.GetNumAtoms(), max_atoms)
            mask[row, component, :count] = 1.0
            for index, atom in enumerate(molecule.GetAtoms()):
                if index >= count:
                    break
                features = (
                    canonical_features(atom)
                    if atom_dim == 74
                    else [
                        min(atom.GetAtomicNum(), 100) / 100.0,
                        atom.GetDegree() / 4.0,
                        atom.GetFormalCharge() / 4.0,
                        atom.GetTotalNumHs() / 4.0,
                        float(atom.GetIsAromatic()),
                        float(atom.IsInRing()),
                    ]
                )
                atoms[row, component, index, : len(features)] = torch.tensor(features)
                adjacency[row, component, index, index] = 1.0
            for bond in molecule.GetBonds():
                first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if first < count and second < count:
                    adjacency[row, component, first, second] = 1.0
                    adjacency[row, component, second, first] = 1.0
    return atoms, adjacency, mask


def _hydrogen_bond_edges(samples: Sequence[VLESample], component_count: int) -> Tensor:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    edges = torch.zeros(len(samples), component_count, component_count)
    for row, sample in enumerate(samples):
        counts: list[tuple[int, int]] = []
        for smiles in sample.smiles[:component_count]:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"RDKit could not parse SMILES: {smiles}")
            counts.append(
                (rdMolDescriptors.CalcNumHBA(molecule), rdMolDescriptors.CalcNumHBD(molecule))
            )
        for first in range(component_count):
            for second in range(component_count):
                if first == second:
                    edges[row, first, second] = min(counts[first])
                else:
                    edges[row, first, second] = (
                        min(counts[first][0], counts[second][1])
                        + min(counts[first][1], counts[second][0])
                    )
    return edges


def _molecule_vectors(samples: Sequence[VLESample], components: int, width: int = 32) -> tuple[Tensor, Tensor, Tensor]:
    vectors = torch.zeros(len(samples), components, width)
    x = torch.zeros(len(samples), components)
    mask = torch.zeros_like(x)
    for row, sample in enumerate(samples):
        for component, smiles in enumerate(sample.smiles[:components]):
            digest = hashlib.sha256(smiles.encode()).digest()
            vectors[row, component] = torch.tensor([value / 255.0 for value in digest[:width]])
            x[row, component] = sample.liquid_composition[component]
            mask[row, component] = 1.0
    return vectors, x, mask


def _one_step(model: torch.nn.Module, loss: Tensor, learning_rate: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite = [gradient for parameter in model.parameters() if (gradient := parameter.grad) is not None]
    if not finite or not all(bool(torch.isfinite(value).all()) for value in finite):
        raise RuntimeError("Smoke backward pass produced missing or non-finite gradients")
    optimizer.step()


def _run_native_contract(
    project_root: Path,
    capability: BaselineCapability,
    samples: Sequence[VLESample],
    config: SmokeConfig,
) -> tuple[str, int, str]:
    binary = [sample for sample in samples if sample.component_count == 2]
    if not binary:
        return "not_applicable", 0, "no binary rows in the smoke training subset"
    rows = binary[: config.max_train_rows]
    if capability.key == "descriptor_ann":
        model = DescriptorANN()
        values = torch.randn(len(rows), 23)
        loss = model(values).square().mean()
        architecture_only_detail = (
            "architecture update passed; formal reproduction still requires the author-defined 21-descriptor table"
        )
    elif capability.key == "smiles_rnn":
        values = _smiles_vector(rows)
        model = SMILESRNN(values.shape[-1])
        target = torch.tensor([[row.temperature_k / 500.0, row.vapor_composition[0]] for row in rows])
        loss = (model(values) - target).square().mean()
    elif capability.key == "ualf_gnn":
        atoms, adjacency, atom_mask = _graph_tensors(rows)
        model = UALFGNN()
        conditions = torch.tensor([[row.pressure_kpa / 500.0, row.liquid_composition[0]] for row in rows])
        target = torch.tensor([[row.temperature_k / 500.0, row.vapor_composition[0]] for row in rows])
        mean, log_variance = model(atoms, adjacency, atom_mask, conditions)
        loss = model.heteroscedastic_loss(mean, log_variance, target)
    elif capability.key in {"solvgnn", "gdi_gnn"}:
        components = max(capability.native_component_counts)
        compatible = [row for row in samples if row.component_count == components][: config.max_train_rows]
        atoms, adjacency, atom_mask = _graph_tensors(compatible, components, atom_dim=74)
        _, x, mask = _molecule_vectors(compatible, components)
        hydrogen_bond_edges = _hydrogen_bond_edges(compatible, components)
        x = x.requires_grad_(capability.key == "gdi_gnn")
        model = GDIGNN() if capability.key == "gdi_gnn" else SolvGNN()
        log_gamma = model(atoms, adjacency, atom_mask, x, mask, hydrogen_bond_edges)
        loss = log_gamma.square().mean()
        if capability.key == "gdi_gnn":
            loss = loss + GDIGNN.gibbs_duhem_residual(log_gamma, x).square().mean()
    elif capability.key == "ge_gnn":
        atoms, adjacency, atom_mask = _graph_tensors(rows, 2, atom_dim=74)
        _, x, _ = _molecule_vectors(rows, 2)
        hydrogen_bond_edges = _hydrogen_bond_edges(rows, 2)
        model = GEGNN()
        loss = model.forward_graphs(
            atoms, adjacency, atom_mask, x, hydrogen_bond_edges
        ).square().mean()
    elif capability.key == "hanna":
        from .formal import FormalConfig, _hanna_result
        from .hanna import OfficialHANNAAssets

        assets = OfficialHANNAAssets.default(project_root)
        try:
            assets.validate()
        except FileNotFoundError:
            return "blocked_external_assets", 0, "official HANNA ensemble assets are required"
        selected: list[VLESample] = []
        for component_count in (2, 3):
            for direction in ("isothermal", "isobaric"):
                selected.extend([
                    sample for sample in samples
                    if sample.component_count == component_count
                    and sample.experiment_mode in (direction, "full_state")
                ][:1])
        result = _hanna_result(
            project_root,
            samples,
            tuple(selected),
            "official_pretrained_to_joint_test",
            FormalConfig(device="cuda" if torch.cuda.is_available() else "cpu"),
        )
        if not result.predictions:
            return "failed_no_covered_rows", 0, "official HANNA produced no covered smoke predictions"
        return (
            "passed_official_pretrained",
            0,
            f"official ten-model ensemble and shared VLE solver produced {len(result.predictions)} registered-row predictions",
        )
    else:
        asset_root = Path(config.asset_root) / capability.key
        assets = ExternalModelAssets(
            checkpoint=str(asset_root / "checkpoint.pt"),
            overlap_inventory=str(asset_root / "training_systems.csv"),
        )
        if not assets.complete:
            return "blocked_external_assets", 0, "official checkpoint and overlap inventory are required"
        return "passed_asset_preflight", 0, "official assets found; execution is delegated to the upstream adapter"
    _one_step(model, loss, config.learning_rate)
    if capability.key == "descriptor_ann":
        return "passed_architecture_only", trainable_parameter_count(model), architecture_only_detail
    if capability.temperature_mode == "fixed_298k":
        return (
            "passed_architecture_only",
            trainable_parameter_count(model),
            "architecture update passed; native evaluation remains restricted to 298.15 K rows",
        )
    return "passed", trainable_parameter_count(model), "finite forward/backward update on train-only rows"


def run_smoke_suite(
    project_root: Path,
    split_path: Path,
    output_dir: Path,
    config: SmokeConfig = SmokeConfig(),
) -> dict[str, Any]:
    """Run seed-0 architecture smoke without evaluating or selecting on test rows."""
    _set_seed(config.seed)
    loaded = load_vle_dataset(
        project_root / "datasets" / "vle",
        source_filter="_vle_",
        failed_weight=0.0,
        max_pressure_kpa=500.0,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=2,
    )
    split = load_split_assignment(split_path, samples)
    if split.seed != config.seed:
        raise ValueError("Smoke split seed does not match SmokeConfig.seed")
    # Preserve the registered partition identity and only truncate after each
    # adapter has had access to the full train partition for capability filtering.
    train = tuple(split.train)
    validation = tuple(split.validation[: config.max_validation_rows])
    if not train or not validation:
        raise ValueError("Smoke requires non-empty train and validation partitions")
    test_ids = {sample_id(sample) for sample in split.test}
    if test_ids & {sample_id(sample) for sample in (*train, *validation)}:
        raise RuntimeError("Smoke train/validation rows overlap the registered test partition")
    benchmark_assignments: dict[str, dict[str, object]] = {}
    for benchmark in OVERALL_BENCHMARKS:
        assignment = registered_assignment(
            project_root,
            benchmark,
            config.seed,
            dataset=samples,
        )
        benchmark_assignments[benchmark.key] = {
            "split_protocol": benchmark.split_protocol,
            "train_samples": len(assignment.train_sample_ids),
            "validation_samples": len(assignment.validation_sample_ids),
            "test_samples": len(assignment.test_sample_ids),
            "train_systems": len(assignment.train_system_ids),
            "validation_systems": len(assignment.validation_system_ids),
            "test_systems": len(assignment.test_system_ids),
            "train_sample_id_sha256": hashlib.sha256(
                "\n".join(assignment.train_sample_ids).encode()
            ).hexdigest(),
            "validation_sample_id_sha256": hashlib.sha256(
                "\n".join(assignment.validation_sample_ids).encode()
            ).hexdigest(),
            "test_sample_id_sha256": hashlib.sha256(
                "\n".join(assignment.test_sample_ids).encode()
            ).hexdigest(),
        }

    rows: list[dict[str, Any]] = []
    model_config_artifacts: dict[str, dict[str, str]] = {}
    for capability in BASELINE_CAPABILITIES.values():
        status, parameters, detail = _run_native_contract(project_root, capability, train, config)
        row = {
            "baseline": capability.key,
            "display_name": capability.display_name,
            "status": status,
            "implementation_source": capability.implementation_source,
            "native_component_counts": "/".join(map(str, capability.native_component_counts)),
            "native_directions": "/".join(capability.native_directions),
            "temperature_mode": capability.temperature_mode,
            "trainable_parameters": parameters,
            "reference_trainable_parameters": capability.reference_trainable_parameters or "",
            "reference_parameter_note": capability.reference_parameter_note,
            "detail": detail,
        }
        rows.append(row)
        model_config_path = output_dir / capability.key / "config.json"
        _atomic_json(model_config_path, {"capability": capability.to_dict(), "smoke": asdict(config)})
        model_config_artifacts[capability.key] = {
            "path": f"{capability.key}/config.json",
            "sha256": _sha256(model_config_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "smoke_summary.csv"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, summary_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except (ImportError, AttributeError):
        rdkit_version = "unavailable"
    manifest = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "diagnostic_smoke",
        "seed": config.seed,
        "selection_partition": "validation",
        "evaluation_partition": "validation",
        "test_ids_audited": True,
        "test_labels_consumed": False,
        "analysis_status": "diagnostic",
        "protocol": split.protocol,
        "benchmark_assignments": benchmark_assignments,
        "dataset_sha256": dataset_digest(samples),
        "split_sha256": _sha256(split_path),
        "train_sample_ids": [sample_id(row) for row in train[: config.max_train_rows]],
        "validation_sample_ids": [sample_id(row) for row in validation],
        "train_system_ids": sorted({system_id(row) for row in train[: config.max_train_rows]}),
        "validation_system_ids": sorted({system_id(row) for row in validation}),
        "config": asdict(config),
        "runtime": {
            "git_commit": git_commit,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit_version,
            "device": "cpu",
        },
        "model_config_artifacts": model_config_artifacts,
        "summary": {"path": "smoke_summary.csv", "sha256": _sha256(summary_path)},
        "models": rows,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest

