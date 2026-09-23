"""RDKit Morgan-fingerprint, UMAP, and SMARTS-family analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from umap import UMAP

SEED = 42
FP_RADIUS = 2
FP_BITS = 2048

SMARTS = {
    "acid": "[CX3](=O)[OX2H1,OX1-]",
    "amide": "[CX3](=O)[NX3]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][#6]",
    "amine": "[NX3;!$(N[C,S,P]=O)]",
    "ether": "[OD2]([#6])[#6]",
    "sulfur-containing": "[S,s]",
    "halogenated": "[F,Cl,Br,I]",
    "aromatic": "[a]",
}
FAMILY_PRIORITY = [
    "water",
    "acid",
    "amide",
    "ester",
    "ketone",
    "alcohol",
    "amine",
    "ether",
    "sulfur-containing",
    "halogenated",
    "aromatic",
    "hydrocarbon",
    "other",
]


def _families(mol: Chem.Mol) -> tuple[str, str]:
    matches = []
    if mol.GetNumHeavyAtoms() == 1 and mol.GetAtomWithIdx(0).GetAtomicNum() == 8:
        matches.append("water")
    for family, smarts in SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            matches.append(family)
    atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if atomic_numbers and atomic_numbers <= {1, 6} and "aromatic" not in matches:
        matches.append("hydrocarbon")
    if not matches:
        matches.append("other")
    primary = next(family for family in FAMILY_PRIORITY if family in matches)
    return primary, ";".join(family for family in FAMILY_PRIORITY if family in matches)


def analyze_molecular_space(
    component_statistics: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write molecular coordinates and family counts; unresolved identities remain explicit."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    valid_indices = []
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
    )
    for index, component in component_statistics.reset_index(drop=True).iterrows():
        canonical = str(component.get("canonical_smiles", "") or "")
        mol = Chem.MolFromSmiles(canonical) if canonical else None
        row = component.to_dict()
        row.update(
            {
                "umap_1": np.nan,
                "umap_2": np.nan,
                "umap_status": "unresolved_smiles",
                "molecular_weight_g_mol": np.nan,
                "logp": np.nan,
                "tpsa_a2": np.nan,
                "primary_functional_family": "unresolved",
                "functional_families": "unresolved",
                "fingerprint_radius": FP_RADIUS,
                "fingerprint_bits": FP_BITS,
            }
        )
        if mol is not None:
            primary, families = _families(mol)
            row.update(
                {
                    "molecular_weight_g_mol": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "tpsa_a2": Descriptors.TPSA(mol),
                    "primary_functional_family": primary,
                    "functional_families": families,
                }
            )
            fingerprint = generator.GetFingerprint(mol)
            array = np.zeros(FP_BITS, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            fingerprints.append(array)
            valid_indices.append(index)
        rows.append(row)
    molecular = pd.DataFrame(rows)
    if len(fingerprints) >= 3:
        matrix = np.stack(fingerprints)
        reducer = UMAP(
            n_neighbors=min(15, len(matrix) - 1),
            min_dist=0.15,
            n_components=2,
            metric="jaccard",
            random_state=SEED,
            transform_seed=SEED,
            n_jobs=1,
        )
        coordinates = reducer.fit_transform(matrix)
        molecular.loc[valid_indices, "umap_1"] = coordinates[:, 0]
        molecular.loc[valid_indices, "umap_2"] = coordinates[:, 1]
        graph_degree = np.asarray(reducer.graph_.sum(axis=1)).ravel()
        connected = graph_degree > 0
        molecular.loc[valid_indices, "umap_status"] = np.where(
            connected,
            "projected",
            "disconnected",
        )
    # Publication-facing aliases retain the requested column names while the
    # snake-case columns remain available to existing analysis code.
    molecular["UMAP1"] = molecular["umap_1"]
    molecular["UMAP2"] = molecular["umap_2"]
    molecular.to_csv(results_dir / "molecular_space.csv", index=False)

    families = (
        molecular.groupby("primary_functional_family", dropna=False)
        .agg(
            molecule_count=("component_id", "nunique"),
            binary_only=("dataset_membership", lambda values: int((values == "Binary only").sum())),
            ternary_only=("dataset_membership", lambda values: int((values == "Ternary only").sum())),
            shared=("dataset_membership", lambda values: int((values == "Shared").sum())),
        )
        .reset_index()
        .sort_values("molecule_count", ascending=False)
    )
    families.to_csv(results_dir / "functional_family_statistics.csv", index=False)
    return molecular, families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    results_dir = args.analysis_root / 'experiments/reference_results'
    components = pd.read_csv(results_dir / "component_statistics.csv")
    molecular, _ = analyze_molecular_space(components, results_dir)
    print(
        "Wrote molecular_space.csv with "
        f"{(molecular['umap_status'] == 'projected').sum()} projected molecules and "
        f"{(molecular['umap_status'] == 'disconnected').sum()} disconnected molecules"
    )


if __name__ == "__main__":
    main()
