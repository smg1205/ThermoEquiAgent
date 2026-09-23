"""Locked variants and staged evaluation matrix for multi-view ThermoFormer."""

from __future__ import annotations

from typing import NamedTuple


class MultiViewVariant(NamedTuple):
    label: str
    config: str


MULTIVIEW_VARIANTS = {
    "v1_rdkit_only": MultiViewVariant(
        "V1 RDKit descriptors only",
        "configs/vle/ablation/studies/molecular_representation/rdkit_only/config.yaml",
    ),
    "v3_functional_group_only": MultiViewVariant(
        "V3 Functional groups only",
        "configs/vle/ablation/studies/molecular_representation/functional_groups_only/config.yaml",
    ),
    "v4_rdkit_unimol_naive": MultiViewVariant(
        "V4 RDKit + Uni-Mol naive fusion",
        "configs/vle/ablation/studies/molecular_representation/rdkit_unimol/config.yaml",
    ),
}

PREDICTIVE_VARIANTS = tuple(MULTIVIEW_VARIANTS)
PREDICTIVE_PROTOCOLS = ("overall_binary_ternary",)
MULTIVIEW_SEEDS = (0, 1, 2, 3, 4)
