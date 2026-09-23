"""Locked pilot and formal matrices for chemical-attention experiments."""

from __future__ import annotations

from typing import NamedTuple


class ChemicalAttentionVariant(NamedTuple):
    label: str
    config: str


CHEMICAL_ATTENTION_VARIANTS = {
    "c0_current_vanilla": ChemicalAttentionVariant(
        "C0 Current Uni-Mol vanilla Transformer",
        "configs/vle/ablation/studies/molecular_representation/unimol_v2_only/config.yaml",
    ),
    "c1_three_view_vanilla": ChemicalAttentionVariant(
        "C1 RDKit + Uni-Mol + FG, vanilla Transformer",
        "configs/vle/ablation/studies/interaction_architecture/vanilla_transformer/config.yaml",
    ),
    "c2_chemical_bias_full": ChemicalAttentionVariant(
        "C2 Three-view chemical-biased Transformer",
        "configs/vle/ablation/studies/interaction_architecture/chemical_interaction_bias/config.yaml",
    ),
    "c3_no_pair_bias": ChemicalAttentionVariant(
        "C3 Full model without attention pair bias",
        "configs/vle/ablation/studies/interaction_architecture/context_pair_without_attention_bias/config.yaml",
    ),
}

CHEMICAL_ATTENTION_PROTOCOLS = ("overall_binary_ternary",)
CHEMICAL_ATTENTION_FORMAL_PROTOCOLS = CHEMICAL_ATTENTION_PROTOCOLS
CHEMICAL_ATTENTION_SEEDS = (0, 1, 2, 3, 4)
