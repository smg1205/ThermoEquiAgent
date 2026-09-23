"""ThermoFormer molecular-interaction architecture."""

from .thermoformer import DirectVLEOutputs, ModelOutputs, ThermoFormer, ThermoFormerConfig


def final_model_config() -> ThermoFormerConfig:
    """Return the C1 three-view vanilla architecture used in the manuscript."""

    return ThermoFormerConfig(
        feature_dim=820,
        hidden_dim=192,
        layers=3,
        heads=6,
        feedforward_multiplier=4,
        dropout=0.0,
        pair_hidden_dim=192,
        pure_hidden_dim=192,
        film_scale=0.1,
        use_transformer=True,
        use_mixture_token=True,
        use_film=True,
        use_composition_context=True,
        interaction_mode="full",
        activity_mode="excess_gibbs",
        decoder_mode="thermodynamic",
        fusion_mode="naive",
        rdkit_feature_dim=24,
        unimol_feature_dim=768,
        functional_group_feature_dim=28,
        chemical_attention_bias=False,
        context_pair_interaction=False,
    )


__all__ = [
    "DirectVLEOutputs",
    "ModelOutputs",
    "ThermoFormer",
    "ThermoFormerConfig",
    "final_model_config",
]
