"""Single registry for every confirmatory ThermoFormer paper protocol."""

from __future__ import annotations


PAPER_SEEDS = (0, 1, 2, 3, 4)

PROTOCOL_CONFIGS = {
    "overall_binary": "configs/vle/prediction/studies/overall_binary/config.yaml",
    "overall_binary_ternary": "configs/vle/prediction/studies/overall_binary_ternary/config.yaml",
    "state_composition_interpolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "state_composition_edge_extrapolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "state_temperature_low_extrapolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "state_temperature_high_extrapolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "state_pressure_low_extrapolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "state_pressure_high_extrapolation": "configs/vle/prediction/studies/state_generalization/config.yaml",
    "unseen_component": "configs/vle/prediction/studies/unseen_components/config.yaml",
    "binary_to_ternary_zero_shot": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
    "binary_to_ternary_scale_0.05": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
    "binary_to_ternary_scale_0.1": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
    "binary_to_ternary_scale_0.25": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
    "binary_to_ternary_scale_0.5": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
    "binary_to_ternary_scale_1": "configs/vle/prediction/studies/binary_to_ternary/config.yaml",
}
