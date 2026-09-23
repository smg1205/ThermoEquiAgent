"""Permutation-equivariant molecular interaction and thermodynamic decoders."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..thermodynamics.activity_coefficients import activity_coefficients_from_excess_gibbs
from .interaction import (
    ChemicalBiasedTransformer,
    FunctionalGroupCrossInteraction,
    ViewProjection,
)
from .vapor_pressure import PureVaporPressure


@dataclass
class ModelOutputs:
    log_gamma: Tensor
    log_psat: Tensor
    nonideality_tokens: Tensor | None = None
    excess_gibbs_rt: Tensor | None = None
    pair_interactions: Tensor | None = None
    view_weights: Tensor | None = None
    view_interactions: Tensor | None = None
    attention_bias: Tensor | None = None
    attention_bias_penalty: Tensor | None = None


@dataclass
class DirectVLEOutputs:
    pressure_kpa: Tensor
    temperature_k: Tensor
    y: Tensor


@dataclass(frozen=True)
class ThermoFormerConfig:
    """Architecture hyperparameters and explicit ablation switches."""

    feature_dim: int | None = None
    hidden_dim: int = 192
    layers: int = 3
    heads: int = 6
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    pair_hidden_dim: int = 0
    pure_hidden_dim: int = 0
    film_scale: float = 0.1
    use_transformer: bool = True
    use_mixture_token: bool = True
    use_film: bool = True
    use_composition_context: bool = True
    interaction_mode: Literal["full", "pairwise", "independent"] = "full"
    activity_mode: Literal["excess_gibbs", "direct_gamma", "ideal"] = "excess_gibbs"
    decoder_mode: Literal["thermodynamic", "direct_vle"] = "thermodynamic"
    fusion_mode: Literal["legacy", "naive", "interaction_specific"] = "legacy"
    rdkit_feature_dim: int = 0
    unimol_feature_dim: int = 0
    functional_group_feature_dim: int = 0
    chemical_attention_bias: bool = False
    context_pair_interaction: bool = False
    chemical_bias_headwise: bool = False
    chemical_bias_shared_gate: bool = False
    chemical_bias_modality_gates: bool = False
    chemical_bias_gate_init: float = 0.03
    chemical_bias_modality_gate_init: float = 0.95
    chemical_bias_functional_group_gate_init: float = 0.1
    chemical_bias_hidden_dim: int = 0
    chemical_bias_dropout: float = 0.0
    chemical_bias_bound: float = 4.0
    chemical_bias_warmup_start: int = 0
    chemical_bias_warmup_end: int = 0
    chemical_bias_apply_to: Literal["all", "last"] = "all"

    def __post_init__(self) -> None:
        integer_values = (
            self.hidden_dim,
            self.layers,
            self.heads,
            self.feedforward_multiplier,
            self.pair_hidden_dim,
            self.pure_hidden_dim,
            self.rdkit_feature_dim,
            self.unimol_feature_dim,
            self.functional_group_feature_dim,
            self.chemical_bias_hidden_dim,
            self.chemical_bias_warmup_start,
            self.chemical_bias_warmup_end,
        )
        if self.feature_dim is not None:
            integer_values = (self.feature_dim, *integer_values)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            raise ValueError("ThermoFormer dimensions, layers, and heads must be integers")
        numeric_values = (
            self.dropout,
            self.film_scale,
            self.chemical_bias_gate_init,
            self.chemical_bias_modality_gate_init,
            self.chemical_bias_functional_group_gate_init,
            self.chemical_bias_dropout,
            self.chemical_bias_bound,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError("dropout and film_scale must be finite numbers")
        if any(
            not isinstance(value, bool)
            for value in (
                self.use_transformer,
                self.use_mixture_token,
                self.use_film,
                self.use_composition_context,
                self.chemical_attention_bias,
                self.context_pair_interaction,
                self.chemical_bias_headwise,
                self.chemical_bias_shared_gate,
                self.chemical_bias_modality_gates,
            )
        ):
            raise ValueError("ThermoFormer ablation switches must be booleans")
        if self.feature_dim is not None and self.feature_dim < 1:
            raise ValueError("feature_dim must be positive when specified")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.heads < 1 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by a positive heads value")
        if self.use_transformer and self.layers < 1:
            raise ValueError("layers must be positive when the Transformer is enabled")
        if self.layers < 0 or self.feedforward_multiplier < 1:
            raise ValueError("layers cannot be negative and feedforward_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pair_hidden_dim < 0 or self.pure_hidden_dim < 0 or self.film_scale < 0.0:
            raise ValueError("head dimensions and film_scale cannot be negative")
        if self.chemical_bias_hidden_dim < 0:
            raise ValueError("chemical_bias_hidden_dim cannot be negative")
        if not 0.0 <= self.chemical_bias_dropout < 1.0:
            raise ValueError("chemical_bias_dropout must be in [0, 1)")
        if not 0.0 < self.chemical_bias_gate_init < 1.0:
            raise ValueError("chemical_bias_gate_init must be between zero and one")
        if self.chemical_bias_functional_group_gate_init < 0.0:
            raise ValueError("chemical_bias_functional_group_gate_init cannot be negative")
        if not 0.0 < self.chemical_bias_modality_gate_init < 1.0:
            raise ValueError("chemical_bias_modality_gate_init must be between zero and one")
        if not 0.0 < self.chemical_bias_functional_group_gate_init < 1.0:
            raise ValueError(
                "chemical_bias_functional_group_gate_init must be between zero and one"
            )
        if self.chemical_bias_bound <= 0.0:
            raise ValueError("chemical_bias_bound must be positive")
        if self.chemical_bias_warmup_start < 0 or self.chemical_bias_warmup_end < 0:
            raise ValueError("chemical-bias warmup epochs cannot be negative")
        if self.chemical_bias_warmup_end and (
            self.chemical_bias_warmup_end <= self.chemical_bias_warmup_start
        ):
            raise ValueError("chemical_bias_warmup_end must exceed warmup_start")
        if self.chemical_bias_apply_to not in ("all", "last"):
            raise ValueError("chemical_bias_apply_to must be all or last")
        if self.interaction_mode not in ("full", "pairwise", "independent"):
            raise ValueError("interaction_mode must be full, pairwise, or independent")
        if self.activity_mode not in ("excess_gibbs", "direct_gamma", "ideal"):
            raise ValueError("activity_mode must be excess_gibbs, direct_gamma, or ideal")
        if self.decoder_mode not in ("thermodynamic", "direct_vle"):
            raise ValueError("decoder_mode must be thermodynamic or direct_vle")
        if self.fusion_mode not in ("legacy", "naive", "interaction_specific"):
            raise ValueError("fusion_mode must be legacy, naive, or interaction_specific")
        view_total = (
            self.rdkit_feature_dim
            + self.unimol_feature_dim
            + self.functional_group_feature_dim
        )
        if min(
            self.rdkit_feature_dim,
            self.unimol_feature_dim,
            self.functional_group_feature_dim,
        ) < 0:
            raise ValueError("molecular view dimensions cannot be negative")
        if self.fusion_mode == "legacy" and view_total:
            raise ValueError("legacy fusion does not accept molecular view dimensions")
        if self.fusion_mode != "legacy" and view_total != self.feature_dim:
            raise ValueError("view dimensions must sum to feature_dim for multiview fusion")
        if self.fusion_mode == "interaction_specific" and min(
            self.rdkit_feature_dim,
            self.unimol_feature_dim,
            self.functional_group_feature_dim,
        ) < 1:
            raise ValueError("interaction-specific fusion requires all three molecular views")
        if self.chemical_attention_bias and (
            self.fusion_mode == "legacy"
            or self.rdkit_feature_dim < 1
            or self.unimol_feature_dim < 1
        ):
            raise ValueError(
                "Chemical attention bias requires multiview RDKit and Uni-Mol features"
            )
        if self.chemical_attention_bias and not self.use_transformer:
            raise ValueError("Chemical attention bias requires the Transformer")
        if self.chemical_bias_shared_gate and self.chemical_bias_headwise:
            raise ValueError("Use either shared or headwise chemical gates, not both")
        if self.chemical_bias_modality_gates and not self.chemical_bias_headwise:
            raise ValueError("Chemical modality gates require headwise chemical bias")
        if self.context_pair_interaction and self.interaction_mode != "full":
            raise ValueError("Context-conditioned pair interaction requires full interaction mode")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ThermoFormer(nn.Module):
    """Binary/ternary interaction model with distinct gamma and Psat branches.

    Activity coefficients are derived from a learned dimensionless excess-Gibbs
    potential. This makes the component outputs thermodynamically coupled and
    anchors the present pure component at ``gamma=1`` by construction.
    """

    def __init__(self, config: ThermoFormerConfig) -> None:
        super().__init__()
        self.config = config
        if config.feature_dim is None:
            raise ValueError("ThermoFormerConfig.feature_dim must be resolved before model construction")
        feature_dim = config.feature_dim
        hidden_dim = config.hidden_dim
        self.view_order = ("rdkit", "unimol", "functional_group")
        self.view_slices: dict[str, slice] = {}
        self.view_projectors: nn.ModuleDict | None = None
        self.base_fusion: nn.Module | None = None
        if config.fusion_mode == "legacy":
            self.molecular_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.molecular_encoder = None
            dimensions = {
                "rdkit": config.rdkit_feature_dim,
                "unimol": config.unimol_feature_dim,
                "functional_group": config.functional_group_feature_dim,
            }
            offset = 0
            projectors: dict[str, nn.Module] = {}
            for name in self.view_order:
                dimension = dimensions[name]
                if dimension:
                    self.view_slices[name] = slice(offset, offset + dimension)
                    projectors[name] = ViewProjection(dimension, hidden_dim)
                    offset += dimension
            self.view_projectors = nn.ModuleDict(projectors)
            self.base_fusion = nn.Sequential(
                nn.Linear(len(projectors) * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        full_interaction = config.interaction_mode == "full"
        if full_interaction and config.use_mixture_token:
            self.mixture_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.register_parameter("mixture_token", None)
        self.chemical_interaction: ChemicalBiasedTransformer | None = None
        self.chemical_bias_mlp: nn.Module | None = None
        self.chemical_bias_mlps: nn.ModuleDict | None = None
        self.chemical_head_gate_logits: nn.Parameter | None = None
        self.chemical_modality_gate_logits: nn.ParameterDict | None = None
        if full_interaction and config.use_transformer and config.chemical_attention_bias:
            chemical_feature_dim = 3 * hidden_dim + 4
            if config.functional_group_feature_dim:
                chemical_feature_dim += hidden_dim
            bias_hidden_dim = (
                config.chemical_bias_hidden_dim
                or config.pair_hidden_dim
                or hidden_dim
            )
            bias_outputs = config.layers * config.heads if config.chemical_bias_headwise else 1

            def bias_mlp(input_dim: int) -> nn.Sequential:
                modules: list[nn.Module] = [
                    nn.Linear(input_dim, bias_hidden_dim),
                    nn.GELU(),
                ]
                # Keep the legacy zero-dropout state-dict indices unchanged so
                # published C2 checkpoints remain loadable.
                if config.chemical_bias_dropout > 0.0:
                    modules.append(nn.Dropout(config.chemical_bias_dropout))
                modules.append(nn.Linear(bias_hidden_dim, bias_outputs))
                return nn.Sequential(*modules)

            if config.chemical_bias_modality_gates:
                self.chemical_bias_mlps = nn.ModuleDict(
                    {
                        "rdkit": bias_mlp(2 * hidden_dim),
                        "unimol": bias_mlp(hidden_dim),
                        "functional_group": bias_mlp(hidden_dim),
                        "state": bias_mlp(4),
                    }
                )
                def logit(value: float) -> Tensor:
                    return torch.tensor(math.log(value / (1.0 - value)))

                self.chemical_modality_gate_logits = nn.ParameterDict(
                    {
                        "rdkit": nn.Parameter(logit(config.chemical_bias_modality_gate_init)),
                        "unimol": nn.Parameter(logit(config.chemical_bias_modality_gate_init)),
                        "functional_group": nn.Parameter(
                            logit(
                                config.chemical_bias_functional_group_gate_init
                            )
                        ),
                        "state": nn.Parameter(logit(config.chemical_bias_modality_gate_init)),
                    }
                )
            else:
                self.chemical_bias_mlp = bias_mlp(chemical_feature_dim)
            if config.chemical_bias_headwise or config.chemical_bias_shared_gate:
                initial_logit = math.log(
                    config.chemical_bias_gate_init
                    / (1.0 - config.chemical_bias_gate_init)
                )
                self.chemical_head_gate_logits = nn.Parameter(
                    torch.full(
                        (config.layers, config.heads)
                        if config.chemical_bias_headwise
                        else (1, 1),
                        initial_logit,
                    )
                )
                self.register_buffer(
                    "_chemical_bias_curriculum",
                    torch.tensor(self._curriculum_value(0), dtype=torch.float32),
                )
            self.chemical_interaction = ChemicalBiasedTransformer(
                hidden_dim=hidden_dim,
                heads=config.heads,
                feedforward_dim=config.feedforward_multiplier * hidden_dim,
                dropout=config.dropout,
                layers=config.layers,
            )
            self.interaction: nn.Module | None = None
        elif full_interaction and config.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.feedforward_multiplier * hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interaction: nn.Module | None = nn.TransformerEncoder(
                layer,
                num_layers=config.layers,
                enable_nested_tensor=False,
            )
        else:
            self.interaction = None
        self.film: nn.Module | None = None
        self.condition_projection: nn.Module | None = None
        if config.use_film:
            self.film = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            )
        else:
            self.condition_projection = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        pair_hidden_dim = config.pair_hidden_dim or hidden_dim
        self.pair_potential: nn.Module | None = None
        self.context_pair_potential: nn.Module | None = None
        self.component_potential: nn.Module | None = None
        self.direct_activity_head: nn.Module | None = None
        self.vapor_pressure: PureVaporPressure | None = None
        self.direct_state_head: nn.Module | None = None
        self.direct_y_head: nn.Module | None = None
        self.view_potentials: nn.ModuleDict | None = None
        self.view_gate: nn.Module | None = None
        self.functional_group_cross: FunctionalGroupCrossInteraction | None = None
        if config.decoder_mode == "thermodynamic":
            if config.interaction_mode == "independent":
                self.component_potential = nn.Sequential(
                    nn.Linear(hidden_dim, pair_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pair_hidden_dim, 1),
                )
            elif config.activity_mode != "direct_gamma" and config.context_pair_interaction:
                self.context_pair_potential = nn.Sequential(
                    nn.Linear(4 * hidden_dim + 4, pair_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pair_hidden_dim, 1),
                )
            elif (
                config.activity_mode != "direct_gamma"
                and config.fusion_mode != "interaction_specific"
            ):
                self.pair_potential = nn.Sequential(
                    nn.Linear(3 * hidden_dim, pair_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pair_hidden_dim, 1),
                )
            elif (
                config.activity_mode != "direct_gamma"
                and config.fusion_mode == "interaction_specific"
            ):
                self.view_potentials = nn.ModuleDict(
                    {
                        name: nn.Sequential(
                            nn.Linear(4 * hidden_dim, pair_hidden_dim),
                            nn.GELU(),
                            nn.Linear(pair_hidden_dim, 1),
                        )
                        for name in self.view_order
                    }
                )
                self.functional_group_cross = FunctionalGroupCrossInteraction(
                    config.functional_group_feature_dim,
                    hidden_dim,
                    config.heads,
                )
                self.view_gate = nn.Sequential(
                    nn.Linear(3 * hidden_dim + 4, pair_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pair_hidden_dim, 3),
                )
            if config.activity_mode == "direct_gamma":
                self.direct_activity_head = nn.Sequential(
                    nn.Linear(2 * hidden_dim, pair_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pair_hidden_dim, 1),
                )
            self.vapor_pressure = PureVaporPressure(
                hidden_dim,
                config.pure_hidden_dim or hidden_dim,
            )
        else:
            self.direct_state_head = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.direct_y_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def _curriculum_value(self, epoch: int) -> float:
        start = self.config.chemical_bias_warmup_start
        end = self.config.chemical_bias_warmup_end
        if end == 0:
            return 1.0
        if epoch < start:
            return 0.0
        if epoch >= end:
            return 1.0
        return (epoch - start) / float(end - start)

    def set_training_epoch(self, epoch: int) -> None:
        """Set the supervised epoch used by the chemical-bias curriculum."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("training epoch must be a non-negative integer")
        curriculum = getattr(self, "_chemical_bias_curriculum", None)
        if curriculum is not None:
            curriculum.fill_(self._curriculum_value(epoch))

    def chemical_head_gates(self) -> Tensor:
        if self.chemical_head_gate_logits is None:
            raise RuntimeError("Headwise chemical gates are unavailable")
        return torch.sigmoid(self.chemical_head_gate_logits)

    def chemical_modality_gates(self) -> dict[str, Tensor]:
        if self.chemical_modality_gate_logits is None:
            raise RuntimeError("Chemical modality gates are unavailable")
        return {
            name: torch.sigmoid(value)
            for name, value in self.chemical_modality_gate_logits.items()
        }

    def _encode_molecules(
        self, molecules: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], Tensor | None]:
        if self.config.fusion_mode == "legacy":
            if self.molecular_encoder is None:
                raise RuntimeError("Legacy molecular encoder is unavailable")
            return self.molecular_encoder(molecules), {}, None
        if self.view_projectors is None or self.base_fusion is None:
            raise RuntimeError("Multiview molecular encoder is unavailable")
        views = {
            name: self.view_projectors[name](molecules[..., self.view_slices[name]])
            for name in self.view_order
            if name in self.view_slices
        }
        base = self.base_fusion(torch.cat([views[name] for name in self.view_order if name in views], dim=-1))
        functional_counts = (
            molecules[..., self.view_slices["functional_group"]]
            if "functional_group" in self.view_slices
            else None
        )
        return base, views, functional_counts

    def pure_log_psat(self, molecules: Tensor, temperature_k: Tensor) -> Tensor:
        if self.vapor_pressure is None:
            raise RuntimeError("Direct-VLE models do not contain a vapor-pressure decoder")
        molecular_tokens, _, _ = self._encode_molecules(molecules)
        return self.vapor_pressure(molecular_tokens, temperature_k)

    def _chemical_pair_bias(
        self,
        views: dict[str, Tensor],
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.chemical_bias_mlp is None and self.chemical_bias_mlps is None:
            raise RuntimeError("Chemical interaction-bias MLP is unavailable")
        rdkit = views["rdkit"]
        unimol = views["unimol"]
        functional_group = views.get("functional_group")
        batch, component_count, _ = rdkit.shape
        if self.config.chemical_bias_headwise:
            bias = torch.zeros(
                batch,
                self.config.layers,
                self.config.heads,
                component_count,
                component_count,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            bias = torch.zeros(
                batch, component_count, component_count, dtype=x.dtype, device=x.device
            )
        temperature = (temperature_k - 350.0) / 150.0
        pressure = torch.log(pressure_kpa.clamp_min(1e-6) / 101.325)
        for first in range(component_count):
            for second in range(first + 1, component_count):
                rdkit_features = torch.cat(
                    [
                        rdkit[:, first] + rdkit[:, second],
                        torch.abs(rdkit[:, first] - rdkit[:, second]),
                    ],
                    dim=-1,
                )
                unimol_features = unimol[:, first] * unimol[:, second]
                group_features = (
                    functional_group[:, first] * functional_group[:, second]
                    if functional_group is not None
                    else None
                )
                state_features = torch.cat(
                    [
                        temperature,
                        pressure,
                        (x[:, first] + x[:, second]).unsqueeze(-1),
                        torch.abs(x[:, first] - x[:, second]).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                features = [rdkit_features, unimol_features]
                if functional_group is not None:
                    assert group_features is not None
                    features.append(group_features)
                features.append(state_features)
                pair_mask = mask[:, first] * mask[:, second]
                if self.chemical_bias_mlps is not None:
                    if group_features is None:
                        raise RuntimeError("Functional-group bias branch is unavailable")
                    gates = self.chemical_modality_gates()
                    raw = (
                        gates["rdkit"] * self.chemical_bias_mlps["rdkit"](rdkit_features)
                        + gates["unimol"] * self.chemical_bias_mlps["unimol"](unimol_features)
                        + gates["functional_group"]
                        * self.chemical_bias_mlps["functional_group"](group_features)
                        + gates["state"] * self.chemical_bias_mlps["state"](state_features)
                    )
                else:
                    assert self.chemical_bias_mlp is not None
                    raw = self.chemical_bias_mlp(torch.cat(features, dim=-1))
                if self.config.chemical_bias_headwise:
                    value = self.config.chemical_bias_bound * torch.tanh(raw)
                    value = value.reshape(batch, self.config.layers, self.config.heads)
                    value = value * self.chemical_head_gates().unsqueeze(0)
                    value = value * self._chemical_bias_curriculum.to(value)
                    if self.config.chemical_bias_apply_to == "last":
                        layer_mask = torch.zeros(
                            self.config.layers, 1, dtype=value.dtype, device=value.device
                        )
                        layer_mask[-1] = 1.0
                        value = value * layer_mask.unsqueeze(0)
                    value = value * pair_mask[:, None, None]
                    bias[:, :, :, first, second] = value
                    bias[:, :, :, second, first] = value
                else:
                    value = (
                        self.config.chemical_bias_bound
                        * torch.tanh(raw.squeeze(-1))
                        * pair_mask
                    )
                    if self.config.chemical_bias_shared_gate:
                        value = value * self.chemical_head_gates().reshape(())
                        value = value * self._chemical_bias_curriculum.to(value)
                    bias[:, first, second] = value
                    bias[:, second, first] = value
        return bias

    def _structural_context(
        self,
        molecular_tokens: Tensor,
        molecular_views: dict[str, Tensor],
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        batch = molecular_tokens.shape[0]
        masked_mean = (molecular_tokens * mask.unsqueeze(-1)).sum(1, keepdim=True)
        masked_mean = masked_mean / mask.sum(-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        learned_mixture = (
            self.mixture_token.expand(batch, -1, -1)
            if self.mixture_token is not None
            else masked_mean
        )
        if self.chemical_interaction is not None:
            component_bias = self._chemical_pair_bias(
                molecular_views, temperature_k, pressure_kpa, x, mask
            )
            sequence = torch.cat([learned_mixture, molecular_tokens], dim=1)
            padding = torch.cat(
                [torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask.bool()],
                dim=1,
            )
            if component_bias.ndim == 5:
                sequence_bias = torch.zeros(
                    batch,
                    self.config.layers,
                    self.config.heads,
                    molecular_tokens.shape[1] + 1,
                    molecular_tokens.shape[1] + 1,
                    dtype=molecular_tokens.dtype,
                    device=molecular_tokens.device,
                )
                sequence_bias[:, :, :, 1:, 1:] = component_bias
                diagnostic_bias = component_bias.mean(dim=(1, 2))
                bias_penalty = component_bias.square().mean()
            else:
                sequence_bias = torch.zeros(
                    batch,
                    molecular_tokens.shape[1] + 1,
                    molecular_tokens.shape[1] + 1,
                    dtype=molecular_tokens.dtype,
                    device=molecular_tokens.device,
                )
                sequence_bias[:, 1:, 1:] = component_bias
                diagnostic_bias = component_bias
                bias_penalty = component_bias.square().mean()
            interacted = self.chemical_interaction(sequence, sequence_bias, padding)
            return (
                interacted[:, 1:] * mask.unsqueeze(-1),
                interacted[:, :1],
                diagnostic_bias,
                bias_penalty,
            )
        if self.interaction is None:
            return molecular_tokens * mask.unsqueeze(-1), learned_mixture, None, None
        if self.mixture_token is None:
            interacted = self.interaction(molecular_tokens, src_key_padding_mask=~mask.bool())
            mixture = (interacted * mask.unsqueeze(-1)).sum(1, keepdim=True)
            mixture = mixture / mask.sum(-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            return interacted * mask.unsqueeze(-1), mixture, None, None
        sequence = torch.cat([learned_mixture, molecular_tokens], dim=1)
        padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask.bool()],
            dim=1,
        )
        interacted = self.interaction(sequence, src_key_padding_mask=padding)
        return interacted[:, 1:] * mask.unsqueeze(-1), interacted[:, :1], None, None

    def _nonideality_tokens(
        self,
        components: Tensor,
        mixture: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized_x = x * mask
        normalized_x = normalized_x / normalized_x.sum(-1, keepdim=True).clamp_min(1e-8)
        composition_context = (normalized_x.unsqueeze(-1) * components).sum(1, keepdim=True)
        if self.config.interaction_mode != "full":
            context = components
        elif self.config.use_composition_context:
            composition_context = 0.5 * (composition_context + mixture)
            context = composition_context.expand_as(components)
        else:
            composition_context = mixture
            context = composition_context.expand_as(components)
        temperature = ((temperature_k - 350.0) / 150.0).expand_as(x).unsqueeze(-1)
        pressure = torch.log(pressure_kpa.clamp_min(1e-6) / 101.325).expand_as(x).unsqueeze(-1)
        condition = torch.cat([context, temperature, pressure, x.unsqueeze(-1)], dim=-1)
        if self.config.use_film:
            if self.film is None:
                raise RuntimeError("FiLM condition projection is unavailable")
            scale, shift = self.film(condition).chunk(2, dim=-1)
            tokens = components * (
                1.0 + self.config.film_scale * torch.tanh(scale)
            ) + shift
        else:
            if self.condition_projection is None:
                raise RuntimeError("Condition concatenation projection is unavailable")
            tokens = self.condition_projection(condition)
        return tokens * mask.unsqueeze(-1), composition_context

    def _excess_gibbs(
        self,
        tokens: Tensor,
        context: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if self.config.interaction_mode == "independent":
            if self.component_potential is None:
                raise RuntimeError("Independent component potential is unavailable")
            unary = self.component_potential(tokens).squeeze(-1) * mask
            total = (x * (1.0 - x) * unary).sum(-1, keepdim=True)
            return total, None
        if self.pair_potential is None:
            raise RuntimeError("Pair interaction potential is unavailable")
        total = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        component_count = x.shape[1]
        pair_interactions = torch.zeros(
            x.shape[0], component_count, component_count, dtype=x.dtype, device=x.device
        )
        for first in range(component_count):
            for second in range(first + 1, component_count):
                pair_mask = (mask[:, first] * mask[:, second]).unsqueeze(-1)
                pair_context = (
                    context.squeeze(1)
                    if self.config.interaction_mode == "full"
                    else 0.5 * (tokens[:, first] + tokens[:, second])
                )
                pair = torch.cat(
                    [
                        tokens[:, first] + tokens[:, second],
                        torch.abs(tokens[:, first] - tokens[:, second]),
                        pair_context,
                    ],
                    dim=-1,
                )
                interaction = self.pair_potential(pair) * pair_mask
                pair_interactions[:, first, second] = interaction.squeeze(-1)
                pair_interactions[:, second, first] = interaction.squeeze(-1)
                total = total + (x[:, first] * x[:, second]).unsqueeze(-1) * interaction
        return total, pair_interactions

    def _contextual_excess_gibbs(
        self,
        tokens: Tensor,
        mixture: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Symmetric pair potential conditioned on the full mixture environment."""
        if self.context_pair_potential is None:
            raise RuntimeError("Context-conditioned pair potential is unavailable")
        total = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        component_count = x.shape[1]
        pair_interactions = torch.zeros(
            x.shape[0], component_count, component_count, dtype=x.dtype, device=x.device
        )
        mixture_context = mixture.squeeze(1)
        temperature = (temperature_k - 350.0) / 150.0
        pressure = torch.log(pressure_kpa.clamp_min(1e-6) / 101.325)
        for first in range(component_count):
            for second in range(first + 1, component_count):
                pair_mask = (mask[:, first] * mask[:, second]).unsqueeze(-1)
                pair_features = torch.cat(
                    [
                        tokens[:, first] + tokens[:, second],
                        torch.abs(tokens[:, first] - tokens[:, second]),
                        tokens[:, first] * tokens[:, second],
                        mixture_context,
                        temperature,
                        pressure,
                        (x[:, first] + x[:, second]).unsqueeze(-1),
                        torch.abs(x[:, first] - x[:, second]).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                interaction = self.context_pair_potential(pair_features) * pair_mask
                pair_interactions[:, first, second] = interaction.squeeze(-1)
                pair_interactions[:, second, first] = interaction.squeeze(-1)
                total = total + (x[:, first] * x[:, second]).unsqueeze(-1) * interaction
        return total, pair_interactions

    def _multiview_excess_gibbs(
        self,
        state_tokens: Tensor,
        context: Tensor,
        views: dict[str, Tensor],
        functional_counts: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
        return_view_weights: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if (
            self.view_potentials is None
            or self.view_gate is None
            or self.functional_group_cross is None
        ):
            raise RuntimeError("Interaction-specific multiview modules are unavailable")
        total = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        component_count = x.shape[1]
        pair_interactions = torch.zeros(
            x.shape[0], component_count, component_count, dtype=x.dtype, device=x.device
        )
        weights_matrix = (
            torch.zeros(
                x.shape[0], component_count, component_count, 3,
                dtype=x.dtype, device=x.device,
            )
            if return_view_weights
            else None
        )
        view_values_matrix = torch.zeros(
            x.shape[0], component_count, component_count, 3,
            dtype=x.dtype, device=x.device,
        ) if return_view_weights else None
        mixture_context = context.squeeze(1)
        temperature = ((temperature_k - 350.0) / 150.0)
        pressure = torch.log(pressure_kpa.clamp_min(1e-6) / 101.325)
        for first in range(component_count):
            for second in range(first + 1, component_count):
                pair_mask = (mask[:, first] * mask[:, second]).unsqueeze(-1)
                branch_values: list[Tensor] = []
                for name in ("rdkit", "unimol"):
                    first_view = views[name][:, first]
                    second_view = views[name][:, second]
                    pair_features = torch.cat(
                        [
                            first_view + second_view,
                            torch.abs(first_view - second_view),
                            first_view * second_view,
                            mixture_context,
                        ],
                        dim=-1,
                    )
                    branch_values.append(self.view_potentials[name](pair_features))
                group_pair = self.functional_group_cross(
                    functional_counts[:, first], functional_counts[:, second]
                )
                branch_values.append(
                    self.view_potentials["functional_group"](
                        torch.cat([group_pair, mixture_context], dim=-1)
                    )
                )
                view_values = torch.cat(branch_values, dim=-1)
                first_state = state_tokens[:, first]
                second_state = state_tokens[:, second]
                gate_features = torch.cat(
                    [
                        first_state + second_state,
                        torch.abs(first_state - second_state),
                        mixture_context,
                        temperature,
                        pressure,
                        (x[:, first] + x[:, second]).unsqueeze(-1),
                        torch.abs(x[:, first] - x[:, second]).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                weights = torch.softmax(self.view_gate(gate_features), dim=-1)
                interaction = (weights * view_values).sum(-1, keepdim=True) * pair_mask
                pair_interactions[:, first, second] = interaction.squeeze(-1)
                pair_interactions[:, second, first] = interaction.squeeze(-1)
                if weights_matrix is not None and view_values_matrix is not None:
                    pair_weights = weights * pair_mask
                    pair_views = view_values * pair_mask
                    weights_matrix[:, first, second] = pair_weights
                    weights_matrix[:, second, first] = pair_weights
                    view_values_matrix[:, first, second] = pair_views
                    view_values_matrix[:, second, first] = pair_views
                total = total + (x[:, first] * x[:, second]).unsqueeze(-1) * interaction
        return total, pair_interactions, weights_matrix, view_values_matrix

    def forward(
        self,
        molecules: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
        *,
        return_view_weights: bool = False,
    ) -> ModelOutputs:
        if self.config.decoder_mode != "thermodynamic":
            raise RuntimeError("Use predict_direct for a direct-VLE model")
        if molecules.shape[1] not in (2, 3):
            raise ValueError("ThermoFormer supports binary and ternary mixtures only")
        molecular_tokens, molecular_views, functional_counts = self._encode_molecules(molecules)
        if self.vapor_pressure is None:
            raise RuntimeError("Thermodynamic decoder is unavailable")
        log_psat = self.vapor_pressure(molecular_tokens, temperature_k) * mask
        if self.config.activity_mode == "ideal":
            components, _, attention_bias, attention_bias_penalty = self._structural_context(
                molecular_tokens,
                molecular_views,
                temperature_k,
                pressure_kpa,
                x,
                mask,
            )
            return ModelOutputs(
                log_gamma=torch.zeros_like(x) * mask,
                log_psat=log_psat,
                nonideality_tokens=components,
                excess_gibbs_rt=torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device),
                pair_interactions=None,
                attention_bias=attention_bias,
                attention_bias_penalty=attention_bias_penalty,
            )

        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            x_variable = x if x.requires_grad else x.detach().clone().requires_grad_(True)
            (
                components,
                mixture,
                attention_bias,
                attention_bias_penalty,
            ) = self._structural_context(
                molecular_tokens,
                molecular_views,
                temperature_k,
                pressure_kpa,
                x_variable,
                mask,
            )
            tokens, context = self._nonideality_tokens(
                components,
                mixture,
                temperature_k,
                pressure_kpa,
                x_variable,
                mask,
            )
            if self.config.activity_mode == "direct_gamma":
                if self.direct_activity_head is None:
                    raise RuntimeError("Direct activity decoder is unavailable")
                activity_context = context.expand_as(tokens)
                raw_log_gamma = self.direct_activity_head(
                    torch.cat([tokens, activity_context], dim=-1)
                ).squeeze(-1)
                log_gamma = raw_log_gamma * (1.0 - x_variable) * mask
                excess_gibbs = None
                pair_interactions = None
                view_weights = None
                view_interactions = None
            else:
                if self.config.context_pair_interaction:
                    excess_gibbs, pair_interactions = self._contextual_excess_gibbs(
                        tokens,
                        mixture,
                        temperature_k,
                        pressure_kpa,
                        x_variable,
                        mask,
                    )
                    view_weights = None
                    view_interactions = None
                elif self.config.fusion_mode == "interaction_specific":
                    if functional_counts is None:
                        raise RuntimeError("Functional-group counts are unavailable")
                    (
                        excess_gibbs,
                        pair_interactions,
                        view_weights,
                        view_interactions,
                    ) = self._multiview_excess_gibbs(
                        tokens,
                        context,
                        molecular_views,
                        functional_counts,
                        temperature_k,
                        pressure_kpa,
                        x_variable,
                        mask,
                        return_view_weights,
                    )
                else:
                    excess_gibbs, pair_interactions = self._excess_gibbs(
                        tokens, context, x_variable, mask
                    )
                    view_weights = None
                    view_interactions = None
                log_gamma = activity_coefficients_from_excess_gibbs(
                    excess_gibbs,
                    x_variable,
                    mask,
                    create_graph=outer_grad_enabled,
                )

        if not outer_grad_enabled:
            log_gamma = log_gamma.detach()
            tokens = tokens.detach()
            if excess_gibbs is not None:
                excess_gibbs = excess_gibbs.detach()
            if pair_interactions is not None:
                pair_interactions = pair_interactions.detach()
            if view_weights is not None:
                view_weights = view_weights.detach()
            if view_interactions is not None:
                view_interactions = view_interactions.detach()
        return ModelOutputs(
            log_gamma=log_gamma,
            log_psat=log_psat,
            nonideality_tokens=tokens,
            excess_gibbs_rt=excess_gibbs,
            pair_interactions=pair_interactions,
            view_weights=view_weights,
            view_interactions=view_interactions,
            attention_bias=(
                attention_bias
                if outer_grad_enabled or attention_bias is None
                else attention_bias.detach()
            ),
            attention_bias_penalty=(
                attention_bias_penalty
                if outer_grad_enabled or attention_bias_penalty is None
                else attention_bias_penalty.detach()
            ),
        )

    def predict_direct(
        self,
        molecules: Tensor,
        temperature_k: Tensor,
        pressure_kpa: Tensor,
        x: Tensor,
        mask: Tensor,
        *,
        direction: Literal["isothermal", "isobaric"],
    ) -> DirectVLEOutputs:
        """Predict a bounded bubble state without gamma, Psat, or VLE equations."""
        if self.config.decoder_mode != "direct_vle":
            raise RuntimeError("predict_direct requires decoder_mode='direct_vle'")
        if direction not in ("isothermal", "isobaric"):
            raise ValueError("direction must be isothermal or isobaric")
        if self.direct_state_head is None or self.direct_y_head is None:
            raise RuntimeError("Direct-VLE heads are unavailable")
        molecular_tokens, molecular_views, _ = self._encode_molecules(molecules)
        components, mixture, _, _ = self._structural_context(
            molecular_tokens,
            molecular_views,
            temperature_k,
            pressure_kpa,
            x,
            mask,
        )
        model_temperature = (
            temperature_k if direction == "isothermal" else torch.full_like(temperature_k, 350.0)
        )
        model_pressure = (
            torch.full_like(pressure_kpa, 101.325) if direction == "isothermal" else pressure_kpa
        )
        tokens, _ = self._nonideality_tokens(
            components, mixture, model_temperature, model_pressure, x, mask
        )
        normalized_x = x * mask
        normalized_x = normalized_x / normalized_x.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (normalized_x.unsqueeze(-1) * tokens).sum(1)
        temperature_feature = (model_temperature - 350.0) / 150.0
        pressure_feature = torch.log(model_pressure.clamp_min(1e-6) / 101.325)
        direction_feature = torch.full_like(
            temperature_feature, 1.0 if direction == "isothermal" else -1.0
        )
        raw_state = self.direct_state_head(
            torch.cat([pooled, temperature_feature, pressure_feature, direction_feature], dim=-1)
        )
        predicted_pressure = torch.exp(
            math.log(101.325) + raw_state.clamp(min=-5.0, max=5.0)
        )
        predicted_temperature = 150.0 + 1350.0 * torch.sigmoid(raw_state)
        global_context = pooled.unsqueeze(1).expand_as(tokens)
        component_direction = direction_feature.expand_as(x).unsqueeze(-1)
        y_logits = self.direct_y_head(
            torch.cat(
                [tokens, global_context, x.unsqueeze(-1), component_direction], dim=-1
            )
        ).squeeze(-1)
        y = torch.softmax(y_logits.masked_fill(~mask.bool(), -1e9), dim=-1) * mask
        return DirectVLEOutputs(
            pressure_kpa=(predicted_pressure if direction == "isothermal" else pressure_kpa),
            temperature_k=(temperature_k if direction == "isothermal" else predicted_temperature),
            y=y,
        )
