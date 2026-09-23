"""Train-only SPT-NRTL adaptation for unseen molecular pairs.

This module intentionally does not claim byte-level reproduction of the
authors' confidentially trained SPT-NRTL model.  It retains the published
character-level molecular-pair Transformer and ten-parameter NRTL decoder,
while deriving supervision exclusively from the registered ThermoFormer
training partition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ...data.loading import VLESample
from ...data.splitting import canonical_smiles
from ..thermodynamic_fitting import LogLinearVaporPressure
from .spt_nrtl import SPTNRTLPairParameters, multicomponent_nrtl_log_gamma


PARAMETER_LOWER = np.asarray(
    [0.10, 0.0, -30.0, -4000.0, -8.0, -0.03, -30.0, -4000.0, -8.0, -0.03],
    dtype=np.float64,
)
PARAMETER_UPPER = np.asarray(
    [0.50, 1.0e-3, 30.0, 4000.0, 8.0, 0.03, 30.0, 4000.0, 8.0, 0.03],
    dtype=np.float64,
)
PARAMETER_INITIAL = np.asarray([0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class AdaptedSPTConfig:
    max_sequence_length: int = 128
    embedding_dimension: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    feedforward_dimension: int = 256
    dropout: float = 0.10
    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 3.0e-4
    patience: int = 30
    minimum_pair_rows: int = 5
    label_fit_evaluations: int = 250
    label_regularization: float = 1.0e-3

    def validate(self) -> None:
        if self.max_sequence_length < 8 or self.embedding_dimension < 8:
            raise ValueError("SPT-NRTL adapted sequence and embedding dimensions are too small")
        if self.embedding_dimension % self.attention_heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        if min(self.transformer_layers, self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("SPT-NRTL adapted training counts must be positive")
        if self.minimum_pair_rows < 2 or self.label_fit_evaluations < 1:
            raise ValueError("SPT-NRTL pair-label fitting settings are invalid")
        if not 0.0 <= self.dropout < 1.0 or self.learning_rate <= 0.0:
            raise ValueError("SPT-NRTL adapted optimizer settings are invalid")


@dataclass(frozen=True)
class PairLabel:
    smiles_i: str
    smiles_j: str
    parameters: tuple[float, ...]
    rows: int
    initial_rmse: float
    final_rmse: float
    success: bool

    def reversed(self) -> "PairLabel":
        values = self.parameters
        return PairLabel(
            self.smiles_j,
            self.smiles_i,
            tuple(values[:2] + values[6:10] + values[2:6]),
            self.rows,
            self.initial_rmse,
            self.final_rmse,
            self.success,
        )


def vector_to_pair(values: Sequence[float]) -> SPTNRTLPairParameters:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,) or not np.isfinite(array).all():
        raise ValueError("SPT-NRTL parameter vector must contain ten finite values")
    array = np.clip(array, PARAMETER_LOWER, PARAMETER_UPPER)
    return SPTNRTLPairParameters(
        alpha=(float(array[0]), float(array[1])),
        tau_ij=tuple(float(value) for value in array[2:6]),
        tau_ji=tuple(float(value) for value in array[6:10]),
    )


def _ordered_binary_state(sample: VLESample, pair: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    positions = {canonical_smiles(value): index for index, value in enumerate(sample.smiles)}
    order = [positions[value] for value in pair]
    return (
        np.asarray([sample.liquid_composition[index] for index in order], dtype=np.float64),
        np.asarray([sample.vapor_composition[index] for index in order], dtype=np.float64),
    )


def _binary_log_gamma_torch(values: Tensor, temperature: Tensor, composition: Tensor) -> Tensor:
    """Vectorized differentiable binary NRTL equation for pair-label fitting."""
    alpha_value = values[0] + values[1] * temperature
    tau_12 = values[2] + values[3] / temperature + values[4] * torch.log(temperature) + values[5] * temperature
    tau_21 = values[6] + values[7] / temperature + values[8] * torch.log(temperature) + values[9] * temperature
    tau = torch.zeros((temperature.shape[0], 2, 2), dtype=values.dtype, device=values.device)
    alpha = torch.zeros_like(tau)
    tau[:, 0, 1], tau[:, 1, 0] = tau_12, tau_21
    alpha[:, 0, 1], alpha[:, 1, 0] = alpha_value, alpha_value
    interaction = torch.exp((-alpha * tau).clamp(-60.0, 60.0))
    denominators = torch.einsum("ni,nij->nj", composition, interaction).clamp_min(1.0e-15)
    weighted_tau = torch.einsum("ni,nij,nij->nj", composition, tau, interaction)
    first = torch.einsum(
        "nj,nji,nji,ni->ni", composition, tau, interaction, denominators.reciprocal()
    )
    bracket = tau - (weighted_tau / denominators).unsqueeze(1)
    second = torch.einsum(
        "nj,nij,nij,nj->ni",
        composition,
        interaction,
        bracket,
        denominators.reciprocal(),
    )
    return first + second


def fit_pair_label(
    samples: Sequence[VLESample],
    vapor_pressure: Mapping[str, LogLinearVaporPressure],
    config: AdaptedSPTConfig,
) -> PairLabel | None:
    """Fit one binary pair from a non-test partition only."""
    if not samples:
        return None
    pair = tuple(sorted(canonical_smiles(value) for value in samples[0].smiles))
    if len(pair) != 2 or any(tuple(sorted(canonical_smiles(value) for value in row.smiles)) != pair for row in samples):
        raise ValueError("Pair-label fitting requires one binary chemical system")
    if any(value not in vapor_pressure for value in pair):
        return None
    temperatures: list[float] = []
    compositions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for sample in samples:
        x, y = _ordered_binary_state(sample, pair)
        psat = np.asarray([vapor_pressure[value].pressure_kpa(sample.temperature_k) for value in pair])
        valid = (x > 1.0e-5) & (y > 1.0e-5) & np.isfinite(psat) & (psat > 0.0)
        if not valid.all():
            continue
        target = np.log(y * float(sample.pressure_kpa) / (x * psat))
        if not np.isfinite(target).all() or np.any(np.abs(target) > 12.0):
            continue
        temperatures.append(float(sample.temperature_k))
        compositions.append(x)
        targets.append(target)
        weights.append(np.full(2, np.sqrt(max(float(sample.quality_weight), 1.0e-12))))
    if len(temperatures) < config.minimum_pair_rows:
        return None
    temperature_array = np.asarray(temperatures)
    composition_array = np.asarray(compositions)
    target_array = np.asarray(targets)
    weight_array = np.asarray(weights)
    lower = torch.tensor(PARAMETER_LOWER, dtype=torch.float64)
    upper = torch.tensor(PARAMETER_UPPER, dtype=torch.float64)
    scale = upper - lower
    initial = torch.tensor(PARAMETER_INITIAL, dtype=torch.float64)
    initial_fraction = ((initial - lower) / scale).clamp(1.0e-6, 1.0 - 1.0e-6)
    raw = nn.Parameter(torch.logit(initial_fraction))
    optimizer = torch.optim.Adam([raw], lr=5.0e-2)
    temperature_tensor = torch.tensor(temperature_array, dtype=torch.float64)
    composition_tensor = torch.tensor(composition_array, dtype=torch.float64)
    target_tensor = torch.tensor(target_array, dtype=torch.float64)
    weight_tensor = torch.tensor(weight_array, dtype=torch.float64)

    def parameter_values() -> Tensor:
        return lower + scale * torch.sigmoid(raw)

    with torch.no_grad():
        initial_residual = (
            _binary_log_gamma_torch(initial, temperature_tensor, composition_tensor)
            - target_tensor
        ) * weight_tensor
        initial_rmse = float(torch.sqrt(initial_residual.square().mean()))
    best_values = initial.clone()
    best_rmse = initial_rmse
    for _ in range(config.label_fit_evaluations):
        optimizer.zero_grad(set_to_none=True)
        values = parameter_values()
        residual = (
            _binary_log_gamma_torch(values, temperature_tensor, composition_tensor)
            - target_tensor
        ) * weight_tensor
        regularizer = config.label_regularization * ((values - initial) / scale).square().mean()
        loss = residual.square().mean() + regularizer
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current_values = parameter_values()
            current_residual = (
                _binary_log_gamma_torch(current_values, temperature_tensor, composition_tensor)
                - target_tensor
            ) * weight_tensor
            current_rmse = float(torch.sqrt(current_residual.square().mean()))
            if np.isfinite(current_rmse) and current_rmse < best_rmse:
                best_rmse = current_rmse
                best_values = current_values.detach().clone()
    fitted_values = best_values.cpu().numpy()
    final_rmse = best_rmse
    success = bool(np.isfinite(fitted_values).all() and final_rmse <= initial_rmse + 1.0e-10)
    if not success:
        return None
    return PairLabel(
        pair[0], pair[1], tuple(float(value) for value in fitted_values), len(temperatures),
        initial_rmse, final_rmse, True,
    )


def fit_partition_pair_labels(
    samples: Sequence[VLESample],
    vapor_pressure: Mapping[str, LogLinearVaporPressure],
    config: AdaptedSPTConfig,
) -> tuple[tuple[PairLabel, ...], dict[str, object]]:
    grouped: dict[tuple[str, str], list[VLESample]] = defaultdict(list)
    for sample in samples:
        if sample.component_count == 2:
            grouped[tuple(sorted(canonical_smiles(value) for value in sample.smiles))].append(sample)
    labels = tuple(
        label
        for pair in sorted(grouped)
        for label in [fit_pair_label(grouped[pair], vapor_pressure, config)]
        if label is not None
    )
    label_payload = [asdict(value) for value in labels]
    return labels, {
        "attempted_pairs": len(grouped),
        "fitted_pairs": len(labels),
        "pair_label_coverage": len(labels) / len(grouped) if grouped else 0.0,
        "minimum_pair_rows": config.minimum_pair_rows,
        "fit_rows": sum(value.rows for value in labels),
        "mean_initial_rmse": float(np.mean([value.initial_rmse for value in labels])) if labels else None,
        "mean_final_rmse": float(np.mean([value.final_rmse for value in labels])) if labels else None,
        "fitted_label_sha256": hashlib.sha256(
            json.dumps(label_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


class PairTokenizer:
    special = ("<PAD>", "<UNK>", "<SOS>", "<MOS>", "<EOS>")

    def __init__(self, vocabulary: Sequence[str], maximum_length: int) -> None:
        self.tokens = tuple(vocabulary)
        self.index = {value: index for index, value in enumerate(self.tokens)}
        self.maximum_length = maximum_length

    @classmethod
    def from_labels(cls, labels: Sequence[PairLabel], maximum_length: int) -> "PairTokenizer":
        characters = sorted({character for label in labels for smiles in (label.smiles_i, label.smiles_j) for character in smiles})
        return cls((*cls.special, *characters), maximum_length)

    def encode(self, first: str, second: str) -> tuple[list[int], list[bool]]:
        sequence = ["<SOS>", *canonical_smiles(first), "<MOS>", *canonical_smiles(second), "<EOS>"]
        if len(sequence) > self.maximum_length:
            raise ValueError("Canonical SMILES pair exceeds the configured SPT-NRTL sequence length")
        ids = [self.index.get(value, self.index["<UNK>"]) for value in sequence]
        padding = self.maximum_length - len(ids)
        return ids + [self.index["<PAD>"]] * padding, [False] * len(ids) + [True] * padding


class AdaptedSPTParameterNetwork(nn.Module):
    """Causal character Transformer followed by a ten-parameter NRTL head."""

    def __init__(self, vocabulary_size: int, config: AdaptedSPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(vocabulary_size, config.embedding_dimension, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.transformer_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(config.embedding_dimension),
            nn.Linear(config.embedding_dimension, config.embedding_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dimension, 10),
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = hidden.masked_fill(~valid, torch.finfo(hidden.dtype).min).amax(dim=1)
        return self.output(pooled)


@dataclass(frozen=True)
class AdaptedSPTTrainingResult:
    model_state: dict[str, Tensor]
    vocabulary: tuple[str, ...]
    parameter_mean: tuple[float, ...]
    parameter_scale: tuple[float, ...]
    best_validation_loss: float
    best_epoch: int
    history: tuple[dict[str, float], ...]
    trainable_parameters: int
    config: dict[str, object]


def _label_tensors(labels: Sequence[PairLabel], tokenizer: PairTokenizer) -> tuple[Tensor, Tensor, Tensor]:
    augmented = tuple(labels) + tuple(value.reversed() for value in labels)
    encoded = [tokenizer.encode(value.smiles_i, value.smiles_j) for value in augmented]
    return (
        torch.tensor([value[0] for value in encoded], dtype=torch.long),
        torch.tensor([value[1] for value in encoded], dtype=torch.bool),
        torch.tensor([value.parameters for value in augmented], dtype=torch.float32),
    )


def train_adapted_spt(
    training_labels: Sequence[PairLabel],
    validation_labels: Sequence[PairLabel],
    config: AdaptedSPTConfig,
    device: torch.device,
) -> AdaptedSPTTrainingResult:
    config.validate()
    if not training_labels or not validation_labels:
        raise ValueError("SPT-NRTL adapted requires fitted training and validation pair labels")
    tokenizer = PairTokenizer.from_labels(training_labels, config.max_sequence_length)
    train_tokens, train_padding, train_targets = _label_tensors(training_labels, tokenizer)
    validation_tokens, validation_padding, validation_targets = _label_tensors(validation_labels, tokenizer)
    mean = train_targets.mean(dim=0)
    scale = train_targets.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    train_targets = (train_targets - mean) / scale
    validation_targets = (validation_targets - mean) / scale
    generator = torch.Generator().manual_seed(torch.initial_seed())
    loader = DataLoader(
        TensorDataset(train_tokens, train_padding, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = AdaptedSPTParameterNetwork(len(tokenizer.tokens), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4)
    best_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for tokens, padding, targets in loader:
            tokens, padding, targets = tokens.to(device), padding.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(tokens, padding), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * tokens.shape[0]
            count += tokens.shape[0]
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.nn.functional.smooth_l1_loss(
                model(validation_tokens.to(device), validation_padding.to(device)),
                validation_targets.to(device),
            ))
        history.append({"epoch": float(epoch + 1), "train_loss": total / count, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1.0e-10:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("SPT-NRTL adapted did not produce a finite validation checkpoint")
    return AdaptedSPTTrainingResult(
        best_state,
        tokenizer.tokens,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        best_loss,
        best_epoch,
        tuple(history),
        sum(value.numel() for value in model.parameters() if value.requires_grad),
        asdict(config),
    )


class AdaptedSPTNRTLPredictor:
    def __init__(self, training: AdaptedSPTTrainingResult, device: torch.device) -> None:
        config = AdaptedSPTConfig(**training.config)
        self.tokenizer = PairTokenizer(training.vocabulary, config.max_sequence_length)
        self.model = AdaptedSPTParameterNetwork(len(training.vocabulary), config).to(device)
        self.model.load_state_dict(training.model_state, strict=True)
        self.model.eval()
        self.device = device
        self.mean = torch.tensor(training.parameter_mean, dtype=torch.float32, device=device)
        self.scale = torch.tensor(training.parameter_scale, dtype=torch.float32, device=device)
        self._cache: dict[tuple[str, str], SPTNRTLPairParameters] = {}

    def _pair(self, first: str, second: str) -> SPTNRTLPairParameters:
        requested = (canonical_smiles(first), canonical_smiles(second))
        canonical = tuple(sorted(requested))
        if canonical not in self._cache:
            tokens, padding = self.tokenizer.encode(*canonical)
            with torch.no_grad():
                standardized = self.model(
                    torch.tensor([tokens], dtype=torch.long, device=self.device),
                    torch.tensor([padding], dtype=torch.bool, device=self.device),
                )[0]
            values = (standardized * self.scale + self.mean).detach().cpu().numpy()
            self._cache[canonical] = vector_to_pair(values)
        return self._cache[canonical] if requested == canonical else self._cache[canonical].reversed()

    def system_pairs(self, smiles: Sequence[str]) -> dict[tuple[int, int], SPTNRTLPairParameters]:
        return {(i, j): self._pair(smiles[i], smiles[j]) for i in range(len(smiles)) for j in range(i + 1, len(smiles))}

    def require_coverage(self, smiles: Sequence[str]) -> None:
        self.system_pairs(smiles)

    def log_gamma(self, smiles: Sequence[str], temperature_k: float, composition: Sequence[float]) -> Tensor:
        return multicomponent_nrtl_log_gamma(
            torch.as_tensor(composition, dtype=torch.float64),
            temperature_k,
            self.system_pairs(smiles),
        ).to(torch.float32)
