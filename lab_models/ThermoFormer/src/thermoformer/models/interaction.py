"""Molecular-view projection and multicomponent interaction layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

class ViewProjection(nn.Module):
    """Independent projection used by one molecular information view."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class ChemicalBiasedTransformerLayer(nn.Module):
    """Pre-norm self-attention layer with a batch-specific additive pair bias."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, bias: Tensor, padding: Tensor) -> Tensor:
        batch, length, _ = values.shape
        additive_mask = (
            bias.unsqueeze(1).expand(batch, self.heads, length, length)
            if bias.ndim == 3
            else bias
        )
        if additive_mask.shape != (batch, self.heads, length, length):
            raise ValueError("attention bias must have shape [batch, heads, tokens, tokens]")
        invalid_keys = padding[:, None, None, :].expand_as(additive_mask)
        additive_mask = additive_mask.masked_fill(invalid_keys, float("-inf"))
        additive_mask = additive_mask.reshape(batch * self.heads, length, length)
        normalized = self.norm1(values)
        # The thermodynamic decoder differentiates log(gamma) through the
        # composition derivative of G^E, so training needs a second backward
        # pass through this composition-conditioned attention operation. CUDA's
        # flash/efficient SDPA kernels do not implement that derivative; the
        # math kernel does and preserves the exact same attention equation.
        with sdpa_kernel(SDPBackend.MATH):
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=additive_mask,
                need_weights=False,
            )
        values = values + self.dropout1(attended)
        return values + self.feedforward(self.norm2(values))


class ChemicalBiasedTransformer(nn.Module):
    """Stack of Transformer layers sharing a chemically constructed pair bias."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ChemicalBiasedTransformerLayer(
                hidden_dim, heads, feedforward_dim, dropout
            )
            for _ in range(layers)
        )

    def forward(self, values: Tensor, bias: Tensor, padding: Tensor) -> Tensor:
        if bias.ndim not in (3, 5):
            raise ValueError("chemical bias must be scalar or layer/head resolved")
        for index, layer in enumerate(self.layers):
            layer_bias = bias[:, index] if bias.ndim == 5 else bias
            values = layer(values, layer_bias, padding)
        return values


class FunctionalGroupCrossInteraction(nn.Module):
    """Symmetric bidirectional cross-attention over functional-group tokens."""

    def __init__(self, group_count: int, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.group_count = group_count
        self.group_embeddings = nn.Parameter(torch.empty(group_count + 1, hidden_dim))
        nn.init.normal_(self.group_embeddings, std=0.02)
        self.count_projection = nn.Linear(1, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )

    def _tokens(self, counts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        counts = counts.clamp_min(0.0)
        present = counts > 0.0
        empty = ~present.any(-1, keepdim=True)
        extended_counts = torch.cat([counts, empty.to(counts.dtype)], dim=-1)
        valid = torch.cat([present, empty], dim=-1)
        embeddings = self.group_embeddings.unsqueeze(0).expand(counts.shape[0], -1, -1)
        tokens = embeddings + self.count_projection(
            torch.log1p(extended_counts).unsqueeze(-1)
        )
        weights = extended_counts.clamp_min(0.0)
        return tokens, valid, weights

    @staticmethod
    def _pool(values: Tensor, valid: Tensor, weights: Tensor) -> Tensor:
        normalized = weights * valid.to(weights.dtype)
        normalized = normalized / normalized.sum(-1, keepdim=True).clamp_min(1.0)
        return (values * normalized.unsqueeze(-1)).sum(1)

    def forward(self, first_counts: Tensor, second_counts: Tensor) -> Tensor:
        first, first_valid, first_weights = self._tokens(first_counts)
        second, second_valid, second_weights = self._tokens(second_counts)
        first_from_second, _ = self.cross_attention(
            first,
            second,
            second,
            key_padding_mask=~second_valid,
            need_weights=False,
        )
        second_from_first, _ = self.cross_attention(
            second,
            first,
            first,
            key_padding_mask=~first_valid,
            need_weights=False,
        )
        pooled_first = self._pool(first_from_second, first_valid, first_weights)
        pooled_second = self._pool(second_from_first, second_valid, second_weights)
        return torch.cat(
            [
                pooled_first + pooled_second,
                torch.abs(pooled_first - pooled_second),
                pooled_first * pooled_second,
            ],
            dim=-1,
        )
