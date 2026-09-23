"""Learned pure-component vapor-pressure decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PureVaporPressure(nn.Module):
    """Monotonic learned Clausius-Clapeyron branch in kPa."""

    def __init__(self, hidden_dim: int, head_hidden_dim: int | None = None) -> None:
        super().__init__()
        head_hidden_dim = head_hidden_dim or hidden_dim
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 2),
        )
        output = self.parameter_head[-1]
        nn.init.zeros_(output.weight)
        with torch.no_grad():
            output.bias.copy_(torch.tensor([12.0, 2.4]))

    def forward(self, molecular_tokens: Tensor, temperature_k: Tensor) -> Tensor:
        raw = self.parameter_head(molecular_tokens)
        intercept = raw[..., 0]
        inverse_temperature = 1000.0 * torch.nn.functional.softplus(raw[..., 1]) + 1.0
        temperature = temperature_k.clamp_min(100.0).expand(molecular_tokens.shape[:2])
        return intercept - inverse_temperature / temperature

__all__ = ["PureVaporPressure"]
