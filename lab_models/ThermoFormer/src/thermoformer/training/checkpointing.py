"""Model-state snapshots shared by both training stages."""

import torch
from torch import nn


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state onto CPU for stable best-epoch restoration."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


__all__ = ["cpu_state_dict"]
