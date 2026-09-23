"""Activity coefficients from the composition derivative of excess Gibbs energy."""

import torch
from torch import Tensor


def activity_coefficients_from_excess_gibbs(
    excess_gibbs_rt: Tensor,
    composition: Tensor,
    mask: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    """Return ``ln(gamma_i)`` from a scalar molar ``gE/RT`` representation."""

    composition_gradient = torch.autograd.grad(
        excess_gibbs_rt.sum(),
        composition,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    weighted_gradient = (composition * composition_gradient * mask).sum(-1, keepdim=True)
    return (excess_gibbs_rt + composition_gradient - weighted_gradient) * mask


__all__ = ["activity_coefficients_from_excess_gibbs"]
