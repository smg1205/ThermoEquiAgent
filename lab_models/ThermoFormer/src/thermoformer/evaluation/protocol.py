"""High-level evaluation of a fixed protocol partition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn

from ..data import VLESample
from ..data.splitting import DatasetPartitions, load_split_assignment
from ..thermodynamics.vapor_pressure import PurePropertyCatalog
from .prediction import predict_vle, prediction_metric_rows


@dataclass(frozen=True)
class ProtocolEvaluation:
    """Predictions and metrics for one immutable protocol partition."""

    protocol: str
    seed: int
    partition: str
    predictions: list[dict[str, Any]]
    metrics: list[dict[str, Any]]


def evaluate_protocol(
    model: nn.Module,
    dataset: Sequence[VLESample],
    split: DatasetPartitions | Path,
    feature_map: dict[str, np.ndarray],
    *,
    partition: Literal["validation", "test"] = "test",
    batch_size: int = 128,
    device: torch.device | None = None,
    solver_iterations: int = 48,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> ProtocolEvaluation:
    """Evaluate a model on a registered validation or test partition."""

    assignment = load_split_assignment(split, dataset) if isinstance(split, Path) else split
    samples = assignment.validation if partition == "validation" else assignment.test
    predictions = predict_vle(
        model,
        samples,
        feature_map,
        batch_size=batch_size,
        device=device or torch.device("cpu"),
        solver_iterations=solver_iterations,
        pure_property_catalog=pure_property_catalog,
    )
    return ProtocolEvaluation(
        protocol=assignment.protocol,
        seed=assignment.seed,
        partition=partition,
        predictions=predictions,
        metrics=prediction_metric_rows(predictions),
    )
