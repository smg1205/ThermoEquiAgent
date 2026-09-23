"""Mask-aware regression metrics for variable-component batches."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def masked_r2(actual: Tensor, predicted: Tensor, mask: Tensor) -> float:
    valid = mask.bool()
    y = actual[valid].float()
    y_hat = predicted[valid].float()
    if y.numel() == 0:
        raise ValueError("masked_r2 requires at least one valid value")
    residual = torch.sum((y - y_hat) ** 2)
    variance = torch.sum((y - y.mean()) ** 2).clamp_min(1e-12)
    return float((1.0 - residual / variance).detach().cpu())


def summarize_fold_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return a complete fold summary, failing loudly on partial/invalid metrics."""
    if not rows:
        return {}
    expected_names = set(rows[0])
    if any(set(row) != expected_names for row in rows[1:]):
        raise ValueError("Every fold must report the same metric names")
    names = sorted(expected_names)
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            failed_folds = [index + 1 for index, value in enumerate(values) if not np.isfinite(value)]
            raise ValueError(f"Metric '{name}' is non-finite in folds {failed_folds}")
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "folds": float(len(rows)),
        }
    return summary
