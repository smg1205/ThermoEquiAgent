from pathlib import Path

import numpy as np

from src.thermoformer.lle_tp.data import load
from src.thermoformer.lle_tp.generalization import partition_generalization
from src.thermoformer.lle_tp.metrics import score


ROOT = Path(__file__).resolve().parents[1]


def test_lle_nist_counts_and_generalization_coverage():
    conditions, audit = load(ROOT / "datasets/lle")
    assert audit["by_components"]["2"]["conditions"] == 2794
    assert audit["by_components"]["3"]["conditions"] == 1584
    for protocol, eligible in (
        ("binary-temperature-low", 169),
        ("binary-temperature-high", 169),
        ("binary-pressure-low", 14),
        ("binary-pressure-high", 14),
    ):
        parts, metadata = partition_generalization(conditions, protocol, seed=0)
        assert metadata["eligible_systems"] == eligible
        keys = [{row.key for row in parts[name]} for name in ("train", "validation", "test")]
        assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        assert all(parts.values())


def test_unseen_components_are_absent_from_training():
    conditions, _ = load(ROOT / "datasets/lle")
    parts, metadata = partition_generalization(
        conditions, "binary-unseen-component", seed=0
    )
    held = set(metadata["held_out_components"])
    train = {component for row in parts["train"] for component in row.key[0]}
    test = {component for row in parts["test"] for component in row.key[0]}
    assert held.isdisjoint(train)
    assert held <= test


def test_lle_metrics_include_endpoint_rmse():
    observed = [[[0.1, 0.9], [0.8, 0.2]]]
    predicted = [[[0.2, 0.8], [0.7, 0.3]]]
    result = score([{"observed": observed, "predicted": predicted, "solver": {}}])
    assert np.isclose(result["matched_endpoint_mae_successful_only"], 0.1)
    assert np.isclose(result["matched_endpoint_rmse_successful_only"], 0.1)
