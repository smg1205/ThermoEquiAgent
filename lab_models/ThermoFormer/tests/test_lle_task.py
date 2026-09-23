import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.thermoformer.configuration import ExperimentConfig, LLEConfig
from src.thermoformer.data.lle import (
    LLEBatch,
    LLESample,
    LLETensorDataset,
    build_lle_split,
    collate_lle,
    load_lle_dataset,
)
from src.thermoformer.thermodynamics.lle_solver import solve_lle
from src.thermoformer.training.lle import (
    lle_fugacity_loss,
    lle_stage3_objective,
)
from src.thermoformer.training.supervised import TrainingConfig


class ToyActivityModel(torch.nn.Module):
    """Minimal differentiable shared-gamma stand-in for solver invariants."""

    def __init__(self) -> None:
        super().__init__()
        self.pair_potential = torch.nn.Parameter(torch.tensor(0.35))

    def forward(self, molecules, temperature_k, pressure_kpa, composition, mask):
        centered = composition - (composition * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        return SimpleNamespace(log_gamma=self.pair_potential * centered * mask)


def sample(index: int, smiles: tuple[str, str, str]) -> LLESample:
    return LLESample(
        record_id=f"tie-{index}", smiles=smiles, names=smiles,
        temperature_k=298.15, pressure_kpa=101.325,
        phase_alpha=(0.15, 0.35, 0.50), phase_beta=(0.70, 0.20, 0.10),
        quality_weight=1.0, source="synthetic", doi="unit",
    )


class LLETaskTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_default_is_vle_and_joint_is_rejected(self) -> None:
        self.assertEqual(ExperimentConfig().task_mode, "vle")
        with self.assertRaisesRegex(ValueError, "joint mode"):
            ExperimentConfig(task_mode="joint")
        with self.assertRaisesRegex(ValueError, "cannot exceed 80"):
            ExperimentConfig(task_mode="lle", lle=LLEConfig(), training=TrainingConfig(epochs_supervised=81))

    def test_system_split_precedes_bidirectional_expansion(self) -> None:
        rows = [
            sample(0, ("CCO", "O", "CC")), sample(1, ("CCO", "O", "CC")),
            sample(2, ("CO", "O", "CCC")), sample(3, ("CO", "O", "CCC")),
            sample(4, ("CCN", "O", "CCCC")), sample(5, ("CCN", "O", "CCCC")),
        ]
        split = build_lle_split(rows, seed=0)
        partitions = (split.train, split.validation, split.test)
        keys = [set(row.system_key for row in partition) for partition in partitions]
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        features = {smiles: np.full(4, index, dtype=np.float32) for index, smiles in enumerate({s for row in rows for s in row.smiles})}
        expanded = LLETensorDataset(split.train, features)
        self.assertEqual(len(expanded), 2 * len(split.train))
        self.assertEqual({row[-1] for row in expanded.rows[::2]}, {row[-1] for row in expanded.rows[1::2]})

    def test_solver_simplex_and_pair_potential_gradient(self) -> None:
        model = ToyActivityModel()
        batch = LLEBatch(
            molecules=torch.ones(1, 3, 2), temperature_k=torch.tensor([[298.15]]),
            pressure_kpa=torch.tensor([[101.325]]), source_phase=torch.tensor([[0.15, 0.35, 0.50]]),
            target_phase=torch.tensor([[0.70, 0.20, 0.10]]), mask=torch.ones(1, 3),
            quality_weight=torch.ones(1, 1), record_ids=("unit",),
        )
        state = solve_lle(model, batch.molecules, batch.temperature_k, batch.pressure_kpa, batch.source_phase, batch.mask, multistarts=2, iterations=2)
        self.assertTrue(torch.all(state.x_target > 0.0))
        self.assertTrue(torch.allclose(state.x_target.sum(-1), torch.ones(1), atol=1e-6))
        state.log_fugacity_residual.square().mean().backward()
        self.assertIsNotNone(model.pair_potential.grad)
        self.assertTrue(torch.isfinite(model.pair_potential.grad).all())

    def test_observed_endpoint_fugacity_is_phase_swap_invariant(self) -> None:
        model = ToyActivityModel()
        common = dict(molecules=torch.ones(1, 3, 2), temperature_k=torch.tensor([[298.15]]), pressure_kpa=torch.tensor([[101.325]]), mask=torch.ones(1, 3), quality_weight=torch.ones(1, 1), record_ids=("unit",))
        alpha = torch.tensor([[0.15, 0.35, 0.50]])
        beta = torch.tensor([[0.70, 0.20, 0.10]])
        forward = LLEBatch(source_phase=alpha, target_phase=beta, **common)
        reverse = LLEBatch(source_phase=beta, target_phase=alpha, **common)
        self.assertTrue(torch.allclose(lle_fugacity_loss(model, forward, 1e-4), lle_fugacity_loss(model, reverse, 1e-4)))

    def test_zero_fugacity_weight_reduces_stage3_to_endpoint_supervision(self) -> None:
        supervised = torch.tensor(1.25)
        fugacity = torch.tensor(9.0)
        value = lle_stage3_objective(supervised, fugacity, LLEConfig(fugacity_weight=0.0))
        self.assertEqual(value.item(), supervised.item())

    def test_actual_workbooks_have_normalized_ternary_records_when_available(self) -> None:
        data_root = self.ROOT / 'datasets/vle_reference'
        if not (data_root / "binary_lle_english.xlsx").is_file() or not (data_root / "ternary_lle_english.xlsx").is_file():
            self.skipTest("User-provided LLE workbooks are not installed")
        loaded = load_lle_dataset(data_root, component_count=3)
        self.assertGreater(loaded.audit["loaded_samples"], 0)
        for row in loaded.samples[:20]:
            self.assertEqual(row.component_count, 3)
            self.assertAlmostEqual(sum(row.phase_alpha), 1.0, places=7)
            self.assertAlmostEqual(sum(row.phase_beta), 1.0, places=7)
            self.assertGreater(row.temperature_k, 273.15)
            self.assertGreater(row.pressure_kpa, 0.0)


if __name__ == "__main__":
    unittest.main()
