import unittest

import torch

from src.data import VLEBatch
from src.losses import (
    direct_vle_objective,
    experimental_objective,
    with_teacher_forced_fugacity_equilibrium,
)
from src.model import ModelOutputs, ThermoFormer, ThermoFormerConfig
from src.pure_properties import CORRELATION_PARAMETER_COUNT
from src.thermo import equilibrium_at_tp


class ConstantIdealModel(torch.nn.Module):
    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        psat = torch.tensor([100.0, 50.0], device=x.device).expand_as(x)
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=torch.log(psat))


class ExperimentalObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ConstantIdealModel()
        self.molecules = torch.zeros(1, 2, 1)
        self.temperature = torch.tensor([[350.0]])
        self.x = torch.tensor([[0.4, 0.6]])
        self.mask = torch.ones_like(self.x)

    def test_exact_equilibrium_has_zero_data_loss(self) -> None:
        pressure = torch.tensor([[70.0]])
        state = equilibrium_at_tp(
            self.model, self.molecules, self.temperature, pressure, self.x, self.mask
        )
        objective = experimental_objective(
            state,
            observed_y=torch.tensor([[4.0 / 7.0, 3.0 / 7.0]]),
            observed_pressure_kpa=pressure,
            quality_weight=torch.ones(1, 1),
            mask=self.mask,
        )

        self.assertAlmostEqual(objective.total.item(), 0.0, places=6)

    def test_pressure_mismatch_is_penalized(self) -> None:
        observed_pressure = torch.tensor([[140.0]])
        state = equilibrium_at_tp(
            self.model, self.molecules, self.temperature, observed_pressure, self.x, self.mask
        )
        objective = experimental_objective(
            state,
            observed_y=state.y,
            observed_pressure_kpa=observed_pressure,
            quality_weight=torch.ones(1, 1),
            mask=self.mask,
        )

        self.assertGreater(objective.pressure.item(), 0.1)

    def test_fugacity_equilibrium_loss_distinguishes_balanced_and_unbalanced_states(self) -> None:
        pressure = torch.tensor([[70.0]])
        equilibrium_y = torch.tensor([[4.0 / 7.0, 3.0 / 7.0]])
        state = equilibrium_at_tp(
            self.model, self.molecules, self.temperature, pressure, self.x, self.mask
        )
        base = experimental_objective(
            state,
            observed_y=equilibrium_y,
            observed_pressure_kpa=pressure,
            quality_weight=torch.ones(1, 1),
            mask=self.mask,
        )

        def batch(y: torch.Tensor) -> VLEBatch:
            return VLEBatch(
                molecules=self.molecules,
                temperature_k=self.temperature,
                pressure_kpa=pressure,
                x=self.x,
                y=y,
                mask=self.mask,
                quality_weight=torch.ones(1, 1),
                experiment_mode=torch.tensor([0]),
                pure_property_parameters=torch.zeros(
                    1, 2, CORRELATION_PARAMETER_COUNT
                ),
            )

        balanced = with_teacher_forced_fugacity_equilibrium(
            base, state, batch(equilibrium_y), 1.0
        )
        unbalanced = with_teacher_forced_fugacity_equilibrium(
            base, state, batch(torch.tensor([[0.5, 0.5]])), 1.0
        )

        self.assertAlmostEqual(
            balanced.teacher_forced_fugacity.item(), 0.0, places=6
        )
        self.assertGreater(unbalanced.teacher_forced_fugacity.item(), 1e-3)

    def test_direct_vle_objective_trains_both_inference_directions(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=4,
                hidden_dim=16,
                layers=1,
                heads=4,
                decoder_mode="direct_vle",
            )
        )
        batch = VLEBatch(
            molecules=torch.randn(2, 3, 4),
            temperature_k=torch.tensor([[330.0], [350.0]]),
            pressure_kpa=torch.tensor([[80.0], [101.325]]),
            x=torch.tensor([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]),
            y=torch.tensor([[0.5, 0.5, 0.0], [0.3, 0.2, 0.5]]),
            mask=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
            quality_weight=torch.ones(2, 1),
            experiment_mode=torch.tensor([0, 1]),
            pure_property_parameters=torch.zeros(
                2, 3, CORRELATION_PARAMETER_COUNT
            ),
        )

        objective = direct_vle_objective(model, batch)
        objective.total.backward()

        self.assertTrue(torch.isfinite(objective.total))
        self.assertGreater(objective.pressure.item(), 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
