import unittest
import math

import torch
from torch import nn

from src.model import ModelOutputs
from src.pure_properties import AntoineParameters, DIPPR101Parameters
from src.thermo import (
    ConvergenceError,
    equilibrium_at_tp,
    solve_isobaric,
    solve_isothermal,
)


class ConstantIdealModel(nn.Module):
    def __init__(self, psat: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("psat", torch.tensor(psat))

    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        log_psat = torch.log(self.psat).expand_as(x)
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=log_psat)


class TemperatureIdealModel(nn.Module):
    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        log_psat = torch.log(temperature_k.expand_as(x))
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=log_psat)


class ScaledIdealModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        log_psat = self.log_scale.expand_as(x) + torch.log(torch.tensor(100.0)).expand_as(x)
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=log_psat)


class ScaledTemperatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        log_psat = self.log_scale.expand_as(x) + torch.log(temperature_k.expand_as(x))
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=log_psat)


class ExtremeLogModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_psat = nn.Parameter(torch.tensor(1000.0))

    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        return ModelOutputs(
            log_gamma=torch.zeros_like(x),
            log_psat=self.log_psat.expand_as(x),
        )


class DifferentiableSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.molecules = torch.zeros(1, 2, 1)
        self.x = torch.tensor([[0.4, 0.6]])
        self.mask = torch.ones_like(self.x)

    def test_isothermal_solver_matches_ideal_raoult_law(self) -> None:
        state = solve_isothermal(
            ConstantIdealModel((100.0, 50.0)),
            self.molecules,
            torch.tensor([[350.0]]),
            self.x,
            self.mask,
        )

        torch.testing.assert_close(state.pressure_kpa, torch.tensor([[70.0]]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state.y, torch.tensor([[4.0 / 7.0, 3.0 / 7.0]]), atol=1e-4, rtol=1e-4)
        self.assertTrue(bool(state.converged.all()))

    def test_reliable_antoine_correlation_overrides_learned_psat(self) -> None:
        water = torch.from_numpy(
            AntoineParameters(8.07131, 1730.63, 233.426, 274.15, 373.15).as_array()
        )
        parameters = torch.stack((water, torch.zeros_like(water))).unsqueeze(0)
        state = equilibrium_at_tp(
            ConstantIdealModel((1.0, 1.0)),
            self.molecules,
            torch.tensor([[373.15]]),
            torch.tensor([[101.325]]),
            torch.tensor([[1.0, 0.0]]),
            self.mask,
            pure_property_parameters=parameters,
        )

        self.assertAlmostEqual(state.psat_kpa[0, 0].item(), 101.336, places=2)

    def test_antoine_correlation_outside_valid_range_falls_back_to_model(self) -> None:
        water = torch.from_numpy(
            AntoineParameters(8.07131, 1730.63, 233.426, 274.15, 373.15).as_array()
        )
        parameters = torch.stack((water, torch.zeros_like(water))).unsqueeze(0)
        state = equilibrium_at_tp(
            ConstantIdealModel((1.0, 1.0)),
            self.molecules,
            torch.tensor([[400.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[1.0, 0.0]]),
            self.mask,
            pure_property_parameters=parameters,
        )

        self.assertAlmostEqual(state.psat_kpa[0, 0].item(), 1.0, places=6)

    def test_dippr101_correlation_is_dispatched_and_converted_from_pa(self) -> None:
        water = torch.from_numpy(
            DIPPR101Parameters(
                a=math.log(100_000.0),
                b=0.0,
                c=0.0,
                d=0.0,
                e=1.0,
                minimum_temperature_k=250.0,
                maximum_temperature_k=450.0,
                pressure_unit="Pa",
            ).as_array()
        )
        parameters = torch.stack((water, torch.zeros_like(water))).unsqueeze(0)

        state = equilibrium_at_tp(
            ConstantIdealModel((1.0, 1.0)),
            self.molecules,
            torch.tensor([[350.0]]),
            torch.tensor([[100.0]]),
            torch.tensor([[1.0, 0.0]]),
            self.mask,
            pure_property_parameters=parameters,
        )

        self.assertAlmostEqual(state.psat_kpa[0, 0].item(), 100.0, places=3)

    def test_isobaric_solver_recovers_bubble_temperature(self) -> None:
        state = solve_isobaric(
            TemperatureIdealModel(),
            self.molecules,
            torch.tensor([[350.0]]),
            self.x,
            self.mask,
            initial_temperature_k=torch.tensor([[330.0]]),
        )

        torch.testing.assert_close(state.temperature_k, torch.tensor([[350.0]]), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(state.y, self.x, atol=1e-4, rtol=1e-4)
        self.assertTrue(bool(state.converged.all()))

    def test_isobaric_solver_keeps_parameter_gradients(self) -> None:
        model = ScaledTemperatureModel()
        state = solve_isobaric(
            model,
            self.molecules,
            torch.tensor([[350.0]]),
            self.x,
            self.mask,
            initial_temperature_k=torch.tensor([[330.0]]),
        )
        state.temperature_k.sum().backward()

        self.assertIsNotNone(model.log_scale.grad)
        self.assertLess(model.log_scale.grad.item(), 0.0)

    def test_isobaric_solver_reports_unbracketed_failure(self) -> None:
        with self.assertRaises(ConvergenceError):
            solve_isobaric(
                ConstantIdealModel((100.0, 50.0)),
                self.molecules,
                torch.tensor([[1000.0]]),
                self.x,
                self.mask,
                iterations=4,
            )

    def test_isothermal_solver_keeps_parameter_gradients(self) -> None:
        model = ScaledIdealModel()
        state = solve_isothermal(model, self.molecules, torch.tensor([[350.0]]), self.x, self.mask)
        state.pressure_kpa.sum().backward()

        self.assertIsNotNone(model.log_scale.grad)
        self.assertGreater(abs(model.log_scale.grad.item()), 0.0)

    def test_equilibrium_state_reports_pressure_residual(self) -> None:
        state = equilibrium_at_tp(
            ConstantIdealModel((100.0, 50.0)),
            self.molecules,
            torch.tensor([[350.0]]),
            torch.tensor([[80.0]]),
            self.x,
            self.mask,
        )

        torch.testing.assert_close(state.calculated_pressure_kpa, torch.tensor([[70.0]]))
        torch.testing.assert_close(state.pressure_residual_kpa, torch.tensor([[-10.0]]))

    def test_extreme_log_pressure_does_not_create_nan_gradients(self) -> None:
        model = ExtremeLogModel()
        state = equilibrium_at_tp(
            model,
            self.molecules,
            torch.tensor([[350.0]]),
            torch.tensor([[100.0]]),
            self.x,
            self.mask,
        )
        state.calculated_pressure_kpa.sum().backward()

        self.assertTrue(torch.isfinite(state.calculated_pressure_kpa).all())
        self.assertTrue(torch.isfinite(model.log_psat.grad))


if __name__ == "__main__":
    unittest.main()
