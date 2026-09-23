import unittest

import numpy as np
import torch

from src.data import VLESample
from src.evaluation.thermodynamic_consistency import (
    _batch,
    _pure_limit_errors,
    _pure_reference_correlations,
    combined_nonphysical_rate,
    composition_closure_metrics,
    evaluate_thermodynamic_consistency,
    gibbs_duhem_residuals,
    phase_curve_smoothness,
    simplex_tangent_directions,
)
from src.model import ModelOutputs
from src.model import ThermoFormer, ThermoFormerConfig


class IdealActivityModel(torch.nn.Module):
    def forward(self, molecules, temperature_k, pressure_kpa, x, mask):
        log_psat = torch.log(torch.full_like(x, 100.0))
        return ModelOutputs(log_gamma=torch.zeros_like(x), log_psat=log_psat)


class ThermodynamicConsistencyTests(unittest.TestCase):
    @staticmethod
    def _sample(temperature, pressure, x=(1.0, 0.0)) -> VLESample:
        return VLESample(
            smiles=("A", "B"), names=("A", "B"), temperature_k=temperature,
            pressure_kpa=pressure, liquid_composition=x, vapor_composition=x,
            quality_weight=1.0, quality_status="passed", source="synthetic.xlsx",
            doi="test", experiment_mode="full_state", experiment_mode_confidence=1.0,
        )

    def test_direct_pure_limit_uses_independent_endpoint_reference(self) -> None:
        anchors = [self._sample(300.0, 50.0), self._sample(400.0, 200.0)]
        representative = self._sample(350.0, 100.0, (0.5, 0.5))
        features = {"A": np.zeros(4, dtype=np.float32), "B": np.ones(4, dtype=np.float32)}
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=4, hidden_dim=16, layers=1, heads=4,
                decoder_mode="direct_vle",
            )
        ).eval()
        batch = _batch([representative], features, torch.device("cpu"))

        errors = _pure_limit_errors(
            model, batch, True, 2, [representative],
            _pure_reference_correlations(anchors),
        )

        self.assertTrue(np.isfinite(errors["pressure_relative"][0]))
        self.assertGreater(errors["pressure_relative"][0], 0.0)

    def test_direct_physical_rate_is_unavailable_without_pure_references(self) -> None:
        sample = self._sample(350.0, 100.0, (0.5, 0.5))
        features = {"A": np.zeros(4, dtype=np.float32), "B": np.ones(4, dtype=np.float32)}
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=4, hidden_dim=16, layers=1, heads=4,
                decoder_mode="direct_vle",
            )
        ).eval()

        metrics = evaluate_thermodynamic_consistency(
            model, [sample], features, torch.device("cpu"), solver_iterations=2,
            grid_points=5, max_systems=1, pure_reference_samples=[],
        )

        self.assertEqual(metrics["pure_limit_reference_coverage"], 0.0)
        self.assertIsNone(metrics["pure_limit_failure_rate"])
        self.assertIsNone(metrics["nonphysical_prediction_rate"])

    def test_ternary_gibbs_duhem_uses_only_simplex_tangent_directions(self) -> None:
        directions = simplex_tangent_directions(3)

        torch.testing.assert_close(directions.sum(-1), torch.zeros(2))
        gram = directions @ directions.T
        torch.testing.assert_close(gram, torch.eye(2), atol=1e-6, rtol=1e-6)

    def test_ideal_activity_has_zero_autograd_gibbs_duhem_residual(self) -> None:
        residuals = gibbs_duhem_residuals(
            IdealActivityModel(),
            molecules=torch.zeros(1, 3, 2),
            temperature_k=torch.tensor([[350.0]]),
            pressure_kpa=torch.tensor([[101.325]]),
            x=torch.tensor([[0.2, 0.3, 0.5]]),
            mask=torch.ones(1, 3),
        )

        torch.testing.assert_close(residuals, torch.zeros(2))

    def test_batched_gibbs_duhem_matches_individual_states(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(feature_dim=4, hidden_dim=16, layers=1, heads=4)
        )
        molecules = torch.randn(2, 3, 4)
        temperature = torch.tensor([[330.0], [370.0]])
        pressure = torch.tensor([[90.0], [120.0]])
        x = torch.tensor([[0.3, 0.7, 0.0], [0.2, 0.3, 0.5]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        batched = gibbs_duhem_residuals(
            model, molecules, temperature, pressure, x, mask
        )
        individual = torch.cat(
            [
                gibbs_duhem_residuals(
                    model, molecules[i : i + 1], temperature[i : i + 1],
                    pressure[i : i + 1], x[i : i + 1], mask[i : i + 1],
                )
                for i in range(2)
            ]
        )

        torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)

    def test_composition_closure_reports_bounds_and_both_phase_sums(self) -> None:
        metrics = composition_closure_metrics(
            x=torch.tensor([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]),
            y=torch.tensor([[0.5, 0.5, 0.0], [-0.1, 0.4, 0.7]]),
            mask=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        )

        self.assertAlmostEqual(metrics["x_closure_mean_abs"], 0.0)
        self.assertAlmostEqual(metrics["y_closure_mean_abs"], 0.0)
        self.assertAlmostEqual(metrics["fraction_y_below_zero"], 1.0 / 5.0)
        self.assertAlmostEqual(metrics["fraction_y_above_one"], 0.0)

    def test_phase_smoothness_does_not_penalize_a_smooth_nonmonotonic_curve(self) -> None:
        coordinate = np.linspace(0.0, 1.0, 101)
        values = np.sin(2.0 * np.pi * coordinate)[:, None]

        metrics = phase_curve_smoothness(coordinate, values)

        self.assertGreater(metrics["total_variation"], 0.0)
        self.assertLess(metrics["first_derivative_jump_p95"], 1.0)
        self.assertLess(metrics["second_derivative_magnitude_mean"], 50.0)

    def test_nonphysical_rate_includes_near_pure_predictions(self) -> None:
        rate = combined_nonphysical_rate(
            ordinary_failures=1,
            ordinary_predictions=9,
            pure_limit_errors=[0.01, 0.2],
            pure_limit_threshold=0.05,
        )

        self.assertAlmostEqual(rate, 2.0 / 11.0)

    def test_integrated_evaluator_reports_accuracy_independent_physical_metrics(self) -> None:
        sample = VLESample(
            smiles=("A", "B"),
            names=("A", "B"),
            temperature_k=350.0,
            pressure_kpa=101.325,
            liquid_composition=(0.4, 0.6),
            vapor_composition=(0.5, 0.5),
            quality_weight=1.0,
            quality_status="passed",
            source="synthetic.xlsx",
            doi="test",
            experiment_mode="full_state",
            experiment_mode_confidence=1.0,
        )
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=4,
                hidden_dim=16,
                layers=1,
                heads=4,
                activity_mode="ideal",
            )
        )

        metrics = evaluate_thermodynamic_consistency(
            model,
            [sample],
            {"A": np.zeros(4, dtype=np.float32), "B": np.ones(4, dtype=np.float32)},
            torch.device("cpu"),
            solver_iterations=4,
            grid_points=5,
            max_systems=1,
        )

        self.assertEqual(metrics["evaluated_systems"], 1)
        self.assertAlmostEqual(metrics["gibbs_duhem_mean_abs"], 0.0, places=6)
        self.assertIn("permutation_y_max_abs", metrics)
        self.assertIn("permutation_pressure_kpa_max_abs", metrics)
        self.assertIn("permutation_temperature_k_max_abs", metrics)
        self.assertIn("pure_limit_pressure_relative_mean_abs", metrics)
        self.assertIn("pure_limit_temperature_k_mean_abs", metrics)
        self.assertIn("equilibrium_residual_p95_abs_kpa", metrics)
        self.assertIn("nonphysical_prediction_rate", metrics)


if __name__ == "__main__":
    unittest.main()
