import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.thermoformer.baselines.thermodynamic_models import (
    ActivityModelParameters,
    LogLinearVaporPressure,
    activity_coefficients,
    solve_isobaric_state,
    solve_isothermal_state,
)
from src.thermoformer.baselines.thermodynamic_runner import BaselineCampaignSettings
from src.thermoformer.baselines.thermodynamic_fitting import (
    FittedActivityModel,
    fit_activity_model,
    fit_shared_activity_model,
    fit_vapor_pressure_correlations,
)
from src.thermoformer.baselines.thermodynamic_evaluation import (
    predict_fitted_activity_model,
)
from src.thermoformer.evaluation import prediction_metric_rows
from src.thermoformer.data import VLESample


class ThermodynamicBaselineTests(unittest.TestCase):
    @staticmethod
    def _sample(
        temperature: float,
        pressure: float,
        x: tuple[float, float],
        y: tuple[float, float],
    ) -> VLESample:
        return VLESample(
            smiles=("CCO", "O"),
            names=("ethanol", "water"),
            temperature_k=temperature,
            pressure_kpa=pressure,
            liquid_composition=x,
            vapor_composition=y,
            quality_weight=1.0,
            quality_status="passed",
            source="synthetic.xlsx",
            doi="synthetic",
            experiment_mode="full_state",
            experiment_mode_confidence=1.0,
        )

    def test_ideal_parameters_recover_raoult_bubble_state(self) -> None:
        vapor_pressure = (
            LogLinearVaporPressure(intercept=math.log(100.0), inverse_temperature=0.0),
            LogLinearVaporPressure(intercept=math.log(50.0), inverse_temperature=0.0),
        )
        parameters = ActivityModelParameters(
            model="nrtl",
            interaction_b=np.zeros((2, 2)),
        )

        state = solve_isothermal_state(
            temperature_k=350.0,
            liquid_composition=np.asarray([0.25, 0.75]),
            vapor_pressure=vapor_pressure,
            parameters=parameters,
        )

        self.assertTrue(state.converged)
        self.assertAlmostEqual(state.pressure_kpa, 62.5, places=8)
        np.testing.assert_allclose(state.vapor_composition, [0.4, 0.6], atol=1e-8)

    def test_isobaric_solver_does_not_require_observed_temperature(self) -> None:
        vapor_pressure = (
            LogLinearVaporPressure(intercept=12.0, inverse_temperature=-2400.0),
            LogLinearVaporPressure(intercept=11.7, inverse_temperature=-2200.0),
        )
        parameters = ActivityModelParameters(
            model="wilson",
            interaction_b=np.zeros((2, 2)),
        )
        reference = solve_isothermal_state(
            temperature_k=360.0,
            liquid_composition=np.asarray([0.4, 0.6]),
            vapor_pressure=vapor_pressure,
            parameters=parameters,
        )

        solved = solve_isobaric_state(
            pressure_kpa=reference.pressure_kpa,
            liquid_composition=np.asarray([0.4, 0.6]),
            vapor_pressure=vapor_pressure,
            parameters=parameters,
            temperature_bounds_k=(200.0, 700.0),
        )

        self.assertTrue(solved.converged)
        self.assertAlmostEqual(solved.temperature_k, 360.0, places=5)
        np.testing.assert_allclose(
            solved.vapor_composition, reference.vapor_composition, atol=1e-7
        )

    def test_activity_models_return_positive_finite_coefficients(self) -> None:
        composition = np.asarray([0.2, 0.3, 0.5])
        interaction = np.asarray(
            [[0.0, 180.0, -90.0], [-120.0, 0.0, 75.0], [40.0, -60.0, 0.0]]
        )
        for model in ("nrtl", "wilson", "uniquac"):
            parameters = ActivityModelParameters(
                model=model,
                interaction_b=interaction,
                uniquac_r=np.asarray([0.92, 2.1055, 3.1878]) if model == "uniquac" else None,
                uniquac_q=np.asarray([1.4, 1.972, 2.4]) if model == "uniquac" else None,
            )
            gamma = activity_coefficients(350.0, composition, parameters)
            self.assertTrue(np.isfinite(gamma).all(), model)
            self.assertTrue((gamma > 0.0).all(), model)

    def test_training_endpoint_calibration_recovers_log_linear_psat(self) -> None:
        true_models = (
            LogLinearVaporPressure(12.0, -2400.0),
            LogLinearVaporPressure(11.7, -2200.0),
        )
        samples = []
        for temperature in (320.0, 350.0, 380.0):
            samples.append(
                self._sample(
                    temperature,
                    true_models[0].pressure_kpa(temperature),
                    (1.0, 0.0),
                    (1.0, 0.0),
                )
            )
            samples.append(
                self._sample(
                    temperature,
                    true_models[1].pressure_kpa(temperature),
                    (0.0, 1.0),
                    (0.0, 1.0),
                )
            )

        fitted, audit = fit_vapor_pressure_correlations(samples)

        self.assertEqual(audit["covered_components"], 2)
        for smiles, expected in zip(("CCO", "O"), true_models):
            self.assertAlmostEqual(fitted[smiles].intercept, expected.intercept, places=8)
            self.assertAlmostEqual(
                fitted[smiles].inverse_temperature,
                expected.inverse_temperature,
                places=6,
            )

    def test_nrtl_fit_uses_training_states_to_recover_unseen_state(self) -> None:
        vapor_pressure = {
            "CCO": LogLinearVaporPressure(12.0, -2400.0),
            "O": LogLinearVaporPressure(11.7, -2200.0),
        }
        true_parameters = ActivityModelParameters(
            model="nrtl",
            interaction_b=np.asarray([[0.0, 420.0], [-180.0, 0.0]]),
        )
        samples = []
        for temperature in (330.0, 360.0, 390.0):
            for x1 in (0.15, 0.35, 0.65, 0.85):
                state = solve_isothermal_state(
                    temperature_k=temperature,
                    liquid_composition=np.asarray([x1, 1.0 - x1]),
                    vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
                    parameters=true_parameters,
                )
                samples.append(
                    self._sample(
                        temperature,
                        state.pressure_kpa,
                        (x1, 1.0 - x1),
                        tuple(state.vapor_composition),
                    )
                )

        fitted = fit_activity_model("nrtl", samples, vapor_pressure)
        predicted = solve_isothermal_state(
            temperature_k=375.0,
            liquid_composition=np.asarray([0.45, 0.55]),
            vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
            parameters=fitted.parameters,
        )
        expected = solve_isothermal_state(
            temperature_k=375.0,
            liquid_composition=np.asarray([0.45, 0.55]),
            vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
            parameters=true_parameters,
        )

        self.assertTrue(fitted.success)
        self.assertAlmostEqual(predicted.pressure_kpa, expected.pressure_kpa, places=3)
        np.testing.assert_allclose(
            predicted.vapor_composition, expected.vapor_composition, atol=1e-6
        )

    def test_prediction_records_use_the_shared_four_task_metric_schema(self) -> None:
        vapor_pressure = {
            "CCO": LogLinearVaporPressure(12.0, -2400.0),
            "O": LogLinearVaporPressure(11.7, -2200.0),
        }
        parameters = ActivityModelParameters(
            model="nrtl",
            interaction_b=np.asarray([[0.0, 200.0], [-100.0, 0.0]]),
        )
        reference = solve_isothermal_state(
            temperature_k=360.0,
            liquid_composition=np.asarray([0.4, 0.6]),
            vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
            parameters=parameters,
        )
        sample = self._sample(
            360.0,
            reference.pressure_kpa,
            (0.4, 0.6),
            tuple(reference.vapor_composition),
        )
        fitted = FittedActivityModel(
            model="nrtl",
            components=("CCO", "O"),
            parameters=parameters,
            success=True,
            training_samples=3,
            fitted_component_observations=6,
            rmse_log_gamma=0.0,
            optimizer_status=0,
            optimizer_message="synthetic",
        )

        records = predict_fitted_activity_model(
            [sample], fitted, vapor_pressure, temperature_bounds_k=(200.0, 700.0)
        )
        metrics = prediction_metric_rows(records)
        directions = {
            row["direction"]: row
            for row in metrics
            if row["scope"] == "direction"
        }

        self.assertEqual({"isothermal", "isobaric"}, set(directions))
        self.assertAlmostEqual(directions["isothermal"]["pressure_mae_kpa"], 0.0, places=6)
        self.assertAlmostEqual(directions["isobaric"]["temperature_mae_k"], 0.0, places=5)
        self.assertAlmostEqual(directions["isothermal"]["y_mae"], 0.0, places=6)
        self.assertAlmostEqual(directions["isobaric"]["y_mae"], 0.0, places=6)

    def test_shared_pair_fit_optimizes_direct_vle_residuals(self) -> None:
        vapor_pressure = {
            "CCO": LogLinearVaporPressure(12.0, -2400.0),
            "O": LogLinearVaporPressure(11.7, -2200.0),
        }
        true_parameters = ActivityModelParameters(
            model="nrtl",
            interaction_b=np.asarray([[0.0, 360.0], [-140.0, 0.0]]),
        )
        samples = []
        for temperature in (330.0, 360.0, 390.0):
            for x1 in (0.2, 0.4, 0.6, 0.8):
                state = solve_isothermal_state(
                    temperature_k=temperature,
                    liquid_composition=np.asarray([x1, 1.0 - x1]),
                    vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
                    parameters=true_parameters,
                )
                samples.append(
                    self._sample(
                        temperature,
                        state.pressure_kpa,
                        (x1, 1.0 - x1),
                        tuple(state.vapor_composition),
                    )
                )

        shared = fit_shared_activity_model(
            "nrtl", samples, vapor_pressure, maximum_iterations=80
        )
        fitted = shared.for_components(("CCO", "O"))
        self.assertIsNotNone(fitted)
        predicted = solve_isothermal_state(
            temperature_k=375.0,
            liquid_composition=np.asarray([0.45, 0.55]),
            vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
            parameters=fitted.parameters,
        )
        expected = solve_isothermal_state(
            temperature_k=375.0,
            liquid_composition=np.asarray([0.45, 0.55]),
            vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
            parameters=true_parameters,
        )

        self.assertTrue(shared.success)
        self.assertLess(shared.final_training_loss, shared.initial_training_loss)
        self.assertAlmostEqual(predicted.pressure_kpa, expected.pressure_kpa, places=2)

    def test_campaign_settings_reject_nonexecutable_scientific_forms(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = (
            project_root
            / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["activity_parameter_temperature_form"] = "constant"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "temperature form"):
                BaselineCampaignSettings.load(path)

    def test_shared_uniquac_fit_accepts_audited_size_overrides(self) -> None:
        vapor_pressure = {
            "CCO": LogLinearVaporPressure(12.0, -2400.0),
            "O": LogLinearVaporPressure(11.7, -2200.0),
        }
        sizes = {"CCO": (2.1, 1.9), "O": (1.0, 1.4)}
        truth = ActivityModelParameters(
            model="uniquac",
            interaction_b=np.asarray([[0.0, 240.0], [-90.0, 0.0]]),
            uniquac_r=np.asarray([2.1, 1.0]),
            uniquac_q=np.asarray([1.9, 1.4]),
        )
        samples = []
        for x1 in (0.2, 0.4, 0.6, 0.8):
            state = solve_isothermal_state(
                temperature_k=350.0,
                liquid_composition=(x1, 1.0 - x1),
                vapor_pressure=(vapor_pressure["CCO"], vapor_pressure["O"]),
                parameters=truth,
            )
            samples.append(
                self._sample(350.0, state.pressure_kpa, (x1, 1.0 - x1), tuple(state.vapor_composition))
            )
        shared = fit_shared_activity_model(
            "uniquac", samples, vapor_pressure, maximum_iterations=60, uniquac_sizes=sizes
        )
        fitted = shared.for_components(("CCO", "O"))
        self.assertTrue(shared.success)
        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(fitted.parameters.uniquac_r, [2.1, 1.0])
        np.testing.assert_allclose(fitted.parameters.uniquac_q, [1.9, 1.4])


if __name__ == "__main__":
    unittest.main()
