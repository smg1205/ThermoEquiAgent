import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.thermoformer.data import VLESample
from src.thermoformer.baselines.overall_phase_diagrams import (
    OverallPhaseDiagramSettings,
    overall_candidate_metrics,
    select_six_representative_cases,
)
from src.thermoformer.baselines.phasepy_adapter import (
    fit_phasepy_activity_model,
    predict_phasepy_activity_model,
)
from src.thermoformer.baselines.external_vapor_pressure import (
    validated_external_vapor_pressures,
)


class OverallPhaseDiagramTests(unittest.TestCase):
    def test_phasepy_backend_fits_and_predicts_with_validated_psat(self) -> None:
        correlations, _ = validated_external_vapor_pressures(
            {"CCO": "ethanol", "O": "water"},
            {"CCO": (330.0, 370.0), "O": (330.0, 370.0)},
        )
        samples = []
        for temperature, first_fraction in ((340.0, 0.3), (350.0, 0.7)):
            x = np.asarray([first_fraction, 1.0 - first_fraction])
            psat = np.asarray(
                [correlations[value].pressure_kpa(temperature) for value in ("CCO", "O")]
            )
            pressure = float(np.dot(x, psat))
            y = x * psat / pressure
            samples.append(
                VLESample(
                    smiles=("CCO", "O"),
                    names=("ethanol", "water"),
                    temperature_k=temperature,
                    pressure_kpa=pressure,
                    liquid_composition=tuple(x),
                    vapor_composition=tuple(y),
                    quality_weight=1.0,
                    quality_status="passed",
                    source="synthetic",
                    doi="",
                    experiment_mode="isothermal",
                    experiment_mode_confidence=1.0,
                )
            )
        fitted = fit_phasepy_activity_model(
            "nrtl", samples, correlations, maximum_evaluations=50
        )
        self.assertTrue(fitted.success)
        self.assertLessEqual(fitted.final_training_loss, fitted.initial_training_loss + 1e-12)
        records = predict_phasepy_activity_model(
            samples, fitted, correlations, temperature_bounds_k=(330.0, 370.0)
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["converged"] for record in records))
        self.assertTrue(all(record["predicted_pressure_kpa"] > 0.0 for record in records))

    def test_external_psat_is_in_range_monotone_and_unit_audited(self) -> None:
        correlations, audit = validated_external_vapor_pressures(
            {"COC=O": "methyl formate"},
            {"COC=O": (304.7, 399.0)},
        )

        correlation = correlations["COC=O"]
        self.assertEqual(correlation.method, "DIPPR_PERRY_8E")
        self.assertGreater(correlation.pressure_kpa(399.0), correlation.pressure_kpa(304.7))
        record = audit["entries"]["COC=O"]
        self.assertEqual(record["native_pressure_unit"], "Pa")
        self.assertEqual(record["output_pressure_unit"], "kPa")
        self.assertTrue(record["monotonic_in_required_range"])
        self.assertGreater(record["minimum_dpsat_dt_kpa_per_k"], 0.0)

    def test_external_psat_marks_out_of_range_component_unavailable(self) -> None:
        correlations, audit = validated_external_vapor_pressures(
            {"COC=O": "methyl formate"},
            {"COC=O": (304.7, 600.0)},
        )

        self.assertNotIn("COC=O", correlations)
        record = audit["entries"]["COC=O"]
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["reason"], "no_valid_monotone_external_correlation")

    def test_candidate_metrics_use_all_composition_components(self) -> None:
        frame = pd.DataFrame(
            {
                "target_pressure_kpa": [10.0, 20.0],
                "predicted_pressure_kpa": [11.0, 18.0],
                "y_true_1": [0.2, 0.7],
                "y_true_2": [0.8, 0.3],
                "y_pred_1": [0.3, 0.6],
                "y_pred_2": [0.7, 0.4],
            }
        )
        metrics = overall_candidate_metrics(
            frame,
            direction="isothermal",
            component_count=2,
            pressure_range_floor_kpa=5.0,
            temperature_range_floor_k=5.0,
        )
        self.assertAlmostEqual(metrics["thermoformer_state_mae"], 1.5)
        self.assertAlmostEqual(metrics["thermoformer_y_mae"], 0.1)
        self.assertGreater(metrics["thermoformer_curve_score"], 0.0)

    def test_six_case_selection_contains_only_one_high_error_case(self) -> None:
        candidates = [
            {"case_id": "a", "system_id": "a", "direction": "isothermal", "thermoformer_curve_score": 0.1},
            {"case_id": "b", "system_id": "b", "direction": "isobaric", "thermoformer_curve_score": 0.2},
            {"case_id": "c", "system_id": "c", "direction": "isothermal", "thermoformer_curve_score": 0.3},
            {"case_id": "d", "system_id": "d", "direction": "isobaric", "thermoformer_curve_score": 0.9},
            {"case_id": "e", "system_id": "e", "direction": "isothermal", "thermoformer_curve_score": 0.4},
            {"case_id": "f", "system_id": "f", "direction": "isobaric", "thermoformer_curve_score": 0.25},
            {"case_id": "g", "system_id": "g", "direction": "isothermal", "thermoformer_curve_score": 0.5},
        ]
        selected = select_six_representative_cases(candidates)
        categories = [row["selection_category"] for row in selected]
        self.assertEqual(len(selected), 6)
        self.assertEqual(categories.count("high_error"), 1)
        self.assertEqual(categories.count("well_predicted"), 4)
        self.assertEqual(categories.count("representative"), 1)

    def test_config_is_frozen_to_seed_zero(self) -> None:
        payload = """{
          "seed": 1,
          "minimum_points": 5,
          "pressure_range_floor_kpa": 5.0,
          "temperature_range_floor_k": 5.0,
          "binary_candidate_output": "a",
          "binary_plot_data_output": "b",
          "binary_figure_stem": "c",
          "ternary_candidate_output": "d",
          "ternary_plot_data_output": "e",
          "ternary_figure_stem": "f",
          "combined_figure_stem": "combined",
          "classical_fit_audit_output": "audit.json",
          "report_output": "g",
          "manifest_output": "h"
        }"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed 0"):
                OverallPhaseDiagramSettings.load(path)


if __name__ == "__main__":
    unittest.main()
