import unittest

import numpy as np
import torch

from src.data import VLESample
from src.evaluation import predict_vle, prediction_metric_rows
from src.model import ThermoFormer, ThermoFormerConfig


def prediction(
    sample,
    system,
    true_primary,
    predicted_primary,
    true_y,
    predicted_y,
    components=("mol_a", "mol_b"),
    cardinality=2,
    direction="isothermal",
    converged=True,
    nonphysical=False,
):
    row = {
        "sample_id": sample,
        "system_id": system,
        "component_count": cardinality,
        "direction": direction,
        "converged": converged,
        "nonphysical": nonphysical,
        "target_pressure_kpa": true_primary if direction == "isothermal" else None,
        "predicted_pressure_kpa": predicted_primary if direction == "isothermal" else None,
        "target_temperature_k": true_primary if direction == "isobaric" else None,
        "predicted_temperature_k": predicted_primary if direction == "isobaric" else None,
    }
    for index in range(3):
        present = index < len(components)
        row[f"component_id_{index + 1}"] = components[index] if present else None
        row[f"y_true_{index + 1}"] = true_y[index] if present else None
        row[f"y_pred_{index + 1}"] = predicted_y[index] if present else None
    return row


class PaperMetricTests(unittest.TestCase):
    def test_reports_point_system_and_component_macro_errors(self) -> None:
        rows = [
            prediction("a1", "A", 100.0, 101.0, (0.2, 0.8), (0.3, 0.7)),
            prediction("a2", "A", 100.0, 101.0, (0.4, 0.6), (0.5, 0.5)),
            prediction("a3", "A", 100.0, 101.0, (0.6, 0.4), (0.7, 0.3)),
            prediction("b1", "B", 100.0, 109.0, (0.5, 0.5), (0.5, 0.5)),
        ]

        result = prediction_metric_rows(rows)
        overall = next(row for row in result if row["scope"] == "all")

        self.assertAlmostEqual(overall["pressure_mae_kpa"], 3.0)
        self.assertAlmostEqual(overall["pressure_system_macro_mae_kpa"], 5.0)
        self.assertAlmostEqual(overall["y_mae"], 0.075)
        self.assertAlmostEqual(overall["y_system_macro_mae"], 0.05)
        self.assertAlmostEqual(overall["y_component_macro_mae"], 0.075)
        self.assertEqual(overall["attempted_samples"], 4)
        self.assertEqual(overall["valid_samples"], 4)

    def test_failed_or_nonphysical_rows_are_counted_but_excluded_from_errors(self) -> None:
        rows = [
            prediction("ok", "A", 100.0, 101.0, (0.5, 0.5), (0.6, 0.4)),
            prediction(
                "failed",
                "B",
                100.0,
                1000.0,
                (0.5, 0.5),
                (1.2, -0.2),
                converged=False,
                nonphysical=True,
            ),
        ]

        overall = next(row for row in prediction_metric_rows(rows) if row["scope"] == "all")

        self.assertEqual(overall["attempted_samples"], 2)
        self.assertEqual(overall["valid_samples"], 1)
        self.assertAlmostEqual(overall["valid_coverage"], 0.5)
        self.assertAlmostEqual(overall["solver_failure_rate"], 0.5)
        self.assertAlmostEqual(overall["nonphysical_rate"], 0.5)
        self.assertAlmostEqual(overall["pressure_mae_kpa"], 1.0)

    def test_produces_separate_cardinality_and_direction_rows(self) -> None:
        rows = [
            prediction("b", "B", 100.0, 101.0, (0.5, 0.5), (0.6, 0.4)),
            prediction(
                "t",
                "T",
                350.0,
                352.0,
                (0.2, 0.3, 0.5),
                (0.25, 0.25, 0.5),
                components=("a", "b", "c"),
                cardinality=3,
                direction="isobaric",
            ),
        ]

        keys = {(row["scope"], row["direction"], row["component_count"]) for row in prediction_metric_rows(rows)}

        self.assertIn(("direction_cardinality", "isothermal", 2), keys)
        self.assertIn(("direction_cardinality", "isobaric", 3), keys)

    def test_reports_declared_generalization_subgroups(self) -> None:
        first = prediction("a", "A", 100.0, 101.0, (0.5, 0.5), (0.6, 0.4))
        first["binary_subsystem_coverage"] = 3
        first["strict_unseen"] = True
        second = prediction("b", "B", 100.0, 102.0, (0.5, 0.5), (0.55, 0.45))
        second["binary_subsystem_coverage"] = 1
        second["strict_unseen"] = False

        metrics = prediction_metric_rows([first, second])
        groups = {(row["scope"], row.get("subgroup")) for row in metrics}

        self.assertIn(("binary_subsystem_coverage", "3/3"), groups)
        self.assertIn(("binary_subsystem_coverage", "1/3"), groups)
        self.assertIn(("unseen_subset", "strict_all_components_unseen"), groups)

    def test_direct_vle_prediction_exports_both_tasks_without_gamma_or_psat(self) -> None:
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
        features = {"A": np.zeros(4, dtype=np.float32), "B": np.ones(4, dtype=np.float32)}
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=4,
                hidden_dim=16,
                layers=1,
                heads=4,
                decoder_mode="direct_vle",
            )
        )

        records = predict_vle(
            model,
            [sample],
            features,
            batch_size=1,
            device=torch.device("cpu"),
        )

        self.assertEqual({record["direction"] for record in records}, {"isothermal", "isobaric"})
        self.assertTrue(all(record["gamma_pred_1"] is None for record in records))
        self.assertTrue(all(record["psat_pred_kpa_1"] is None for record in records))


if __name__ == "__main__":
    unittest.main()
