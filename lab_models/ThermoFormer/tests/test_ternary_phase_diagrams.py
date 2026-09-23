import unittest

import numpy as np
import pandas as pd

from src.thermoformer.baselines.ternary_phase_diagrams import (
    EXPERIMENT_VAPOR_ZORDER,
    SQRT3_OVER_2,
    THERMOFORMER_ZORDER,
    barycentric_to_cartesian,
    select_ternary_extremes,
    ternary_candidate_metrics,
    ternary_model_marker_style,
)


class TernaryPhaseDiagramTests(unittest.TestCase):
    def test_thermoformer_marker_is_hollow_and_topmost(self) -> None:
        style = ternary_model_marker_style("thermoformer")
        self.assertEqual(style["facecolor"], "none")
        self.assertEqual(style["zorder"], THERMOFORMER_ZORDER)
        self.assertGreater(THERMOFORMER_ZORDER, EXPERIMENT_VAPOR_ZORDER)
        self.assertGreater(
            THERMOFORMER_ZORDER,
            int(ternary_model_marker_style("nrtl")["zorder"]),
        )

    def test_barycentric_vertices_map_to_equilateral_triangle(self) -> None:
        mapped = barycentric_to_cartesian(np.eye(3))
        np.testing.assert_allclose(
            mapped,
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3_OVER_2]]),
        )

    def test_curve_score_uses_all_three_vapor_components(self) -> None:
        payload = {
            "target_pressure_kpa": [50.0, 60.0],
            "y_true_1": [0.2, 0.3],
            "y_true_2": [0.3, 0.4],
            "y_true_3": [0.5, 0.3],
        }
        for model in ("thermoformer", "nrtl", "wilson", "uniquac"):
            payload[f"{model}_state_mean"] = [50.0, 60.0]
            payload[f"{model}_y_mean_1"] = [0.2, 0.3]
            payload[f"{model}_y_mean_2"] = [0.3, 0.4]
            payload[f"{model}_y_mean_3"] = [0.5, 0.3]
        metrics = ternary_candidate_metrics(
            pd.DataFrame(payload),
            direction="isothermal",
            pressure_range_floor_kpa=5.0,
            temperature_range_floor_k=5.0,
        )
        self.assertEqual(metrics["thermoformer_curve_score"], 0.0)
        self.assertEqual(metrics["thermoformer_y_mae"], 0.0)

    def test_selection_keeps_one_global_high_error_case(self) -> None:
        protocol_scores = {
            "state_temperature_high_extrapolation": (0.1, 0.35, 0.7),
            "state_temperature_low_extrapolation": (0.08, 0.45, 0.9),
        }
        candidates = [
            {
                "case_id": f"{protocol}-{index}",
                "protocol": protocol,
                "direction": "isothermal",
                "system_id": f"{protocol}-system-{index}",
                "thermoformer_curve_score": score,
            }
            for protocol, scores in protocol_scores.items()
            for index, score in enumerate(scores)
        ]
        selected = select_ternary_extremes(candidates)
        self.assertEqual(
            [row["selection_category"] for row in selected],
            [
                "well_predicted",
                "well_predicted",
                "representative",
                "high_error",
            ],
        )
        self.assertEqual(sum(row["selection_category"] == "high_error" for row in selected), 1)
        self.assertEqual(selected[0]["thermoformer_curve_score"], 0.1)
        self.assertEqual(selected[1]["thermoformer_curve_score"], 0.08)
        self.assertEqual(selected[-1]["thermoformer_curve_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
