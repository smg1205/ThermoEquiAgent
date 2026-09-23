import unittest

import pandas as pd

from src.thermoformer.baselines.phase_diagrams import (
    candidate_metrics,
    select_extreme_cases,
)


class PhaseDiagramTests(unittest.TestCase):
    def test_curve_score_is_zero_for_exact_prediction(self) -> None:
        frame = pd.DataFrame(
            {
                "target_pressure_kpa": [50.0, 60.0, 70.0],
                "y_true_1": [0.2, 0.5, 0.8],
                **{
                    f"{model}_{field}": values
                    for model in ("thermoformer", "nrtl", "wilson", "uniquac")
                    for field, values in (
                        ("state_mean", [50.0, 60.0, 70.0]),
                        ("y_mean", [0.2, 0.5, 0.8]),
                    )
                },
            }
        )
        metrics = candidate_metrics(
            frame,
            direction="isothermal",
            pressure_range_floor_kpa=5.0,
            temperature_range_floor_k=5.0,
        )
        self.assertEqual(metrics["thermoformer_curve_score"], 0.0)
        self.assertEqual(metrics["thermoformer_state_mae"], 0.0)

    def test_selection_keeps_one_global_high_error_case(self) -> None:
        candidates = []
        scores = {
            "isothermal": (0.1, 0.2, 0.7),
            "isobaric": (0.05, 0.4, 0.9),
        }
        for direction, values in scores.items():
            for index, score in enumerate(values):
                candidates.append(
                    {
                        "case_id": f"{direction}-{index}",
                        "direction": direction,
                        "system_id": f"{direction}-system-{index}",
                        "thermoformer_curve_score": score,
                    }
                )
        selected = select_extreme_cases(candidates)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [row["selection_category"] for row in selected],
            ["well_predicted", "well_predicted", "representative", "high_error"],
        )
        self.assertEqual(sum(row["selection_category"] == "high_error" for row in selected), 1)
        self.assertEqual(selected[0]["thermoformer_curve_score"], 0.1)
        self.assertEqual(selected[1]["thermoformer_curve_score"], 0.05)
        self.assertEqual(selected[-1]["thermoformer_curve_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
