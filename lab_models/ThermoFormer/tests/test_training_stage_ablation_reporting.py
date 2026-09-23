"""Tests for the public cumulative-stage reporting summaries."""

from __future__ import annotations

import unittest


def _directional(seed: int, offset: float) -> dict[str, dict[str, float]]:
    return {
        "isothermal": {
            "pressure_mae_kpa": 10.0 + seed + offset,
            "pressure_rmse_kpa": 20.0 + seed + offset,
            "pressure_r2": 0.90 - offset / 100.0,
            "y_mae": 0.05 + offset / 100.0,
            "y_rmse": 0.08 + offset / 100.0,
            "y_r2": 0.80 - offset / 100.0,
            "valid_coverage": 1.0,
            "solver_failure_rate": 0.0,
            "nonphysical_rate": 0.0,
        },
        "isobaric": {
            "temperature_mae_k": 3.0 + seed + offset,
            "temperature_rmse_k": 4.0 + seed + offset,
            "temperature_r2": 0.90 - offset / 100.0,
            "y_mae": 0.04 + offset / 100.0,
            "y_rmse": 0.07 + offset / 100.0,
            "y_r2": 0.81 - offset / 100.0,
            "valid_coverage": 1.0,
            "solver_failure_rate": 0.0,
            "nonphysical_rate": 0.0,
        },
    }


class TrainingStageAblationReportingTests(unittest.TestCase):
    def test_stage_summary_and_paired_deltas_keep_five_seed_pairing(self) -> None:
        from src.thermoformer.reporting.training_stage_ablation import (
            build_stage_summary,
            paired_stage_deltas,
        )

        payload = {
            "stage0": {seed: _directional(seed, 1.0) for seed in range(5)},
            "stage1": {seed: _directional(seed, 0.0) for seed in range(5)},
        }
        summaries = build_stage_summary(payload)
        iso_stage1 = next(
            row
            for row in summaries
            if row["stage"] == "stage1" and row["direction"] == "isothermal"
        )
        self.assertEqual(iso_stage1["available_seeds"], 5)
        self.assertAlmostEqual(iso_stage1["state_mae_mean"], 12.0)
        self.assertAlmostEqual(iso_stage1["y_mae_mean"], 0.05)

        deltas = paired_stage_deltas(payload)
        iso = next(
            row
            for row in deltas
            if row["from_stage"] == "stage0"
            and row["to_stage"] == "stage1"
            and row["direction"] == "isothermal"
            and row["metric"] == "state_mae"
        )
        self.assertEqual(iso["available_seeds"], 5)
        self.assertAlmostEqual(iso["mean_delta"], -1.0)
        self.assertEqual(iso["improved_seed_count"], 5)

        iso_rmse = next(
            row
            for row in deltas
            if row["from_stage"] == "stage0"
            and row["to_stage"] == "stage1"
            and row["direction"] == "isothermal"
            and row["metric"] == "state_rmse"
        )
        self.assertAlmostEqual(iso_rmse["mean_delta"], -1.0)
        iso_r2 = next(
            row
            for row in deltas
            if row["from_stage"] == "stage0"
            and row["to_stage"] == "stage1"
            and row["direction"] == "isothermal"
            and row["metric"] == "state_r2"
        )
        self.assertAlmostEqual(iso_r2["mean_delta"], 0.01)
        self.assertEqual(iso_r2["improved_seed_count"], 5)


if __name__ == "__main__":
    unittest.main()
