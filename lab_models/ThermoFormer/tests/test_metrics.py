import unittest

import numpy as np
import torch

from src.metrics import masked_r2, summarize_fold_metrics


class MaskedMetricTests(unittest.TestCase):
    def test_padding_does_not_change_r2(self) -> None:
        actual = torch.tensor([[0.2, 0.8, 0.0], [0.1, 0.3, 0.6]])
        predicted = torch.tensor([[0.3, 0.7, 999.0], [0.2, 0.2, 0.6]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        score = masked_r2(actual, predicted, mask)
        expected = 1.0 - 0.04 / 0.34
        self.assertAlmostEqual(score, expected, places=6)

    def test_fold_summary_reports_mean_and_standard_deviation(self) -> None:
        summary = summarize_fold_metrics(
            [{"vapor_r2": 0.7}, {"vapor_r2": 0.9}]
        )

        self.assertAlmostEqual(summary["vapor_r2"]["mean"], 0.8)
        self.assertAlmostEqual(summary["vapor_r2"]["std"], 0.1)
        self.assertEqual(summary["vapor_r2"]["folds"], 2)

    def test_fold_summary_rejects_nonfinite_or_missing_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_fold_metrics([{"vapor_r2": 0.7}, {"vapor_r2": np.nan}])
        with self.assertRaisesRegex(ValueError, "same metric names"):
            summarize_fold_metrics([{"vapor_r2": 0.7}, {"pressure_rmse": 0.1}])


if __name__ == "__main__":
    unittest.main()
