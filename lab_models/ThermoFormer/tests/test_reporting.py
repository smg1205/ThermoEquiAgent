import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ExperimentConfig
from src.reporting import write_experiment_results


class ExperimentReportingTests(unittest.TestCase):
    def test_skipped_validation_is_reported_as_smoke_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.md"

            write_experiment_results(
                results,
                ExperimentConfig(name="smoke_report"),
                {
                    "run_status": "smoke",
                    "evaluation_mode": "kfold",
                    "n_samples": 10,
                    "n_samples_removed_by_pure_anchor_filter": 2,
                    "data_loading_audit": {
                        "raw_rows": 15,
                        "accepted_before_deduplication": 12,
                        "duplicates_removed": 1,
                        "loaded_samples": 11,
                        "rejected": {"failed_quality": 3},
                    },
                    "final_test_metrics": {"r2": 0.9},
                    "cross_validation_summary": {},
                },
                root / "run",
            )

            text = results.read_text(encoding="utf-8")
            self.assertIn("Status: smoke", text)
            self.assertNotIn("Status: completed", text)
            self.assertIn("failed_quality", text)
            self.assertIn("Raw workbook rows | 15", text)

    def test_failed_atomic_replace_preserves_previous_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.md"
            results.write_text("previous results", encoding="utf-8")
            with patch("src.reporting.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_experiment_results(
                        results,
                        ExperimentConfig(name="atomic_report"),
                        {
                            "evaluation_mode": "kfold",
                            "n_samples": 10,
                            "final_test_metrics": {"r2": 0.9},
                            "cross_validation_summary": {
                                "r2": {"mean": 0.8, "std": 0.1}
                            },
                        },
                        root / "run",
                    )

            self.assertEqual(results.read_text(encoding="utf-8"), "previous results")
            self.assertEqual(list(root.glob(".results.md.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
