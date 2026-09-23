import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from src.results import aggregate_protocol_results


class ResultAggregationTests(unittest.TestCase):
    @staticmethod
    def manifest(seed: int, root: Optional[Path] = None, **updates):
        artifacts = {}
        if root is not None:
            for name in (
                "checkpoint", "history", "training_curves", "predictions",
                "metrics", "resolved_config",
            ):
                path = root / f"artifact_{name}.bin"
                path.write_bytes(f"{seed}-{name}".encode("utf-8"))
                artifacts[name] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        payload = {
            "status": "completed",
            "seed": seed,
            "protocol": "protocol_a",
            "dataset_sha256": "data-a",
            "git_commit": "commit-a",
            "resolved_config_sha256": "config-a",
            "feature_subset_sha256": "features-a",
            "environment_sha256": "environment-a",
            "artifacts": artifacts,
        }
        payload.update(updates)
        return payload

    def test_aggregates_identical_metric_scopes_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, mae in ((0, 1.0), (1, 3.0)):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "manifest.json").write_text(
                    json.dumps(self.manifest(seed, seed_dir)),
                    encoding="utf-8",
                )
                (seed_dir / "metrics.json").write_text(
                    json.dumps(
                        [
                            {
                                "scope": "direction",
                                "direction": "isothermal",
                                "component_count": None,
                                "pressure_mae_kpa": mae,
                                "valid_samples": 10,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            summary = aggregate_protocol_results(
                root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
            )

            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(summary[0]["pressure_mae_kpa_mean"], 2.0)
            self.assertAlmostEqual(summary[0]["pressure_mae_kpa_std"], 2**0.5)
            self.assertEqual(summary[0]["seeds"], 2)
            self.assertTrue((root / "diagnostic_metrics_by_seed.csv").is_file())
            self.assertTrue((root / "diagnostic_metrics_summary.csv").is_file())
            aggregate_manifest = json.loads(
                (root / "diagnostic_aggregate_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate_manifest["status"], "diagnostic")

            history = Path(
                json.loads((root / "seed_0" / "manifest.json").read_text(encoding="utf-8"))
                ["artifacts"]["history"]["path"]
            )
            history.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history.*changed"):
                aggregate_protocol_results(
                    root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
                )

    def test_formal_aggregate_requires_five_seeds_and_writes_completion_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(5):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "manifest.json").write_text(
                    json.dumps(self.manifest(seed, seed_dir)), encoding="utf-8"
                )
                rows = [
                    {
                        "scope": "all",
                        "direction": None,
                        "component_count": None,
                        "subgroup": None,
                        "y_mae": 0.1 + seed,
                    },
                    {
                        "scope": "cardinality",
                        "direction": None,
                        "component_count": 3,
                        "subgroup": None,
                        "y_mae": 0.2 + seed,
                        "pressure_mae_kpa": None if seed == 1 else 1.0 + seed,
                    },
                ]
                if seed in (0, 2, 4):
                    rows.append(
                        {
                            "scope": "direction_cardinality",
                            "direction": "isothermal",
                            "component_count": 3,
                            "subgroup": None,
                            "y_mae": 0.3 + seed,
                        }
                    )
                (seed_dir / "metrics.json").write_text(
                    json.dumps(rows),
                    encoding="utf-8",
                )

            with patch(
                "src.results._git_aggregate_state",
                return_value=("aggregate-commit", False, []),
            ):
                summary = aggregate_protocol_results(root)

            manifest = json.loads(
                (root / "aggregate_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(manifest["training_git_commit"], "commit-a")
            self.assertEqual(manifest["aggregation_git_commit"], "aggregate-commit")
            sparse = next(
                row for row in summary if row["scope"] == "direction_cardinality"
            )
            self.assertEqual(sparse["scope_available_seeds"], 3)
            self.assertEqual(sparse["scope_seed_ids"], "0;2;4")
            self.assertEqual(sparse["seeds"], 5)
            ternary = next(
                row
                for row in summary
                if row["scope"] == "cardinality" and row["component_count"] == 3
            )
            self.assertEqual(ternary["pressure_mae_kpa_available_seeds"], 4)
            self.assertEqual(ternary["pressure_mae_kpa_seed_ids"], "0;2;3;4")
            self.assertAlmostEqual(ternary["pressure_mae_kpa_mean"], 3.25)

    def test_formal_aggregate_rejects_incomplete_headline_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(5):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "manifest.json").write_text(
                    json.dumps(self.manifest(seed, seed_dir)), encoding="utf-8"
                )
                rows = [
                    {
                        "scope": "all",
                        "direction": None,
                        "component_count": None,
                        "subgroup": None,
                        "y_mae": 0.1,
                    }
                ]
                if seed != 4:
                    rows.append(
                        {
                            "scope": "cardinality",
                            "direction": None,
                            "component_count": 3,
                            "subgroup": None,
                            "y_mae": 0.2,
                        }
                    )
                (seed_dir / "metrics.json").write_text(
                    json.dumps(rows), encoding="utf-8"
                )

            with patch(
                "src.results._git_aggregate_state",
                return_value=("aggregate-commit", False, []),
            ):
                with self.assertRaisesRegex(ValueError, "Headline metric scope.*all seeds"):
                    aggregate_protocol_results(root)

    def test_refuses_partial_seed_sets_or_mismatched_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_dir = root / "seed_0"
            seed_dir.mkdir()
            (seed_dir / "manifest.json").write_text(
                json.dumps(self.manifest(0, seed_dir)), encoding="utf-8"
            )
            (seed_dir / "metrics.json").write_text(
                json.dumps([{"scope": "all", "direction": None, "component_count": None, "y_mae": 0.1}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing completed seeds"):
                aggregate_protocol_results(
                    root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
                )

    def test_refuses_mixed_provenance_and_duplicate_metric_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                manifest = self.manifest(seed, seed_dir)
                (seed_dir / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                row = {
                    "scope": "all",
                    "direction": None,
                    "component_count": None,
                    "subgroup": None,
                    "y_mae": 0.1,
                }
                (seed_dir / "metrics.json").write_text(
                    json.dumps([row, row] if seed == 0 else [row]), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "Duplicate metric identity"):
                aggregate_protocol_results(
                    root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
                )

            seed_zero_metrics = root / "seed_0" / "metrics.json"
            rows = json.loads(seed_zero_metrics.read_text(encoding="utf-8"))
            seed_zero_metrics.write_text(json.dumps(rows[:1]), encoding="utf-8")
            seed_one_manifest = root / "seed_1" / "manifest.json"
            manifest = json.loads(seed_one_manifest.read_text(encoding="utf-8"))
            manifest["git_commit"] = "commit-b"
            seed_one_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                aggregate_protocol_results(
                    root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
                )

    def test_explicitly_approved_equivalent_commits_preserve_seed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, commit in ((0, "commit-a"), (1, "commit-b")):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "manifest.json").write_text(
                    json.dumps(self.manifest(seed, seed_dir, git_commit=commit)),
                    encoding="utf-8",
                )
                (seed_dir / "metrics.json").write_text(
                    json.dumps(
                        [
                            {
                                "scope": "all",
                                "direction": None,
                                "component_count": None,
                                "subgroup": None,
                                "y_mae": 0.1 + seed,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "provenance"):
                aggregate_protocol_results(
                    root, expected_seeds=(0, 1), aggregate_kind="diagnostic"
                )

            with patch(
                "src.results._git_aggregate_state",
                return_value=("aggregate-commit", False, []),
            ):
                aggregate_protocol_results(
                    root,
                    expected_seeds=(0, 1),
                    aggregate_kind="diagnostic",
                    compatible_training_git_commits=("commit-a", "commit-b"),
                    mixed_commit_justification="Only progress logging changed.",
                )

            manifest = json.loads(
                (root / "diagnostic_aggregate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["training_git_commits"], ["commit-a", "commit-b"]
            )
            self.assertEqual(
                manifest["training_git_commits_by_seed"],
                {"0": "commit-a", "1": "commit-b"},
            )
            self.assertEqual(
                manifest["mixed_commit_justification"],
                "Only progress logging changed.",
            )

    def test_validation_failure_preserves_existing_aggregate_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_by_seed = root / "metrics_by_seed.csv"
            old_summary = root / "metrics_summary.csv"
            old_by_seed.write_text("old-by-seed", encoding="utf-8")
            old_summary.write_text("old-summary", encoding="utf-8")
            seed_dir = root / "seed_0"
            seed_dir.mkdir()
            (seed_dir / "manifest.json").write_text(
                json.dumps(self.manifest(0, seed_dir)), encoding="utf-8"
            )
            (seed_dir / "metrics.json").write_text(
                json.dumps(
                    [
                        {
                            "scope": "all",
                            "direction": None,
                            "component_count": None,
                            "subgroup": None,
                            "y_mae": float("nan"),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Non-finite"):
                aggregate_protocol_results(
                    root, expected_seeds=(0,), aggregate_kind="diagnostic"
                )
            self.assertEqual(old_by_seed.read_text(encoding="utf-8"), "old-by-seed")
            self.assertEqual(old_summary.read_text(encoding="utf-8"), "old-summary")

    def test_refuses_empty_or_duplicate_expected_seed_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "non-empty"):
                aggregate_protocol_results(root, expected_seeds=())
            with self.assertRaisesRegex(ValueError, "unique"):
                aggregate_protocol_results(root, expected_seeds=(0, 0))
            with self.assertRaisesRegex(ValueError, "exactly seeds"):
                aggregate_protocol_results(root, expected_seeds=(0, 1))


if __name__ == "__main__":
    unittest.main()
