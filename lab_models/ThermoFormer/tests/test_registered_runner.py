import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openpyxl

from src.data import load_vle_samples
from src.experiment_runner import result_protocol_name, run_paper_experiment
from src.thermoformer.protocols.runner import _require_generated_stage1_checkpoint
from src.splits import DatasetPartitions, save_split_assignment
from scripts.run_registered_experiment import output_roots, parser
from scripts.run_registered_suite import smoke_overrides


class RegisteredRunnerTests(unittest.TestCase):
    def test_generated_stage1_checkpoint_requires_matching_completed_manifest(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        artifact_root = project_root / 'experiments/run_records/test_generated_stage1_checkpoint'
        with tempfile.TemporaryDirectory(dir=artifact_root.parent) as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"audited-stage0")
            import hashlib
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_kind": "formal",
                        "analysis_status": "confirmatory",
                        "git_commit": "abc123",
                        "git_dirty": False,
                        "seed": 2,
                        "artifacts": {
                            "checkpoint": {
                                "path": str(checkpoint.relative_to(project_root)).replace("\\", "/"),
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            _require_generated_stage1_checkpoint(
                checkpoint,
                manifest,
                expected_git_commit="abc123",
                expected_seed=2,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifacts"]["checkpoint"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                _require_generated_stage1_checkpoint(
                    checkpoint,
                    manifest,
                    expected_git_commit="abc123",
                    expected_seed=2,
                )

    def test_registered_suite_smoke_remains_supervised_only(self) -> None:
        overrides = smoke_overrides()
        self.assertIn("training.epochs_supervised=2", overrides)
        self.assertIn("training.epochs_physics=0", overrides)
        self.assertIn("training.minimum_physics_epochs=0", overrides)

    def test_ablation_result_namespace_reuses_split_without_overwriting_reference(self) -> None:
        self.assertEqual(
            result_protocol_name("overall_binary_ternary", "overall_binary_ternary"),
            "overall_binary_ternary",
        )
        self.assertEqual(
            result_protocol_name("ablation_pairwise", "overall_binary_ternary"),
            "ablation_pairwise.on.overall_binary_ternary",
        )

    def test_single_run_smoke_defaults_are_isolated_from_formal_roots(self) -> None:
        args = parser().parse_args(["--split", "split.json", "--seed", "0", "--smoke"])
        run_root, checkpoint_root, results_root = output_roots(args)
        self.assertIn("single_smoke", str(run_root))
        self.assertIn("single_smoke", str(checkpoint_root))
        self.assertIn("single_smoke", str(results_root))
        self.assertNotEqual(results_root, Path("D:/VLE/VLE/results"))

    def test_end_to_end_run_exports_checkpoint_predictions_metrics_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'datasets/vle_reference'
            data_root.mkdir(parents=True)
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(
                [
                    "name1", "cas1", "smiles1", "name2", "cas2", "smiles2",
                    "check1", "check2", "P", "T", "x1", "y1", "doi",
                ]
            )
            systems = [("C", "O"), ("CC", "O"), ("CCC", "O")]
            for system_index, (first, second) in enumerate(systems):
                for point in range(3):
                    x = 0.2 + 0.2 * point
                    sheet.append(
                        [
                            first, "", first, second, "", second, 1, 1,
                            760.0 + 5.0 * system_index, 30.0 + point,
                            x, x, f"series-{system_index}",
                        ]
                    )
            workbook.save(data_root / "binary.xlsx")
            samples = load_vle_samples(data_root)
            split = DatasetPartitions(
                train=tuple(samples[:3]),
                validation=tuple(samples[3:6]),
                test=tuple(samples[6:]),
                protocol="runner_contract",
                seed=0,
            )
            split_path = root / "split.json"
            save_split_assignment(split_path, samples, split)
            config = {
                "name": "runner_contract",
                "seed": 0,
                "model": {
                    "hidden_dim": 8,
                    "layers": 1,
                    "heads": 2,
                    "activity_mode": "ideal",
                },
                "encoder": {"representation": "unimol_v2"},
                "data": {
                    "root": str(data_root),
                    "minimum_pure_anchor_temperatures": 0,
                },
                "training": {
                    "batch_size": 3,
                    "epochs_supervised": 1,
                    "epochs_physics": 0,
                    "solver_iterations_eval": 2,
                },
                "runtime": {"device": "cpu"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            cache = root / "unimol.npz"
            unique = sorted({smiles for sample in samples for smiles in sample.smiles})
            np.savez_compressed(
                cache,
                smiles=np.asarray(unique),
                features=np.arange(len(unique) * 4, dtype=np.float32).reshape(len(unique), 4),
                model=np.asarray("unimolv2"),
                model_size=np.asarray("84m"),
            )

            manifest = run_paper_experiment(
                config_path=config_path,
                split_path=split_path,
                seed=0,
                run_root=root / 'experiments/run_records',
                checkpoint_root=root / 'models/vle',
                results_root=root / 'experiments/reference_results',
                feature_cache=cache,
                run_kind="smoke",
                evaluation_partition="validation",
            )

            run_dir = root / 'experiments/run_records/runner_contract/seed_0'
            result_dir = root / 'experiments/reference_results/runner_contract/seed_0'
            checkpoint = root / 'models/vle/runner_contract/seed_0/best_model.pt'
            self.assertEqual(manifest["status"], "smoke")
            self.assertEqual(manifest["evaluation_partition"], "validation")
            self.assertEqual(manifest["selection_partition"], "validation")
            self.assertFalse(manifest["test_metrics_used_for_selection"])
            self.assertEqual(manifest["rows"]["evaluated_rows"], len(split.validation))
            self.assertEqual(
                manifest["physical_consistency"]["status"], "not_evaluated"
            )
            self.assertRegex(manifest["git_commit"], r"^[0-9a-f]{7,40}$")
            self.assertIsInstance(manifest["git_dirty"], bool)
            self.assertRegex(manifest["resolved_config_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["feature_cache_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["feature_subset_sha256"], r"^[0-9a-f]{64}$")
            self.assertIsNone(manifest["pure_property_catalog_sha256"])
            self.assertRegex(manifest["environment_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("numpy", manifest["runtime"])
            self.assertIn("rdkit", manifest["runtime"])
            self.assertIn("unimol_tools", manifest["runtime"])
            self.assertGreaterEqual(manifest["inference_seconds"], 0.0)
            self.assertGreaterEqual(manifest["inference_ms_per_attempt"], 0.0)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((run_dir / "history.json").is_file())
            self.assertTrue((run_dir / "training_curves.csv").is_file())
            self.assertTrue((result_dir / "predictions.csv").is_file())
            self.assertTrue((result_dir / "metrics.json").is_file())
            self.assertTrue((result_dir / "physical_consistency.json").is_file())
            self.assertTrue((result_dir / "resolved_config.json").is_file())
            prediction_lines = (result_dir / "predictions.csv").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            self.assertGreater(len(prediction_lines), len(split.test))

            protocol_dir = root / 'experiments/reference_results/runner_contract'
            (protocol_dir / "aggregate_manifest.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (protocol_dir / "diagnostic_aggregate_manifest.json").write_text(
                json.dumps({"status": "diagnostic"}), encoding="utf-8"
            )
            (protocol_dir / "metrics_by_seed.csv").write_text(
                "old", encoding="utf-8"
            )
            (protocol_dir / "metrics_summary.csv").write_text(
                "old", encoding="utf-8"
            )

            with patch("src.experiment_runner.fit_model", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    run_paper_experiment(
                        config_path=config_path,
                        split_path=split_path,
                        seed=0,
                        run_root=root / 'experiments/run_records',
                        checkpoint_root=root / 'models/vle',
                        results_root=root / 'experiments/reference_results',
                        feature_cache=cache,
                        allow_overwrite=True,
                        run_kind="smoke",
                    )
            interrupted = json.loads(
                (result_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(interrupted["status"], "completed")
            self.assertNotEqual(interrupted["status"], "smoke")
            aggregate = json.loads(
                (protocol_dir / "aggregate_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["status"], "invalidated")
            diagnostic_aggregate = json.loads(
                (protocol_dir / "diagnostic_aggregate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(diagnostic_aggregate["status"], "invalidated")


if __name__ == "__main__":
    unittest.main()
