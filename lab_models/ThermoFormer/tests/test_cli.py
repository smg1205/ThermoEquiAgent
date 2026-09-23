import contextlib
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import numpy as np
import openpyxl
import torch

from src.cli import PROJECT_ROOT, _clear_stale_artifacts, _project_path, build_parser, main
from src.model import ThermoFormer, ThermoFormerConfig


class OutputLifecycleTests(unittest.TestCase):
    def test_default_config_and_relative_paths_are_anchored_to_project_root(self) -> None:
        arguments = build_parser().parse_args([])
        self.assertEqual(
            arguments.config,
            PROJECT_ROOT
            / "configs"
            / "training"
            / "supervised.yaml",
        )
        self.assertEqual(_project_path("datasets/vle_reference"), PROJECT_ROOT / 'datasets/vle_reference')
        self.assertEqual(
            _project_path("experiments/run_records/public/c1_supervised"),
            PROJECT_ROOT
            / 'experiments/run_records/public/c1_supervised',
        )

    def test_stale_training_outputs_are_removed_but_unimolv2_cache_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stale = [
                output / "fold_1.pt",
                output / "fold_2_best.pt",
                output / "holdout.pt",
                output / "final_model.pt",
                output / "best.pt",
                output / "history.json",
                output / "dataset_manifest.json",
                output / "experiment_config.json",
                output / "unimol_features.npy",
                output / "training.stdout.log",
                output / "training.stderr.log",
            ]
            for path in stale:
                path.write_bytes(b"old")
            cache = output / "unimolv2_features.npz"
            cache.write_bytes(b"expensive-cache")
            unrelated = output / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")

            _clear_stale_artifacts(output)

            self.assertTrue(all(not path.exists() for path in stale))
            self.assertEqual(cache.read_bytes(), b"expensive-cache")
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_holdout_cli_writes_reloadable_thermoformer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output = root / "run"
            data_root.mkdir()
            output.mkdir()
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(
                [
                    "name1", "cas1", "smiles1", "name2", "cas2", "smiles2",
                    "check1", "check2", "P", "T", "x1", "y1", "doi",
                ]
            )
            smiles = []
            for index in range(10):
                first, second = "C" * (2 * index + 1), "C" * (2 * index + 2)
                smiles.extend((first, second))
                sheet.append(
                    [first, "", first, second, "", second, 1, 1, 760.0, 30.0, 0.5, 0.5, "synthetic"]
                )
            workbook.save(data_root / "binary.xlsx")
            np.savez_compressed(
                output / "unimolv2_84m.npz",
                smiles=np.asarray(sorted(smiles)),
                features=np.arange(80, dtype=np.float32).reshape(20, 4) / 80.0,
                model=np.asarray("unimolv2"),
                model_size=np.asarray("84m"),
            )
            config = {
                "name": "cli_smoke",
                "model": {
                    "hidden_dim": 8,
                    "layers": 1,
                    "heads": 2,
                    "activity_mode": "ideal",
                },
                "data": {
                    "root": str(data_root),
                    "minimum_pure_anchor_temperatures": 0,
                },
                "evaluation": {
                    "mode": "holdout",
                    "folds": 2,
                    "test_fraction": 0.2,
                    "validation_fraction": 0.2,
                },
                "training": {
                    "batch_size": 4,
                    "epochs_supervised": 1,
                    "epochs_physics": 0,
                },
                "runtime": {
                    "output_dir": str(output),
                    "device": "cpu",
                    "results_file": str(root / "results.md"),
                },
            }
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with contextlib.redirect_stdout(StringIO()):
                main(["--config", str(config_path)])

            checkpoint = torch.load(output / "final_model.pt", map_location="cpu", weights_only=False)
            model = ThermoFormer(ThermoFormerConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model"])
            manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["model_name"], "ThermoFormer")
            self.assertEqual(manifest["evaluation_mode"], "holdout")
            self.assertEqual(manifest["run_status"], "completed")
            self.assertEqual(manifest["data_loading_audit"]["raw_rows"], 10)
            self.assertIn("isothermal_pressure_log_rmse", manifest["final_test_metrics"])
            self.assertIn("isobaric_temperature_rmse_k", manifest["final_test_metrics"])
            self.assertTrue((output / "holdout.pt").exists())
            results = (root / "results.md").read_text(encoding="utf-8")
            self.assertIn("Status: completed", results)
            self.assertIn("Final test metrics", results)
            formal_checkpoint = (output / "final_model.pt").read_bytes()
            formal_manifest = (output / "dataset_manifest.json").read_text(encoding="utf-8")

            with contextlib.redirect_stdout(StringIO()):
                main(["--config", str(config_path), "--skip-validation"])

            self.assertEqual((root / "results.md").read_text(encoding="utf-8"), results)
            self.assertEqual((output / "final_model.pt").read_bytes(), formal_checkpoint)
            self.assertEqual(
                (output / "dataset_manifest.json").read_text(encoding="utf-8"),
                formal_manifest,
            )
            smoke_output = output / "smoke"
            smoke_results = (smoke_output / "smoke_results.md").read_text(encoding="utf-8")
            self.assertIn("Status: smoke", smoke_results)
            self.assertTrue((smoke_output / "final_model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
