import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_c1_direct_ge_training import (
    archive_interrupted_seed_outputs,
    experiment_config_path,
    output_roots,
    parser as campaign_parser,
    resolve_seeds,
    split_path,
    smoke_overrides,
    stage0_checkpoint_path,
)
from src.thermoformer.training.direct_ge_pipeline import format_direct_ge_progress
from src.thermoformer.training.direct_ge_reporting import (
    load_joint_baseline_metrics,
    summarize_direct_ge_campaign,
)


def metric_rows(pressure: float, temperature: float, y_iso: float, y_isob: float):
    rows = []
    for component_count, scale in ((None, 1.0), (2, 1.1), (3, 0.8)):
        scope = "direction" if component_count is None else "direction_cardinality"
        rows.extend(
            [
                {
                    "scope": scope,
                    "direction": "isothermal",
                    "component_count": component_count,
                    "pressure_mae_kpa": pressure * scale,
                    "pressure_rmse_kpa": pressure * scale * 2.0,
                    "pressure_r2": 0.9,
                    "y_mae": y_iso * scale,
                    "y_rmse": y_iso * scale * 2.0,
                    "y_r2": 0.8,
                    "valid_coverage": 1.0,
                    "solver_failure_rate": 0.0,
                    "nonphysical_rate": 0.0,
                },
                {
                    "scope": scope,
                    "direction": "isobaric",
                    "component_count": component_count,
                    "temperature_mae_k": temperature * scale,
                    "temperature_rmse_k": temperature * scale * 2.0,
                    "temperature_r2": 0.92,
                    "y_mae": y_isob * scale,
                    "y_rmse": y_isob * scale * 2.0,
                    "y_r2": 0.82,
                    "valid_coverage": 1.0,
                    "solver_failure_rate": 0.0,
                    "nonphysical_rate": 0.0,
                },
            ]
        )
    return rows


def comparison(seed: int, *, selected: str) -> dict[str, object]:
    stage0 = {
        "metrics": metric_rows(6.0 + seed, 3.0 + seed, 0.03, 0.04),
        "physics_residuals": {"teacher_forced_fugacity": 0.01},
        "validation_loss": 1.0,
        "evaluation_label_metrics": {
            "excess_gibbs_rt_mae": 0.05,
            "log_gamma_mae": 0.12,
            "ge_label_coverage": 0.6,
            "gamma_label_coverage": 0.7,
        },
    }
    stage3 = {
        "metrics": metric_rows(2.0 + 2.0 * seed, 2.0, 0.025, 0.03),
        "physics_residuals": {"teacher_forced_fugacity": 0.007},
        "validation_loss": 0.9 + 0.2 * seed,
        "evaluation_label_metrics": {
            "excess_gibbs_rt_mae": 0.04,
            "log_gamma_mae": 0.10,
            "ge_label_coverage": 0.6,
            "gamma_label_coverage": 0.7,
        },
    }
    return {
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "selected_stage": selected,
        "trained_candidate_stage": "stage3",
        "thermodynamic_loss_weights": {
            "excess_gibbs": 1.0,
            "activity_coefficient": 0.5,
            "vle": 1.0,
            "teacher_forced_fugacity": 0.01,
        },
        "parameter_summary": {
            "total_parameters": 100,
            "trainable_parameters": 20,
        },
        "stages": {"stage0": stage0, "stage3": stage3},
    }


class DirectGECampaignTests(unittest.TestCase):
    def test_formal_defaults_to_five_seeds_and_smoke_is_seed_zero_only(self) -> None:
        self.assertEqual(resolve_seeds(campaign_parser().parse_args([])), tuple(range(5)))
        self.assertEqual(
            resolve_seeds(campaign_parser().parse_args(["--seeds", "1", "3"])),
            (1, 3),
        )
        self.assertEqual(resolve_seeds(campaign_parser().parse_args(["--smoke"])), (0,))
        with self.assertRaisesRegex(ValueError, "Smoke mode"):
            resolve_seeds(
                campaign_parser().parse_args(["--smoke", "--seeds", "0", "1"])
            )

    def test_seed_paths_are_dynamic_and_restricted_to_overall_protocol(self) -> None:
        root = Path("project")
        self.assertEqual(
            split_path(root, 4), root / "datasets/splits/vle/overall_binary_ternary/seed_4.json"
        )
        self.assertEqual(
            stage0_checkpoint_path(root, 2),
            root
            / 'models/vle/multiview/chemical_attention/formal/c1_three_view_vanilla.on.overall_binary_ternary/seed_2/best_model.pt',
        )

    def test_binary_only_protocol_uses_registered_split_and_stage0_checkpoint(self) -> None:
        arguments = campaign_parser().parse_args(
            ["--protocol", "overall_binary", "--seeds", "0", "1"]
        )
        self.assertEqual(arguments.protocol, "overall_binary")
        root = Path("project")
        self.assertEqual(
            split_path(root, 4, arguments.protocol),
            root / "datasets/splits/vle/overall_binary/seed_4.json",
        )
        self.assertEqual(
            stage0_checkpoint_path(root, 2, arguments.protocol),
            root / "models/vle/overall_binary/seed_2/best_model.pt",
        )
        self.assertEqual(
            experiment_config_path(root, arguments.protocol),
            root
            / "configs/vle/comparison/studies/binary_only/thermoformer_direct_ge.yaml",
        )
        run_root, checkpoint_root, result_root = output_roots(
            root,
            smoke=False,
            split_protocol=arguments.protocol,
        )
        self.assertEqual(
            run_root,
            root / "experiments/vle/generalization/training_records/comparisons/binary_only/direct_ge",
        )
        self.assertEqual(
            checkpoint_root,
            root / "models/vle/experiments/comparisons/binary_only/direct_ge",
        )
        self.assertEqual(
            result_root,
            root / "experiments/vle/generalization/evaluations/comparisons/binary_only/direct_ge",
        )

    def test_smoke_overrides_isolate_validation_evaluation(self) -> None:
        self.assertIn("protocol.evaluation_partition=validation", smoke_overrides())
        self.assertIn("direct_ge_supervision.pretrain_epochs=1", smoke_overrides())

    def test_campaign_summary_is_seed_complete_paired_and_sample_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, selected in ((0, "stage0"), (1, "stage3")):
                path = root / f"seed_{seed}" / "stage_comparison.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(comparison(seed, selected=selected)))
                paths.append(path)
            summary = summarize_direct_ge_campaign(paths, expected_seeds=(0, 1))
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(summary["selected_stage_counts"], {"stage0": 1, "stage3": 1})
        self.assertAlmostEqual(
            summary["variants"]["stage3_candidate"]["joint"]["isothermal"]
            ["pressure_mae_kpa"]["mean"],
            3.0,
        )
        self.assertAlmostEqual(
            summary["variants"]["stage3_candidate"]["joint"]["isothermal"]
            ["pressure_mae_kpa"]["sample_std"],
            2.0**0.5,
        )
        self.assertIn("binary", summary["variants"]["selected"])
        self.assertIn("ternary", summary["variants"]["selected"])
        self.assertEqual(len(summary["seed_rows"]), 2 * 3 * 3)

    def test_epoch_progress_is_human_readable_and_complete(self) -> None:
        line = format_direct_ge_progress(
            seed=3,
            stage_index=2,
            stage_count=3,
            stage_name="joint VLE",
            epoch=17,
            epoch_count=80,
            train_metrics={"total": 0.123456},
            validation_metrics={
                "isothermal_pressure_mae_kpa": 2.5,
                "isothermal_y_mae": 0.012,
                "isobaric_temperature_mae_k": 1.75,
                "isobaric_y_mae": 0.018,
            },
            validation_composite=0.87654,
            best_validation_composite=0.81234,
            improved=False,
            epoch_seconds=12.3,
            total_seconds=45.6,
        )
        self.assertIn("[seed 3]", line)
        self.assertIn("[stage 2/3: joint VLE]", line)
        self.assertIn("[epoch 17/80]", line)
        self.assertIn("train_total=0.123456", line)
        self.assertIn("val_score=0.876540", line)
        self.assertIn("best=0.812340", line)
        self.assertIn("P_MAE=2.5000 kPa", line)
        self.assertIn("T_MAE=1.7500 K", line)
        self.assertIn("improved=no", line)
        self.assertIn("epoch_time=12.3s", line)
        self.assertIn("total_time=45.6s", line)

    def test_interrupted_seed_outputs_are_archived_without_touching_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / 'experiments/run_records'
            checkpoint_root = root / 'models/vle'
            results_root = root / 'experiments/reference_results'
            protocol = "model.on.overall_binary_ternary"
            for output_root in (run_root, checkpoint_root, results_root):
                seed_dir = output_root / protocol / "seed_1"
                seed_dir.mkdir(parents=True)
                (seed_dir / "marker.txt").write_text("partial")
            manifest_path = results_root / protocol / "seed_1/manifest.json"
            manifest_path.write_text(json.dumps({"status": "running", "seed": 1}))

            archived = archive_interrupted_seed_outputs(
                run_root=run_root,
                checkpoint_root=checkpoint_root,
                results_root=results_root,
                protocol=protocol,
                seed=1,
                archive_label="interrupted_test",
            )

            self.assertEqual(len(archived), 3)
            for output_root in (run_root, checkpoint_root, results_root):
                self.assertFalse((output_root / protocol / "seed_1").exists())
                archived_dir = output_root / protocol / "seed_1.interrupted_test"
                self.assertEqual((archived_dir / "marker.txt").read_text(), "partial")

            completed_dir = results_root / protocol / "seed_2"
            completed_dir.mkdir(parents=True)
            completed_manifest = completed_dir / "manifest.json"
            completed_manifest.write_text(
                json.dumps({"status": "completed", "seed": 2})
            )
            with self.assertRaisesRegex(RuntimeError, "will not archive"):
                archive_interrupted_seed_outputs(
                    run_root=run_root,
                    checkpoint_root=checkpoint_root,
                    results_root=results_root,
                    protocol=protocol,
                    seed=2,
                    archive_label="interrupted_test",
                )
            self.assertTrue(completed_manifest.exists())

    def test_campaign_summary_reports_missing_subgroup_seed_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in (0, 1):
                payload = comparison(seed, selected="stage0")
                if seed == 1:
                    payload["stages"]["stage3"]["metrics"] = [
                        row
                        for row in payload["stages"]["stage3"]["metrics"]
                        if not (
                            row["scope"] == "direction_cardinality"
                            and row["direction"] == "isothermal"
                            and row["component_count"] == 3
                        )
                    ]
                path = root / f"seed_{seed}" / "stage_comparison.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
                paths.append(path)

            summary = summarize_direct_ge_campaign(paths, expected_seeds=(0, 1))

        metric = summary["variants"]["stage3_candidate"]["ternary"][
            "isothermal"
        ]["pressure_mae_kpa"]
        self.assertEqual(metric["available_seeds"], 1)
        self.assertEqual(metric["seed_ids"], [0])
        missing = next(
            row
            for row in summary["seed_rows"]
            if row["seed"] == 1
            and row["variant"] == "stage3_candidate"
            and row["scope"] == "ternary"
        )
        self.assertIsNone(missing["isothermal_pressure_mae_kpa"])

    def test_campaign_summary_rejects_missing_or_duplicate_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_0" / "stage_comparison.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(comparison(0, selected="stage0")))
            with self.assertRaisesRegex(ValueError, "exact seeds"):
                summarize_direct_ge_campaign([path], expected_seeds=(0, 1))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize_direct_ge_campaign([path, path], expected_seeds=(0,))

    def test_joint_baseline_loader_validates_split_and_uses_sample_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for seed, pressure in ((0, 2.0), (1, 4.0)):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                manifest = {
                    "status": "completed",
                    "seed": seed,
                    "benchmark": "joint_train_joint_test",
                    "evaluation_partition": "test",
                    "test_labels_used_for_selection": False,
                    "dataset_sha256": "dataset",
                    "split_sha256": f"split-{seed}",
                }
                (seed_dir / "manifest.json").write_text(json.dumps(manifest))
                with (seed_dir / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=(
                            "direction", "component_count", "status",
                            "pressure_mae_kpa", "temperature_mae_k", "y_mae",
                            "valid_coverage", "solver_failure_rate", "nonphysical_rate",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "direction": "isothermal", "component_count": "2+3",
                            "status": "evaluated", "pressure_mae_kpa": pressure,
                            "y_mae": 0.02, "valid_coverage": 1.0,
                            "solver_failure_rate": 0.0, "nonphysical_rate": 0.0,
                        }
                    )
                    writer.writerow(
                        {
                            "direction": "isobaric", "component_count": "2+3",
                            "status": "evaluated", "temperature_mae_k": 8.0,
                            "y_mae": 0.08, "valid_coverage": 0.96,
                            "solver_failure_rate": 0.04, "nonphysical_rate": 0.0,
                        }
                    )
                expected[seed] = {"dataset_sha256": "dataset", "split_sha256": f"split-{seed}"}
            summary = load_joint_baseline_metrics(
                root, expected_provenance=expected, expected_seeds=(0, 1)
            )
            self.assertAlmostEqual(
                summary["directions"]["isothermal"]["pressure_mae_kpa"]["mean"],
                3.0,
            )
            self.assertAlmostEqual(
                summary["directions"]["isothermal"]["pressure_mae_kpa"]
                ["sample_std"],
                2.0**0.5,
            )
            bad = json.loads((root / "seed_1/manifest.json").read_text())
            bad["split_sha256"] = "wrong"
            (root / "seed_1/manifest.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "split_sha256"):
                load_joint_baseline_metrics(
                    root, expected_provenance=expected, expected_seeds=(0, 1)
                )


if __name__ == "__main__":
    unittest.main()
