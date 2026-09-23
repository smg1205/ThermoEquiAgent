from __future__ import annotations

from copy import deepcopy
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import scripts.run_overall_binary_three_stage_ablations as campaign
from scripts.run_overall_binary_three_stage_ablations import (
    FORMAL_NAMESPACE,
    SMOKE_NAMESPACE,
    SPLIT_PROTOCOL,
    Stage0Reference,
    VARIANT_IDS,
    artifact_roots,
    config_path,
    legacy_c1_stage0_paths,
    legacy_stage0_matches_current_recipe,
    stage0_recipe_fingerprint,
    parser,
    protocol_name,
    resolve_seeds,
    selected_variants,
    smoke_overrides,
    stage0_config_path,
)
from src.thermoformer.configuration import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OverallBinaryThreeStageAblationCampaignTests(unittest.TestCase):
    def test_registry_has_the_seven_retained_table_two_variants(self) -> None:
        self.assertEqual(
            VARIANT_IDS,
            (
                "c0_current_vanilla",
                "v1_rdkit_only",
                "v3_functional_group_only",
                "v4_rdkit_unimol_naive",
                "c1_three_view_vanilla",
                "c2_chemical_bias_full",
                "c3_no_pair_bias",
            ),
        )
        self.assertEqual(
            tuple(protocol_name(variant) for variant in VARIANT_IDS),
            tuple(f"{variant}.on.overall_binary" for variant in VARIANT_IDS),
        )

    def test_all_configs_use_binary_only_registered_splits_and_schedule(self) -> None:
        for variant in VARIANT_IDS:
            stage0 = load_experiment_config(stage0_config_path(PROJECT_ROOT, variant))
            three_stage = load_experiment_config(config_path(PROJECT_ROOT, variant))
            self.assertEqual(stage0.name, variant)
            self.assertEqual(three_stage.name, variant)
            self.assertEqual(stage0.protocol.registered_splits, (SPLIT_PROTOCOL,))
            self.assertEqual(three_stage.protocol.registered_splits, (SPLIT_PROTOCOL,))
            self.assertEqual(stage0.protocol.evaluation_partition, "validation")
            self.assertEqual(three_stage.protocol.evaluation_partition, "test")
            self.assertEqual(stage0.protocol.seeds, tuple(range(5)))
            self.assertEqual(three_stage.protocol.seeds, tuple(range(5)))
            self.assertIsNone(stage0.direct_ge_supervision)
            self.assertEqual(stage0.training.epochs_supervised, 80)
            self.assertEqual(stage0.training.epochs_physics, 0)
            self.assertIsNotNone(three_stage.direct_ge_supervision)
            self.assertIsNotNone(three_stage.physics_finetuning)
            self.assertEqual(three_stage.direct_ge_supervision.pretrain_epochs, 20)
            self.assertEqual(three_stage.training.epochs_supervised, 80)
            self.assertEqual(three_stage.direct_ge_supervision.fugacity_epochs, 10)
            self.assertEqual(three_stage.direct_ge_supervision.warmup_epochs, 2)
            self.assertEqual(
                three_stage.physics_finetuning.teacher_forced_fugacity_weight, 0.01
            )

    def test_variant_feature_switches_match_retained_ablation_definitions(self) -> None:
        expected = {
            "c0_current_vanilla": ("unimol_v2", "legacy", False, True, False, False, False),
            "v1_rdkit_only": ("rdkit_2d", "legacy", True, False, False, False, False),
            "v3_functional_group_only": (
                "functional_groups", "legacy", False, False, True, False, False
            ),
            "v4_rdkit_unimol_naive": ("multiview", "naive", True, True, False, False, False),
            "c1_three_view_vanilla": ("multiview", "naive", True, True, True, False, False),
            "c2_chemical_bias_full": ("multiview", "naive", True, True, True, True, True),
            "c3_no_pair_bias": ("multiview", "naive", True, True, True, False, True),
        }
        for variant, switches in expected.items():
            encoder = load_experiment_config(config_path(PROJECT_ROOT, variant)).encoder
            observed = (
                encoder.representation,
                encoder.fusion_mode,
                encoder.use_rdkit_descriptors,
                encoder.use_unimol,
                encoder.use_functional_groups,
                encoder.chemical_attention_bias,
                encoder.context_pair_interaction,
            )
            self.assertEqual(observed, switches, variant)

    def test_formal_and_smoke_roots_are_isolated(self) -> None:
        root = Path("project")
        formal = artifact_roots(root, stage="three_stage", smoke=False)
        stage0 = artifact_roots(root, stage="stage0", smoke=False)
        smoke = artifact_roots(root, stage="three_stage", smoke=True)
        self.assertEqual(
            formal,
            (
                root / 'experiments/run_records' / FORMAL_NAMESPACE,
                root / 'models/vle' / FORMAL_NAMESPACE,
                root / 'experiments/reference_results' / FORMAL_NAMESPACE,
            ),
        )
        self.assertEqual(
            stage0,
            (
                root / 'experiments/run_records' / FORMAL_NAMESPACE / "stage0",
                root / 'models/vle' / FORMAL_NAMESPACE / "stage0",
                root / 'experiments/reference_results' / FORMAL_NAMESPACE / "stage0",
            ),
        )
        self.assertNotEqual(stage0, formal)
        self.assertTrue(all(str(path).startswith(str(root / SMOKE_NAMESPACE)) for path in smoke))
        self.assertTrue(all("overall_binary_ternary" not in str(path) for path in (*formal, *stage0, *smoke)))

    def test_cli_defaults_and_smoke_overrides_cover_every_training_stage(self) -> None:
        self.assertEqual(resolve_seeds(parser().parse_args([])), tuple(range(5)))
        self.assertEqual(resolve_seeds(parser().parse_args(["--smoke"])), (0,))
        self.assertEqual(
            resolve_seeds(parser().parse_args(["--seeds", "1", "3"])), (1, 3)
        )
        with self.assertRaisesRegex(ValueError, "Smoke mode"):
            resolve_seeds(parser().parse_args(["--smoke", "--seeds", "0", "1"]))
        arguments = parser().parse_args(
            ["--variant", "c1_three_view_vanilla", "--variant", "c2_chemical_bias_full"]
        )
        self.assertEqual(
            selected_variants(arguments),
            ("c1_three_view_vanilla", "c2_chemical_bias_full"),
        )
        with self.assertRaisesRegex(ValueError, "at most once"):
            selected_variants(
                parser().parse_args(
                    ["--variant", "c1_three_view_vanilla", "--variant", "c1_three_view_vanilla"]
                )
            )
        self.assertIn("training.epochs_supervised=1", smoke_overrides("stage0"))
        self.assertNotIn("direct_ge_supervision.pretrain_epochs=1", smoke_overrides("stage0"))
        self.assertIn(
            "direct_ge_supervision.pretrain_epochs=1", smoke_overrides("three_stage")
        )
        self.assertIn(
            "direct_ge_supervision.fugacity_epochs=1", smoke_overrides("three_stage")
        )

    def test_registered_binary_split_is_binary_and_system_disjoint(self) -> None:
        for seed in range(5):
            split = PROJECT_ROOT / 'datasets/splits/vle' / SPLIT_PROTOCOL / f"seed_{seed}.json"
            payload = json.loads(split.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol"], SPLIT_PROTOCOL)
            rows = payload["metadata"]["component_count_rows"]
            for partition in ("train", "validation", "test"):
                self.assertTrue(payload["partitions"][partition])
                self.assertEqual(set(rows[partition]), {"2"})
            self.assertEqual(
                payload["metadata"]["split_rule"],
                "unordered canonical system disjoint, row-balanced by cardinality",
            )

    def test_legacy_c1_reference_paths_remain_read_only_inputs(self) -> None:
        checkpoint, manifest = legacy_c1_stage0_paths(PROJECT_ROOT, 3)
        self.assertEqual(
            checkpoint,
            PROJECT_ROOT / 'models/vle/overall_binary/seed_3/best_model.pt',
        )
        self.assertEqual(
            manifest,
            PROJECT_ROOT / 'experiments/vle/prediction/reference/overall_binary/seed_3/manifest.json',
        )


    def test_smoke_runs_stage_zero_then_all_three_stages_in_isolated_roots(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            variant = Path(kwargs["config_path"]).stem
            seed = int(kwargs["seed"])
            protocol = protocol_name(variant)
            checkpoint = kwargs["checkpoint_root"] / protocol / f"seed_{seed}" / "best_model.pt"
            manifest = kwargs["results_root"] / protocol / f"seed_{seed}" / "manifest.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"smoke-checkpoint")
            manifest.write_text("{}", encoding="utf-8")
            return {"status": "smoke", "selected_stage": "stage3"}

        # The mock must never share the live smoke namespace.  In particular,
        # its fixture files have the same names as real diagnostic artifacts.
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            shutil.copytree(
                PROJECT_ROOT / FORMAL_NAMESPACE,
                temporary_root / FORMAL_NAMESPACE,
            )
            shutil.copytree(PROJECT_ROOT / "configs", temporary_root / "configs", dirs_exist_ok=True)
            source_split = PROJECT_ROOT / 'datasets/splits/vle' / SPLIT_PROTOCOL / "seed_0.json"
            temporary_split = temporary_root / 'datasets/splits/vle' / SPLIT_PROTOCOL / "seed_0.json"
            temporary_split.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_split, temporary_split)

            # This test verifies stage order and smoke-root isolation, not
            # recovery of a previously completed bundle.
            with patch.object(campaign, "PROJECT_ROOT", temporary_root), patch.object(
                campaign, "_recover_if_complete", return_value=None
            ), patch.object(campaign, "run_paper_experiment", side_effect=fake_run), patch.object(
                campaign, "_require_three_stage_artifacts"
            ):
                campaign.main(
                    ["--variant", "v1_rdkit_only", "--smoke", "--device", "cpu"]
                )

            self.assertEqual(len(calls), 2)
            stage0_call, three_stage_call = calls
            self.assertEqual(
                stage0_call["config_path"],
                stage0_config_path(temporary_root, "v1_rdkit_only"),
            )
            self.assertEqual(
                three_stage_call["config_path"],
                config_path(temporary_root, "v1_rdkit_only"),
            )
            self.assertEqual(stage0_call["evaluation_partition"], "validation")
            self.assertEqual(three_stage_call["evaluation_partition"], "validation")
            self.assertEqual(stage0_call["stage1_checkpoint"], None)
            self.assertEqual(
                three_stage_call["stage1_checkpoint"],
                artifact_roots(temporary_root, stage="stage0", smoke=True)[1]
                / "v1_rdkit_only.on.overall_binary/seed_0/best_model.pt",
            )
            self.assertEqual(
                three_stage_call["generated_stage1_manifest"],
                artifact_roots(temporary_root, stage="stage0", smoke=True)[2]
                / "v1_rdkit_only.on.overall_binary/seed_0/manifest.json",
            )
            smoke_root = temporary_root / SMOKE_NAMESPACE
            self.assertTrue(
                all(str(call["run_root"]).startswith(str(smoke_root)) for call in calls)
            )

    def test_recovery_requires_hash_verified_stage_prediction_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed_dir = root / 'experiments/reference_results/seed_0'
            manifest_path = seed_dir / "manifest.json"

            def record(path: Path) -> dict[str, str]:
                return {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": campaign.artifact_sha256(path),
                }

            artifacts: dict[str, dict[str, str]] = {}
            stages: dict[str, dict[str, str]] = {}
            for stage in campaign.STAGES:
                checkpoint = root / 'models/vle' / f"{stage}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(stage.encode("utf-8"))
                prediction = seed_dir / f"{stage}_predictions.csv"
                prediction.parent.mkdir(parents=True, exist_ok=True)
                prediction.write_text("sample_id,prediction\nfixture,0\n", encoding="utf-8")
                checkpoint_record, prediction_record = record(checkpoint), record(prediction)
                artifacts[stage + "_checkpoint"] = checkpoint_record
                artifacts[stage + "_predictions"] = prediction_record
                stages[stage] = {
                    "checkpoint": checkpoint_record["path"],
                    "checkpoint_sha256": checkpoint_record["sha256"],
                    "predictions": prediction_record["path"],
                    "predictions_sha256": prediction_record["sha256"],
                }
            comparison_path = seed_dir / "stage_comparison.json"
            comparison_path.write_text(json.dumps({"stages": stages}), encoding="utf-8")
            artifacts["stage_comparison"] = record(comparison_path)
            manifest = {"artifacts": artifacts}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            arguments = {
                "manifest_path": manifest_path,
                "config_path": root / "config.yaml",
                "split": root / "split.json",
                "feature_cache": root / "features.pt",
                "seed": 0,
                "device": "cpu",
                "overrides": (),
                "run_kind": "formal",
                "evaluation_partition": "test",
                "stage1_checkpoint": None,
                "aggregate_expected": False,
                "analysis_status": "confirmatory",
                "variant": "c1_three_view_vanilla",
                "stage": "three_stage",
            }
            with patch.object(campaign, "PROJECT_ROOT", root), patch.object(
                campaign, "requested_run_fingerprint", return_value="request"
            ), patch.object(
                campaign, "recover_completed_seed_manifest", return_value=manifest
            ):
                self.assertIs(campaign._recover_if_complete(**arguments), manifest)
                missing = json.loads(json.dumps(manifest))
                missing["artifacts"].pop("stage0_predictions")
                with patch.object(
                    campaign, "recover_completed_seed_manifest", return_value=missing
                ), self.assertRaisesRegex(RuntimeError, "stage0_predictions"):
                    campaign._recover_if_complete(**arguments)
                (seed_dir / "stage2_predictions.csv").write_text(
                    "sample_id,prediction\nfixture,tampered\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "stage2_predictions failed SHA-256"
                ):
                    campaign._recover_if_complete(**arguments)

    def test_formal_c1_can_reuse_only_an_explicitly_validated_legacy_reference(self) -> None:
        legacy = Stage0Reference(
            PROJECT_ROOT / "models/vle/overall_binary/seed_0/best_model.pt",
            None,
            "legacy_c1",
        )
        calls: list[dict[str, object]] = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return {"status": "completed", "selected_stage": "stage3"}

        with patch.object(campaign, "legacy_c1_stage0_reference", return_value=legacy), patch.object(
            campaign, "run_paper_experiment", side_effect=fake_run
        ), patch.object(campaign, "_require_three_stage_artifacts"):
            campaign.main(
                ["--variant", "c1_three_view_vanilla", "--seeds", "0", "--device", "cpu"]
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["config_path"], config_path(PROJECT_ROOT, "c1_three_view_vanilla")
        )
        self.assertEqual(calls[0]["stage1_checkpoint"], legacy.checkpoint)
        self.assertIsNone(calls[0]["generated_stage1_manifest"])
        self.assertEqual(calls[0]["run_kind"], "formal")

        self.assertFalse(calls[0]["aggregate_expected"])
    def test_legacy_stage0_recipe_requires_complete_semantic_match(self) -> None:
        recipe = {
            "schema": "thermoformer-stage0-recipe-v1",
            "variant": "c1_three_view_vanilla",
            "protocol": {"split": "current", "partitions": {"train": "a"}},
            "data": {"dataset_sha256": "data", "catalog_sha256": "catalog"},
            "features": {"cache_sha256": "features", "encoder": {"unimol": True}},
            "model": {"hidden_dim": 192, "layers": 3},
            "training": {"epochs_supervised": 80, "learning_rate": 2e-4},
            "loss": {"pressure_weight": 1.0, "pure_weight": 0.5},
            "selection": {"partition": "validation", "test_metrics_used_for_selection": False},
        }
        valid, reasons = legacy_stage0_matches_current_recipe(
            recipe, stage0_recipe_fingerprint(recipe), recipe
        )
        self.assertTrue(valid)
        self.assertEqual(reasons, ())

        changes = {
            "protocol": {"split": "changed", "partitions": {"train": "a"}},
            "data": {"dataset_sha256": "other", "catalog_sha256": "catalog"},
            "features": {"cache_sha256": "other", "encoder": {"unimol": True}},
            "model": {"hidden_dim": 256, "layers": 3},
            "training": {"epochs_supervised": 79, "learning_rate": 2e-4},
            "loss": {"pressure_weight": 0.5, "pure_weight": 0.5},
            "selection": {"partition": "test", "test_metrics_used_for_selection": True},
        }
        for field, changed_value in changes.items():
            candidate = deepcopy(recipe)
            candidate[field] = changed_value
            valid, reasons = legacy_stage0_matches_current_recipe(
                candidate, stage0_recipe_fingerprint(candidate), recipe
            )
            self.assertFalse(valid, field)
            self.assertIn(field, reasons)

        valid, reasons = legacy_stage0_matches_current_recipe(None, None, recipe)
        self.assertFalse(valid)
        self.assertEqual(reasons, ("missing stage0_recipe",))
        valid, reasons = legacy_stage0_matches_current_recipe(
            recipe, "not-the-recipe-hash", recipe
        )
        self.assertFalse(valid)
        self.assertEqual(reasons, ("stage0_recipe_sha256 mismatch",))

    def test_existing_legacy_c1_manifest_falls_back_without_versioned_recipe(self) -> None:
        # The historical binary C1 result predates validation-only Stage 0
        # recipe provenance, so it must be regenerated rather than reused.
        with patch.object(campaign, "_tracked_and_clean", return_value=True):
            self.assertIsNone(campaign.legacy_c1_stage0_reference(PROJECT_ROOT, 0))
if __name__ == "__main__":
    unittest.main()
