"""Public CLI and configuration checks for the joint training-stage ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.thermoformer.configuration import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ThreeStageTrainingAblationCampaignTests(unittest.TestCase):
    def test_campaign_uses_only_the_registered_joint_protocol(self) -> None:
        import scripts.run_three_stage_training_ablation as campaign

        self.assertEqual(campaign.SPLIT_PROTOCOL, "overall_binary_ternary")
        self.assertEqual(
            campaign.resolve_seeds(campaign.parser().parse_args([])),
            (0, 1, 2, 3, 4),
        )
        self.assertEqual(
            campaign.resolve_seeds(campaign.parser().parse_args(["--smoke"])),
            (0,),
        )
        self.assertEqual(campaign.STAGES, ("stage0", "stage1", "stage2", "stage3"))

    def test_configs_preserve_the_approved_three_stage_budget(self) -> None:
        import scripts.run_three_stage_training_ablation as campaign

        stage0 = load_experiment_config(campaign.stage0_config_path(PROJECT_ROOT))
        three_stage = load_experiment_config(campaign.three_stage_config_path(PROJECT_ROOT))
        self.assertEqual(stage0.protocol.registered_splits, ("overall_binary_ternary",))
        self.assertEqual(three_stage.protocol.registered_splits, ("overall_binary_ternary",))
        self.assertEqual(stage0.protocol.evaluation_partition, "validation")
        self.assertEqual(three_stage.protocol.evaluation_partition, "test")
        self.assertEqual(stage0.training.epochs_supervised, 80)
        self.assertEqual(three_stage.direct_ge_supervision.pretrain_epochs, 20)
        self.assertEqual(three_stage.training.epochs_supervised, 80)
        self.assertEqual(three_stage.direct_ge_supervision.fugacity_epochs, 10)
        self.assertEqual(three_stage.direct_ge_supervision.fugacity_weight, 0.01)

    def test_formal_and_smoke_artifacts_are_strictly_isolated(self) -> None:
        import scripts.run_three_stage_training_ablation as campaign

        root = Path("project")
        formal = campaign.artifact_roots(root, smoke=False)
        smoke = campaign.artifact_roots(root, smoke=True)
        self.assertNotEqual(formal, smoke)
        self.assertTrue(all("experiments/run_records/smoke" in path.as_posix() for path in smoke))
        self.assertTrue(
            all(
                "three_stage_training/overall_binary_ternary" in path.as_posix()
                for path in formal
            )
        )

    def test_registered_splits_have_binary_and_ternary_training_and_test_rows(self) -> None:
        for seed in range(5):
            path = PROJECT_ROOT / 'datasets/splits/vle/overall_binary_ternary' / f"seed_{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["metadata"]["component_count_rows"]
            self.assertEqual(payload["protocol"], "overall_binary_ternary")
            for partition in ("train", "validation", "test"):
                self.assertGreater(rows[partition]["2"], 0)
                self.assertGreater(rows[partition]["3"], 0)
                self.assertEqual(sum(rows[partition].values()), payload["metadata"]["rows"][partition])
