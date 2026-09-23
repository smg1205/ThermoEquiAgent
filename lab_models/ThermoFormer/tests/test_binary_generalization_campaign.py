from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.run_binary_generalization_three_stage import (
    BINARY_STATE_PROTOCOLS,
    BINARY_TRANSFER_PROTOCOLS,
    BINARY_UNSEEN_PROTOCOLS,
    aggregation_provenance_overrides,
    parser,
    artifact_roots,
    campaign_config_path,
    smoke_overrides,
    supervised_config_path,
)
from src.thermoformer.configuration import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BinaryGeneralizationCampaignTests(unittest.TestCase):
    def test_campaign_registers_state_unseen_and_transfer_protocols(self) -> None:
        self.assertEqual(len(BINARY_STATE_PROTOCOLS), 6)
        self.assertEqual(BINARY_UNSEEN_PROTOCOLS, ("binary_unseen_component",))
        self.assertEqual(
            BINARY_TRANSFER_PROTOCOLS,
            (
                "binary_to_ternary_zero_shot",
                "binary_to_ternary_scale_0.05",
                "binary_to_ternary_scale_0.1",
                "binary_to_ternary_scale_0.25",
                "binary_to_ternary_scale_0.5",
                "binary_to_ternary_scale_1",
            ),
        )

    def test_configs_reuse_the_current_three_stage_c1_method(self) -> None:
        for group, protocols in (
            ("state", BINARY_STATE_PROTOCOLS),
            ("unseen", BINARY_UNSEEN_PROTOCOLS),
            ("transfer", BINARY_TRANSFER_PROTOCOLS),
        ):
            direct = load_experiment_config(campaign_config_path(PROJECT_ROOT, group))
            supervised = load_experiment_config(supervised_config_path(PROJECT_ROOT, group))
            self.assertEqual(direct.protocol.registered_splits, protocols)
            self.assertEqual(supervised.protocol.registered_splits, protocols)
            self.assertEqual(supervised.protocol.evaluation_partition, "validation")
            self.assertEqual(direct.protocol.evaluation_partition, "test")
            self.assertEqual(direct.direct_ge_supervision.pretrain_epochs, 20)
            self.assertEqual(direct.training.epochs_supervised, 80)
            self.assertEqual(direct.direct_ge_supervision.fugacity_epochs, 10)
            self.assertFalse(direct.encoder.chemical_attention_bias)
            self.assertFalse(direct.encoder.context_pair_interaction)

    def test_registered_transfer_splits_keep_fixed_ternary_test_rows(self) -> None:
        for seed in range(5):
            payloads = []
            for protocol in BINARY_TRANSFER_PROTOCOLS:
                split_path = PROJECT_ROOT / f"datasets/splits/vle/{protocol}/seed_{seed}.json"
                payloads.append(
                    json.loads(
                        split_path.read_text(encoding="utf-8")
                    )
                )
            expected_test_rows = payloads[0]["partitions"]["test"]
            self.assertTrue(expected_test_rows)
            for payload in payloads:
                self.assertEqual(payload["partitions"]["test"], expected_test_rows)
                self.assertEqual(
                    payload["metadata"]["component_count_rows"]["test"],
                    {"3": len(expected_test_rows)},
                )

    def test_smoke_is_isolated_and_shortens_all_three_stages(self) -> None:
        formal = artifact_roots(PROJECT_ROOT, "state", "three_stage", smoke=False)
        smoke = artifact_roots(PROJECT_ROOT, "state", "three_stage", smoke=True)
        self.assertNotEqual(formal, smoke)
        self.assertTrue(all("experiments/run_records/smoke".replace("/", "\\") in str(path) for path in smoke))
        overrides = smoke_overrides("direct")
        self.assertIn("direct_ge_supervision.pretrain_epochs=1", overrides)
        self.assertIn("training.epochs_supervised=1", overrides)
        self.assertIn("direct_ge_supervision.fugacity_epochs=1", overrides)

    def test_cli_exposes_audited_mixed_commit_aggregation(self) -> None:
        arguments = parser().parse_args(
            (
                "--group",
                "state",
                "--compatible-training-commits",
                "commit-a",
                "commit-b",
                "--mixed-commit-justification",
                "Only a results report was committed between resumed runs.",
            )
        )
        self.assertEqual(
            aggregation_provenance_overrides(arguments),
            {
                "compatible_training_git_commits": ("commit-a", "commit-b"),
                "mixed_commit_justification": "Only a results report was committed between resumed runs.",
            },
        )

    def test_cli_rejects_mixed_commits_without_justification(self) -> None:
        arguments = parser().parse_args(
            ("--group", "state", "--compatible-training-commits", "commit-a", "commit-b")
        )
        with self.assertRaisesRegex(ValueError, "justification"):
            aggregation_provenance_overrides(arguments)


if __name__ == "__main__":
    unittest.main()
