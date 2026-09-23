import unittest
from dataclasses import fields
from pathlib import Path

from src.config import PhysicsFineTuningConfig, load_experiment_config
from src.training import TrainingConfig


class FinalExperimentScopeTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_final_c1_configuration_is_three_view_vanilla(self) -> None:
        config = load_experiment_config(
            self.ROOT
            / "configs/vle/ablation/studies/fugacity_finetuning/config.yaml"
        )
        self.assertEqual(config.encoder.representation, "multiview")
        self.assertEqual(config.encoder.fusion_mode, "naive")
        self.assertTrue(config.encoder.use_rdkit_descriptors)
        self.assertTrue(config.encoder.use_unimol)
        self.assertTrue(config.encoder.use_functional_groups)
        self.assertFalse(config.encoder.chemical_attention_bias)
        self.assertFalse(config.encoder.context_pair_interaction)

    def test_stage2_configuration_exposes_only_fugacity_loss(self) -> None:
        training_fields = {field.name for field in fields(TrainingConfig)}
        self.assertFalse(
            {
                "continuity_weight",
                "boundary_weight",
                "solver_weight",
                "chemical_bias_weight",
                "solver_batches_per_epoch",
            }
            & training_fields
        )
        finetuning_fields = {field.name for field in fields(PhysicsFineTuningConfig)}
        self.assertIn("teacher_forced_fugacity_weight", finetuning_fields)
        self.assertNotIn(
            "additional_pure_vapor_pressure_anchor_weight", finetuning_fields
        )

    def test_canonical_ablation_campaign_is_binary_only(self) -> None:
        config = load_experiment_config(
            self.ROOT
            / "configs/vle/ablation/studies/overall_binary_three_stage/three_stage_base.yaml"
        )
        self.assertEqual(config.protocol.registered_splits, ("overall_binary",))
        self.assertEqual(config.protocol.seeds, tuple(range(5)))
        self.assertEqual(config.direct_ge_supervision.pretrain_epochs, 20)
        self.assertEqual(config.training.epochs_supervised, 80)
        self.assertEqual(config.direct_ge_supervision.fugacity_epochs, 10)

    def test_only_fugacity_physics_experiment_remains(self) -> None:
        self.assertFalse((self.ROOT / "experiments/physics_finetuning").exists())
        self.assertTrue(
            (self.ROOT / "configs/vle/ablation/studies/fugacity_finetuning/config.yaml").is_file()
        )


if __name__ == "__main__":
    unittest.main()
