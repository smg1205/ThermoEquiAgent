from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.training.direct_ge_pipeline import fit_direct_ge_stages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegisteredDirectGEExperimentTests(unittest.TestCase):
    def test_registered_config_keeps_final_c1_architecture(self) -> None:
        config = load_experiment_config(
            PROJECT_ROOT
            / 'configs/vle/ablation/studies/direct_ge_supervision/config.yaml'
        )

        self.assertEqual(config.encoder.representation, "multiview")
        self.assertEqual(config.encoder.fusion_mode, "naive")
        self.assertTrue(config.encoder.use_rdkit_descriptors)
        self.assertTrue(config.encoder.use_unimol)
        self.assertTrue(config.encoder.use_functional_groups)
        self.assertFalse(config.encoder.chemical_attention_bias)
        self.assertFalse(config.encoder.context_pair_interaction)
        self.assertEqual(config.training.epochs_supervised, 80)
        self.assertEqual(config.direct_ge_supervision.fugacity_epochs, 10)
        self.assertEqual(config.data.pure_property_catalog,
                         "datasets/derived/thermodynamic_labels/external_psat_catalog.json")

    def test_binary_only_config_uses_same_three_stage_method(self) -> None:
        config = load_experiment_config(
            PROJECT_ROOT
            / 'configs/vle/comparison/studies/binary_only/thermoformer_direct_ge.yaml'
        )

        self.assertEqual(config.protocol.registered_splits, ("overall_binary",))
        self.assertEqual(config.encoder.representation, "multiview")
        self.assertEqual(config.encoder.fusion_mode, "naive")
        self.assertTrue(config.encoder.use_rdkit_descriptors)
        self.assertTrue(config.encoder.use_unimol)
        self.assertTrue(config.encoder.use_functional_groups)
        self.assertFalse(config.encoder.chemical_attention_bias)
        self.assertFalse(config.encoder.context_pair_interaction)
        self.assertEqual(config.direct_ge_supervision.pretrain_epochs, 20)
        self.assertEqual(config.training.epochs_supervised, 80)
        self.assertEqual(config.direct_ge_supervision.fugacity_epochs, 10)
        self.assertEqual(config.direct_ge_supervision.excess_gibbs_weight, 1.0)
        self.assertEqual(config.direct_ge_supervision.activity_coefficient_weight, 0.5)
        self.assertEqual(config.direct_ge_supervision.vle_weight, 1.0)
        self.assertEqual(config.direct_ge_supervision.fugacity_weight, 0.01)
        self.assertEqual(
            config.data.pure_property_catalog,
            "datasets/derived/thermodynamic_labels/external_psat_catalog.json",
        )

    def test_stage_fitter_cannot_receive_test_samples(self) -> None:
        parameters = inspect.signature(fit_direct_ge_stages).parameters
        self.assertIn("validation_samples", parameters)
        self.assertNotIn("test_samples", parameters)
        self.assertNotIn("evaluation_samples", parameters)


if __name__ == "__main__":
    unittest.main()
