from dataclasses import asdict
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_public_and_legacy_model_imports_are_identical(self) -> None:
        from src.model import ThermoFormer as LegacyThermoFormer
        from src.thermoformer import ThermoFormer

        self.assertIs(ThermoFormer, LegacyThermoFormer)

    def test_final_architecture_preset_matches_manuscript(self) -> None:
        from src.model import ThermoFormerConfig
        from src.thermoformer import final_model_config

        config = final_model_config()
        self.assertEqual(config.feature_dim, 820)
        self.assertEqual(config.rdkit_feature_dim, 24)
        self.assertEqual(config.unimol_feature_dim, 768)
        self.assertEqual(config.functional_group_feature_dim, 28)
        self.assertFalse(config.chemical_attention_bias)
        self.assertFalse(config.context_pair_interaction)
        self.assertEqual(config.interaction_mode, "full")
        self.assertTrue(config.use_film)
        self.assertTrue(config.use_mixture_token)

        resolved_path = (
            PROJECT_ROOT
            / 'experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity/c1_three_view_vanilla_fugacity_finetune.on.overall_binary_ternary/seed_0/resolved_config.json'
        )
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        formal_config = ThermoFormerConfig(**payload["model"])
        self.assertEqual(asdict(config), asdict(formal_config))

    def test_public_solver_imports_are_identical(self) -> None:
        from src.thermo import solve_isobaric as legacy_solve_isobaric
        from src.thermoformer import solve_isobaric

        self.assertIs(solve_isobaric, legacy_solve_isobaric)

    def test_public_scientific_modules_import(self) -> None:
        from src.thermoformer import configuration, data, evaluation, features, training

        self.assertTrue(callable(configuration.load_experiment_config))
        self.assertTrue(callable(data.load_vle_dataset))
        self.assertTrue(callable(evaluation.predict_vle))
        self.assertTrue(callable(features.build_molecular_encoder))
        self.assertTrue(callable(training.fit_physics_stage))

    def test_protocol_evaluation_is_not_a_row_prediction_alias(self) -> None:
        from src.thermoformer.evaluation import evaluate_protocol, predict_vle

        self.assertIsNot(evaluate_protocol, predict_vle)
        self.assertIn("split", evaluate_protocol.__annotations__)

    def test_method_classes_are_owned_by_their_scientific_modules(self) -> None:
        from src.thermoformer.features.functional_groups import FunctionalGroupEncoder
        from src.thermoformer.features.rdkit_descriptors import RDKit2DEncoder
        from src.thermoformer.features.unimol_v2 import UniMolV2Encoder
        from src.thermoformer.models.interaction import ChemicalBiasedTransformer
        from src.thermoformer.models.vapor_pressure import PureVaporPressure

        classes = (
            FunctionalGroupEncoder,
            RDKit2DEncoder,
            UniMolV2Encoder,
            ChemicalBiasedTransformer,
            PureVaporPressure,
        )
        for value in classes:
            self.assertFalse(value.__module__.endswith(".features.fusion"))
            self.assertFalse(value.__module__.endswith(".models.thermoformer"))


if __name__ == "__main__":
    unittest.main()
