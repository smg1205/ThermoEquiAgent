import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.config import DataConfig, EncoderConfig, load_experiment_config
from src.model import ThermoFormer, ThermoFormerConfig
from src.training import TrainingConfig
from src.thermoformer.cli import validate_generic_training_config
from src.thermoformer.configuration import ProtocolConfig


class ExperimentConfigTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_ROOT = ROOT / "configs"
    BASE_CONFIG = ROOT / "configs" / "training" / "supervised.yaml"

    def assert_only_named_config_changes(
        self,
        candidate_path: Path,
        section_changes: dict[str, dict[str, object]],
    ) -> None:
        base = load_experiment_config(self.BASE_CONFIG)
        candidate = load_experiment_config(candidate_path)
        expected = base.to_dict()
        expected["name"] = candidate.name
        expected["runtime"]["output_dir"] = candidate.runtime.output_dir
        expected["runtime"]["results_file"] = candidate.runtime.results_file
        for section, changes in section_changes.items():
            expected[section].update(changes)
        self.assertEqual(candidate.to_dict(), expected)

    def test_json_config_and_dotted_overrides_define_an_ablation(self) -> None:
        payload = {
            "model": {"hidden_dim": 64, "layers": 2, "heads": 4},
            "evaluation": {"mode": "kfold", "folds": 5},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_experiment_config(
                path,
                overrides=["model.use_film=false", "evaluation.folds=3"],
            )

        self.assertEqual(config.model.hidden_dim, 64)
        self.assertFalse(config.model.use_film)
        self.assertEqual(config.evaluation.folds, 3)
        self.assertEqual(config.evaluation.mode, "kfold")

    def test_default_encoder_preserves_legacy_unimol_baseline(self) -> None:
        encoder = EncoderConfig()

        self.assertEqual(encoder.representation, "unimol_v2")
        self.assertEqual(encoder.fusion_mode, "legacy")

    def test_multiview_encoder_requires_views_and_explicit_fusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one branch"):
            EncoderConfig(
                representation="multiview",
                fusion_mode="naive",
                use_rdkit_descriptors=False,
                use_unimol=False,
                use_functional_groups=False,
            )
        encoder = EncoderConfig(
            representation="multiview",
            fusion_mode="interaction_specific",
        )
        self.assertEqual(encoder.fusion_mode, "interaction_specific")

    def test_chemical_attention_requires_multiview_rdkit_and_unimol(self) -> None:
        with self.assertRaisesRegex(ValueError, "RDKit and Uni-Mol"):
            ThermoFormerConfig(
                feature_dim=3,
                hidden_dim=12,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=3,
                chemical_attention_bias=True,
            )
        config = ThermoFormerConfig(
            feature_dim=7,
            hidden_dim=12,
            heads=3,
            fusion_mode="naive",
            rdkit_feature_dim=3,
            unimol_feature_dim=4,
            chemical_attention_bias=True,
            context_pair_interaction=True,
        )
        self.assertTrue(config.chemical_attention_bias)
        self.assertTrue(config.context_pair_interaction)

    def test_repository_configs_construct_thermoformer(self) -> None:
        paths = sorted(
            set()
            | set(self.CONFIG_ROOT.rglob("config.yaml"))
        )

        self.assertGreater(len(paths), 0)

        for path in paths:
            with self.subTest(config=path):
                experiment = load_experiment_config(path)
                model = ThermoFormer(replace(experiment.model, feature_dim=8))
                self.assertEqual(model.config.feature_dim, 8)

    def test_registered_configs_are_separate_from_results(self) -> None:
        catalog = json.loads((self.CONFIG_ROOT / "catalog.json").read_text(encoding="utf-8"))
        for item in catalog["experiments"]:
            config = self.CONFIG_ROOT / item["config"]
            self.assertTrue(config.is_file())
            entry = json.loads(config.read_text(encoding="utf-8"))
            for relative in entry["configs"]:
                self.assertTrue((self.ROOT / relative).is_file(), relative)

    def test_cyclic_config_inheritance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text('{"extends": "b.json"}', encoding="utf-8")
            (root / "b.json").write_text('{"extends": "a.json"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cyclic experiment configuration"):
                load_experiment_config(root / "a.json")

    def test_unknown_config_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"modell": {"use_film": False}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown experiment configuration"):
                load_experiment_config(path)

            valid = Path(directory) / "valid.json"
            valid.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration section"):
                load_experiment_config(valid, overrides=["modell.use_film=false"])

    def test_conflicting_training_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(
                json.dumps({"seed": 7, "training": {"seed": 8}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "training.seed"):
                load_experiment_config(path)

    def test_invalid_training_hyperparameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            TrainingConfig(epochs_supervised=1.5)
        with self.assertRaises(ValueError):
            TrainingConfig(solver_iterations_eval=0)
        with self.assertRaises(ValueError):
            ThermoFormerConfig(film_scale=float("inf"))
        with self.assertRaises(ValueError):
            ThermoFormerConfig(layers=1.5)
        with self.assertRaises(ValueError):
            DataConfig(max_pressure_kpa=float("nan"))
        with self.assertRaises(ValueError):
            EncoderConfig(representation="unknown")

    def test_ablation_switches_are_explicit_and_validated(self) -> None:
        config = ThermoFormerConfig(
            interaction_mode="pairwise",
            activity_mode="direct_gamma",
            decoder_mode="thermodynamic",
        )

        self.assertEqual(config.interaction_mode, "pairwise")
        self.assertEqual(config.activity_mode, "direct_gamma")
        with self.assertRaises(ValueError):
            ThermoFormerConfig(interaction_mode="implicit")
        with self.assertRaises(ValueError):
            ThermoFormerConfig(decoder_mode="black_box")

    def test_protocol_config_rejects_undeclared_requests(self) -> None:
        protocol = ProtocolConfig(
            registered_splits=("overall_binary_ternary",),
            seeds=(0, 1),
            evaluation_partition="test",
        )
        protocol.validate_request("overall_binary_ternary", 0, "test")
        with self.assertRaisesRegex(ValueError, "not declared"):
            protocol.validate_request("unseen_component", 0, "test")
        with self.assertRaisesRegex(ValueError, "Seed 2"):
            protocol.validate_request("overall_binary_ternary", 2, "test")
        with self.assertRaisesRegex(ValueError, "partition"):
            protocol.validate_request("overall_binary_ternary", 0, "validation")

    def test_generic_trainer_rejects_registered_and_physics_configs(self) -> None:
        supervised = load_experiment_config(
            self.ROOT / "configs" / "training" / "supervised.yaml"
        )
        validate_generic_training_config(supervised)
        with self.assertRaisesRegex(ValueError, "Stage-1 supervised"):
            validate_generic_training_config(
                load_experiment_config(
                    self.ROOT / "configs" / "training" / "fugacity_finetuning.yaml"
                )
            )
        with self.assertRaisesRegex(ValueError, "Stage-1 supervised"):
            validate_generic_training_config(
                load_experiment_config(
                    self.ROOT
                    / "configs"
                    / "protocols"
                    / "overall_binary_ternary.yaml"
                )
            )


if __name__ == "__main__":
    unittest.main()
