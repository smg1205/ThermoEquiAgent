from dataclasses import asdict
import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _evaluation_script():
    path = PROJECT_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("thermoformer_public_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PublicEvaluationTests(unittest.TestCase):
    def test_checkpoint_provenance_accepts_exact_inputs(self) -> None:
        from src.thermoformer.models import ThermoFormerConfig

        module = _evaluation_script()
        model_config = ThermoFormerConfig()
        expected = {"dataset_sha256": "data", "split_sha256": "split", "seed": 0}
        checkpoint = {**expected, "model_config": asdict(model_config)}
        module.validate_checkpoint_provenance(checkpoint, expected, model_config)

    def test_checkpoint_provenance_rejects_mismatched_split(self) -> None:
        from src.thermoformer.models import ThermoFormerConfig

        module = _evaluation_script()
        model_config = ThermoFormerConfig()
        expected = {"dataset_sha256": "data", "split_sha256": "expected", "seed": 0}
        checkpoint = {
            "dataset_sha256": "data",
            "split_sha256": "different",
            "seed": 0,
            "model_config": asdict(model_config),
        }
        with self.assertRaisesRegex(RuntimeError, "split_sha256"):
            module.validate_checkpoint_provenance(checkpoint, expected, model_config)

    def test_checkpoint_provenance_rejects_model_mismatch(self) -> None:
        from src.thermoformer.models import ThermoFormerConfig

        module = _evaluation_script()
        expected_config = ThermoFormerConfig()
        checkpoint = {"model_config": asdict(ThermoFormerConfig(hidden_dim=96))}
        with self.assertRaisesRegex(RuntimeError, "model configuration"):
            module.validate_checkpoint_provenance(checkpoint, {}, expected_config)

    def test_test_partition_rejects_unselected_stage2_checkpoint(self) -> None:
        module = _evaluation_script()
        checkpoint = {
            "checkpoint_role": "best_physics_epoch_by_validation",
            "selected_as_final": False,
        }
        module.validate_checkpoint_selection(checkpoint, "validation")
        with self.assertRaisesRegex(RuntimeError, "not selected on validation"):
            module.validate_checkpoint_selection(checkpoint, "test")


if __name__ == "__main__":
    unittest.main()
