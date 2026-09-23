from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch

from src.thermoformer.configuration import (
    DirectGESupervisionConfig,
    PhysicsFineTuningConfig,
    load_experiment_config,
)
from src.thermoformer.models import ThermoFormer, ThermoFormerConfig
from src.thermoformer.training.supervised import TrainingConfig
from src.thermoformer.training.direct_ge_pipeline import (
    configure_ge_pretraining,
    fit_direct_ge_stages,
    validation_composite_score,
)


class DirectGEPipelineTests(unittest.TestCase):
    def test_configuration_exposes_all_three_stage_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                """{
                  "name": "direct_ge",
                  "direct_ge_supervision": {
                    "enabled": true,
                    "pretrain_epochs": 20,
                    "excess_gibbs_weight": 1.0,
                    "activity_coefficient_weight": 0.5,
                    "vle_weight": 1.0,
                    "fugacity_weight": 0.01,
                    "minimum_fraction": 0.0001,
                    "warmup_epochs": 2
                  }
                }""",
                encoding="utf-8",
            )
            config = load_experiment_config(path)

        self.assertTrue(config.direct_ge_supervision.enabled)
        self.assertEqual(config.direct_ge_supervision.pretrain_epochs, 20)
        self.assertEqual(config.direct_ge_supervision.fugacity_epochs, 10)
        self.assertAlmostEqual(config.direct_ge_supervision.fugacity_weight, 0.01)

    def test_ge_pretraining_optimizer_excludes_frozen_vapor_pressure(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(feature_dim=6, hidden_dim=12, layers=1, heads=3)
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("vapor_pressure")
        }
        optimizer = configure_ge_pretraining(model, learning_rate=2e-4, weight_decay=1e-4)
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(before)
        for name, parameter in model.named_parameters():
            if name.startswith("vapor_pressure"):
                self.assertFalse(parameter.requires_grad)
                self.assertNotIn(id(parameter), optimized)
                torch.testing.assert_close(parameter, before[name])

    def test_stage_states_record_independent_validation_best_epochs(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

        def validation_metrics(scale: float) -> dict[str, float]:
            return {
                "isothermal_pressure_mae_kpa": 10.0 * scale,
                "isothermal_y_mae": 0.03 * scale,
                "isobaric_temperature_mae_k": 3.0 * scale,
                "isobaric_y_mae": 0.03 * scale,
                "isothermal_coverage": 1.0,
                "isobaric_coverage": 1.0,
                "isothermal_solver_failure_rate": 0.0,
                "isobaric_solver_failure_rate": 0.0,
            }

        model = TinyModel()
        with torch.no_grad():
            model.value.fill_(5.0)
        baseline_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        setup = SimpleNamespace(
            optimizer=object(),
            total_parameters=1,
            trainable_parameters=1,
            record=lambda: {"groups": []},
        )

        def fake_epoch(current_model, *args, **kwargs):
            with torch.no_grad():
                current_model.value.add_(1.0)
            return {"total": 0.0}

        label_metrics = {
            "excess_gibbs_rt_mae": 0.0,
            "log_gamma_mae": 0.0,
            "ge_label_coverage": 1.0,
            "gamma_label_coverage": 1.0,
            "ge_labeled_states": 1.0,
            "gamma_labeled_components": 2.0,
        }
        with (
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline._loader",
                return_value=object(),
            ),
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline._validation_summary",
                side_effect=[
                    validation_metrics(1.0),
                    validation_metrics(0.9),
                    validation_metrics(float("inf")),
                    validation_metrics(float("inf")),
                ],
            ),
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline.evaluate_direct_label_metrics",
                return_value=label_metrics,
            ),
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline._run_direct_epoch",
                side_effect=fake_epoch,
            ),
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline.configure_ge_pretraining",
                return_value=object(),
            ),
            mock.patch(
                "src.thermoformer.training.direct_ge_pipeline.configure_physics_finetuning",
                return_value=setup,
            ),
        ):
            result = fit_direct_ge_stages(
                model,
                train_samples=[object()],
                feature_map={},
                config=TrainingConfig(
                    batch_size=1,
                    epochs_supervised=1,
                    early_stopping_patience=0,
                    solver_iterations_eval=1,
                    seed=7,
                ),
                supervision=DirectGESupervisionConfig(
                    pretrain_epochs=1,
                    fugacity_epochs=1,
                ),
                finetuning=PhysicsFineTuningConfig(),
                device=torch.device("cpu"),
                validation_samples=[object()],
                pure_property_catalog=None,
                baseline_state=baseline_state,
            )

        self.assertEqual(
            tuple(result.stage_states), ("stage0", "stage1", "stage2", "stage3")
        )
        self.assertEqual(
            result.stage_best_epochs,
            {"stage0": 0, "stage1": 1, "stage2": 1, "stage3": 1},
        )
        self.assertEqual(result.selected_stage, "stage1")
        self.assertEqual(result.selection_partitions, ("validation",))
        self.assertEqual(result.stage_validation_losses["stage0"], 1.0)
        self.assertEqual(result.stage_validation_losses["stage3"], float("inf"))
        for stage_name, expected in (("stage0", 5.0), ("stage1", 6.0), ("stage2", 7.0), ("stage3", 8.0)):
            self.assertEqual(result.stage_states[stage_name]["value"].item(), expected)

    def test_validation_composite_rejects_coverage_regression(self) -> None:
        baseline = {
            "isothermal_pressure_mae_kpa": 10.0,
            "isothermal_y_mae": 0.03,
            "isobaric_temperature_mae_k": 3.0,
            "isobaric_y_mae": 0.03,
            "isothermal_coverage": 1.0,
            "isobaric_coverage": 1.0,
        }
        improved = {
            **baseline,
            "isothermal_pressure_mae_kpa": 8.0,
            "isothermal_y_mae": 0.02,
            "isobaric_temperature_mae_k": 2.0,
            "isobaric_y_mae": 0.02,
        }
        reduced_coverage = {**improved, "isobaric_coverage": 0.99}

        self.assertLess(validation_composite_score(improved, baseline), 1.0)
        self.assertEqual(
            validation_composite_score(reduced_coverage, baseline), float("inf")
        )


if __name__ == "__main__":
    unittest.main()
