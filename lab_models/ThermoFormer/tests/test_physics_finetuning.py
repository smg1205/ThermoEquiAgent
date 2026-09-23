import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_c1_physics_finetune import (
    output_roots,
    parser as physics_runner_parser,
    physics_report_path,
    recover_completed_seed_manifest,
    require_materialized_checkpoint,
    stage1_checkpoint_path,
)
from src.artifacts import artifact_sha256
from src.config import PhysicsFineTuningConfig, load_experiment_config
from src.data import VLESample
from src.model import ThermoFormer, ThermoFormerConfig
from src.physics_finetuning import (
    configure_physics_finetuning,
    fit_physics_stage,
    load_stage1_checkpoint,
    physics_finetune_objective,
    physics_warmup_scale,
    summarize_physics_finetuning,
    write_multiseed_physics_finetune_report,
)
from src.training import TrainingConfig, _loader


class PhysicsFineTuningTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def model() -> ThermoFormer:
        return ThermoFormer(
            ThermoFormerConfig(
                feature_dim=6,
                hidden_dim=8,
                layers=1,
                heads=2,
                pair_hidden_dim=8,
                pure_hidden_dim=8,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=2,
                functional_group_feature_dim=2,
                chemical_attention_bias=False,
                context_pair_interaction=False,
            )
        )

    @staticmethod
    def samples() -> list[VLESample]:
        return [
            VLESample(
                smiles=("A", "B"),
                names=("A", "B"),
                temperature_k=340.0 + index,
                pressure_kpa=70.0 + index,
                liquid_composition=(x, 1.0 - x),
                vapor_composition=(min(0.95, x + 0.08), max(0.05, 0.92 - x)),
                quality_weight=1.0,
                quality_status="passed",
                source="synthetic.xlsx",
                doi=f"physics-{index}",
                experiment_mode="isothermal" if index % 2 == 0 else "isobaric",
            )
            for index, x in enumerate((0.2, 0.4, 0.6, 0.8))
        ]

    @staticmethod
    def features() -> dict[str, np.ndarray]:
        return {
            "A": np.asarray([1.0, 0.2, -0.1, 0.4, 1.0, 0.0], dtype=np.float32),
            "B": np.asarray([0.1, 1.0, 0.5, -0.2, 0.0, 1.0], dtype=np.float32),
        }

    @staticmethod
    def config(**overrides: object) -> TrainingConfig:
        values = dict(
            batch_size=4,
            epochs_supervised=80,
            epochs_physics=1,
            minimum_physics_epochs=1,
            early_stopping_patience=0,
            solver_iterations_eval=4,
        )
        values.update(overrides)
        return TrainingConfig(**values)

    @staticmethod
    def finetuning() -> PhysicsFineTuningConfig:
        return PhysicsFineTuningConfig(teacher_forced_fugacity_weight=1.0)

    def test_final_configuration_is_c1_with_fugacity_only(self) -> None:
        config = load_experiment_config(
            self.ROOT
            / "configs/vle/ablation/studies/fugacity_finetuning/config.yaml"
        )
        self.assertEqual(config.encoder.representation, "multiview")
        self.assertEqual(config.encoder.fusion_mode, "naive")
        self.assertFalse(config.encoder.chemical_attention_bias)
        self.assertFalse(config.encoder.context_pair_interaction)
        self.assertEqual(config.training.epochs_physics, 10)
        self.assertEqual(config.training.minimum_physics_epochs, 10)
        self.assertEqual(config.physics_finetuning.teacher_forced_fugacity_weight, 1.0)

    def test_partial_optimizer_contains_only_declared_unfrozen_groups(self) -> None:
        model = self.model()
        setup = configure_physics_finetuning(model, self.config(), self.finetuning())
        optimizer_ids = {
            id(parameter)
            for group in setup.optimizer.param_groups
            for parameter in group["params"]
        }
        expected_prefixes = ("pair_potential.", "vapor_pressure.", "film.")
        for name, parameter in model.named_parameters():
            expected = name == "mixture_token" or name.startswith(expected_prefixes)
            self.assertEqual(parameter.requires_grad, expected, name)
            self.assertEqual(id(parameter) in optimizer_ids, expected, name)
        self.assertLess(setup.trainable_parameters, setup.total_parameters)

    def test_c1_keeps_pair_potential_group_and_learning_rate(self) -> None:
        finetuning = self.finetuning()
        setup = configure_physics_finetuning(self.model(), self.config(), finetuning)

        self.assertEqual(
            [group.name for group in setup.groups],
            ["pair_potential", "vapor_pressure", "film", "mixture_token"],
        )
        self.assertEqual(setup.groups[0].learning_rate, finetuning.pair_potential_lr)
        self.assertTrue(
            all(name.startswith("pair_potential.") for name in setup.groups[0].parameter_names)
        )

    def test_context_pair_variants_resolve_the_interaction_group(self) -> None:
        finetuning = self.finetuning()
        base = dict(
            feature_dim=6,
            hidden_dim=8,
            layers=1,
            heads=2,
            pair_hidden_dim=8,
            pure_hidden_dim=8,
            fusion_mode="naive",
            rdkit_feature_dim=2,
            unimol_feature_dim=2,
            functional_group_feature_dim=2,
            context_pair_interaction=True,
        )
        for variant_name, chemical_attention_bias in (
            ("c2_chemical_bias_full", True),
            ("c3_no_pair_bias", False),
        ):
            with self.subTest(variant=variant_name):
                model = ThermoFormer(
                    ThermoFormerConfig(
                        **base,
                        chemical_attention_bias=chemical_attention_bias,
                    )
                )
                self.assertIsNone(model.pair_potential)
                self.assertIsNotNone(model.context_pair_potential)

                setup = configure_physics_finetuning(model, self.config(), finetuning)
                context_group = next(
                    group
                    for group in setup.groups
                    if group.name == "context_pair_potential"
                )
                self.assertEqual(context_group.learning_rate, finetuning.pair_potential_lr)
                self.assertTrue(context_group.parameter_names)
                self.assertTrue(
                    all(
                        name.startswith("context_pair_potential.")
                        for name in context_group.parameter_names
                    )
                )
                optimized = {
                    id(parameter)
                    for optimizer_group in setup.optimizer.param_groups
                    for parameter in optimizer_group["params"]
                }
                for name, parameter in model.named_parameters():
                    if name.startswith("context_pair_potential."):
                        self.assertTrue(parameter.requires_grad, name)
                        self.assertIn(id(parameter), optimized)

    def test_missing_interaction_potential_fails_clearly(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=6,
                hidden_dim=8,
                layers=1,
                heads=2,
                pair_hidden_dim=8,
                pure_hidden_dim=8,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=2,
                functional_group_feature_dim=2,
                interaction_mode="independent",
            )
        )
        self.assertIsNone(model.pair_potential)
        self.assertIsNone(model.context_pair_potential)
        with self.assertRaisesRegex(
            ValueError,
            "interaction-potential.*pair_potential.*context_pair_potential",
        ):
            configure_physics_finetuning(model, self.config(), self.finetuning())

    def test_frozen_parameters_stay_fixed_and_unfrozen_groups_receive_gradients(self) -> None:
        model = self.model()
        config = self.config()
        setup = configure_physics_finetuning(model, config, self.finetuning())
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        batch = next(iter(_loader(self.samples(), self.features(), config, False)))
        objective = physics_finetune_objective(
            model,
            batch,
            config,
            physics_scale=1.0,
            teacher_forced_fugacity_weight=1.0,
        )
        setup.optimizer.zero_grad(set_to_none=True)
        objective.total.backward()
        for group in setup.groups:
            gradients = [p.grad for p in group.parameters if p.grad is not None]
            self.assertTrue(gradients, group.name)
            norm = torch.stack([gradient.norm() for gradient in gradients]).sum()
            self.assertTrue(torch.isfinite(norm), group.name)
            self.assertGreater(float(norm), 0.0, group.name)
        setup.optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(parameter.detach(), frozen_before[name])

    def test_fugacity_loss_has_real_pair_potential_gradient(self) -> None:
        model = self.model()
        config = self.config()
        configure_physics_finetuning(model, config, self.finetuning())
        batch = next(iter(_loader(self.samples(), self.features(), config, False)))
        objective = physics_finetune_objective(
            model,
            batch,
            config,
            physics_scale=1.0,
            teacher_forced_fugacity_weight=1.0,
        )
        objective.teacher_forced_fugacity.backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("pair_potential.") and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        norm = torch.stack([gradient.norm() for gradient in gradients]).sum()
        self.assertTrue(torch.isfinite(norm))
        self.assertGreater(float(norm), 0.0)

    def test_stage2_loads_stage1_and_uses_validation_only_for_selection(self) -> None:
        model = self.model()
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "training_config": {"epochs_physics": 0},
                },
                checkpoint,
            )
            for parameter in model.parameters():
                parameter.data.add_(10.0)
            load_stage1_checkpoint(model, checkpoint)
            stage1 = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            result = fit_physics_stage(
                model,
                self.samples(),
                self.features(),
                config,
                self.finetuning(),
                torch.device("cpu"),
                validation_samples=self.samples(),
            )
        for name in stage1:
            torch.testing.assert_close(result.stage_states["stage1"][name], stage1[name])
        self.assertEqual(result.selection_partitions, ("validation",))

    def test_stage1_loader_rejects_a_physics_trained_checkpoint(self) -> None:
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "physics.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "training_config": {"epochs_physics": 5},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "supervised-only"):
                load_stage1_checkpoint(model, checkpoint)

    def test_warmup_and_smoke_isolation(self) -> None:
        self.assertEqual(
            [physics_warmup_scale(epoch, 2) for epoch in (1, 2, 3)],
            [0.5, 1.0, 1.0],
        )
        formal = output_roots(self.ROOT, smoke=False)
        smoke = output_roots(self.ROOT, smoke=True)
        self.assertNotEqual(formal, smoke)
        self.assertIn("smoke", str(smoke[0]))
        self.assertIn("c1_three_view_vanilla_fugacity", formal[0].as_posix())

    def test_generalization_protocol_selects_its_own_split_and_stage1_checkpoint(self) -> None:
        args = physics_runner_parser().parse_args(
            ["--protocol", "unseen_component", "--seeds", "0", "1"]
        )
        self.assertEqual(args.protocol, "unseen_component")
        self.assertEqual(args.seeds, [0, 1])
        self.assertEqual(
            stage1_checkpoint_path(self.ROOT, "unseen_component", 3),
            self.ROOT / "models/vle/unseen_component/seed_3/best_model.pt",
        )
        self.assertIn(
            "c1_three_view_vanilla.on.overall_binary_ternary",
            stage1_checkpoint_path(self.ROOT, "overall_binary_ternary", 3).as_posix(),
        )

    def test_physics_runner_rejects_unregistered_protocol(self) -> None:
        with self.assertRaises(SystemExit):
            physics_runner_parser().parse_args(["--protocol", "made_up_protocol"])

    def test_checkpoint_preflight_rejects_an_unhydrated_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            checkpoint.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "0" * 64 + "\nsize 123\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
                require_materialized_checkpoint(checkpoint)

    def test_nonoverall_report_stays_in_the_result_namespace(self) -> None:
        protocol_dir = self.ROOT / "experiments/reference_results/example.on.unseen_component"
        self.assertEqual(
            physics_report_path(
                self.ROOT,
                protocol_dir,
                "unseen_component",
                smoke=False,
            ),
            protocol_dir / "results.md",
        )
        self.assertIn("experiments/", physics_report_path(
            self.ROOT, protocol_dir, "unseen_component", smoke=False
        ).as_posix())

    def test_multiseed_summary_is_paired_and_seed_aware(self) -> None:
        source = (
            self.ROOT
            / 'experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity/c1_three_view_vanilla_fugacity_finetune.on.overall_binary_ternary/seed_0/stage_comparison.json'
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in (0, 1):
                path = root / f"seed_{seed}/stage_comparison.json"
                path.parent.mkdir(parents=True)
                candidate = json.loads(json.dumps(payload))
                candidate["selected_stage"] = "stage2" if seed == 0 else "stage1"
                candidate["stages"]["stage2"]["validation_loss"] += 0.01 * seed
                path.write_text(json.dumps(candidate), encoding="utf-8")
                paths.append(path)
            summary = summarize_physics_finetuning(paths)
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(summary["selected_stage_counts"], {"stage1": 1, "stage2": 1})
        self.assertGreater(summary["stages"]["stage2"]["validation_loss"]["std"], 0.0)
        expected_pressure = (
            payload["stages"]["stage2"]["metrics"][2]["pressure_mae_kpa"]
            + payload["stages"]["stage1"]["metrics"][2]["pressure_mae_kpa"]
        ) / 2.0
        self.assertAlmostEqual(
            summary["stages"]["selected"]["directions"]["isothermal"][
                "pressure_mae_kpa"
            ]["mean"],
            expected_pressure,
        )

    def test_multiseed_report_includes_final_metrics_and_parameter_audit(self) -> None:
        result_root = (
            self.ROOT
            / 'experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity/c1_three_view_vanilla_fugacity_finetune.on.overall_binary_ternary'
        )
        paths = [result_root / f"seed_{seed}/stage_comparison.json" for seed in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            report, _ = write_multiseed_physics_finetune_report(
                paths,
                Path(directory) / "results.md",
                physics_epochs=10,
            )
            text = report.read_text(encoding="utf-8")
        self.assertIn("Validation-selected final test metrics", text)
        self.assertIn("physics fine-tuning: `10` epoch", text)
        self.assertIn("Total parameters: `2,015,043`", text)
        self.assertIn("solver failure", text)
        self.assertIn("pair_potential", text)

    def test_report_recovery_rejects_stale_status_or_corrupt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "metrics.json"
            artifact.write_text("{}\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = {
                "status": "completed",
                "protocol": "example",
                "seed": 0,
                "evaluation_partition": "test",
                "request_sha256": "current",
                "analysis_status": "confirmatory",
                "artifacts": {
                    "metrics": {
                        "path": str(artifact),
                        "sha256": artifact_sha256(artifact),
                    }
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            recovered = recover_completed_seed_manifest(
                manifest_path,
                expected_status="completed",
                expected_protocol="example",
                expected_seed=0,
                expected_evaluation_partition="test",
                expected_request_sha256="current",
                expected_analysis_status="confirmatory",
            )
            self.assertEqual(recovered["request_sha256"], "current")
            manifest["analysis_status"] = "diagnostic"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale analysis_status"):
                recover_completed_seed_manifest(
                    manifest_path,
                    expected_status="completed",
                    expected_protocol="example",
                    expected_seed=0,
                    expected_evaluation_partition="test",
                    expected_request_sha256="current",
                    expected_analysis_status="confirmatory",
                )
            manifest["analysis_status"] = "confirmatory"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            artifact.write_text("corrupt\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed SHA"):
                recover_completed_seed_manifest(
                    manifest_path,
                    expected_status="completed",
                    expected_protocol="example",
                    expected_seed=0,
                    expected_evaluation_partition="test",
                    expected_request_sha256="current",
                    expected_analysis_status="confirmatory",
                )


if __name__ == "__main__":
    unittest.main()
