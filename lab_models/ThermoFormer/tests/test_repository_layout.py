import hashlib
import io
import json
from pathlib import Path
import re
import tokenize
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u9fff]")


class RepositoryLayoutTests(unittest.TestCase):
    def test_release_dataset_contains_only_two_workbooks(self) -> None:
        files = sorted(
            path.relative_to(PROJECT_ROOT / 'datasets/vle_reference').as_posix()
            for path in (PROJECT_ROOT / 'datasets/vle_reference').rglob("*")
            if path.is_file()
        )
        self.assertEqual(
            files,
            ["binary_vle_english.xlsx", "ternary_vle_english.xlsx"],
        )

    def test_active_source_paths_are_ascii(self) -> None:
        for root_name in ("analysis", "src", "scripts", "tests", "configs", "experiments"):
            for path in (PROJECT_ROOT / root_name).rglob("*"):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                self.assertTrue(relative.isascii(), relative)

    def test_active_python_comments_are_english(self) -> None:
        for root_name in ("analysis", "src", "scripts", "tests"):
            for path in (PROJECT_ROOT / root_name).rglob("*.py"):
                source = path.read_text(encoding="utf-8-sig")
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                for token in tokens:
                    if token.type == tokenize.COMMENT:
                        self.assertIsNone(
                            HAN.search(token.string),
                            f"Non-English comment in {path.relative_to(PROJECT_ROOT)}:{token.start[0]}",
                        )

    def test_reference_code_is_isolated_from_active_imports(self) -> None:
        self.assertTrue((PROJECT_ROOT / 'docs/reference_records/README.md').is_file())
        for root_name in ("src", "scripts"):
            for path in (PROJECT_ROOT / root_name).rglob("*.py"):
                source = path.read_text(encoding="utf-8-sig")
                self.assertNotIn("archive.legacy_code", source)
                self.assertNotIn("docs/reference_records", source)

    def test_experiment_navigation_declares_incomplete_studies(self) -> None:
        experiment_map = (PROJECT_ROOT / "docs" / "experiment_code_map.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Machine-learning and thermodynamic VLE baselines", experiment_map)
        self.assertIn("Agent-assisted separation design", experiment_map)
        self.assertIn("unavailable", experiment_map.lower())

    def test_active_experiments_cover_existing_research_tasks(self) -> None:
        active = {
            path.name
            for path in (PROJECT_ROOT / "experiments").iterdir()
            if path.is_dir() and any(path.rglob("*"))
        }
        self.assertEqual(active, {"vle", "lle", "summary", "data_quality", "run_records", "reference_results"})
        representation_variants = {
            path.name
            for path in (
                PROJECT_ROOT / 'configs/vle/ablation/studies/molecular_representation'
            ).iterdir()
            if path.is_dir()
        }
        self.assertEqual(
            representation_variants,
            {
                "unimol_v2_only",
                "rdkit_only",
                "functional_groups_only",
                "rdkit_unimol",
                "full_three_view",
            },
        )
        interaction_variants = {
            path.name
            for path in (
                PROJECT_ROOT / 'configs/vle/ablation/studies/interaction_architecture'
            ).iterdir()
            if path.is_dir()
        }
        self.assertEqual(
            interaction_variants,
            {
                "vanilla_transformer",
                "chemical_interaction_bias",
                "context_pair_without_attention_bias",
            },
        )

    def test_result_figures_are_inside_experiments(self) -> None:
        self.assertFalse((PROJECT_ROOT / "analysis").exists())
        figure = PROJECT_ROOT / "experiments/vle/interpretability/molecular_interactions/figures/vle_interpretability_complete.png"
        self.assertTrue(figure.is_file())
        self.assertEqual(hashlib.sha256(figure.read_bytes()).hexdigest(), "aad0801b940baae279d1346a5213038e4f2c27be6c6315029d59c56251089a54")

    def test_active_implementation_uses_manuscript_subpackages(self) -> None:
        expected = {
            "data",
            "evaluation",
            "features",
            "interpretability",
            "models",
            "protocols",
            "reporting",
            "thermodynamics",
            "training",
        }
        package_root = PROJECT_ROOT / "src" / "thermoformer"
        actual = {path.name for path in package_root.iterdir() if path.is_dir()}
        self.assertTrue(expected.issubset(actual))
        for name in expected:
            self.assertTrue((package_root / name / "__init__.py").is_file(), name)

    def test_manuscript_configs_resolve(self) -> None:
        from src.thermoformer.configuration import load_experiment_config

        config_root = PROJECT_ROOT / "configs"
        paths = [
            config_root / "model" / "c1_three_view_vanilla.yaml",
            config_root / "training" / "supervised.yaml",
            config_root / "training" / "fugacity_finetuning.yaml",
            config_root / "protocols" / "overall_binary_ternary.yaml",
        ]
        configs = [load_experiment_config(path) for path in paths]
        self.assertTrue(all(config.encoder.representation == "multiview" for config in configs))
        self.assertFalse(configs[-1].encoder.chemical_attention_bias)
        self.assertEqual(configs[-1].training.epochs_physics, 10)
        self.assertEqual(configs[-1].protocol.registered_splits, ("overall_binary_ternary",))
        self.assertEqual(configs[-1].protocol.seeds, (0, 1, 2, 3, 4))

    def test_checkpoint_documentation_matches_manuscript_artifacts(self) -> None:
        optional_root = PROJECT_ROOT / 'models/vle/experiments/physics_finetuning'
        if not optional_root.is_dir() or not any(optional_root.rglob("*.pt")):
            self.skipTest("Ablation checkpoints are optional and are not distributed")
        readme = (PROJECT_ROOT / 'models/vle/README.md').read_text(
            encoding="utf-8"
        )
        documented_stages = {}
        for line in readme.splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) != 5 or not cells[2].startswith("`"):
                continue
            if not cells[3].startswith(("S1", "S2")):
                continue
            protocol = cells[2].strip("`")
            documented_stages[protocol] = cells[3].split(", ")

        result_root = (
            PROJECT_ROOT
            / 'experiments/vle/generalization/evaluations/physics_finetuning/c1_three_view_vanilla_fugacity'
        )
        checkpoint_root = (
            PROJECT_ROOT
            / 'models/vle/experiments/physics_finetuning'
            / "c1_three_view_vanilla_fugacity"
        )
        actual_stages = {}
        for protocol_dir in sorted(result_root.iterdir()):
            if not protocol_dir.is_dir():
                continue
            protocol = protocol_dir.name.removeprefix(
                "c1_three_view_vanilla_fugacity_finetune.on."
            )
            stages = []
            for seed in range(5):
                seed_dir = protocol_dir / f"seed_{seed}"
                comparison = json.loads(
                    (seed_dir / "stage_comparison.json").read_text(encoding="utf-8")
                )
                stages.append(comparison["selected_stage"].upper().replace("STAGE", "S"))
                self.assertTrue((seed_dir / "manifest.json").is_file())
                weight_dir = checkpoint_root / protocol_dir.name / f"seed_{seed}"
                self.assertTrue((weight_dir / "best_model.pt").is_file())
                self.assertTrue((weight_dir / "stage2_best_model.pt").is_file())
                self.assertTrue(
                    (PROJECT_ROOT / comparison["stage1_checkpoint"]).is_file()
                )
            actual_stages[protocol] = stages
        self.assertEqual(documented_stages, actual_stages)

        ablation_roots = (
            "models/vle/multiview/chemical_attention/formal/"
            "c0_current_vanilla.on.overall_binary_ternary",
            "models/vle/multiview/formal/"
            "v1_rdkit_only.on.overall_binary_ternary",
            "models/vle/multiview/predictive/"
            "v3_functional_group_only.on.overall_binary_ternary",
            "models/vle/multiview/predictive/"
            "v4_rdkit_unimol_naive.on.overall_binary_ternary",
            "models/vle/multiview/chemical_attention/formal/"
            "c1_three_view_vanilla.on.overall_binary_ternary",
            "models/vle/multiview/chemical_attention/formal/"
            "c2_chemical_bias_full.on.overall_binary_ternary",
            "models/vle/multiview/chemical_attention/formal/"
            "c3_no_pair_bias.on.overall_binary_ternary",
        )
        for relative_root in ablation_roots:
            self.assertIn(relative_root, readme)
            for seed in range(5):
                self.assertTrue(
                    (
                        PROJECT_ROOT
                        / relative_root
                        / f"seed_{seed}"
                        / "best_model.pt"
                    ).is_file()
                )

        self.assertRegex(readme, r"no\s+seed-specific checkpoint is claimed")


if __name__ == "__main__":
    unittest.main()








