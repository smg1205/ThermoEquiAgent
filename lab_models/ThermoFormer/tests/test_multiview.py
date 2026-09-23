import unittest
from pathlib import Path

import numpy as np
import torch

from src.model import ThermoFormer, ThermoFormerConfig
from src.representation import RDKitDescriptorScaler
from src.thermo import solve_isobaric, solve_isothermal


def multiview_model() -> ThermoFormer:
    torch.manual_seed(7)
    return ThermoFormer(
        ThermoFormerConfig(
            feature_dim=9,
            hidden_dim=12,
            layers=1,
            heads=3,
            pair_hidden_dim=12,
            rdkit_feature_dim=2,
            unimol_feature_dim=4,
            functional_group_feature_dim=3,
            fusion_mode="interaction_specific",
        )
    )


class MultiViewRepresentationTests(unittest.TestCase):
    def test_headwise_chemical_bias_is_gated_symmetric_and_curriculum_controlled(self) -> None:
        torch.manual_seed(13)
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=9,
                hidden_dim=12,
                layers=2,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=3,
                chemical_attention_bias=True,
                context_pair_interaction=True,
                chemical_bias_headwise=True,
                chemical_bias_gate_init=0.03,
                chemical_bias_warmup_start=5,
                chemical_bias_warmup_end=20,
                chemical_bias_hidden_dim=8,
                chemical_bias_dropout=0.05,
            )
        ).eval()
        molecules = torch.randn(1, 3, 9)
        temperature = torch.tensor([[345.0]])
        pressure = torch.tensor([[120.0]])
        composition = torch.tensor([[0.2, 0.3, 0.5]])
        mask = torch.ones(1, 3)

        self.assertEqual(tuple(model.chemical_head_gates().shape), (2, 3))
        torch.testing.assert_close(
            model.chemical_head_gates(), torch.full((2, 3), 0.03), atol=1e-6, rtol=0.0
        )
        model.set_training_epoch(4)
        before = model(molecules, temperature, pressure, composition, mask)
        assert before.attention_bias is not None
        torch.testing.assert_close(before.attention_bias, torch.zeros_like(before.attention_bias))
        model.set_training_epoch(12)
        middle = model(molecules, temperature, pressure, composition, mask)
        model.set_training_epoch(20)
        full = model(molecules, temperature, pressure, composition, mask)
        assert middle.attention_bias is not None and full.attention_bias is not None
        torch.testing.assert_close(
            middle.attention_bias,
            full.attention_bias * (7.0 / 15.0),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(full.attention_bias, full.attention_bias.transpose(1, 2))
        self.assertTrue(full.attention_bias.requires_grad)
        assert full.attention_bias_penalty is not None
        full.attention_bias_penalty.backward()
        self.assertIsNotNone(model.chemical_head_gate_logits.grad)

    def test_modality_gates_start_with_reduced_functional_group_contribution(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=9,
                hidden_dim=12,
                layers=2,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=3,
                chemical_attention_bias=True,
                context_pair_interaction=True,
                chemical_bias_headwise=True,
                chemical_bias_modality_gates=True,
                chemical_bias_gate_init=0.03,
                chemical_bias_functional_group_gate_init=0.1,
                chemical_bias_hidden_dim=8,
            )
        )
        gates = model.chemical_modality_gates()
        self.assertEqual(set(gates), {"rdkit", "unimol", "functional_group", "state"})
        self.assertAlmostEqual(float(gates["rdkit"]), 0.95, places=6)
        self.assertAlmostEqual(float(gates["unimol"]), 0.95, places=6)
        self.assertAlmostEqual(float(gates["functional_group"]), 0.1, places=6)
        self.assertAlmostEqual(float(gates["state"]), 0.95, places=6)

    def test_chemical_attention_bias_is_symmetric_and_permutation_equivariant(self) -> None:
        torch.manual_seed(17)
        config = ThermoFormerConfig(
            feature_dim=10,
            hidden_dim=12,
            layers=2,
            heads=3,
            fusion_mode="naive",
            rdkit_feature_dim=2,
            unimol_feature_dim=5,
            functional_group_feature_dim=3,
            chemical_attention_bias=True,
            context_pair_interaction=True,
        )
        model = ThermoFormer(config).eval()
        molecules = torch.randn(2, 3, 10)
        temperature = torch.tensor([[330.0], [360.0]])
        pressure = torch.tensor([[101.325], [150.0]])
        x = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
        mask = torch.ones(2, 3)

        output = model(molecules, temperature, pressure, x, mask)
        self.assertIsNotNone(output.attention_bias)
        assert output.attention_bias is not None
        torch.testing.assert_close(
            output.attention_bias,
            output.attention_bias.transpose(1, 2),
        )

        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            molecules[:, permutation],
            temperature,
            pressure,
            x[:, permutation],
            mask[:, permutation],
        )
        torch.testing.assert_close(
            output.log_gamma[:, permutation], permuted.log_gamma, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            output.pair_interactions[:, permutation][:, :, permutation],
            permuted.pair_interactions,
            rtol=2e-5,
            atol=2e-5,
        )
        output.log_gamma.square().sum().backward()
        assert model.chemical_bias_mlp is not None
        bias_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.chemical_bias_mlp.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(bias_gradient, 0.0)

    def test_context_pair_changes_with_third_component_environment(self) -> None:
        torch.manual_seed(23)
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=7,
                hidden_dim=12,
                layers=1,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=3,
                functional_group_feature_dim=2,
                chemical_attention_bias=True,
                context_pair_interaction=True,
            )
        ).eval()
        ab = torch.randn(1, 2, 7)
        abc = torch.cat([ab, torch.randn(1, 1, 7)], dim=1)
        binary = model(
            ab,
            torch.tensor([[340.0]]),
            torch.tensor([[101.325]]),
            torch.tensor([[0.4, 0.6]]),
            torch.ones(1, 2),
        )
        ternary = model(
            abc,
            torch.tensor([[340.0]]),
            torch.tensor([[101.325]]),
            torch.tensor([[0.3, 0.45, 0.25]]),
            torch.ones(1, 3),
        )
        self.assertFalse(
            torch.allclose(binary.pair_interactions[:, 0, 1], ternary.pair_interactions[:, 0, 1])
        )

    def test_chemical_attention_supports_functional_group_ablation(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=6,
                hidden_dim=12,
                layers=1,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=0,
                chemical_attention_bias=True,
                context_pair_interaction=True,
            )
        )
        output = model(
            torch.randn(1, 3, 6),
            torch.tensor([[350.0]]),
            torch.tensor([[101.325]]),
            torch.tensor([[0.2, 0.3, 0.5]]),
            torch.ones(1, 3),
        )
        self.assertTrue(torch.isfinite(output.log_gamma).all())

    def test_chemical_attention_gamma_is_independent_of_input_grad_flag(self) -> None:
        torch.manual_seed(29)
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=9,
                hidden_dim=12,
                layers=1,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=3,
                chemical_attention_bias=True,
                context_pair_interaction=True,
            )
        ).eval()
        molecules = torch.randn(1, 3, 9)
        temperature = torch.tensor([[345.0]])
        pressure = torch.tensor([[120.0]])
        composition = torch.tensor([[0.2, 0.3, 0.5]])
        mask = torch.ones(1, 3)
        without_flag = model(molecules, temperature, pressure, composition, mask)
        with_flag = model(
            molecules,
            temperature,
            pressure,
            composition.clone().requires_grad_(True),
            mask,
        )
        torch.testing.assert_close(without_flag.log_gamma, with_flag.log_gamma)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_chemical_attention_supports_cuda_higher_order_backward(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=9,
                hidden_dim=12,
                layers=1,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=3,
                chemical_attention_bias=True,
                context_pair_interaction=True,
            )
        ).cuda()
        output = model(
            torch.randn(2, 3, 9, device="cuda"),
            torch.full((2, 1), 345.0, device="cuda"),
            torch.full((2, 1), 120.0, device="cuda"),
            torch.tensor(
                [[0.2, 0.3, 0.5], [0.4, 0.6, 0.0]], device="cuda"
            ),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], device="cuda"),
        )
        output.log_gamma.square().mean().backward()
        self.assertTrue(
            torch.isfinite(model.chemical_bias_mlp[-1].weight.grad).all()
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_headwise_modality_bias_supports_cuda_higher_order_backward(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(
                feature_dim=9,
                hidden_dim=12,
                layers=2,
                heads=3,
                fusion_mode="naive",
                rdkit_feature_dim=2,
                unimol_feature_dim=4,
                functional_group_feature_dim=3,
                chemical_attention_bias=True,
                context_pair_interaction=True,
                chemical_bias_headwise=True,
                chemical_bias_modality_gates=True,
                chemical_bias_hidden_dim=8,
                chemical_bias_warmup_start=5,
                chemical_bias_warmup_end=20,
            )
        ).cuda()
        model.set_training_epoch(20)
        output = model(
            torch.randn(1, 3, 9, device="cuda"),
            torch.full((1, 1), 345.0, device="cuda"),
            torch.full((1, 1), 120.0, device="cuda"),
            torch.tensor([[0.2, 0.3, 0.5]], device="cuda"),
            torch.ones(1, 3, device="cuda"),
        )
        output.log_gamma.square().mean().backward()
        assert model.chemical_bias_mlps is not None
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.chemical_bias_mlps.parameters()
            )
        )
        self.assertIsNotNone(model.chemical_head_gate_logits.grad)

    def test_rdkit_scaler_uses_train_molecules_only(self) -> None:
        raw = {
            "train-a": np.asarray([0.0, 10.0], dtype=np.float32),
            "train-b": np.asarray([2.0, 14.0], dtype=np.float32),
            "held-out": np.asarray([100.0, 100.0], dtype=np.float32),
        }
        scaler = RDKitDescriptorScaler.fit(raw, ["train-a", "train-b"], ("a", "b"))
        np.testing.assert_allclose(scaler.mean, [1.0, 12.0])
        np.testing.assert_allclose(scaler.std, [1.0, 2.0])
        self.assertEqual(scaler.fit_smiles, ("train-a", "train-b"))
        self.assertGreater(float(scaler.transform(raw["held-out"])[0]), 90.0)

    def test_multiview_forward_binary_and_ternary(self) -> None:
        model = multiview_model()
        molecules = torch.randn(2, 3, 9)
        temperature = torch.full((2, 1), 350.0)
        pressure = torch.full((2, 1), 101.325)
        x = torch.tensor([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        output = model(molecules, temperature, pressure, x, mask, return_view_weights=True)
        self.assertEqual(output.log_gamma.shape, (2, 3))
        self.assertEqual(output.view_weights.shape, (2, 3, 3, 3))
        self.assertTrue(torch.isfinite(output.log_gamma).all())
        self.assertTrue(torch.isfinite(output.excess_gibbs_rt).all())
        self.assertTrue(torch.allclose(output.view_weights[0, 0, 1].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.all(output.view_weights[0, 0, 2] == 0.0))

    def test_pair_interaction_and_component_permutation_symmetry(self) -> None:
        model = multiview_model().eval()
        molecules = torch.randn(1, 3, 9)
        temperature = torch.tensor([[360.0]])
        pressure = torch.tensor([[120.0]])
        x = torch.tensor([[0.2, 0.3, 0.5]])
        mask = torch.ones_like(x)
        permutation = torch.tensor([2, 0, 1])
        original = model(molecules, temperature, pressure, x, mask, return_view_weights=True)
        permuted = model(
            molecules[:, permutation], temperature, pressure, x[:, permutation],
            mask[:, permutation], return_view_weights=True,
        )
        self.assertTrue(torch.allclose(original.excess_gibbs_rt, permuted.excess_gibbs_rt, atol=1e-6))
        self.assertTrue(torch.allclose(original.log_gamma[:, permutation], permuted.log_gamma, atol=1e-5))
        expected_pairs = original.pair_interactions[:, permutation][:, :, permutation]
        self.assertTrue(torch.allclose(expected_pairs, permuted.pair_interactions, atol=1e-6))
        expected_weights = original.view_weights[:, permutation][:, :, permutation]
        self.assertTrue(torch.allclose(expected_weights, permuted.view_weights, atol=1e-6))
        original_state = solve_isothermal(
            model, molecules, temperature, x, mask, iterations=3, strict=False
        )
        permuted_state = solve_isothermal(
            model, molecules[:, permutation], temperature, x[:, permutation],
            mask[:, permutation], iterations=3, strict=False,
        )
        self.assertTrue(torch.allclose(original_state.pressure_kpa, permuted_state.pressure_kpa, atol=1e-5))
        self.assertTrue(torch.allclose(original_state.y[:, permutation], permuted_state.y, atol=1e-5))
        original_isobaric = solve_isobaric(
            model, molecules, pressure, x, mask, iterations=3, strict=False
        )
        permuted_isobaric = solve_isobaric(
            model, molecules[:, permutation], pressure, x[:, permutation],
            mask[:, permutation], iterations=3, strict=False,
        )
        self.assertTrue(torch.allclose(
            original_isobaric.temperature_k, permuted_isobaric.temperature_k, atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            original_isobaric.y[:, permutation], permuted_isobaric.y, atol=1e-5
        ))

    def test_excess_gibbs_activity_and_solver_gradients_are_finite(self) -> None:
        model = multiview_model()
        molecules = torch.randn(1, 3, 9)
        temperature = torch.tensor([[350.0]])
        x = torch.tensor([[0.25, 0.35, 0.40]], requires_grad=True)
        mask = torch.ones_like(x)
        output = model(molecules, temperature, torch.tensor([[101.325]]), x, mask)
        composition_gradient = torch.autograd.grad(
            output.excess_gibbs_rt.sum(), x, create_graph=True
        )[0]
        self.assertTrue(torch.isfinite(composition_gradient).all())
        self.assertTrue(torch.isfinite(output.log_gamma).all())
        state = solve_isothermal(
            model, molecules, temperature, x, mask, iterations=2, strict=False
        )
        state.pressure_kpa.sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_legacy_unimol_state_dict_contract_is_preserved(self) -> None:
        model = ThermoFormer(ThermoFormerConfig(feature_dim=4, hidden_dim=8, layers=1, heads=2))
        self.assertIn("molecular_encoder.0.weight", model.state_dict())
        self.assertFalse(any(name.startswith("view_projectors") for name in model.state_dict()))

    def test_legacy_unimol_regression_from_published_checkpoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            root / 'models/vle/overall_binary_ternary/seed_0/best_model.pt'
        )
        if not checkpoint_path.is_file():
            self.skipTest("Optional ablation checkpoint is not distributed")
        self.assertFalse(
            checkpoint_path.read_bytes().startswith(b"version https://git-lfs"),
            "Git LFS checkpoint was not materialized; run `git lfs pull` before tests",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ThermoFormer(ThermoFormerConfig(**checkpoint["model_config"])).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        feature_dim = int(checkpoint["model_config"]["feature_dim"])
        molecules = torch.linspace(-1.0, 1.0, steps=2 * feature_dim).reshape(
            1, 2, feature_dim
        )
        x = torch.tensor([[0.4, 0.6]])
        output = model(
            molecules,
            torch.tensor([[350.0]]),
            torch.tensor([[101.325]]),
            x,
            torch.ones_like(x),
        )
        torch.testing.assert_close(
            output.excess_gibbs_rt.detach(), torch.tensor([[0.08884782]]),
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            output.log_gamma.detach(), torch.tensor([[0.04087700, 0.12082836]]),
            rtol=1e-5, atol=1e-6,
        )

    def test_original_chemical_bias_checkpoint_remains_loadable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            root / 'models/vle/multiview/chemical_attention/formal'
            / "c2_chemical_bias_full.on.overall_binary_ternary" / "seed_0"
            / "best_model.pt"
        )
        if not checkpoint_path.is_file():
            self.skipTest("Optional ablation checkpoint is not distributed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = ThermoFormer(ThermoFormerConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"], strict=True)


if __name__ == "__main__":
    unittest.main()



