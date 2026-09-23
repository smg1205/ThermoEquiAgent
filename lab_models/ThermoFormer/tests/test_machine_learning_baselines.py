from __future__ import annotations

import unittest
import csv
import hashlib
import tempfile
from pathlib import Path

import torch

from src.thermoformer.baselines.machine_learning.models import (
    DescriptorANN,
    GEGNN,
    SMILESRNN,
    UALFGNN,
    muggianu_binary_weights,
    trainable_parameter_count,
)
from src.thermoformer.baselines.machine_learning.schema import BASELINE_CAPABILITIES
from src.thermoformer.baselines.machine_learning.protocols import (
    OVERALL_BENCHMARKS,
    UNIFIED_METRIC_KEYS,
    benchmark_for_baseline,
    native_evaluation_cells,
)
from src.thermoformer.baselines.machine_learning.artifacts import (
    aggregate_seed_metrics,
    write_seed_metrics,
)
from src.thermoformer.baselines.machine_learning.upstream import audit_pretrained_overlap
from src.thermoformer.baselines.machine_learning.hanna import (
    HANNA_SOURCE_REVISION,
    HANNA_EMBEDDING_DIMENSION,
    HANNAThermodynamicAdapter,
    OfficialHANNAAssets,
    OfficialHANNAInference,
)
from src.thermoformer.baselines.machine_learning.formal import _runtime_provenance
from src.thermoformer.baselines.machine_learning.spt_nrtl import (
    SPTNRTLDatabase,
    SPTNRTLPairParameters,
    multicomponent_nrtl_log_gamma,
)
from src.thermoformer.baselines.machine_learning.tennet_sac import TeNNetSACPredictor
from src.thermoformer.baselines.machine_learning.external_activity import (
    ExternalActivityThermodynamicAdapter,
)
from src.thermoformer.thermodynamics import solve_isobaric, solve_isothermal


OFFICIAL_HANNA_REVISION = "6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365"


class MachineLearningBaselineTests(unittest.TestCase):
    def test_external_activity_bridge_uses_shared_isothermal_and_isobaric_solvers(self) -> None:
        class IdealPredictor:
            @staticmethod
            def log_gamma(smiles, temperature_k, composition):
                return torch.zeros(len(smiles))

        adapter = ExternalActivityThermodynamicAdapter(IdealPredictor(), [("CCO", "O")])
        inverse_temperature = -2000.0
        molecules = torch.tensor([[[
            float(torch.log(torch.tensor(100.0))) - inverse_temperature / 350.0,
            inverse_temperature,
        ], [
            float(torch.log(torch.tensor(50.0))) - inverse_temperature / 350.0,
            inverse_temperature,
        ]]])
        x = torch.tensor([[0.5, 0.5]])
        mask = torch.ones_like(x)
        temperature = torch.tensor([[350.0]])
        isothermal = solve_isothermal(adapter, molecules, temperature, x, mask, strict=True)
        self.assertAlmostEqual(float(isothermal.pressure_kpa), 75.0, places=2)
        isobaric = solve_isobaric(
            adapter,
            molecules,
            torch.tensor([[75.0]]),
            x,
            mask,
            initial_temperature_k=torch.tensor([[340.0]]),
            strict=True,
        )
        self.assertAlmostEqual(float(isobaric.temperature_k), 350.0, places=2)

    def test_tennetsac_dispatches_binary_and_ternary_public_apis(self) -> None:
        calls = []

        def binary(smiles, temperature, x1, version="tuned"):
            calls.append(("binary", tuple(smiles), temperature, tuple(x1), version))
            return [0.1], [0.2]

        def multi(smiles, temperature, composition, version="tuned"):
            calls.append(("multi", tuple(smiles), temperature, tuple(composition), version))
            return [0.3, 0.4, 0.5]

        predictor = TeNNetSACPredictor(binary_predictor=binary, multi_predictor=multi)
        self.assertTrue(torch.allclose(
            predictor.log_gamma(("CCO", "O"), 298.15, (0.25, 0.75)),
            torch.tensor([0.1, 0.2]),
        ))
        self.assertTrue(torch.allclose(
            predictor.log_gamma(("CCO", "O", "CCN"), 310.0, (0.2, 0.3, 0.5)),
            torch.tensor([0.3, 0.4, 0.5]),
        ))
        self.assertEqual([call[0] for call in calls], ["binary", "multi"])
        self.assertTrue(all(call[-1] == "tuned" for call in calls))

    def test_spt_nrtl_pair_reversal_and_multicomponent_permutation(self) -> None:
        pair01 = SPTNRTLPairParameters(
            alpha=(0.25, 1e-4),
            tau_ij=(1.0, -20.0, 0.1, 1e-3),
            tau_ji=(-0.5, 30.0, -0.05, 2e-3),
        )
        reversed_pair = pair01.reversed()
        self.assertEqual(reversed_pair.tau_ij, pair01.tau_ji)
        self.assertEqual(reversed_pair.tau_ji, pair01.tau_ij)
        pair02 = SPTNRTLPairParameters((0.3, 2e-4), (0.2, 5.0, 0.02, 5e-4), (-0.1, 8.0, 0.03, 8e-4))
        pair12 = SPTNRTLPairParameters((0.2, 3e-4), (0.4, 2.0, -0.01, 7e-4), (0.7, -3.0, 0.01, 9e-4))
        x = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        original = multicomponent_nrtl_log_gamma(x, 330.0, {(0, 1): pair01, (0, 2): pair02, (1, 2): pair12})
        permuted = multicomponent_nrtl_log_gamma(
            x[[2, 0, 1]],
            330.0,
            {(0, 1): pair02.reversed(), (0, 2): pair12.reversed(), (1, 2): pair01},
        )
        self.assertTrue(torch.allclose(permuted, original[[2, 0, 1]], atol=1e-10, rtol=1e-10))

    def test_spt_nrtl_pure_endpoint_is_finite_and_unity(self) -> None:
        pair = SPTNRTLPairParameters((0.3, 0.0), (0.5, 10.0, 0.0, 0.0), (-0.2, 5.0, 0.0, 0.0))
        result = multicomponent_nrtl_log_gamma(
            torch.tensor([1.0, 0.0], dtype=torch.float64), 298.15, {(0, 1): pair}
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(float(result[0]), 0.0, places=10)

    def test_spt_nrtl_parses_fixed_official_row_and_matches_binary_reference(self) -> None:
        text = (
            "SMILES0,SMILES1,a_1,a_2,t_12_1,t_12_2,t_12_3,t_12_4,"
            "t_21_1,t_21_2,t_21_3,t_21_4\n"
            "CCO,O,0.242151,0.000169,15.340546,-104.544922,-2.894455,0.005724,"
            "15.432343,-355.378906,-2.293152,0.001618\n"
        )
        parameters = SPTNRTLDatabase._parse_row(
            text, "O", "fixed-revision-url", "fixed-file-sha"
        )
        self.assertEqual(parameters.alpha, (0.242151, 0.000169))
        self.assertEqual(
            parameters.selected_row_sha256,
            "cae534ab24abc3c2d65065edcb6e17948c2f148459e0c4d9b754b4a018433d05",
        )
        result = multicomponent_nrtl_log_gamma(
            torch.tensor([0.3, 0.7], dtype=torch.float64), 350.0, {(0, 1): parameters}
        )
        self.assertTrue(torch.allclose(
            result,
            torch.tensor([0.5903029528810406, 0.16799923088253196], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        ))

    def test_cached_fixed_revision_ethanol_file_matches_official_sha(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/run_records/cache/spt_nrtl/v3/C/3/CCO.csv"
        )
        if not path.is_file():
            self.skipTest("Fixed-revision SPT-NRTL cache is not materialized")
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "903e2fcbd174ec2dac61fbf56c093beb029219b7622f2f5d446b014695723981",
        )

    def test_spt_nrtl_requires_every_ternary_pair(self) -> None:
        pair = SPTNRTLPairParameters((0.3, 0.0), (0.5, 10.0, 0.0, 0.0), (-0.2, 5.0, 0.0, 0.0))
        with self.assertRaises(KeyError):
            multicomponent_nrtl_log_gamma(
                torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
                298.15,
                {(0, 1): pair, (0, 2): pair},
            )

    def test_spt_nrtl_lookup_persists_reverse_row_provenance(self) -> None:
        header = ",".join(SPTNRTLDatabase.columns) + "\n"
        reverse = header + (
            "O,CCO,0.2,0.0,1,2,3,4,5,6,7,8\n"
        )

        class InMemoryDatabase(SPTNRTLDatabase):
            def _load_file(self, smiles):
                if smiles == "CCO":
                    return header, "left-url", "left-sha"
                return reverse, "right-url", "right-sha"

        database = InMemoryDatabase(Path("unused"))
        pair = database.lookup("CCO", "O")
        self.assertEqual(pair.lookup_direction, "reverse")
        self.assertEqual(pair.tau_ij, (5.0, 6.0, 7.0, 8.0))
        self.assertEqual(database.pair_audit["CCO|O"]["lookup_direction"], "reverse")
        self.assertEqual(
            database.pair_audit["CCO|O"]["selected_row_sha256"],
            pair.selected_row_sha256,
        )

    def test_hanna_runtime_provenance_discloses_compatibility_environment(self) -> None:
        runtime = _runtime_provenance("hanna", "cpu")
        self.assertEqual(runtime["execution_profile"], "thermoformer_ggnn39_compatibility")
        self.assertEqual(runtime["upstream_environment_parity"], "not_established")
        self.assertIn("pandas", runtime["hanna_dependencies"])

    def test_official_hanna_assets_are_pinned_and_complete(self) -> None:
        assets = OfficialHANNAAssets.default(Path(__file__).resolve().parents[1])
        if not all(path.is_file() for path in assets.ensemble_paths):
            self.skipTest("Optional HANNA comparison weights are obtained from the upstream source")
        self.assertEqual(HANNA_SOURCE_REVISION, OFFICIAL_HANNA_REVISION)
        self.assertEqual(len(assets.ensemble_paths), 10)
        self.assertTrue(assets.chemberta_path.joinpath("model.safetensors").is_file())
        self.assertTrue(all(path.is_file() for path in assets.required_files()))

    def test_official_hanna_ensemble_runs_through_shared_solver(self) -> None:
        assets = OfficialHANNAAssets.default(Path(__file__).resolve().parents[1])
        if not all(path.is_file() for path in assets.ensemble_paths):
            self.skipTest("Optional HANNA comparison weights are obtained from the upstream source")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HANNAThermodynamicAdapter(assets, device)
        embeddings = torch.randn(2, 3, HANNA_EMBEDDING_DIMENSION, device=device) * 0.05
        psat_coefficients = torch.tensor(
            [[[8.0, -1800.0], [8.1, -1900.0], [7.9, -1700.0]]],
            device=device,
        ).repeat(2, 1, 1)
        molecules = torch.cat([embeddings, psat_coefficients], dim=-1)
        x = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]], device=device)
        state = solve_isothermal(
            model,
            molecules,
            torch.full((2, 1), 330.0, device=device),
            x,
            torch.ones_like(x),
            strict=False,
        )
        self.assertEqual(tuple(state.gamma.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(state.gamma).all()))
        self.assertTrue(torch.allclose(state.y.sum(-1), torch.ones(2, device=device), atol=1e-5))

    def test_hanna_adapter_matches_official_binary_demo_reference(self) -> None:
        assets = OfficialHANNAAssets.default(Path(__file__).resolve().parents[1])
        if not all(path.is_file() for path in assets.ensemble_paths):
            self.skipTest("Optional HANNA comparison weights are obtained from the upstream source")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference = OfficialHANNAInference(assets, device)
        embeddings = torch.stack([
            inference.embedding("CCOC(=O)OCC"),
            inference.embedding("CCCCCCCCO"),
        ])
        molecules = torch.cat(
            [embeddings, torch.zeros(2, 2, device=device)], dim=-1
        ).unsqueeze(0).repeat(2, 1, 1)
        x = torch.tensor([[0.3, 0.7], [0.1, 0.9]], device=device)
        output = inference.model(
            molecules,
            torch.full((2, 1), 300.0, device=device),
            torch.full((2, 1), 101.325, device=device),
            x,
            torch.ones_like(x),
        )
        reference = torch.tensor(
            [[0.5127999, 0.08112263], [0.7983205, 0.01013201]], device=device
        )
        self.assertTrue(torch.allclose(output.log_gamma, reference, atol=2e-5, rtol=2e-5))
        self.assertFalse(any(parameter.requires_grad for parameter in inference.model.ensemble.parameters()))
        self.assertEqual(
            sum(parameter.numel() for parameter in inference.model.ensemble.parameters()),
            651900,
        )

    def test_registry_contains_exact_requested_baselines(self) -> None:
        self.assertEqual(
            set(BASELINE_CAPABILITIES),
            {
                "descriptor_ann", "smiles_rnn", "ualf_gnn", "solvgnn", "gdi_gnn",
                "ge_gnn", "hanna", "tennet_sac", "spt_nrtl", "spt_nrtl_adapted",
            },
        )

    def test_native_support_is_not_silently_expanded(self) -> None:
        descriptor = BASELINE_CAPABILITIES["descriptor_ann"]
        self.assertIsNotNone(descriptor.support_reason(3, "isobaric"))
        self.assertIsNotNone(descriptor.support_reason(2, "isothermal"))
        solvgnn = BASELINE_CAPABILITIES["solvgnn"]
        self.assertIsNotNone(solvgnn.support_reason(2, "isothermal", 330.0))
        self.assertIsNone(solvgnn.support_reason(3, "isothermal", 298.15))

    def test_descriptor_ann_matches_audited_parameter_count(self) -> None:
        model = DescriptorANN()
        self.assertEqual(trainable_parameter_count(model), 9921)
        self.assertEqual(model(torch.randn(3, 23)).shape, (3, 1))

    def test_smiles_rnn_is_direction_specific_two_output_model(self) -> None:
        model = SMILESRNN(48)
        self.assertEqual(model(torch.randn(4, 48)).shape, (4, 2))

    def test_ualf_has_heteroscedastic_outputs_and_gradients(self) -> None:
        model = UALFGNN()
        atoms = torch.randn(2, 2, 5, 16)
        adjacency = torch.eye(5).repeat(2, 2, 1, 1)
        mask = torch.ones(2, 2, 5)
        mean, log_variance = model(atoms, adjacency, mask, torch.rand(2, 2))
        loss = model.heteroscedastic_loss(mean, log_variance, torch.rand(2, 2))
        loss.backward()
        self.assertEqual(mean.shape, (2, 2))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        predictive_mean, predictive_variance = model.mc_predict(
            atoms, adjacency, mask, torch.rand(2, 2), samples=3
        )
        self.assertEqual(predictive_mean.shape, (2, 2))
        self.assertTrue(bool((predictive_variance > 0).all()))

    def test_ge_gnn_activity_coefficients_retain_composition_autograd(self) -> None:
        model = GEGNN()
        values = model(torch.randn(3, 2, 32), torch.softmax(torch.randn(3, 2), -1))
        self.assertEqual(values.shape, (3, 2))
        values.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_solvgnn_includes_molecular_graph_encoder(self) -> None:
        from src.thermoformer.baselines.machine_learning.models import SolvGNN

        model = SolvGNN()
        atoms = torch.randn(2, 3, 5, 16)
        adjacency = torch.eye(5).repeat(2, 3, 1, 1)
        atom_mask = torch.ones(2, 3, 5)
        x = torch.softmax(torch.randn(2, 3), -1)
        output = model(
            torch.randn(2, 3, 5, 74),
            adjacency,
            atom_mask,
            x,
            torch.ones_like(x),
            torch.zeros(2, 3, 3),
        )
        self.assertEqual(output.shape, (2, 3))

    def test_muggianu_projection_is_normalized_and_permutation_consistent(self) -> None:
        x = torch.tensor([[0.2, 0.3, 0.5]])
        weights = muggianu_binary_weights(x)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1)))
        permuted = muggianu_binary_weights(x[:, [2, 1, 0]])
        self.assertTrue(torch.allclose(weights.sort(-1).values, permuted.sort(-1).values))

    def test_overall_protocol_matrix_is_assigned_by_native_cardinality(self) -> None:
        self.assertEqual(
            [value.key for value in OVERALL_BENCHMARKS],
            ["binary_train_binary_test", "ternary_train_ternary_test", "joint_train_joint_test"],
        )
        self.assertEqual(
            [value.split_protocol for value in OVERALL_BENCHMARKS],
            ["vle_overall_binary", "vle_overall_ternary", "vle_overall_binary_ternary"],
        )
        self.assertEqual(benchmark_for_baseline("descriptor_ann").key, "binary_train_binary_test")
        self.assertEqual(benchmark_for_baseline("solvgnn").key, "joint_train_joint_test")
        self.assertEqual(
            benchmark_for_baseline("hanna").key,
            "official_pretrained_to_joint_test",
        )
        rows = native_evaluation_cells()
        self.assertEqual(len(rows), 10 * 2)
        descriptor = [row for row in rows if row["baseline"] == "descriptor_ann"]
        self.assertEqual({row["benchmark"] for row in descriptor}, {"binary_train_binary_test"})
        self.assertEqual(
            {row["benchmark"] for row in rows if row["baseline"] == "solvgnn"},
            {"joint_train_joint_test"},
        )
        self.assertEqual(
            benchmark_for_baseline("tennet_sac").key,
            "external_fixed_to_joint_test",
        )
        self.assertEqual(
            benchmark_for_baseline("spt_nrtl").key,
            "external_fixed_to_joint_test",
        )
        self.assertEqual(
            benchmark_for_baseline("spt_nrtl_adapted").key,
            "joint_train_joint_test",
        )
        self.assertEqual(
            {row["status"] for row in descriptor if row["direction"] == "isothermal"},
            {"not_applicable"},
        )

    def test_metric_contract_contains_all_requested_metrics(self) -> None:
        self.assertEqual(
            UNIFIED_METRIC_KEYS,
            (
                "pressure_mae_kpa", "pressure_rmse_kpa", "pressure_r2",
                "temperature_mae_k", "temperature_rmse_k", "temperature_r2",
                "y_mae", "y_rmse", "y_r2", "valid_coverage",
                "nonphysical_rate", "solver_failure_rate",
            ),
        )

    def test_external_activity_adapter_memoizes_identical_thermodynamic_state(self) -> None:
        class CountingPredictor:
            def __init__(self):
                self.calls = 0

            def log_gamma(self, smiles, temperature_k, composition):
                self.calls += 1
                return torch.tensor([0.1, 0.2])

        predictor = CountingPredictor()
        adapter = ExternalActivityThermodynamicAdapter(predictor, [("CCO", "O")])
        molecules = torch.tensor([[[12.0, -3000.0], [13.0, -3500.0]]])
        temperature = torch.tensor([[330.0]])
        x = torch.tensor([[0.3, 0.7]])
        mask = torch.ones_like(x)
        first = adapter(molecules, temperature, torch.tensor([[80.0]]), x, mask)
        second = adapter(molecules, temperature, torch.tensor([[120.0]]), x, mask)
        self.assertEqual(predictor.calls, 1)
        self.assertTrue(torch.equal(first.log_gamma, second.log_gamma))

    def test_formal_aggregation_uses_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for seed in range(5):
                path = root / f"seed_{seed}.csv"
                write_seed_metrics(
                    path,
                    [{
                        "baseline": "smiles_rnn",
                        "benchmark": "binary_train_binary_test",
                        "direction": "isobaric",
                        "component_count": 2,
                        "status": "evaluated",
                        "temperature_mae_k": float(seed),
                        "valid_coverage": 1.0,
                    }],
                    seed,
                )
                files.append(path)
            rows = aggregate_seed_metrics(files, root / "summary.csv")
            self.assertAlmostEqual(rows[0]["temperature_mae_k_mean"], 2.0)
            self.assertAlmostEqual(rows[0]["temperature_mae_k_sample_std"], 2.5 ** 0.5)

    def test_formal_aggregation_keeps_one_row_when_native_coverage_varies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for seed in range(5):
                path = root / f"seed_{seed}.csv"
                evaluated = seed < 3
                write_seed_metrics(
                    path,
                    [{
                        "baseline": "gdi_gnn",
                        "benchmark": "binary_train_binary_test",
                        "direction": "isothermal",
                        "component_count": 2,
                        "status": "evaluated" if evaluated else "not_evaluated_no_native_coverage",
                        "pressure_mae_kpa": float(seed + 1) if evaluated else "",
                        "valid_coverage": 0.04 if evaluated else 0.0,
                    }],
                    seed,
                )
                files.append(path)
            rows = aggregate_seed_metrics(files, root / "summary.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "evaluated_partial_seeds")
            self.assertEqual(rows[0]["seed_count"], 5)
            self.assertEqual(rows[0]["evaluated_seed_count"], 3)
            self.assertAlmostEqual(rows[0]["pressure_mae_kpa_mean"], 2.0)
            self.assertAlmostEqual(rows[0]["valid_coverage_mean"], 0.024)

    def test_pretrained_overlap_requires_normalized_system_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_systems.csv"
            path.write_text("system_id\n2c_a\n2c_b\n", encoding="utf-8")
            audit = audit_pretrained_overlap(path, ["2c_b", "2c_c"])
            self.assertEqual(audit.overlapping_systems, ("2c_b",))
            self.assertEqual(audit.overlap_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
