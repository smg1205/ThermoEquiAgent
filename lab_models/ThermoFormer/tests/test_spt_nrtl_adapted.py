import unittest

from src.thermoformer.baselines.machine_learning.spt_nrtl import (
    SPTNRTLPairParameters,
    multicomponent_nrtl_log_gamma,
)
from src.thermoformer.baselines.machine_learning.spt_nrtl_adapted import (
    AdaptedSPTConfig,
    AdaptedSPTNRTLPredictor,
    PairLabel,
    _binary_log_gamma_torch,
    fit_pair_label,
    train_adapted_spt,
    vector_to_pair,
)
from src.thermoformer.baselines.thermodynamic_fitting import LogLinearVaporPressure
from src.thermoformer.data.loading import VLESample

import numpy as np
import torch


class AdaptedSPTNRTLTests(unittest.TestCase):
    @staticmethod
    def _samples():
        pair = SPTNRTLPairParameters(
            (0.30, 0.0), (1.0, 120.0, -0.2, 0.0), (-0.5, 80.0, 0.1, 0.0)
        )
        vapor = {
            "CCO": LogLinearVaporPressure(12.0, -3000.0),
            "O": LogLinearVaporPressure(13.0, -3500.0),
        }
        samples = []
        for index, x0 in enumerate(np.linspace(0.1, 0.9, 9)):
            temperature = 320.0 + index * 3.0
            x = torch.tensor([x0, 1.0 - x0], dtype=torch.float64)
            gamma = torch.exp(multicomponent_nrtl_log_gamma(x, temperature, {(0, 1): pair})).numpy()
            psat = np.asarray([value.pressure_kpa(temperature) for value in vapor.values()])
            pressure = float(np.sum(x.numpy() * gamma * psat))
            y = x.numpy() * gamma * psat / pressure
            samples.append(VLESample(
                smiles=("CCO", "O"), names=("ethanol", "water"), temperature_k=temperature,
                pressure_kpa=pressure, liquid_composition=tuple(x.numpy()),
                vapor_composition=tuple(y), quality_weight=1.0, quality_status="passed",
                source="synthetic", doi="synthetic", experiment_mode="isothermal",
            ))
        return samples, vapor

    def test_pair_fit_uses_finite_ten_parameter_label(self):
        samples, vapor = self._samples()
        label = fit_pair_label(samples, vapor, AdaptedSPTConfig(
            minimum_pair_rows=5, label_fit_evaluations=80
        ))
        self.assertIsNotNone(label)
        self.assertEqual(len(label.parameters), 10)
        self.assertLessEqual(label.final_rmse, label.initial_rmse)

    def test_vectorized_label_equation_matches_shared_nrtl(self):
        values = torch.tensor(
            [0.3, 1e-4, 1.0, 120.0, -0.2, 1e-3, -0.5, 80.0, 0.1, -1e-3],
            dtype=torch.float64,
        )
        temperatures = torch.tensor([320.0, 350.0], dtype=torch.float64)
        compositions = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float64)
        vectorized = _binary_log_gamma_torch(values, temperatures, compositions)
        pair = vector_to_pair(values.numpy())
        reference = torch.stack([
            multicomponent_nrtl_log_gamma(x, t, {(0, 1): pair})
            for x, t in zip(compositions, temperatures)
        ])
        self.assertTrue(torch.allclose(vectorized, reference, atol=1e-10, rtol=1e-10))

    def test_pair_reversal_swaps_directional_parameters(self):
        label = PairLabel("CCO", "O", tuple(range(10)), 5, 1.0, 0.5, True)
        self.assertEqual(label.reversed().parameters, (0, 1, 6, 7, 8, 9, 2, 3, 4, 5))
        self.assertEqual(vector_to_pair(label.parameters).reversed().tau_ij, vector_to_pair(label.reversed().parameters).tau_ij)

    def test_validation_selected_network_reloads_and_predicts_ternary(self):
        train = (
            PairLabel("CCO", "O", (0.3, 0.0, 1, 2, 0, 0, -1, 3, 0, 0), 8, 1, .2, True),
            PairLabel("CC", "O", (0.3, 0.0, 2, 1, 0, 0, -2, 4, 0, 0), 8, 1, .2, True),
        )
        validation = (PairLabel("CCC", "O", (0.3, 0.0, 1, 1, 0, 0, -1, 2, 0, 0), 8, 1, .2, True),)
        config = AdaptedSPTConfig(
            max_sequence_length=32, embedding_dimension=16, attention_heads=4,
            transformer_layers=1, feedforward_dimension=32, dropout=0.0,
            batch_size=2, epochs=2, patience=2,
        )
        result = train_adapted_spt(train, validation, config, torch.device("cpu"))
        self.assertGreater(result.trainable_parameters, 0)
        self.assertGreaterEqual(result.best_epoch, 1)
        predictor = AdaptedSPTNRTLPredictor(result, torch.device("cpu"))
        output = predictor.log_gamma(("CCO", "O", "CC"), 330.0, (0.2, 0.3, 0.5))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.isfinite(output).all())
        permutation = (2, 0, 1)
        permuted = predictor.log_gamma(
            tuple(("CCO", "O", "CC")[index] for index in permutation),
            330.0,
            tuple((0.2, 0.3, 0.5)[index] for index in permutation),
        )
        self.assertTrue(torch.allclose(permuted, output[list(permutation)], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
