from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src.thermoformer.data import VLEBatch
from src.thermoformer.models import ThermoFormer, ThermoFormerConfig
from src.thermoformer.thermodynamics.vapor_pressure import (
    AntoineParameters,
    CORRELATION_PARAMETER_COUNT,
)
from src.thermoformer.training.direct_ge import (
    build_direct_thermodynamic_targets,
    direct_thermodynamic_supervision,
)


def _parameters(*rows: np.ndarray | None) -> torch.Tensor:
    result = torch.zeros(1, 3, CORRELATION_PARAMETER_COUNT)
    for index, row in enumerate(rows):
        if row is not None:
            result[0, index] = torch.from_numpy(row)
    return result


def _batch(
    x: tuple[float, float, float],
    y: tuple[float, float, float],
    parameters: torch.Tensor,
    *,
    components: int = 2,
    pressure_kpa: float = 72.0,
) -> VLEBatch:
    mask = torch.zeros(1, 3)
    mask[:, :components] = 1.0
    return VLEBatch(
        molecules=torch.randn(1, 3, 6),
        temperature_k=torch.tensor([[350.0]]),
        pressure_kpa=torch.tensor([[pressure_kpa]]),
        x=torch.tensor([x]),
        y=torch.tensor([y]),
        mask=mask,
        quality_weight=torch.ones(1, 1),
        experiment_mode=torch.zeros(1, dtype=torch.long),
        pure_property_parameters=parameters,
    )


class DirectThermodynamicTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        # log10(P/kPa) = A at 350 K gives exact constant values for these
        # worked examples while B>0 preserves dPsat/dT > 0.
        self.psat_100 = AntoineParameters(
            a=2.0,
            b=1.0e-6,
            c=0.0,
            minimum_temperature_k=300.0,
            maximum_temperature_k=400.0,
            pressure_unit="kPa",
            temperature_unit="K",
        ).as_array()
        self.psat_50 = AntoineParameters(
            a=math.log10(50.0),
            b=1.0e-6,
            c=0.0,
            minimum_temperature_k=300.0,
            maximum_temperature_k=400.0,
            pressure_unit="kPa",
            temperature_unit="K",
        ).as_array()

    def test_binary_targets_recover_known_gamma_and_excess_gibbs(self) -> None:
        gamma = (1.2, 0.8)
        x = (0.4, 0.6, 0.0)
        partial = (x[0] * gamma[0] * 100.0, x[1] * gamma[1] * 50.0)
        pressure = sum(partial)
        y = (partial[0] / pressure, partial[1] / pressure, 0.0)
        targets = build_direct_thermodynamic_targets(
            _batch(x, y, _parameters(self.psat_100, self.psat_50), pressure_kpa=pressure)
        )

        torch.testing.assert_close(
            targets.log_gamma[0, :2],
            torch.log(torch.tensor(gamma)),
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertTrue(bool(targets.gamma_mask[0, :2].all()))
        self.assertTrue(bool(targets.ge_mask[0, 0]))
        self.assertAlmostEqual(
            targets.excess_gibbs_rt[0, 0].item(),
            x[0] * math.log(gamma[0]) + x[1] * math.log(gamma[1]),
            places=6,
        )

    def test_missing_or_nonphysical_psat_masks_labels_without_fallback(self) -> None:
        invalid = self.psat_50.copy()
        invalid[2] = -1.0  # Antoine B <= 0 implies dPsat/dT <= 0.
        batch = _batch(
            (0.4, 0.6, 0.0),
            (2.0 / 3.0, 1.0 / 3.0, 0.0),
            _parameters(self.psat_100, invalid),
        )
        targets = build_direct_thermodynamic_targets(batch)

        self.assertTrue(bool(targets.gamma_mask[0, 0]))
        self.assertFalse(bool(targets.gamma_mask[0, 1]))
        self.assertFalse(bool(targets.ge_mask[0, 0]))
        self.assertEqual(targets.covered_components, 1)
        self.assertEqual(targets.covered_states, 0)

    def test_small_fraction_is_masked_and_pure_endpoint_has_zero_ge(self) -> None:
        interior = build_direct_thermodynamic_targets(
            _batch(
                (0.99995, 0.00005, 0.0),
                (0.99995, 0.00005, 0.0),
                _parameters(self.psat_100, self.psat_50),
                pressure_kpa=99.995,
            ),
            minimum_fraction=1e-4,
        )
        self.assertFalse(bool(interior.gamma_mask[0, 1]))
        self.assertFalse(bool(interior.ge_mask[0, 0]))

        endpoint = build_direct_thermodynamic_targets(
            _batch(
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                _parameters(self.psat_100, None),
                pressure_kpa=100.0,
            )
        )
        self.assertTrue(bool(endpoint.ge_mask[0, 0]))
        self.assertEqual(endpoint.excess_gibbs_rt[0, 0].item(), 0.0)
        self.assertTrue(bool(endpoint.gamma_mask[0, 0]))
        self.assertFalse(bool(endpoint.gamma_mask[0, 1]))

    def test_direct_losses_backpropagate_to_pair_potential(self) -> None:
        model = ThermoFormer(
            ThermoFormerConfig(feature_dim=6, hidden_dim=12, layers=1, heads=3)
        )
        batch = _batch(
            (0.4, 0.6, 0.0),
            (2.0 / 3.0, 1.0 / 3.0, 0.0),
            _parameters(self.psat_100, self.psat_50),
        )
        outputs = model(
            batch.molecules,
            batch.temperature_k,
            batch.pressure_kpa,
            batch.x,
            batch.mask,
        )
        objective = direct_thermodynamic_supervision(outputs, batch)
        objective.total.backward()

        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("pair_potential")
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and bool(gradient.abs().sum() > 0)
                for gradient in gradients
            )
        )


if __name__ == "__main__":
    unittest.main()
