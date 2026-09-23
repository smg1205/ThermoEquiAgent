from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.generate_splits import _ternary_only_projection
from scripts.run_joint_activity_adapted import (
    benchmark_partitions,
    psat_calibration_partition,
)
from src.thermoformer.baselines.machine_learning.joint_activity_adapted import (
    equilibrium_log_gamma,
)
from src.thermoformer.baselines.thermodynamic_fitting import (
    fit_vapor_pressure_correlations,
)
from src.thermoformer.data.splitting import DatasetPartitions


class TernaryOnlyProtocolTests(unittest.TestCase):
    @staticmethod
    def _row(component_count: int, label: str):
        return SimpleNamespace(component_count=component_count, label=label)

    def test_projection_preserves_partition_membership_and_removes_binary_rows(self) -> None:
        split = DatasetPartitions(
            train=(self._row(2, "b-train"), self._row(3, "t-train")),
            validation=(self._row(3, "t-validation"), self._row(2, "b-validation")),
            test=(self._row(2, "b-test"), self._row(3, "t-test")),
            protocol="overall_binary_ternary",
            seed=3,
        )
        projected = _ternary_only_projection(split)
        self.assertEqual(projected.protocol, "overall_ternary")
        self.assertEqual(projected.seed, 3)
        self.assertEqual([row.label for row in projected.train], ["t-train"])
        self.assertEqual([row.label for row in projected.validation], ["t-validation"])
        self.assertEqual([row.label for row in projected.test], ["t-test"])
        self.assertEqual(projected.metadata["source_protocol"], "overall_binary_ternary")

    def test_adapted_activity_benchmark_filters_all_three_partitions(self) -> None:
        split = SimpleNamespace(
            train=(self._row(2, "b0"), self._row(3, "t0")),
            validation=(self._row(2, "b1"), self._row(3, "t1")),
            test=(self._row(3, "t2"), self._row(2, "b2")),
        )
        partitions = benchmark_partitions(split, "ternary_train_ternary_test")
        self.assertEqual(
            [[row.label for row in rows] for rows in partitions],
            [["t0"], ["t1"], ["t2"]],
        )
        self.assertTrue(all(row.component_count == 3 for rows in partitions for row in rows))

    def test_ternary_activity_labels_use_training_only_pure_psat_anchors(self) -> None:
        def row(smiles, x, y, temperature, pressure, label):
            return SimpleNamespace(
                smiles=smiles,
                liquid_composition=x,
                vapor_composition=y,
                temperature_k=temperature,
                pressure_kpa=pressure,
                quality_weight=1.0,
                component_count=len(smiles),
                label=label,
            )

        pure_anchors = (
            row(("CCO", "O"), (1.0, 0.0), (1.0, 0.0), 300.0, 50.0, "ethanol-300"),
            row(("CCO", "O"), (1.0, 0.0), (1.0, 0.0), 350.0, 100.0, "ethanol-350"),
            row(("O", "CC"), (1.0, 0.0), (1.0, 0.0), 300.0, 40.0, "water-300"),
            row(("O", "CC"), (1.0, 0.0), (1.0, 0.0), 350.0, 80.0, "water-350"),
            row(("CC", "CCO"), (1.0, 0.0), (1.0, 0.0), 300.0, 30.0, "ethane-300"),
            row(("CC", "CCO"), (1.0, 0.0), (1.0, 0.0), 350.0, 60.0, "ethane-350"),
        )
        ternary = row(
            ("CCO", "O", "CC"),
            (0.2, 0.3, 0.5),
            (0.3, 0.3, 0.4),
            325.0,
            70.0,
            "ternary",
        )
        split = SimpleNamespace(
            train=(*pure_anchors, ternary),
            validation=(ternary,),
            test=(ternary,),
        )

        interaction_train, interaction_validation, _ = benchmark_partitions(
            split, "ternary_train_ternary_test"
        )
        self.assertEqual(interaction_train, (ternary,))
        psat_rows = psat_calibration_partition(split)
        self.assertEqual(psat_rows, split.train)
        vapor_pressure, _ = fit_vapor_pressure_correlations(psat_rows)
        self.assertEqual(set(vapor_pressure), {"CC", "CCO", "O"})
        self.assertIsNotNone(
            equilibrium_log_gamma(ternary, vapor_pressure, minimum_composition=1e-5)
        )
        self.assertEqual(interaction_validation, (ternary,))


if __name__ == "__main__":
    unittest.main()
