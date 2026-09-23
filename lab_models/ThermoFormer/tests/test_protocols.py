import unittest

from src.data import VLESample
from src.protocols import (
    binary_generalization_splits,
    binary_to_ternary_split,
    composition_interpolation_split,
    overall_system_split,
    protect_pure_reference_rows,
    protect_pure_reference_systems,
    state_extreme_split,
    unseen_component_split,
)
from src.auditing import ternary_subsystem_rows
from src.splits import sample_id, system_id


def row(smiles, index, x=None):
    count = len(smiles)
    if x is None:
        first = 0.05 + 0.09 * index
        if count == 2:
            x = (first, 1.0 - first)
        else:
            x = (first, 0.4 * (1.0 - first), 0.6 * (1.0 - first))
    return VLESample(
        smiles=tuple(smiles),
        names=tuple(smiles),
        temperature_k=300.0 + 5.0 * index,
        pressure_kpa=50.0 + 3.0 * index,
        liquid_composition=tuple(x),
        vapor_composition=tuple(x),
        quality_weight=1.0,
        quality_status="passed",
        source="synthetic.xlsx",
        doi=f"series-{smiles}",
        experiment_mode="full_state",
    )


def dataset():
    binary = [
        ("C", "O"), ("C", "N"), ("CC", "O"), ("CC", "N"),
        ("CCC", "O"), ("CCC", "N"), ("CO", "N"), ("CN", "O"),
        ("CCO", "N"), ("CCN", "O"),
    ]
    ternary = [
        ("C", "O", "N"), ("CC", "O", "N"), ("CCC", "O", "N"),
        ("CO", "CN", "O"), ("CCO", "CCN", "N"),
    ]
    return [row(system, index) for system in (*binary, *ternary) for index in range(10)]


class ProtocolSplitTests(unittest.TestCase):
    def test_binary_generalization_protocols_use_only_binary_rows(self) -> None:
        splits = {
            split.protocol: split
            for split in binary_generalization_splits(
                dataset(), seed=2, minimum_anchor_temperatures=0
            )
        }
        self.assertEqual(
            set(splits),
            {
                "binary_state_composition_interpolation",
                "binary_state_composition_edge_extrapolation",
                "binary_state_temperature_low_extrapolation",
                "binary_state_temperature_high_extrapolation",
                "binary_state_pressure_low_extrapolation",
                "binary_state_pressure_high_extrapolation",
                "binary_unseen_component",
            },
        )
        for split in splits.values():
            self.assertTrue(
                all(
                    sample.component_count == 2
                    for partition in (split.train, split.validation, split.test)
                    for sample in partition
                )
            )
            self.assertEqual(split.metadata["component_counts"], [2])
        unseen = splits["binary_unseen_component"]
        held_out = set(unseen.metadata["held_out_components"])
        training_components = {
            smiles for sample in unseen.train for smiles in sample.smiles
        }
        self.assertTrue(held_out.isdisjoint(training_components))

    def test_pure_reference_rows_are_moved_to_train_without_row_overlap(self) -> None:
        anchors = [
            row(("C", "O"), index, x=composition)
            for index in (0, 4)
            for composition in ((1.0, 0.0), (0.0, 1.0))
        ]
        mixtures = [row(("C", "O"), index) for index in (1, 2, 3)]
        from src.splits import DatasetPartitions

        split = DatasetPartitions(
            train=(mixtures[0],),
            validation=(anchors[0], mixtures[1]),
            test=tuple(anchors[1:] + [mixtures[2]]),
            protocol="state_test",
            seed=0,
        )

        protected = protect_pure_reference_rows(
            [*anchors, *mixtures], split, minimum_temperatures=2
        )

        from src.data import pure_anchor_temperatures

        temperatures = pure_anchor_temperatures(protected.train)
        self.assertGreaterEqual(len(temperatures["C"]), 2)
        self.assertGreaterEqual(len(temperatures["O"]), 2)
        partition_ids = [
            {sample_id(value) for value in partition}
            for partition in (protected.train, protected.validation, protected.test)
        ]
        self.assertFalse(partition_ids[0] & partition_ids[1])
        self.assertFalse(partition_ids[0] & partition_ids[2])
        self.assertEqual(protected.metadata["pure_reference_rows"], 4)

    def test_overall_split_is_system_disjoint_and_reproducible(self) -> None:
        rows = dataset()
        first = overall_system_split(rows, seed=2, minimum_anchor_temperatures=0)
        second = overall_system_split(rows, seed=2, minimum_anchor_temperatures=0)

        train_systems = {system_id(value) for value in first.train}
        validation_systems = {system_id(value) for value in first.validation}
        test_systems = {system_id(value) for value in first.test}
        self.assertTrue(train_systems.isdisjoint(validation_systems))
        self.assertTrue(train_systems.isdisjoint(test_systems))
        self.assertTrue(validation_systems.isdisjoint(test_systems))
        self.assertEqual(
            [sample_id(value) for value in first.test],
            [sample_id(value) for value in second.test],
        )

    def test_complete_reference_systems_preserve_strict_state_boundaries(self) -> None:
        evaluated = [row(("C", "O"), index) for index in range(10)]
        reference_c = [
            row(("C", "N"), 20 + index, x=(1.0, 0.0)) for index in range(2)
        ]
        reference_o = [
            row(("O", "F"), 30 + index, x=(1.0, 0.0)) for index in range(2)
        ]
        raw = [*evaluated, *reference_c, *reference_o]
        split = state_extreme_split(
            raw, variable="temperature", tail="high", seed=0
        )
        protected = protect_pure_reference_systems(
            raw,
            split,
            minimum_temperatures=2,
            allowed_components={"C", "O", "N", "F"},
            required_components={"C", "O"},
        )

        reference_systems = set(protected.metadata["pure_reference_system_ids"])
        validation_systems = {system_id(value) for value in protected.validation}
        test_systems = {system_id(value) for value in protected.test}
        self.assertTrue(reference_systems.isdisjoint(validation_systems | test_systems))
        for key in {system_id(value) for value in protected.test}:
            train_values = [
                value.temperature_k
                for value in protected.train
                if system_id(value) == key
            ]
            test_values = [
                value.temperature_k
                for value in protected.test
                if system_id(value) == key
            ]
            self.assertGreater(min(test_values), max(train_values))

    def test_composition_interpolation_holds_out_only_interior_states(self) -> None:
        rows = [row(("C", "O"), index) for index in range(10)]
        split = composition_interpolation_split(rows, seed=0)

        train_x = [value.liquid_composition[0] for value in split.train]
        test_x = [value.liquid_composition[0] for value in split.test]
        self.assertTrue(test_x)
        self.assertGreater(min(test_x), min(train_x))
        self.assertLess(max(test_x), max(train_x))
        self.assertEqual(
            {system_id(value) for value in split.train},
            {system_id(value) for value in split.test},
        )

    def test_state_extreme_split_respects_requested_tail(self) -> None:
        rows = [row(("C", "O"), index) for index in range(10)]
        low = state_extreme_split(rows, variable="temperature", tail="low", seed=0)
        high = state_extreme_split(rows, variable="pressure", tail="high", seed=0)

        self.assertLess(
            max(value.temperature_k for value in low.test),
            min(value.temperature_k for value in low.train),
        )
        self.assertGreater(
            min(value.pressure_kpa for value in high.test),
            max(value.pressure_kpa for value in high.train),
        )

    def test_unseen_component_split_has_no_component_leakage(self) -> None:
        split = unseen_component_split(
            dataset(), seed=3, target_component_fraction=0.25,
            minimum_anchor_temperatures=0,
        )
        training_components = {smiles for value in split.train for smiles in value.smiles}
        test_components = {smiles for value in split.test for smiles in value.smiles}
        held_out = set(split.metadata["held_out_components"])

        self.assertTrue(held_out)
        self.assertTrue(held_out.isdisjoint(training_components))
        self.assertTrue(test_components & held_out)
        self.assertGreater(split.metadata["strict_unseen_test_rows"], 0)
        self.assertTrue(
            {system_id(value) for value in split.train}.isdisjoint(
                {system_id(value) for value in split.validation}
            )
        )

    def test_binary_to_ternary_zero_shot_and_scaling_keep_a_fixed_test(self) -> None:
        rows = dataset()
        zero = binary_to_ternary_split(rows, seed=1, ternary_training_fraction=0.0)
        small = binary_to_ternary_split(rows, seed=1, ternary_training_fraction=0.25)
        full = binary_to_ternary_split(rows, seed=1, ternary_training_fraction=1.0)

        self.assertTrue(all(value.component_count == 2 for value in zero.train))
        self.assertTrue(all(value.component_count == 2 for value in zero.validation))
        self.assertTrue(all(value.component_count == 3 for value in zero.test))
        self.assertEqual(
            {sample_id(value) for value in zero.test},
            {sample_id(value) for value in small.test},
        )
        self.assertEqual(
            {sample_id(value) for value in small.test},
            {sample_id(value) for value in full.test},
        )
        small_ternary = sum(value.component_count == 3 for value in small.train)
        full_ternary = sum(value.component_count == 3 for value in full.train)
        self.assertGreater(full_ternary, small_ternary)

    def test_binary_to_ternary_coverage_uses_only_binary_training_systems(self) -> None:
        split = binary_to_ternary_split(
            dataset(), seed=1, ternary_training_fraction=0.0
        )
        expected = {
            str(item["ternary_system_id"]): int(item["covered_binary_subsystems"])
            for item in ternary_subsystem_rows(
                split.test, binary_reference_samples=split.train
            )
        }
        actual_counts = {}
        for value in split.metadata["test_binary_subsystem_coverage"].items():
            actual_counts[int(value[0])] = int(value[1])

        from collections import Counter

        self.assertEqual(actual_counts, dict(Counter(expected.values())))


if __name__ == "__main__":
    unittest.main()
