import json
import tempfile
import unittest
from pathlib import Path

from src.data import VLESample
from src.splits import (
    DatasetPartitions,
    load_split_assignment,
    sample_id,
    save_split_assignment,
    system_id,
)


def row(smiles=("CCO", "O"), x=(0.4, 0.6), y=(0.5, 0.5), temperature=350.0):
    return VLESample(
        smiles=smiles,
        names=smiles,
        temperature_k=temperature,
        pressure_kpa=101.325,
        liquid_composition=x,
        vapor_composition=y,
        quality_weight=1.0,
        quality_status="passed",
        source="synthetic.xlsx",
        doi="synthetic",
    )


class ReproducibleSplitTests(unittest.TestCase):
    def test_ids_are_invariant_to_component_order(self) -> None:
        original = row()
        reversed_row = row(
            smiles=tuple(reversed(original.smiles)),
            x=tuple(reversed(original.liquid_composition)),
            y=tuple(reversed(original.vapor_composition)),
        )

        self.assertEqual(system_id(original), system_id(reversed_row))
        self.assertEqual(sample_id(original), sample_id(reversed_row))

    def test_saved_assignment_round_trips_and_records_dataset_digest(self) -> None:
        rows = [row(temperature=330.0 + index) for index in range(6)]
        split = DatasetPartitions(
            train=tuple(rows[:3]),
            validation=tuple(rows[3:5]),
            test=tuple(rows[5:]),
            protocol="unit_system_holdout",
            seed=7,
            metadata={"note": "fixed"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_split_assignment(path, rows, split)
            payload = json.loads(path.read_text(encoding="utf-8"))
            restored = load_split_assignment(path, rows)

        self.assertEqual(restored, split)
        self.assertRegex(payload["dataset_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["schema_version"], 1)

    def test_loader_rejects_dataset_drift_or_partition_overlap(self) -> None:
        rows = [row(temperature=330.0 + index) for index in range(4)]
        split = DatasetPartitions(
            train=tuple(rows[:2]),
            validation=(rows[2],),
            test=(rows[3],),
            protocol="unit",
            seed=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_split_assignment(path, rows, split)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["partitions"]["test"].append(payload["partitions"]["train"][0])
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split_assignment(path, rows)

            save_split_assignment(path, rows, split)
            with self.assertRaisesRegex(ValueError, "dataset digest"):
                load_split_assignment(path, rows[:-1])

    def test_loader_rejects_protocol_path_traversal(self) -> None:
        rows = [row(temperature=330.0 + index) for index in range(3)]
        split = DatasetPartitions(
            train=(rows[0],),
            validation=(rows[1],),
            test=(rows[2],),
            protocol="valid_protocol",
            seed=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_split_assignment(path, rows, split)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["protocol"] = "../escape"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "protocol"):
                load_split_assignment(path, rows)


if __name__ == "__main__":
    unittest.main()
