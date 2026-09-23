import csv
import tempfile
import unittest
from pathlib import Path

from src.artifacts import artifact_sha256
from src.c1_generalization_outputs import (
    PROTOCOLS,
    collect_generalization_rows,
    write_c1_generalization_outputs,
)


class C1GeneralizationOutputTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_collects_all_protocols_and_distinguishes_joint_subsets(self) -> None:
        rows = collect_generalization_rows(self.ROOT)
        self.assertEqual({row["protocol"] for row in rows}, {spec.name for spec in PROTOCOLS})
        joint = [row for row in rows if row["protocol"] == "overall_binary_ternary"]
        self.assertEqual(len(joint), 4)
        self.assertEqual({row["test_subset"] for row in joint}, {"binary", "ternary"})
        self.assertEqual({row["joint_outputs"] for row in joint}, {"P,y", "T,y"})
        self.assertTrue(all(row["available_seeds"] >= 4 for row in rows))

    def test_writes_reproducible_task_and_stage_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.md"
            table = root / "tasks.csv"
            outputs = write_c1_generalization_outputs(
                self.ROOT,
                report_path=report,
                table_path=table,
            )
            first_hash = artifact_sha256(report)
            write_c1_generalization_outputs(
                self.ROOT,
                report_path=report,
                table_path=table,
            )
            self.assertEqual(first_hash, artifact_sha256(report))
            with table.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            text = report.read_text(encoding="utf-8")
            self.assertIn("molecules,T,x -> P,y", text)
            self.assertIn("molecules,P,x -> T,y", text)
            self.assertIn("validation retained Stage 1", text)
            self.assertIn("manifest", outputs)
            self.assertNotIn("complete_chinese_report", outputs)
            self.assertIn("146/180", text)


if __name__ == "__main__":
    unittest.main()
