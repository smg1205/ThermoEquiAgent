from __future__ import annotations
import csv
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev

from src.c1_ablation_outputs import (
    C1_ABLATION_SOURCES,
    collect_c1_ablation_rows,
    write_c1_ablation_outputs,
)
from src.thermoformer.reporting.artifacts import artifact_sha256
from src.thermoformer.reporting.c1_ablation_binary import (
    FIELDS,
    RESULT_ROOT,
    SEEDS,
    STAGES,
)


class C1AblationOutputTests(unittest.TestCase):
    def _record(self, path: Path) -> dict[str, str]:
        return {"path": path.relative_to(self.root).as_posix(), "sha256": artifact_sha256(path)}

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def _metric_rows(self, seed: int) -> list[dict[str, float | str]]:
        rows = []
        for direction, fields in FIELDS.items():
            row: dict[str, float | str] = {"scope": "direction", "direction": direction}
            for index, field in enumerate(fields):
                row[field] = (10.0 if direction == "isothermal" else 20.0) + index + 0.1 * seed
            rows.append(row)
        return rows

    def _summary_rows(self) -> list[dict[str, float | int | str]]:
        rows = []
        for direction, fields in FIELDS.items():
            row: dict[str, float | int | str] = {"scope": "direction", "direction": direction, "seeds": 5, "scope_available_seeds": 5}
            for index, field in enumerate(fields):
                values = [(10.0 if direction == "isothermal" else 20.0) + index + 0.1 * seed for seed in SEEDS]
                row[field + "_mean"] = fmean(values)
                row[field + "_std"] = stdev(values)
                row[field + "_available_seeds"] = 5
            rows.append(row)
        return rows

    def _csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(stream.getvalue(), encoding="utf-8")

    def _seed_artifacts(self, directory: Path, source_id: str, seed: int) -> None:
        seed_dir = directory / f"seed_{seed}"
        stage_metrics = self._metric_rows(seed)
        comparison = {
            "selection_partition": "validation",
            "evaluation_partition": "test",
            "selected_stage": "stage3",
            "parameter_summary": {
                "stage1_trainable_parameters": 80,
                "stage2_trainable_parameters": 100,
                "stage3": {"trainable_parameters": 20},
            },
            "stages": {stage: {"metrics": stage_metrics} for stage in STAGES},
        }
        comparison_path = seed_dir / "stage_comparison.json"
        artifacts: dict[str, dict[str, str]] = {}
        for stage in STAGES:
            checkpoint = self.root / 'models/vle' / source_id / f"seed_{seed}" / f"{stage}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes((source_id + stage).encode("utf-8"))
            checkpoint_record = self._record(checkpoint)
            prediction = seed_dir / f"{stage}_predictions.csv"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_text("sample_id,prediction\nfixture,0\n", encoding="utf-8")
            prediction_record = self._record(prediction)
            artifacts[stage + "_checkpoint"] = checkpoint_record
            artifacts[stage + "_predictions"] = prediction_record
            comparison["stages"][stage]["checkpoint"] = checkpoint_record["path"]
            comparison["stages"][stage]["checkpoint_sha256"] = checkpoint_record["sha256"]
            comparison["stages"][stage]["predictions"] = prediction_record["path"]
            comparison["stages"][stage]["predictions_sha256"] = prediction_record["sha256"]
        self._write_json(comparison_path, comparison)
        artifacts["stage_comparison"] = self._record(comparison_path)
        self._write_json(seed_dir / "manifest.json", {
            "status": "completed", "seed": seed,
            "protocol": C1_ABLATION_SOURCES[source_id].result_protocol,
            "split_protocol": "overall_binary", "run_kind": "formal",
            "selection_partition": "validation", "evaluation_partition": "test",
            "selected_stage": "stage3", "total_parameters": 100,
            "initially_trainable_parameters": 100, "artifacts": artifacts,
        })

    def _source(self, source_id: str) -> None:
        source = C1_ABLATION_SOURCES[source_id]
        directory = self.root / RESULT_ROOT / source.result_protocol
        for seed in SEEDS:
            self._seed_artifacts(directory, source_id, seed)
        summary, by_seed = directory / "metrics_summary.csv", directory / "metrics_by_seed.csv"
        self._csv(summary, self._summary_rows())
        self._csv(by_seed, [{"seed": seed, "value": float(seed)} for seed in SEEDS])
        self._write_json(directory / "aggregate_manifest.json", {
            "status": "completed", "aggregate_kind": "formal", "seeds": list(SEEDS),
            "protocol": source.result_protocol,
            "input_provenance": {"protocol": source.result_protocol},
            "outputs": {"metrics_summary": self._record(summary), "metrics_by_seed": self._record(by_seed)},
        })

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for source_id in C1_ABLATION_SOURCES:
            self._source(source_id)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_binary_three_stage_records_have_all_outputs(self) -> None:
        rows = collect_c1_ablation_rows(self.root)
        self.assertEqual(len(rows), 2 * len(C1_ABLATION_SOURCES))
        self.assertEqual({row["protocol"] for row in rows}, {"overall_binary"})
        self.assertEqual({row["direction"] for row in rows}, {"isothermal", "isobaric"})
        self.assertTrue(all("nonphysical_rate_mean" in row for row in rows))

    def test_report_contains_all_training_stages_and_shared_c1_source(self) -> None:
        report, metrics, manifest = write_c1_ablation_outputs(
            self.root, output_root=self.root / "generated", report_path=self.root / "report.md"
        )
        self.assertTrue(report.is_file())
        self.assertTrue(metrics.is_file())
        written = report.read_text(encoding="utf-8")
        self.assertIn("binary train -> binary test", written)
        self.assertIn("Stage 0 supervised reference", written)
        self.assertIn("Stage 3 fugacity-constrained", written)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["shared_source_mappings"]["vanilla_multicomponent_transformer"], "c1_final")
        self.assertEqual(payload["stage_order"], ["stage0", "stage1", "stage2", "stage3", "selected"])
        by_seed_path = Path(payload["outputs"]["selected_stages_by_seed"]["path"])
        counts_path = Path(payload["outputs"]["selected_stage_counts"]["path"])
        self.assertEqual(
            payload["selected_stage_counts"]["c1_final"],
            {"stage0": 0, "stage1": 0, "stage2": 0, "stage3": 5},
        )
        with by_seed_path.open("r", encoding="utf-8", newline="") as handle:
            by_seed = list(csv.DictReader(handle))
        with counts_path.open("r", encoding="utf-8", newline="") as handle:
            counts = list(csv.DictReader(handle))
        self.assertEqual(len(by_seed), len(C1_ABLATION_SOURCES) * len(SEEDS))
        self.assertEqual(len(counts), len(C1_ABLATION_SOURCES) * len(STAGES))
        self.assertEqual({row["selected_stage"] for row in by_seed}, {"stage3"})
        self.assertTrue(all(row["selection_partition"] == "validation" for row in by_seed))
        self.assertTrue(all(row["evaluation_partition"] == "test" for row in by_seed))
        self.assertIn("stage_prediction_artifacts", payload["inputs"]["c1_final"])

    def test_stage_prediction_artifacts_are_required_and_cross_checked(self) -> None:
        source_id = "c0_unimol"
        source = C1_ABLATION_SOURCES[source_id]
        directory = self.root / RESULT_ROOT / source.result_protocol
        manifest_path = directory / "seed_0" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"].pop("stage2_predictions")
        self._write_json(manifest_path, manifest)
        with self.assertRaisesRegex(RuntimeError, "stage2 predictions"):
            collect_c1_ablation_rows(self.root)

        self._source(source_id)
        comparison_path = directory / "seed_0" / "stage_comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["stages"]["stage1"]["predictions_sha256"] = "0" * 64
        self._write_json(comparison_path, comparison)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["stage_comparison"] = self._record(comparison_path)
        self._write_json(manifest_path, manifest)
        with self.assertRaisesRegex(RuntimeError, "stage1 predictions SHA-256"):
            collect_c1_ablation_rows(self.root)

    def test_joint_protocol_and_incomplete_seeds_are_rejected(self) -> None:
        source = C1_ABLATION_SOURCES["c0_unimol"]
        manifest_path = self.root / RESULT_ROOT / source.result_protocol / "aggregate_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["protocol"] = "c0_current_vanilla.on.overall_binary_ternary"
        self._write_json(manifest_path, payload)
        with self.assertRaisesRegex(RuntimeError, "wrong-protocol|binary-only"):
            collect_c1_ablation_rows(self.root)
        payload["protocol"] = source.result_protocol
        payload["seeds"] = [0, 1, 2, 3]
        self._write_json(manifest_path, payload)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            collect_c1_ablation_rows(self.root)


if __name__ == "__main__":
    unittest.main()
