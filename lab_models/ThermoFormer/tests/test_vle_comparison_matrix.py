import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/vle/comparison/v1"


def _digest(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".tex", ".py"}:
        payload = payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def test_vle_comparison_matrix_contract() -> None:
    csv_path = OUTPUT / "VLE_comparison_matrix.csv"
    md_path = OUTPUT / "VLE_comparison_matrix.md"
    tex_path = OUTPUT / "VLE_comparison_matrix.tex"
    manifest_path = OUTPUT / "VLE_comparison_matrix_manifest.json"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["seed_count"] in {"4", "5"} for row in rows)
    assert all(
        row["seed_count"] == "5" or
        (row["model"] == "SolvGNN" and row["direction"] == "isothermal" and row["component_count"] == "2+3")
        for row in rows
    )

    for model in ("HANNA adapted", "TeNNet-SAC adapted"):
        joint = {
            (row["direction"], row["component_count"])
            for row in rows
            if row["model"] == model and row["training_scope"] == "joint"
        }
        assert joint == {
            ("isothermal", "3"), ("isothermal", "2+3"),
            ("isobaric", "3"), ("isobaric", "2+3"),
        }

    ualf_iso = next(
        row for row in rows
        if row["model"] == "UALF-GNN" and row["direction"] == "isothermal"
    )
    assert ualf_iso["status"] == "not_applicable_native_direction_isobaric_only"
    assert ualf_iso["state_mae_mean"] == ""
    assert ualf_iso["y_mae_mean"] == ""

    assert "UALF-GNN isothermal P/y remains N/A" in md_path.read_text(encoding="utf-8")
    assert "UALF-GNN isothermal: N/A" in tex_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == len(rows)
    for relative, expected in manifest["inputs"].items():
        assert _digest(ROOT / relative) == expected
    for relative, expected in manifest["outputs"].items():
        assert _digest(ROOT / relative) == expected


def test_compact_publication_table_contract() -> None:
    csv_path = OUTPUT / "VLE_comparison_table.csv"
    md_path = OUTPUT / "VLE_comparison_table.md"
    manifest_path = OUTPUT / "VLE_comparison_table_manifest.json"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == [
            "Evaluation protocol", "Model", "P (kPa), isothermal", "y, isothermal",
            "T (K), isobaric", "y, isobaric", "Coverage (%)",
        ]

    identities = [(row["Evaluation protocol"], row["Model"]) for row in rows]
    assert len(identities) == len(set(identities))
    group_models = {}
    for row in rows:
        group_models.setdefault(row["Evaluation protocol"], []).append(row["Model"])
    assert group_models == {
        "Matched binary→binary": [
            "ThermoFormer", "HANNA adapted", "TeNNet-SAC adapted",
            "SMILES-RNN", "UALF-GNN", "Descriptor ANN",
        ],
        "Matched ternary→ternary": ["ThermoFormer", "HANNA adapted", "TeNNet-SAC adapted"],
        "Matched joint→ternary": ["ThermoFormer", "HANNA adapted", "TeNNet-SAC adapted", "SPT-NRTL adapted"],
        "Matched joint→joint": ["ThermoFormer", "HANNA adapted", "TeNNet-SAC adapted", "SPT-NRTL adapted"],
        "External direct→ternary": ["HANNA official", "TeNNet-SAC official", "SPT-NRTL official"],
        "External direct→joint": ["HANNA official", "TeNNet-SAC official", "SPT-NRTL official"],
        "Local-temperature reference 298.15±0.5 K": ["SolvGNN", "GDI-GNN", "GE-GNN"],
    }
    for row in rows:
        for field in (
            "P (kPa), isothermal", "y, isothermal",
            "T (K), isobaric", "y, isobaric",
        ):
            assert [line.split(":", 1)[0] for line in row[field].splitlines()] == ["MAE", "RMSE", "R2"]

    ualf = next(row for row in rows if row["Evaluation protocol"] == "Matched binary→binary" and row["Model"] == "UALF-GNN")
    assert ualf["P (kPa), isothermal"].splitlines() == ["MAE: N/A", "RMSE: N/A", "R2: N/A"]
    assert ualf["y, isothermal"].splitlines() == ["MAE: N/A", "RMSE: N/A", "R2: N/A"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocols = [
        "Matched binary→binary", "Matched ternary→ternary",
        "Matched joint→ternary", "Matched joint→joint",
        "External direct→ternary", "External direct→joint",
        "Local-temperature reference 298.15±0.5 K",
    ]
    assert manifest["protocol_groups"] == protocols
    assert manifest["empty_protocol_groups"] == []
    assert manifest["row_count"] == len(rows)
    assert manifest["metric_line_order"] == ["MAE", "RMSE", "R2"]
    md = md_path.read_text(encoding="utf-8")
    assert [md.index(f"## {protocol}") for protocol in protocols] == sorted(md.index(f"## {protocol}") for protocol in protocols)
    for relative, expected in manifest["outputs"].items():
        assert _digest(ROOT / relative) == expected

