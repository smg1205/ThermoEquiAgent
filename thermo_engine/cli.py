"""Command-line interface for LLM-independent execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schemas.domain import TaskManifest
from thermo_engine.service import calculate_equilibrium, validate_equilibrium_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ThermoEqui task manifest")
    parser.add_argument("manifest", type=Path, help="Path to a TaskManifest JSON file")
    parser.add_argument("--output", type=Path, help="Optional output JSON file")
    args = parser.parse_args()
    manifest = TaskManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    result = calculate_equilibrium(manifest)
    validation = validate_equilibrium_result(result)
    payload = json.dumps(
        {
            "result": result.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
