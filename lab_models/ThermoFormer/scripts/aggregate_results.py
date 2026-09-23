"""Aggregate five completed ThermoFormer seed runs for one or all protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.reporting.aggregation import aggregate_protocol_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / 'experiments/reference_results')
    parser.add_argument("--protocol", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow a partial seed set and write diagnostic-prefixed summaries",
    )
    args = parser.parse_args()
    protocol_dirs = (
        [args.results_root / args.protocol]
        if args.protocol
        else sorted(path for path in args.results_root.iterdir() if path.is_dir())
    )
    summary = {
        path.name: aggregate_protocol_results(
            path,
            args.seeds,
            aggregate_kind="diagnostic" if args.diagnostic else "formal",
        )
        for path in protocol_dirs
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
