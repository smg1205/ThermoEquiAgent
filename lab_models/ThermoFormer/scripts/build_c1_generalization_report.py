"""Build the validated final-C1 fugacity generalization report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.reporting.c1_generalization import write_c1_generalization_outputs


def main() -> None:
    """Build the validated final-model generalization report."""
    print(
        json.dumps(
            write_c1_generalization_outputs(PROJECT_ROOT),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

