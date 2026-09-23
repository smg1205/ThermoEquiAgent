"""Generate the five-seed VLE interpretability analysis."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.thermoformer.interpretability.vle_binary import run_c1_final_interpretability

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples-per-seed", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    maximum = 12 if args.smoke and args.max_samples_per_seed == 256 else args.max_samples_per_seed
    output_root = PROJECT_ROOT / "experiments/run_records/interpretability_smoke/vle_binary" if args.smoke else None
    result = run_c1_final_interpretability(PROJECT_ROOT, device_name=args.device, max_samples_per_seed=maximum, batch_size=args.batch_size, output_root=output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
