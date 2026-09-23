"""Shared implementation for descriptive section entry points."""
import sys
from scripts.benchmarks.registry import main
def dispatch(prefix):
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("Usage: python <this-script> SUFFIX [--dry-run] [--run-id VERSION] [--seeds 0 1 2 3 4]")
        print("List suffixes: python scripts/run_experiments.py list; prefix: "+prefix)
        return
    main(["run",prefix+"."+sys.argv[1],*sys.argv[2:]])
