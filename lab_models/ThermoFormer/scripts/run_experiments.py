"""Unified research workflow CLI: list, show, run, index, verify."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmarks.registry import main
if __name__ == "__main__":
    main()
