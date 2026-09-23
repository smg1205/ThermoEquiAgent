"""Research workflow vle/prediction; unchanged scientific backend."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.benchmarks._dispatch import dispatch
if __name__ == "__main__":
    dispatch("vle.prediction")
