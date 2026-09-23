# Reproducibility guide

ThermoFormer supports registered VLE and LLE prediction, generalization, comparison, ablation, and analysis workflows. Scientific definitions, datasets, splits, checkpoints, and result artifacts are identified by explicit paths and hashes.

## Repository entry points

- `scripts/run_experiments.py`: list, inspect, run, index, and verify registered experiments.
- `scripts/benchmarks/vle/`: descriptive VLE prediction and generalization commands.
- `scripts/benchmarks/lle/`: descriptive LLE prediction and generalization commands.
- `configs/catalog.json`: authoritative experiment registry.
- `docs/reproducibility/experiment_map.md`: mapping from a scientific question to its configuration, command, and output directory.
- `docs/reproducibility/scientific_alignment.md`: scope limits and manuscript-alignment findings.
- `datasets/README.md`: dataset registry and provenance.
- `models/README.md`: checkpoint registry and loading guidance.

## Environment

Create and activate the declared Python 3.9 environment:

```powershell
conda env create -f environment.yml
conda activate ggnn39
```

An existing Python 3.9 environment can use `python -m pip install -r requirements.txt`. GPU training requires a PyTorch build compatible with the local CUDA runtime.

## Inspect and verify

Run commands from the repository root:

```powershell
python scripts/run_experiments.py list
python scripts/run_experiments.py show vle.prediction.binary
python scripts/run_experiments.py verify
python -m pytest -q
```

The verification command checks registered scientific-content hashes. The test suite checks loaders, split rules, thermodynamic solvers, model interfaces, reporting, and artifact registries; it does not retrain formal models.

## Run registered benchmarks

Inspect a command without creating outputs:

```powershell
python scripts/benchmarks/vle/run_prediction.py binary --dry-run --run-id example
python scripts/benchmarks/lle/run_prediction.py ternary_random --dry-run --run-id example
```

Audit splits before training:

```powershell
python scripts/benchmarks/vle/run_prediction.py binary --check-splits-only --run-id example
python scripts/benchmarks/lle/run_prediction.py binary_random --check-splits-only --run-id example
```

Run the five registered seeds:

```powershell
python scripts/benchmarks/vle/run_prediction.py binary --seeds 0 1 2 3 4 --run-id example
python scripts/benchmarks/lle/run_prediction.py ternary_random --seeds 0 1 2 3 4 --run-id example
```

Execute VLE and LLE prediction protocols before their generalization protocols. Run at most two GPU training jobs concurrently. Baseline comparisons, ablations, and interpretability analyses apply to VLE only.

## Output identity

`--run-id` creates a versioned execution directory. Each execution records its command, status, input hashes, seed list, and output paths. `--resume` accepts only an identity-matched execution. Formal backends require committed scientific inputs and a clean Git state so that checkpoints and metrics can be traced to a unique source version.

Five-seed summaries use the mean and sample standard deviation. The test set is evaluated only after validation-based checkpoint selection. Missing or undefined metrics remain explicit; they are never replaced with zero.

