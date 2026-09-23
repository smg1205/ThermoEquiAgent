# ThermoFormer

ThermoFormer predicts binary and ternary phase equilibria from molecular representations and thermodynamic state variables.

- **VLE:** SMILES, temperature and liquid composition to pressure and vapor composition; or SMILES, pressure and liquid composition to temperature and vapor composition.
- **LLE:** SMILES, temperature and pressure to liquid coexistence endpoints. Ternary predictions are unordered tie-line sets.
- Baseline comparison, ablation and interpretability studies apply to VLE. LLE benchmarks cover prediction, generalization and thermodynamic stability.

## Installation

Install Git LFS before cloning to download the model weights and molecular feature arrays:

```bash
git lfs install
git clone https://github.com/JinlinYY/ThermoFormer.git
cd ThermoFormer
git lfs pull
```

```bash
conda env create -f environment.yml
conda activate ggnn39
```

## Repository organization

| Directory | Contents |
|---|---|
| `configs/` | Experiment registry, model settings and training configurations |
| `datasets/` | VLE/LLE workbooks, split assignments, molecular resources and thermodynamic labels |
| `src/thermoformer/` | Scientific models, solvers, training, evaluation and baseline adapters |
| `scripts/` | Experiment, inference, evaluation and plotting entry points |
| `models/` | Validation-selected checkpoints and required pretrained resources |
| `experiments/` | Predictions, metrics, figures and reproducibility records, organized by task and study |
| `docs/` | Method, dataset, baseline and reproduction documentation |
| `tests/` | Scientific and interface regression tests |

## Experiments

Run commands from the repository root:

```bash
python scripts/run_experiments.py list
python scripts/run_experiments.py show vle.prediction.binary
python scripts/run_experiments.py run vle.prediction.binary --seeds 0 1 2 3 4 --device cuda --run-id v2
python scripts/run_experiments.py run lle.prediction.binary_random --seeds 0 1 2 3 4 --device cuda --run-id v2
```

Use `--dry-run` to inspect a command, `--check-splits-only` to audit supported campaign splits, and `--resume` for a matching recorded execution. Use a new run identifier for a new experiment.

The principal VLE and LLE result tables and all five-seed summaries are in `experiments/summary/`. Task-specific results and figures are under `experiments/vle/` and `experiments/lle/`. Read [the experiment directory guide](docs/experiment_layout.md) for their organization.

## Checkpoints and verification

`models/registry.json` identifies 50 VLE and 55 LLE final checkpoints selected using validation data. Mixed-training models are shared across their binary, ternary and joint test subsets. Baseline-specific weights are obtained separately from their cited upstream sources.

```bash
python scripts/validate_checkpoint_catalog.py --load
python scripts/verify_phase_equilibrium_results.py
python scripts/run_experiments.py verify
python -m pytest -q
```

LLE uses a two-phase prior, binary RK free-energy parameterization and a continuous parallel tie-line approximation for ternary systems. Finite-grid TPD certification is a numerical stability diagnostic, not a global proof over the continuous composition simplex. Compare metrics only when dataset identity, split protocol and evaluation targets agree.
