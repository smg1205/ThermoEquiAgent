# Reproduction guide

## Prerequisites

1. Create or update the `ggnn39` environment from `environment.yml`.
2. Materialize Git-LFS checkpoints with `git lfs pull`.
3. Keep formal code, configuration, split, and input artifacts committed.
4. Run commands from the repository root.

## Verification

```powershell
conda run -n ggnn39 python -m unittest discover -s tests
conda run -n ggnn39 python scripts\validate_splits.py
```

## Reports

```powershell
conda run -n ggnn39 python scripts\build_c1_generalization_report.py
conda run -n ggnn39 python scripts\build_c1_ablation_report.py
```

## Run semantics

- Formal runs require registered splits and committed scientific inputs.
- Smoke runs are diagnostic and write to isolated directories.
- Checkpoint selection uses validation only.
- Formal aggregation requires seeds 0--4 and validates input and output hashes.
- Direction-resolved reports include MAE, RMSE, R2, valid coverage, solver
  failure rate, and nonphysical rate where applicable.
