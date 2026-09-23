# Commands

Generate the registered ternary-only projections:

```powershell
conda run -n ggnn39 python scripts/generate_ternary_splits.py
```

Run isolated smoke tests:

```powershell
conda run -n ggnn39 python scripts/run_ternary_only_thermoformer.py --smoke --device cuda
conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --benchmark ternary_train_ternary_test --smoke --device cuda
```

Run the five-seed formal experiments:

```powershell
conda run -n ggnn39 python scripts/run_ternary_only_thermoformer.py --seeds 0 1 2 3 4 --device cuda
conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --benchmark ternary_train_ternary_test --seeds 0 1 2 3 4 --device cuda
```
