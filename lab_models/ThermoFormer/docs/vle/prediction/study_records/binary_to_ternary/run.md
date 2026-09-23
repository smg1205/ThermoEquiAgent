# Binary-to-ternary transfer

The same command covers zero-shot and the five ternary-data fractions.

```powershell
conda run -n ggnn39 python scripts\run_binary_generalization_three_stage.py --group transfer --smoke --device cuda
conda run -n ggnn39 python scripts\run_binary_generalization_three_stage.py --group transfer --device cuda
```

All six protocols share the same ternary test systems for a given seed. Formal checkpoint selection uses validation data only.
