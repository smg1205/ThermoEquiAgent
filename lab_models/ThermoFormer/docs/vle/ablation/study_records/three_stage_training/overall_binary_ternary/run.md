# Reproducible commands

## Smoke validation

```powershell
conda run -n ggnn39 python scripts/run_three_stage_training_ablation.py --smoke --device cuda
```

Smoke uses seed 0, one epoch in each trainable stage, validation-only evaluation, and an isolated artifact namespace.

## Formal five-seed campaign

```powershell
conda run -n ggnn39 python scripts/run_three_stage_training_ablation.py --device cuda
```

The command runs seeds 0--4, then aggregates the formal selected-checkpoint metrics and writes the cumulative-stage report. To resume a completed seed without overwriting it, pass a subset:

```powershell
conda run -n ggnn39 python scripts/run_three_stage_training_ablation.py --seeds 2 3 4 --device cuda
```

After all five seed bundles are complete, rebuild only the report with:

```powershell
conda run -n ggnn39 python scripts/run_three_stage_training_ablation.py --report-only
```
