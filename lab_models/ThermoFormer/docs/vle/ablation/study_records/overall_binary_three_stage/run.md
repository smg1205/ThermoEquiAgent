# Binary-only three-stage ablation

## Protocol

All variants use `overall_binary`: binary mixtures only, with mutually exclusive
chemical systems in the training, validation, and test partitions. Formal
results use seeds `0` through `4`; selection among Stage 0, Stage 1, Stage 2,
and Stage 3 is based only on validation data.

## Smoke validation

```powershell
conda run -n ggnn39 python scripts\run_overall_binary_three_stage_ablations.py --smoke --device cuda
```

Smoke runs use seed `0`, one epoch per trainable stage, and validation-set
evaluation only. They are written under `experiments/run_records/smoke/overall_binary_three_stage/`.

## Formal campaign

```powershell
conda run -n ggnn39 python scripts\run_overall_binary_three_stage_ablations.py --device cuda
conda run -n ggnn39 python scripts\build_c1_ablation_report.py
```

To resume a subset explicitly, specify the retained variant and seed set:

```powershell
conda run -n ggnn39 python scripts\run_overall_binary_three_stage_ablations.py --variant c1_three_view_vanilla --seeds 2 3 4 --device cuda
```

Formal run, checkpoint, and result bundles are stored under their respective
`overall_binary_three_stage` namespaces. The report generator rejects incomplete
seed sets, foreign protocols, or test-selected checkpoints.
