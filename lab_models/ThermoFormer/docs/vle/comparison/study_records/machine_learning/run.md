# Reproduction commands

Run the seed-0 diagnostic smoke test in the project environment:

```powershell
conda run -n ggnn39 python scripts/run_ml_baselines.py
```

This command writes only under `experiments/run_records/smoke/ml_vle_baselines/seed_0`.

Run the registered five-seed formal campaign:

```powershell
conda run -n ggnn39 python scripts/run_ml_baselines_formal.py --seeds 0 1 2 3 4
```

Before adding the two fixed external baselines to a formal campaign, run their
validation-only interface and coverage diagnostics:

```powershell
conda run -n ggnn39 python scripts/run_external_activity_baselines_smoke.py --baseline tennet_sac
conda run -n ggnn39 python scripts/run_external_activity_baselines_smoke.py --baseline spt_nrtl
conda run -n ggnn39 python scripts/run_spt_nrtl_adapted_smoke.py
```

Only after both diagnostics pass, run their frozen split evaluations:

```powershell
conda run -n ggnn39 python scripts/run_ml_baselines_formal.py --baselines tennet_sac spt_nrtl --seeds 0 1 2 3 4
```

After its validation-only smoke succeeds, run the same-data adapted comparison with:

```powershell
conda run -n ggnn39 python scripts/run_ml_baselines_formal.py --baselines spt_nrtl_adapted --seeds 0 1 2 3 4
```

The command resolves `formal_settings.json`; the diagnostic smoke continues to use
the separate non-formal `settings.json`.

The formal runner never selects on test. It writes seed-level predictions, metrics,
history, checkpoints, manifests, and mean +/- sample-standard-deviation summaries.
Models lacking author inputs or official pretrained assets are recorded as blocked
rather than replaced with an unlabelled approximation.

Run the validation-only smoke and formal joint adaptations with:

```powershell
conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --baselines hanna_joint_adapted tennet_sac_joint_adapted --seeds 0 --smoke --device cuda
conda run -n ggnn39 python scripts/run_joint_activity_adapted.py --baselines hanna_joint_adapted tennet_sac_joint_adapted --seeds 0 1 2 3 4 --epochs 100 --patience 15 --device cuda
```

The official base parameters remain frozen. Only the 1,345-parameter residual head is
optimized; validation selects its checkpoint and test labels are not used for selection.
