# Reproduction commands

Run the isolated smoke test:

```powershell
conda run --no-capture-output -n ggnn39 python -u scripts/run_c1_direct_ge_training.py --protocol overall_binary --smoke --device cuda
```

Run the registered five-seed campaign with live progress and a persistent log:

```powershell
New-Item experiments/run_records/logs/binary_only -ItemType Directory -Force | Out-Null
python -u scripts/run_c1_direct_ge_training.py --protocol overall_binary --seeds 0 1 2 3 4 --device cuda 2>&1 | Tee-Object -FilePath experiments/run_records/logs/binary_only/thermoformer_direct_ge_seeds_0_4.log
```

The campaign verifies completed seed artifacts before resuming. Use `--overwrite`
only when a deliberate full rerun is required.
