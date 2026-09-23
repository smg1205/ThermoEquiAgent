# Direct excess-Gibbs supervision experiment

The registered campaign compares the fixed C1 checkpoints with validation-selected
three-stage training on `overall_binary_ternary` for seeds 0--4. The Stage-3 candidate
is aggregated separately and is never selected with test metrics. Smoke artifacts are
isolated from formal result directories.

```powershell
conda run -n ggnn39 python scripts\run_c1_direct_ge_training.py --device cuda --smoke --overwrite
```

Run or resume the complete formal campaign:

```powershell
conda activate ggnn39
cd D:\VLE\paper_structure_worktree
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\direct_ge_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python -u scripts\run_c1_direct_ge_training.py --device cuda 2>&1 |
    Tee-Object -FilePath $log
```

The CLI prints and flushes one line per epoch with the seed, stage, epoch,
training loss, validation composite, four VLE validation MAEs, best score, and elapsed time.

Run a registered subset when recovering an interrupted campaign:

```powershell
conda run -n ggnn39 python scripts\run_c1_direct_ge_training.py --device cuda --seeds 2 3 4
```

Completed seed bundles are reused only after request, provenance, and artifact-hash
validation. The five-seed report is published only when seeds 0--4 are complete.
If a process stops while a seed manifest is `running`, the next invocation renames
that seed's run, checkpoint, and result directories with an `interrupted_TIMESTAMP`
suffix before restarting it. Completed seeds are never archived automatically.
