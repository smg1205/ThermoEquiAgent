# Run thermodynamic-model state-generalization baselines

Smoke test:

```powershell
conda run -n ggnn39 python scripts/run_thermodynamic_baselines.py --smoke --overwrite
```

Complete NRTL, Wilson, and UNIQUAC campaign:

```powershell
conda run -n ggnn39 python scripts/run_thermodynamic_baselines.py
```
