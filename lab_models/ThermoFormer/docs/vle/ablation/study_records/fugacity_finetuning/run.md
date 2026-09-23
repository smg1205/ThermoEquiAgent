# Fugacity fine-tuning ablation

```powershell
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol overall_binary_ternary --seeds 0 1 2 3 4
```

The configuration records the same C1 architecture and fugacity-only Stage-2
settings used by the dedicated fine-tuning runner.
