# Final C1 binary-only evaluation

```powershell
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol overall_binary --seeds 0 1 2 3 4
```

The command uses the final three-view C1 architecture, ten fugacity-fine-tuning
epochs, and validation-only checkpoint selection. It consumes the five registered
`overall_binary` splits without regenerating them.
