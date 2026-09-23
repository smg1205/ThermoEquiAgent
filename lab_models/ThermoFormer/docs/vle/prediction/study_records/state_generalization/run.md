# State generalization

Run the six registered state protocols for seeds 0--4:

```powershell
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_composition_interpolation --seeds 0 1 2 3 4
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_composition_edge_extrapolation --seeds 0 1 2 3 4
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_temperature_low_extrapolation --seeds 0 1 2 3 4
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_temperature_high_extrapolation --seeds 0 1 2 3 4
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_pressure_low_extrapolation --seeds 0 1 2 3 4
conda run -n ggnn39 python scripts\run_c1_physics_finetune.py --protocol state_pressure_high_extrapolation --seeds 0 1 2 3 4
```

Each command uses the final C1 model and validation-gated fugacity fine-tuning.
