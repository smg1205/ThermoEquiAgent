# Reproduction

```powershell
conda run -n ggnn39 python scripts\generate_splits.py
conda run -n ggnn39 python scripts\run_binary_generalization_three_stage.py --group unseen --smoke --device cuda
conda run -n ggnn39 python scripts\run_binary_generalization_three_stage.py --group unseen --device cuda
```
