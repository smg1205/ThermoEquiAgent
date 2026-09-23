# Joint binary--ternary VLE ablation benchmark

This benchmark evaluates the molecular-representation and interaction-architecture
variants on the release VLE dataset. Every variant uses the registered
`vle_overall_binary_ternary` assignments for seeds 0--4. For each seed, all
variants receive the same training, validation, and test states.

Training follows the same three-stage schedule as the main VLE benchmark:
20 epochs of thermodynamic pretraining, up to 80 epochs of supervised VLE
training, and 10 epochs of fugacity-consistency fine-tuning. Checkpoint selection
uses validation data only.

The C1 three-view vanilla model is shared with the main VLE benchmark. Its five
completed checkpoints and predictions are referenced directly, avoiding a
duplicate training run. The other five structural variants are trained here.

Run the campaign with:

```powershell
python scripts/run_vle_joint_ablations.py --device cuda
python scripts/build_vle_joint_ablation_report.py
```

Machine-readable results and publication tables are written to
`experiments/vle/ablation/joint/`.

