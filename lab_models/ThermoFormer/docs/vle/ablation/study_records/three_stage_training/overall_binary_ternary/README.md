# C1 cumulative three-stage ablation

This experiment isolates the contribution of the final ThermoFormer C1 training procedure on the registered `overall_binary_ternary` protocol. It retains the C1 three-view molecular representation and vanilla Transformer without chemical-attention bias or context-conditioned pair interaction.

Each random seed produces one shared trajectory and four frozen, validation-selected endpoints:

- A0: supervised reference;
- A1: direct GE/RT and log-gamma stage;
- A2: joint VLE supervision stage;
- A3: fugacity-constrained fine-tuning stage.

All endpoints are evaluated on the same joint test partition only after checkpoint selection. The auxiliary deployment checkpoint selects the best of A0--A3 using validation data only; it does not replace any endpoint in the primary cumulative ablation.

Smoke artifacts are isolated under `experiments/run_records/smoke/three_stage_training/overall_binary_ternary`. Formal artifacts are written under matching `runs`, `checkpoints`, and `results` namespaces.
