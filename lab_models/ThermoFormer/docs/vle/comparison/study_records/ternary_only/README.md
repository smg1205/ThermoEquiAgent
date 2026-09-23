# Ternary-only baseline comparison

This experiment uses the registered `overall_binary_ternary` system assignment but
retains only three-component rows in training, validation, and test partitions. No
systems are reassigned. Checkpoint selection uses the ternary validation partition;
the ternary test partition is evaluated only after model selection.

The strict comparison includes ThermoFormer and ternary-trained residual adaptations
of HANNA and TeNNet-SAC. Fixed official models and database-only baselines are excluded
because they do not perform ternary training on the registered ThermoFormer split.

See `run.md` for the smoke and formal commands. Formal results remain unpublished until
all five seeds are complete and provenance checks pass.
