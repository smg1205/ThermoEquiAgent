# Predictive-performance experiments

This category contains every row of manuscript Table 1:

- `overall_binary/`: binary training to binary test;
- `overall_binary_ternary/`: joint binary/ternary training with binary and ternary test subsets;
- `state_generalization/`: six within-system interpolation and extrapolation protocols;
- `unseen_components/`: system-disjoint unseen-component evaluation;
- `binary_to_ternary/`: zero-shot and five ternary-data fractions.

All reported values use seeds 0--4 and validation-only checkpoint selection. The
machine-readable source is `experiments/vle/prediction/summary/c1_fugacity_generalization_by_task.csv`.
