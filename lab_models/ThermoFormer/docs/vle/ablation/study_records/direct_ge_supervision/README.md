# Three-stage direct GE supervision

This experiment retains the registered C1 three-view vanilla Transformer,
thermodynamic equations, split, and differentiable VLE solver. It changes only
the training signals:

1. direct supervision of `GE/RT` and `ln(gamma)` from externally validated
   Antoine/DIPPR correlations;
2. joint direct-thermodynamic and VLE supervision using the established
   supervised budget;
3. ten epochs of partial fine-tuning with all supervised terms retained and a
   warm-started, low-weight fugacity-equilibrium loss.

Rows without reliable pure-component vapor pressure remain in VLE training and
are masked only from the new direct-label losses.

The formal campaign uses the frozen `overall_binary_ternary` seeds 0--4. It
publishes validation-selected final checkpoints and a separately labelled Stage-3
diagnostic, with joint, binary, and ternary metrics. Existing joint baselines are
joined only after split and test-partition provenance checks; binary-only and
298 K partial-coverage baselines are not used for full-test winner claims.
