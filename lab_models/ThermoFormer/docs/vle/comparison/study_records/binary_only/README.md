# Binary-only comparison protocol

This experiment trains and evaluates ThermoFormer exclusively on the registered
`overall_binary` system split. The model and training method are identical to the
current direct-excess-Gibbs-energy ThermoFormer campaign; only the registered
component-cardinality protocol changes.

The three stages are:

1. 20 epochs of direct excess-Gibbs-energy and activity-coefficient supervision;
2. up to 80 epochs of joint VLE supervision with validation-based early stopping;
3. 10 epochs of low-weight fugacity-equilibrium fine-tuning.

The Stage-0 checkpoint and all later-stage candidates are selected using validation
data. The locked test partition is evaluated only after model selection.

See [run.md](run.md) for reproducible commands. Formal outputs are written under
`experiments/vle/generalization/evaluations/comparisons/binary_only/direct_ge/`.
