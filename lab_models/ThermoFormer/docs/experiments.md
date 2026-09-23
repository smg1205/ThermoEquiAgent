# Experiment design

Experiments are organized by the scientific question posed in the manuscript:
predictive performance, comparison, ablation, interpretation, and separation
design. Runnable leaves contain a strict configuration, a reproducible command,
and a scientific result record. Planned studies are marked `not_evaluated` and
contain no numerical placeholders.

The final C1 ablations use only `overall_binary_ternary` with seeds 0--4 and
compare molecular views, interaction architecture, and fugacity-constrained
fine-tuning. Generalization protocols retain their registered split semantics:
state studies hold out states within systems, whereas unseen-component and
transfer studies impose chemical-system constraints.

See [`docs/experiment_overview.md`](../experiments/README.md) for the registry and
[`experiment_code_map.md`](experiment_code_map.md) for the manuscript mapping.
