# Scientific experiments

The common runnable catalog is configs/catalog.json. Use
python scripts/run_experiments.py list to discover experiment IDs and show to
inspect the registered backend and original configuration.

- VLE: predictive benchmarks; composition/state/chemical generalization;
  machine-learning and classical-model comparisons; representation/architecture
  ablations; physical diagnostics and interpretability.
- LLE: predictive benchmarks; system/state/chemical generalization;
  activity-equilibrium and finite-grid stability diagnostics.
- Data auditing and exploratory separation design are separately labeled.

Existing subdirectories retain their original protocol configurations, records,
and results references. Do not infer that all studies use the latest data version.
See docs/reproducibility/experiment_map.md for the complete mapping.
