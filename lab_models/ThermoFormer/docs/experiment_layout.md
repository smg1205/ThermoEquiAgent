# Experiment outputs

All result data, tables and figures are stored under `experiments/`.

- `vle/prediction/`, `vle/generalization/`: VLE predictive and extrapolation evaluations.
- `vle/comparison/`, `vle/ablation/`, `vle/interpretability/`: VLE reference models, controlled experiments and molecular analysis.
- `vle/separation_design/`: registered separation-design cases and availability information.
- `lle/prediction/`, `lle/generalization/`, `lle/thermodynamics/`: liquid coexistence benchmarks and certification results.
- `summary/`: cross-task five-seed tables in CSV, Markdown and LaTeX.
- `data_quality/`: data and split audits.
- Task-specific `training_records/` directories: recorded selection histories and execution manifests.
- `run_records/`: shared campaign execution records.
- `reference_results/`: additional result bundles whose data and protocol identity must be checked before comparison.

Figures are colocated with their experiment data in `figures/` subdirectories. Final checkpoint paths are supplied by `models/registry.json`; configuration paths are supplied by `configs/catalog.json`. Split assignments are under `datasets/splits/vle/` or the recorded LLE condition partition.

Frozen execution records preserve the recorded scientific settings. A model result is identified by its task, protocol, seed, data digest and checkpoint digest, rather than by the name of its enclosing directory.
