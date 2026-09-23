# Scientific experiment-to-code map

This map links each article result to its runnable implementation and registered result source.

| Article section | Experiment or analysis | Implementation | Results |
|---|---|---|---|
| Phase-equilibrium dataset construction | VLE/LLE data audits | `experiments/data_quality/construction/dataset_distribution/`, `scripts/audit_dataset.py` | `experiments/data_quality/construction/dataset_distribution/results/` |
| VLE predictive performance | Binary, ternary, and mixed training | `scripts/benchmarks/vle/run_prediction.py` | `experiments/vle/prediction/` |
| VLE generalization | Composition, state, and unseen-component protocols | `scripts/benchmarks/vle/run_generalization.py` | `experiments/vle/generalization/` |
| LLE predictive performance | Binary, ternary, and mixed random/system protocols | `scripts/benchmarks/lle/run_prediction.py` | `experiments/lle/prediction/` |
| LLE generalization and stability | State extrapolation, unseen components, activity and TPD diagnostics | `scripts/benchmarks/lle/run_generalization.py` | `experiments/lle/generalization/`, `experiments/lle/thermodynamics/` |
| Model comparison | Machine-learning and thermodynamic VLE baselines | registered VLE comparison backends | `experiments/vle/comparison/` |
| Molecular and interaction ablation | Joint binary--ternary ablation | `scripts/benchmarks/vle/run_ablation.py` | `experiments/vle/ablation/joint/` |
| Staged thermodynamic optimization | Stage 0--3 validation diagnostics | `scripts/evaluate_staged_thermodynamic_optimization.py` | `experiments/vle/ablation/staged_validation/` |
| Learned multicomponent thermodynamics | Molecular-view and composition-dependent attribution | `scripts/benchmarks/vle/generate_interpretability.py` | `experiments/vle/interpretability/molecular_interactions/` |
| Agent-assisted separation design | ThermoEqui-Agent case studies | implementation and case artifacts unavailable | `experiments/vle/separation_design/` |

The validation-selected VLE and LLE prediction/generalization checkpoints are listed in `models/registry.json`. Comparison, ablation, and interpretability checkpoints are optional and are not distributed.
