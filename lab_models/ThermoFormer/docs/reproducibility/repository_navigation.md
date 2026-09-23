# Repository navigation and reproduction order

ThermoFormer organizes every scientific question through a benchmark configuration, a task entry point, and a result directory.

| Scientific task | Configuration | Command | Results |
|---|---|---|---|
| VLE prediction | `configs/vle/prediction/` | `python scripts/benchmarks/vle/run_prediction.py <setting>` | `experiments/vle/prediction/` |
| VLE generalization | `configs/vle/generalization/` | `python scripts/benchmarks/vle/run_generalization.py <setting>` | `experiments/vle/generalization/` |
| LLE prediction | `configs/lle/prediction/` | `python scripts/benchmarks/lle/run_prediction.py <setting>` | `experiments/lle/prediction/` |
| LLE generalization | `configs/lle/generalization/` | `python scripts/benchmarks/lle/run_generalization.py <setting>` | `experiments/lle/generalization/` |
| LLE stability diagnostics | `configs/lle/thermodynamics/` | `python scripts/run_experiments.py show lle.thermodynamics.stability` | `experiments/lle/thermodynamics/` |
| VLE comparisons | `configs/vle/comparison/` | registered comparison backends | `experiments/vle/comparison/` |
| VLE ablations | `configs/vle/ablation/` | `python scripts/benchmarks/vle/run_ablation.py --help` | `experiments/vle/ablation/` |
| VLE interpretability | `configs/vle/interpretability/` | registered analysis backends | `experiments/vle/interpretability/` |
| Separation design | `configs/separation_design/` | inspect the registered artifact scope | `experiments/vle/separation_design/` |

Recommended order:

1. Create the declared `ggnn39` environment and run `python scripts/run_experiments.py verify`.
2. Inspect dataset manifests and run split-only checks.
3. Run VLE and LLE prediction protocols for seeds 0--4.
4. Run state, composition, and unseen-component generalization protocols.
5. Recompute aggregate metrics with `python scripts/verify_phase_equilibrium_results.py`.
6. Run VLE comparisons, ablations, and interpretability analyses when their optional external assets are available.
7. Validate distributed models with `python scripts/validate_checkpoint_catalog.py --load`.

The complete file inventory is `docs/reproducibility/repository_files.txt`; the navigational tree is `docs/reproducibility/directory_tree.txt`; principal path mappings are in `docs/reproducibility/path_mapping.csv`.
