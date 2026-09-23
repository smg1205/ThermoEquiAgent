# Research repository architecture

This organization draws on the separation of configs, datasets, src, scripts,
experiments, models, results, tests, and docs in
[PSMI](https://github.com/JinlinYY/PSMI/tree/main).
The scientific methods and experiment definitions remain those of ThermoFormer.

## Data flow

Dataset catalog → existing task loader and registered split → original model and
training backend → checkpoint and predictions → original metrics and diagnostics
→ experiment-specific result index.

## Configuration and entry points

configs/<task>/<category>/<experiment>.json declares the scientific
backend, existing configuration paths, data version, seed policy, and artifact
sources. These are orchestration declarations, not replacement model hyperparameters.
Shared scientific YAMLs remain in configs/model, configs/training, configs/protocols
and the existing experiments subdirectories.

scripts/run_experiments.py is the common discovery/run/index command.
scripts/benchmarks/registry.py implements its shared management behavior.
scripts/benchmarks/vle and scripts/benchmarks/lle provide descriptive entry names.
The existing registered backend CLI files are now run_registered_experiment.py
and run_registered_suite.py. Their internal scientific functions have not changed.

## Maintained implementation

src/thermoformer/data, features, models, training, protocols, evaluation,
thermodynamics, and reporting remain the maintained VLE implementation.
src/thermoformer/lle_tp retains the confirmed LLE implementation.
src/thermoformer/baselines and interpretability are VLE research components.
Existing src-level compatibility modules remain adapters; do not duplicate their
implementation when adding a new experiment.

## Storage and compatibility

datasets/registry.json discovers versioned workbooks; models/registry.json discovers
checkpoint locations. Original files remain in dataset*/, checkpoints/, and runs/
because existing provenance records reference these paths. The catalogs do not
imply that these physical files have been moved or copied.

New managed results use experiments/reference_results/benchmarks/<task>/<category>/<experiment>/executions/
<run-id>/<formal|split_checks>/. Some historical backends still use fixed output
locations; their native output_mode and source indexes make this explicit.

## Extension rule

To add a protocol already implemented in a scientific backend, register its
configuration and scope in configs and test the command mapping.
Changing a scientific definition requires a separately reviewed scientific change,
not an orchestration override. Unsupported LLE comparisons, ablations, and
interpretability studies are not introduced by this repository organization.
