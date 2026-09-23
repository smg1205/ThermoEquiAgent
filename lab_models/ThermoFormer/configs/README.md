# Configuration organization

- benchmarks/: common experiment catalog and task-specific orchestration JSON.
- model/: scientific model/representation definitions.
- training/: existing optimization configurations.
- protocols/: registered split and evaluation definitions.
- ablation/: retained reference configurations.
- experiments/: existing configuration inheritance notes.

Benchmark JSON references the original YAML/JSON declarations, including those
retained under experiments/. No model parameter, loss, split, or metric is
overridden by the repository reorganization.
Use python scripts/run_experiments.py show EXPERIMENT_ID to inspect a workflow.
