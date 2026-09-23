# Stable Python interfaces

New research code should import maintained functionality from `src.thermoformer`:

| Scientific area | Interface |
|---|---|
| Models | `src.thermoformer.models` |
| Molecular features | `src.thermoformer.features` |
| Data loading | `src.thermoformer.data` |
| Thermodynamics and pure properties | `src.thermoformer.thermodynamics` |
| Training and losses | `src.thermoformer.training` |
| Evaluation and metrics | `src.thermoformer.evaluation` |
| Experiment configuration | `src.thermoformer.configuration` |
| Registered protocols | `src.thermoformer.protocols` |
| Reporting and artifact identity | `src.thermoformer.reporting` |

Top-level `src.*` compatibility modules preserve established imports used by registered experiments. New modules should use the maintained interfaces above. Data preparation and reference analyses under `archive/` are provenance resources and are not public training entry points.
