# Legacy code inventory

This directory contains preserved programs that are not part of the supported
ThermoFormer workflow.

| Directory | Contents | Active replacement |
|---|---|---|
| `data_preparation/` | One-time acquisition, workbook formatting, consistency checking, validation, and repair programs used before the English release dataset was frozen | `datasets/vle_reference/*.xlsx`, `scripts/data/prepare_dataset.py`, and `src.thermoformer.data` |
| `reporting/` | Report generator for an earlier model and its command-line entry point | `scripts/build_c1_generalization_report.py` and `scripts/build_c1_ablation_report.py` |
| `diagnostics/` | One-off solver-iteration study | Formal evaluation through `src.thermoformer.evaluation` |
| `tests/` | Tests coupled to archived report-generation code | Active tests under `tests/` |
| `experiment_records/` | Superseded configurations, commands, and result records | Manuscript-aligned leaves under `experiments/` |
| `interpretability_earlier/` | Earlier four-table interpretability pipeline and compatibility adapters | Frozen manuscript Figure 2 under `analysis/` |
| `interpretability_c1_four_panel/` | Later four-panel C1 grouped-Shapley summary that does not match manuscript Figure 2 | Frozen manuscript Figure 2 under `analysis/` |

Historical data-preparation programs preserve original workbook field names and
literal paths because translating those values would alter their behavior. The
archive directory names and inventory are English, and none of these programs is
imported by active code.
