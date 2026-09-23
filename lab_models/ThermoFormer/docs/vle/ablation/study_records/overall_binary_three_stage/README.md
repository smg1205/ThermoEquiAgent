# Binary-only three-stage ablation campaign

This directory defines the binary train -> binary test ablation campaign.
All variants use the registered `overall_binary` split and the same three-stage
training schedule: direct GE/RT and log-gamma supervision (20 epochs), joint
VLE supervision (up to 80 epochs), and fugacity-constrained fine-tuning (10
epochs with a two-epoch warmup).  `stage0/` contains the supervised reference
configurations used to supply the Stage 0 checkpoint for the corresponding
three-stage configuration.

The seven unique structural variants are listed in the runner and reuse C1
between the representation and interaction sections of Table 2. Formal outputs
are intentionally kept separate from legacy `overall_binary_ternary` ablation
results. See [run.md](run.md) for reproducible smoke, formal, and reporting
commands. `results.md` is generated only after all five formal seeds are
available and validated.
