# ThermoFormer

ThermoFormer is a thermodynamically structured neural model for low-pressure
binary and ternary vapor--liquid equilibrium (VLE). The reported model combines
RDKit descriptors, Uni-Mol v2 embeddings, and SMARTS functional-group features
with a permutation-equivariant vanilla Transformer. Activity coefficients are
obtained by differentiating a learned excess Gibbs energy, pure-component vapor
pressures are modeled separately, and differentiable isothermal and isobaric
solvers reconstruct equilibrium states.

## Scientific scope

- Binary and ternary mixtures at pressures up to 500 kPa.
- Isothermal P--x--y inference: molecules, temperature, and liquid composition
  are provided; bubble pressure and vapor composition are predicted.
- Isobaric T--x--y inference: molecules, pressure, and liquid composition are
  provided; bubble temperature and vapor composition are predicted.
- Supervised training followed by optional validation-gated, fugacity-equilibrium
  fine-tuning.
- Registered evaluation protocols for within-system state generalization,
  system-disjoint prediction, unseen components, and binary-to-ternary transfer.

High-pressure vapor-phase equations of state, mixtures with more than three
components, and autonomous separation design are outside the validated scope.

## Final model

The manuscript model is the C1 three-view vanilla Transformer:

1. 24 standardized RDKit descriptors, a 768-dimensional Uni-Mol v2 embedding,
   and 28 SMARTS functional-group counts are projected independently.
2. The projected views are fused into one molecular token per component.
3. A vanilla Transformer exchanges information among component tokens.
4. A symmetric pair potential produces the interaction terms in
   `gE/RT = sum_(i<j) x_i x_j I_ij`.
5. Automatic differentiation of `gE/RT` yields `ln(gamma_i)`.
6. A separate temperature-dependent branch predicts pure-component vapor
   pressure when a valid Antoine or DIPPR 101 correlation is unavailable.
7. Differentiable VLE solvers reconstruct pressure or temperature together with
   vapor composition.

The final model sets `chemical_attention_bias=false` and
`context_pair_interaction=false`. These modules are retained only for the
interaction-architecture ablation.

## Repository structure

```text
src/thermoformer/       Active implementation organized by manuscript method
scripts/                Train, evaluate, reproduce, and report entry points
experiments/            Experiments aligned to the manuscript Results section
configs/                Shared configuration and frozen reference declarations
dataset/                Two English VLE workbooks used by the model
assets/                 Frozen RDKit descriptor and SMARTS definitions
datasets/splits/vle/                 Registered protocol assignments
results/                Machine-readable predictions, metrics, and manifests
checkpoints/            Git-LFS model checkpoints
reports/                Validated aggregate reports
analysis/               Dataset and interpretability analyses
docs/reference_records/    Preserved code excluded from active scientific workflows
tests/                  Unit, integration, provenance, and scientific-invariant tests
docs/                   Model, training, reproduction, and paper-to-code documentation
```

The active implementation and public research interface are documented in
[`docs/model_architecture.md`](docs/model_architecture.md).

## Environment

All reported experiments use the `ggnn39` Conda environment:

```powershell
conda env update -n ggnn39 -f environment.yml
conda run -n ggnn39 python -m unittest discover -s tests
```

The validated environment includes Python 3.9, PyTorch 2.6 with CUDA 12.6,
NumPy 1.26, pandas 1.5, RDKit, OpenPyXL, and `unimol-tools` with the Uni-Mol v2
84M encoder.

## Data

`dataset/` contains only:

- `binary_vle_english.xlsx`
- `ternary_vle_english.xlsx`

The loader converts degrees Celsius to kelvin and mmHg to kPa, audits quality
flags, excludes failed records, downweights indeterminate records, removes
duplicates, infers experiment direction when required, and records every
filtering decision. Pure-component endpoint systems are protected on the
training side to separate activity-coefficient learning from vapor-pressure
learning. See [`docs/dataset.md`](docs/dataset.md).

## Training

Stage 1 minimizes supervised pressure, temperature, vapor composition, and
pure-component endpoint losses. Stage 2 reloads the best Stage-1 validation
checkpoint, retains the supervised objective, and adds only the teacher-forced
fugacity-equilibrium loss

```text
x_i gamma_i P_i^sat(T) = y_i P.
```

Stage 2 runs for 10 epochs with a two-epoch warmup. Only `pair_potential`,
`vapor_pressure`, `film`, and `mixture_token` are optimized. The Stage-1
checkpoint remains epoch zero of model selection; test data are evaluated only
after validation selects Stage 1 or Stage 2. See [`docs/training.md`](docs/training.md).

## Reproducing experiments

The manuscript experiment registry is [`docs/experiment_overview.md`](docs/experiment_overview.md),
and the exact figure/table mapping is [`docs/paper_code_map.md`](docs/paper_code_map.md).
Each runnable experiment leaf contains `config.json` or `config.yaml`, together
with `run.md` and `results.md`.

Generate the final five-seed generalization report:

```powershell
conda run -n ggnn39 python scripts\build_c1_generalization_report.py
```

Generate the retained C1 ablation report:

```powershell
conda run -n ggnn39 python scripts\build_c1_ablation_report.py
```

The principal reports are:

- [VLE dataset overview](../../experiments/data_quality/construction/dataset_distribution/figures/vle_dataset_overview.png)
- [LLE dataset overview](../../experiments/data_quality/construction/dataset_distribution/figures/lle_dataset_overview.png)
- [`experiments/data_quality/reports/c1_fugacity_generalization_report.md`](experiments/data_quality/reports/c1_fugacity_generalization_report.md)
- [`experiments/data_quality/reports/c1_ablation_overall_binary_ternary.md`](experiments/data_quality/reports/c1_ablation_overall_binary_ternary.md)
- [`experiments/vle/interpretability/molecular_interactions/figures/vle_interpretability_complete.png`](experiments/vle/interpretability/molecular_interactions/figures/vle_interpretability_complete.png)

Detailed commands, artifact conventions, and Git-LFS requirements are described
in [`docs/reproduction.md`](docs/reproduction.md).

The principal public commands are:

```powershell
conda run -n ggnn39 python scripts\train.py --config configs\training\supervised.yaml
conda run -n ggnn39 python scripts\run_experiment.py --config configs\protocols\overall_binary_ternary.yaml --split datasets\splits\vle\overall_binary_ternary\seed_0.json --seed 0
conda run -n ggnn39 python scripts\build_reports.py all
```

## Scientific provenance

Reported values are linked to fixed code, configuration, data, feature, split,
prediction, and checkpoint identities. Full verification details are provided in
[`docs/reproduction.md`](docs/reproduction.md).

Historical source-processing programs and early report generators are preserved
under [`docs/reference_records/`](docs/reference_records/) and are not imported by the
active training or evaluation paths.

