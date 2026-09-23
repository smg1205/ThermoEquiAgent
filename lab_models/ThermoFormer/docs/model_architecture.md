# Model architecture

## Inputs and outputs

For each mixture, the model receives up to three molecular feature vectors, a
component mask, liquid composition `x`, and one specified thermodynamic state.

| Inference direction | Known inputs | Joint outputs |
|---|---|---|
| Isothermal P--x--y | molecules, `T`, `x` | bubble pressure `P`, vapor composition `y` |
| Isobaric T--x--y | molecules, `P`, `x` | bubble temperature `T`, vapor composition `y` |

Padding is masked, and component permutations produce the corresponding output
permutation.

## Molecular views

- RDKit: 24 split-train standardized physical and topological descriptors.
- Uni-Mol v2: frozen 768-dimensional molecular embedding from the 84M model.
- Functional groups: 28 deterministic SMARTS counts.

Each view has an independent projection. The projected views are concatenated
and projected to the Transformer hidden dimension.

## Thermodynamic decoder

The final C1 model uses a vanilla Transformer, a mixture token, state FiLM, and
a symmetric pair potential. It does not use chemical-attention bias or the
context-conditioned pair-interaction variant.

The excess Gibbs energy is

```text
gE/RT = sum_(i<j) x_i x_j I_ij.
```

Activity coefficients are computed from the complete composition derivative of
this scalar. Pure-component vapor pressure depends only on the molecular
representation and temperature, unless a valid Antoine or DIPPR 101 correlation
is supplied.

The active implementation is organized as follows:

| Scientific responsibility | Module |
|---|---|
| Dataset schema, loading, and split logic | `src.thermoformer.data` |
| RDKit, Uni-Mol v2, functional groups, and fusion | `src.thermoformer.features` |
| Molecular interaction and vapor-pressure heads | `src.thermoformer.models` |
| Activity coefficients and differentiable VLE solvers | `src.thermoformer.thermodynamics` |
| Supervised and fugacity-constrained optimization | `src.thermoformer.training` |
| Prediction metrics and consistency analysis | `src.thermoformer.evaluation` |
