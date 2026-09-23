# Thermodynamic-model comparisons

NRTL, Wilson, and UNIQUAC share one controlled low-pressure VLE workflow:

- pure-component vapor pressures use training-only endpoint fits of
  `ln(Psat/kPa) = a + b/T`;
- ordered interaction energies use one `B_ij/T` term per molecular pair;
- NRTL uses a fixed symmetric non-randomness parameter of 0.3;
- Wilson uses `ln(Lambda_ij) = B_ij/T`;
- UNIQUAC uses `ln(tau_ij) = B_ij/T` and standard UNIFAC-derived `r` and `q`;
- isothermal and isobaric predictions use the same ideal-vapor bubble-point
  equations and label-independent temperature bracket.

The first campaign evaluates the six registered within-system composition,
temperature, and pressure generalization protocols. Parameters are fitted on
the training partition only. Test rows are used once for evaluation.

See `run.md` for the canonical command and `results.md` for generated results.

The test-exposed low- and high-error phase-diagram diagnostic is documented in
`phase_diagrams/`. It is exploratory because held-out errors determine the
displayed cases.
