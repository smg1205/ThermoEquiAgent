# Ternary LLE Training and Testing

This configuration evaluates the shared ThermoFormer thermodynamic backbone on system-disjoint ternary liquid-liquid equilibrium tie-lines. It loads a validated VLE Stage 1 checkpoint, performs LLE endpoint supervision, and then applies observed-endpoint fugacity refinement.

The registered split is generated before bidirectional alpha-to-beta and beta-to-alpha expansion. Model selection is validation-only; the ternary test partition is evaluated once after checkpoint selection.
