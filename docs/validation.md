# Validation

Validation checks positive temperature/pressure, bounded normalized compositions, material balance,
equilibrium residuals, solver convergence, parameter range, and stability information. Results are
`passed`, `warning`, or `failed`. Non-convergence and physical failures cannot be overridden by an
LLM. Azeotropes require an internal composition with `x approximately y`, not a visual crossing.
For TP Flash, phase outputs and feed closure are mandatory. The current K-value phase classification
is not a tangent-plane-distance stability proof, so the stability check remains an explicit warning
until a validated TPD implementation is available.
