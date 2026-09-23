# Model routing

Routing applies hard exclusions, then an explainable score composed of phase support, system fit,
conditions, parameter availability, evidence quality, extrapolation penalty, and numerical risk.
Wilson is excluded from LLE. Electrolytes are rejected. Ideal/Wilson are favored for near-ideal
low-pressure VLE, NRTL/UNIQUAC for polar non-ideal low-pressure VLE, and Peng-Robinson for high-
pressure hydrocarbons. A required but missing parameter set makes a candidate non-executable.

Execution uses a separate conservative backend gate. An explicit user model is preserved but must
still pass the backend applicability check. With no model, the gate considers pressure, availability
of reviewed local Ideal/Raoult pure properties, and whether the system is a hydrocarbon/allowlisted
light-gas set supported by the current Peng-Robinson adapter. Unsupported associating systems fail
instead of being sent to PR. LLE is never guessed: it returns `missing_parameters` until a reviewed
NRTL or UNIQUAC parameter set and production backend are available. Peng-Robinson also requires a
ChemSep PR interaction entry for every binary pair; the adapter does not silently replace
missing values with zero. The selected model and rule reason are persisted in the input snapshot.
