# NIST ThermoML LLE datasets/vle_reference

This directory contains source-verified binary and ternary liquid--liquid equilibrium workbooks. It covers records that can be linked uniquely to a DOI and a ThermoML source record; it is not a complete representation of all LLE records in NIST ThermoML.

| Dataset | Source records | Records included | Records requiring further source resolution |
|---|---:|---:|---:|
| Binary | 6,150 | 5,983 | 167 |
| Ternary | 14,895 | 13,861 | 1,034 |

The model reads `binary_lle.xlsx` and `ternary_lle.xlsx`. The matching CSV files expose `x_alpha_i` and `x_beta_i` for every component to support independent inspection. Temperature is expressed in K and pressure in kPa.

For ternary data, `x1` and `x2` are the component-1 and component-2 mole fractions in liquid phase alpha; `y1` and `y2` are the corresponding mole fractions in liquid phase beta. The third mole fraction follows from composition closure. For binary data, `x1` and `y1` are the component-1 mole fractions in the two liquid phases. The two phases may be exchanged as complete vectors, but individual components may not be exchanged independently.

Source verification jointly resolves Property, Variable, and Constraint records and matches components by complete InChIKey. Records with missing pressure, ambiguous phase-variable correspondence, or unmatched components are excluded from the workbooks. Each row retains the DOI, NIST URL, and source-record indices; `manifest.json` records file hashes and datasets/vle_reference sizes.

The training loader groups observations by complete mixture, temperature, and pressure condition, then applies the registered quality, deduplication, and splitting protocols. Workbook row counts and final training-condition counts must be reported separately.
