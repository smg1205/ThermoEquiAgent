# Datasets

`registry.json` records the benchmark workbook paths, sources and SHA-256 digests.

- `vle/`: binary and ternary NIST ThermoML VLE workbooks.
- `lle/`: binary and ternary NIST ThermoML LLE workbooks.
- `vle_reference/`: the distinct workbook version used by the registered reference protocols.
- `splits/vle/`: frozen VLE sample assignments and protocol indexes.
- `molecular_features/`: molecular descriptor and functional-group definitions.
- `derived/thermodynamic_labels/`: thermodynamic supervision resources and their provenance.

VLE uses the registered quality filters, 500 kPa pressure limit and pure-component reference rules. LLE retains complete mixture--temperature--pressure groups. LLE condition partitions are included in their corresponding experiment execution records.

Raw rows, filtered observations, distinct conditions and distinct systems are different counts. Compare results only when data identity, split, target and evaluation protocol agree.
