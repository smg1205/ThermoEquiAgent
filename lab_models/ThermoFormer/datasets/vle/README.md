# NIST ThermoML VLE dataset

This directory contains binary and ternary vapor--liquid equilibrium workbooks. Each retained record includes a DOI, NIST URL, source-record indices, and molecular identity fields for traceability to its ThermoML record.

| Item | Binary | Ternary |
|---|---:|---:|
| Source records | 23,061 | 5,229 |
| Records included | 22,760 | 4,544 |
| Records requiring further source resolution | 301 | 685 |
| Records with resolved SMILES | 679 | 2,935 |
| Source-verified composition resolutions | 0 | 68 |

`binary_vle_english.xlsx` and `ternary_vle_english.xlsx` are the model inputs. The matching CSV files support independent inspection, while `*_quarantine.csv` contains records that could not be linked uniquely to a source record and are excluded from training.

Temperature is stored in degrees Celsius as `temperature_c`, and pressure is stored in mmHg as `pressure_mmhg`; the loader converts them to K and kPa. `x` denotes liquid composition and `y` denotes vapor composition in the order defined by `smiles1`, `smiles2`, and `smiles3`. The third mole fraction in a ternary mixture follows from composition closure.

`source_original_excel_row`, `source_indices`, `source_url`, `source_json_sha256`, and `source_component_InChIKeys` provide row-level provenance. `manifest.json` records workbook hashes, `validation.json` records data and loader checks, and `curation.json` records source-resolution decisions.

The registered VLE evaluation uses `failed_weight=0`, a 500 kPa pressure ceiling, and the registered pure-component reference rules. Source traceability and thermodynamic consistency are evaluated separately.
