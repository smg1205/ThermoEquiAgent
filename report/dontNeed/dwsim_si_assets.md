# DWSIM SI Assets

This file lists the DWSIM project files that support the manuscript statement that ThermoAgent can generate and validate DWSIM-compatible simulations.

## Recommended Screenshots

| Asset | System | File | Recommended view |
|---|---|---|---|
| Fig. S-DWSIM-1 | 2-propanol / water | `report/dwsim/ipa_x0p3_2comp_near_vapor_83p6811C_v1p0e-04.dwxmz` | Near-bubble TP-flash flowsheet with NRTL property package and a small nonzero vapor outlet (`vapor_mol_s = 1.0035e-4`) |
| Fig. S-DWSIM-2 | Ethyl acetate / n-propyl acetate + DMSO | `lunwen/dwsim_demonstration/dmso_etac0p059_3comp_bubble_113.1C.dwxmz` | Ternary TP-flash flowsheet, compound list, and vapor composition result |
| Fig. S-DWSIM-3 | Ethanol / ethyl acetate / water | `lunwen/dwsim_demonstration/ethanol_eac_water_lle_extractor_nrtl.dwxmz` | Native Liquid-Liquid Extractor, feed/solvent streams, raffinate/extract streams |

## Recommended Operation Video

Record the ethyl acetate / n-propyl acetate + DMSO example:

```text
lunwen/dwsim_demonstration/dmso_etac0p059_3comp_bubble_113.1C.dwxmz
```

Suggested sequence:

1. Open the `.dwxmz` file in DWSIM.
2. Show the component list: ethyl acetate, n-propyl acetate, and dimethyl sulfoxide.
3. Show the property package panel.
4. Run or refresh the calculation.
5. Open the vapor stream result table.
6. Highlight the bubble-point temperature and vapor composition used in the three-source validation table.

## Supplementary LLE Example

The water / MIBK LLE files are retained as an additional example and stress test, not as the third main validation system:

```text
lunwen/dwsim_demonstration/water_mibk*.dwxmz
```

The main third validation system remains ethanol / ethyl acetate / water.

## Near-Bubble 2-Propanol / Water Files

The strict bubble-point files can reload with zero vapor flow, which is visually ambiguous in DWSIM. For SI evidence, use the near-bubble files in `report/dwsim/` instead. They were generated slightly above the DWSIM bubble-point temperature and preserve a nonzero vapor stream.

| x_IPA | Bubble T (deg C) | Near-vapor T (deg C) | Vapor mol/s | File |
|---:|---:|---:|---:|---|
| 0.1 | 89.50 | 89.50545 | 1.0022e-4 | `ipa_x0p1_2comp_near_vapor_89p5054C_v1p0e-04.dwxmz` |
| 0.3 | 83.68 | 83.68111 | 1.0035e-4 | `ipa_x0p3_2comp_near_vapor_83p6811C_v1p0e-04.dwxmz` |
| 0.5 | 81.92 | 81.92001 | 7.6448e-3 | `ipa_x0p5_2comp_near_vapor_81p9200C_v7p6e-03.dwxmz` |
| 0.7 | 81.12 | 81.12585 | 3.3496e-2 | `ipa_x0p7_2comp_near_vapor_81p1259C_v3p3e-02.dwxmz` |
| 0.9 | 81.42 | 81.42001 | 1.4178e-2 | `ipa_x0p9_2comp_near_vapor_81p4200C_v1p4e-02.dwxmz` |

The index file is `report/dwsim/ipa_water_near_vapor_files.csv`.
