# TeNNet-SAC and SPT-NRTL source audit

Audit date: 2026-08-27

Scope: primary-source review for integration into the ThermoFormer machine-learning baseline framework. This note fixes the upstream artifacts, records the exact public APIs and equations, and identifies provenance and fairness constraints. It does not report ThermoFormer benchmark results.

## 1. TeNNet-SAC

### Fixed upstream artifacts

- Official repository: [yueyue2299/TeNNet-SAC](https://github.com/yueyue2299/TeNNet-SAC).
- Repository revision inspected: [`2367e89c87c335c4fd27ff7fa22c87f660f459ba`](https://github.com/yueyue2299/TeNNet-SAC/tree/2367e89c87c335c4fd27ff7fa22c87f660f459ba), committed 2026-06-09.
- Official distribution inspected: [`tennetsac==0.1.10`](https://pypi.org/project/tennetsac/0.1.10/), released 2026-01-21.
- Wheel: `tennetsac-0.1.10-py3-none-any.whl`, size 43,459,447 bytes, SHA256 `0ec0e2724273730e4fd8dd70a2fc1c170c092f098462b1065eb20cf5110a05e5` (the same digest is published by PyPI).
- The repository has no release tags, so the repository commit cannot be cryptographically equated with PyPI 0.1.10. The benchmark must bind the wheel name and SHA256 as the executable artifact; the repository commit is the independently fixed source/documentation reference.
- License: MIT, copyright 2025 Yue Yang, from the [fixed-commit LICENSE](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/LICENSE). The wheel metadata also declares MIT, although the wheel itself does not contain a separate `LICENSE` file.

### Installation and runtime assets

The authors recommend `pip install tennetsac` in the [fixed-commit README](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/README.md). PyPI 0.1.10 declares Python `>=3.9,<3.13` and, among others, `numpy>=1.25,<2`, `pandas<2.2`, `pyarrow<15`, `transformers==4.36.2`, `torch>=2.1.2`, RDKit, SciPy, tokenizers and safetensors. Because these pins may conflict with the main ThermoFormer environment, the exact resolved environment must be recorded in the run manifest.

The wheel includes the following TeNNet-SAC checkpoints:

- `base.ckpt`, `prf.ckpt`, and `geo.ckpt`;
- ten experimental fine-tuned checkpoints, `fine-tuned/1.ckpt` through `fine-tuned/10.ckpt`.

The default `version="tuned"` path uses the mean of this ten-model ensemble. No TeNNet-SAC parameters should be optimized during the ThermoFormer comparison.

Two encoder assets are not self-contained in the wheel:

- ChemBERTa is loaded by name from `DeepChem/ChemBERTa-77M-MLM`;
- SMI-TED Light is downloaded through `hf_hub_download(repo_id="ibm/materials.smi-ted", filename="smi-ted-Light_40.pt")`.

Consequently, a reproducible run must cache and hash both downloaded encoder assets, record the Hugging Face revisions, and fail explicitly when they are unavailable. Silent fallback to randomly initialized or substitute encoders is invalid.

### Public inference API

PyPI 0.1.10 exports `profile`, `binary_lng`, `multi_lng`, `fit_nrtl`, and `plot_nrtl_fitting`. For this baseline only the direct activity-coefficient predictors are relevant:

```python
from tennetsac import binary_lng, multi_lng

ln_gamma_1, ln_gamma_2 = binary_lng(
    [smiles_1, smiles_2],
    temperature_kelvin,
    x1_values,
    version="tuned",
)

ln_gamma = multi_lng(
    [smiles_1, smiles_2, smiles_3],
    temperature_kelvin,
    [x1, x2, x3],  # [x1, x2] is also accepted; x3 is completed internally
    version="tuned",
)
```

`binary_lng` requires exactly two SMILES and treats its third argument as a list of first-component mole fractions, returning two equally sized lists. `multi_lng` accepts two or more components, validates a full composition to sum to one within `1e-6`, or completes the final fraction if `n-1` fractions are supplied. Both return natural logarithms `ln(gamma_i)`, not `gamma_i`.

The package canonicalizes SMILES internally with RDKit before embedding: isomeric canonical SMILES for ChemBERTa and non-isomeric canonical SMILES for SMI-TED. The integration should still validate each input with the project-fixed RDKit version and record failures rather than substituting a different molecule.

The authors' README confirms that the base model is trained on synthetic COSMO-SAC data and the fine-tuned model is further optimized on experimental data. The repository contains [`TrainingSystemsList.csv`](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/TrainingSystemsList.csv), but its completeness and precise relation to all base/fine-tuned checkpoints are not documented in the README. Until overlap is audited against this file and clarified by the authors, formal results must use the provenance label `external_pretrained_overlap_unknown` rather than claim an unseen-system comparison.

### Required ThermoFormer adapter behavior

1. Use `binary_lng(..., version="tuned")` for binary rows and `multi_lng(..., version="tuned")` for ternary rows.
2. Convert `ln(gamma)` to activity coefficients only where the shared VLE backend requires it; do not fit an intermediate NRTL model with `fit_nrtl`.
3. Use the same train-only pure-component vapor-pressure source and the same isothermal/isobaric VLE solver as the other activity-coefficient baselines.
4. Treat the model as fixed and deterministic. Seeds 0--4 change the registered split being evaluated, not the TeNNet-SAC weights.
5. Separate failure reasons at least into invalid SMILES, missing external encoder asset, model inference failure, missing vapor pressure, solver failure, and nonphysical prediction.
6. Record model, package, encoder and checkpoint hashes in each manifest. Do not use test labels for tuning or calibration.

## 2. SPT-NRTL parameter database

### Fixed upstream artifacts and licensing

- Paper: Winter et al., *SPT-NRTL: A physics-guided machine learning model to predict thermodynamically consistent activity coefficients*, Fluid Phase Equilibria 568 (2023) 113731, [DOI 10.1016/j.fluid.2023.113731](https://doi.org/10.1016/j.fluid.2023.113731); author manuscript and equations: [arXiv:2209.04135](https://arxiv.org/abs/2209.04135).
- Official Julia wrapper: [ClapeyronThermo/SPTNRTL.jl](https://github.com/ClapeyronThermo/SPTNRTL.jl), fixed at [`691d55ce07a860b9e700877c64b6342a6dc385e1`](https://github.com/ClapeyronThermo/SPTNRTL.jl/tree/691d55ce07a860b9e700877c64b6342a6dc385e1) (v1.0.0, 2025-07-12). The wrapper is MIT licensed.
- Official database: [ClapeyronThermo/spt-nrtl-db](https://github.com/ClapeyronThermo/spt-nrtl-db), fixed at [`f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e`](https://github.com/ClapeyronThermo/spt-nrtl-db/tree/f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e) (2023-05-19).
- The database repository has no `LICENSE` and the GitHub API reports `license=null`. Public accessibility is not a redistribution grant. Do not vendor the full database, copied CSV files, or extracted parameter tables into the public ThermoFormer repository without license clarification. Prefer runtime retrieval from fixed-commit raw URLs, an ignored local cache, and manifest hashes. Publication/reuse permission should be confirmed manually before release.

### Database key and CSV schema

The [fixed database README](https://github.com/ClapeyronThermo/spt-nrtl-db/blob/f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e/README.md) specifies a filesystem key based on canonical SMILES:

```text
v3/<first character of SMILES>/<SMILES string length>/<percent-encoded SMILES>.csv
```

Each CSV contains rows for partner molecules, with the exact columns:

```text
SMILES0,SMILES1,a_1,a_2,
t_12_1,t_12_2,t_12_3,t_12_4,
t_21_1,t_21_2,t_21_3,t_21_4
```

The official wrapper performs exact string lookup and does not canonicalize input itself; see [`src/SPTNRTL.jl`](https://github.com/ClapeyronThermo/SPTNRTL.jl/blob/691d55ce07a860b9e700877c64b6342a6dc385e1/src/SPTNRTL.jl). ThermoFormer must canonicalize using its fixed RDKit version and record that version. The database does not document which canonicalizer/version produced its keys, so an exact-string miss must be `unavailable`; fuzzy matching, stereochemistry removal, synonym matching, or test-data fitting is not permitted.

The lookup file is selected from the first SMILES and `SMILES1` is matched exactly. If the database pair is found only in reverse order, `a` is unchanged while the directional `t_12_*` and `t_21_*` blocks are swapped. Both directions may be queried, but only exact matches are acceptable.

Reference parity fixture: the fixed database file [`v3/C/3/CCO.csv`](https://raw.githubusercontent.com/ClapeyronThermo/spt-nrtl-db/f8e903ad96ac1a6d9c3dbaaa53c1752e8106286e/v3/C/3/CCO.csv) contains the ethanol/water row. The audited file has 9,969 data rows and SHA256 `903e2fcbd174ec2dac61fbf56c093beb029219b7622f2f5d446b014695723981`.

### Parameter mapping and temperature-dependent NRTL equations

Temperature is in kelvin and logarithms are natural logarithms. The CSV mapping used by the official Clapeyron extension is:

```text
a_1, a_2       -> a0, a1
t_*_1..t_*_4   -> t0, t1, t2, t3
```

For each pair:

\[
\alpha_{ij}(T)=a_{0,ij}+a_{1,ij}T,
\]

\[
\tau_{ij}(T)=t_{0,ij}+\frac{t_{1,ij}}{T}+t_{2,ij}\ln T+t_{3,ij}T,
\]

\[
G_{ij}(T)=\exp[-\alpha_{ij}(T)\tau_{ij}(T)].
\]

`alpha` is symmetric for a constituent binary pair; `tau` is directional. Set `tau_ii=0` and `G_ii=1`. These definitions follow the paper's Eqs. 5--8 and the official [`SPTNRTLClapeyronExt.jl`](https://github.com/ClapeyronThermo/SPTNRTL.jl/blob/691d55ce07a860b9e700877c64b6342a6dc385e1/ext/SPTNRTLClapeyronExt.jl), which maps the database into Clapeyron's temperature-dependent `aspenNRTL` implementation.

For a multicomponent mixture, use the standard NRTL expression implied by

\[
\frac{g^E}{RT}=\sum_i x_i
\frac{\sum_j x_j\tau_{ji}G_{ji}}
     {\sum_j x_jG_{ji}}.
\]

An explicit activity-coefficient form is

\[
\ln\gamma_i=
\sum_j \frac{x_j\tau_{ji}G_{ji}}{\sum_k x_kG_{ki}}
+\sum_j\frac{x_jG_{ij}}{\sum_kx_kG_{kj}}
\left(
\tau_{ij}-
\frac{\sum_mx_m\tau_{mj}G_{mj}}{\sum_kx_kG_{kj}}
\right).
\]

At two components this must reproduce the binary equations in the SPT-NRTL paper. For a ternary system all three constituent binary parameter sets, `(1,2)`, `(1,3)`, and `(2,3)`, are required. If any pair is absent, the ternary row/system is unavailable; missing interactions must not be set to zero.

### Scientific interpretation and fairness constraints

- The paper's Transformer natively predicts parameters for binary mixtures. A ternary calculation assembled from three database binary pairs and the standard multicomponent NRTL equation is therefore labeled `SPT-NRTL database + multicomponent NRTL composition`, not native ternary SPT-NRTL.
- The database rows contain no `Tmin`, `Tmax`, uncertainty, or per-pair validity metadata. The equations can be evaluated during an isothermal or isobaric solve, but the scientific temperature applicability of an individual pair cannot be certified. Record `temperature_validity_metadata=unavailable`.
- The paper states that experimental training data are confidential. Dataset overlap with ThermoFormer cannot be fully audited, so this baseline is also `external_pretrained_overlap_unknown`.
- The database `CITATION.bib` points to the earlier limiting-activity SPT paper, DOI `10.1039/D2DD00058J`, rather than the 2023 SPT-NRTL paper. The baseline must cite DOI `10.1016/j.fluid.2023.113731` as its model source.
- Coverage must distinguish parameter coverage, vapor-pressure coverage, solver coverage and valid prediction coverage. A 404, missing row, malformed row, missing vapor pressure or solver failure is not a valid prediction.
- The official wrapper's default URL follows mutable `main`; ThermoFormer must instead use the fixed commit URL and bind each response SHA256 in the manifest.

### Required unavailable reasons and parity tests

Recommended unavailable reasons:

- `missing_smiles_file`
- `missing_pair_row`
- `http_error`
- `malformed_row`
- `missing_psat`
- `solver_failure`

Minimum parity tests:

1. The fixed ethanol/water fixture matches every CSV parameter column and its recorded SHA256.
2. Reversing a pair transposes the directional `tau` parameters and produces composition-permuted `ln(gamma)`.
3. The general multicomponent equation numerically matches the paper's binary equations at `n=2`.
4. The ternary result is equivariant to component permutation.
5. One missing constituent binary pair makes the complete ternary system unavailable.
6. `T>0`, pure-component endpoints remain finite, and the pure component's own `gamma` approaches one.
7. Every query records the fixed database commit, raw URL, response hash and exact lookup direction.

## 3. Integration decision

Both baselines can be connected to the existing ThermoFormer activity-coefficient-to-VLE path without changing registered splits, vapor-pressure data, thermodynamic equations, solver or metrics:

```text
SMILES, T, x
  -> TeNNet-SAC tuned ensemble or SPT-NRTL parameter lookup
  -> ln(gamma)
  -> shared Psat(T)
  -> shared isothermal/isobaric VLE solver
  -> P/T/y metrics and explicit coverage accounting
```

TeNNet-SAC is a fixed experimental fine-tuned neural ensemble. SPT-NRTL is a fixed public parameter-database lookup; its ternary use is an explicit multicomponent NRTL composition of binary predictions. Neither baseline may be retrained, calibrated, completed or selected using ThermoFormer test labels.

## 4. Same-data adapted SPT-NRTL comparison

The separately named `SPT-NRTL adapted (ThermoFormer-train)` experiment is not an
official-model reproduction. It preserves the paper's character-level ordered SMILES
pair, causal Transformer, pooled molecular-pair representation and ten-parameter
temperature-dependent NRTL decoder. It does not reuse the authors' confidential
COSMO pretraining corpus or weights.

For every registered split, ten-parameter targets are fitted only for binary systems
in the training partition. The parameter network is optimized only on these training
targets; fitted binary targets from the validation partition are used only for early
stopping and checkpoint selection. No NRTL target is fitted from test rows. Test
binary pairs are predicted from canonical SMILES, and ternary states use three such
predictions in the standard multicomponent NRTL equation. The resulting activity
coefficients enter the same train-derived pure-vapor-pressure correlations and VLE
solver used by the other activity-coefficient baselines.

This experiment answers a different question from fixed-database SPT-NRTL: it compares
an SPT-like structural parameter-prediction architecture with ThermoFormer under the
same split data. The external database result remains a frozen out-of-domain reference
and must not be described as a same-training-data comparison.
