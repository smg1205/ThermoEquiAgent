# ML/DL VLE baseline source audit

Audit date: 2026-08-26 (Asia/Shanghai)

Scope: source and reproducibility audit for the eight machine-learning/deep-learning VLE baselines requested for the ThermoFormer overall-predictive-performance benchmark. This document records what the cited papers and current official repositories actually implement. It does **not** report ThermoFormer benchmark results and does not authorize use of the test set for model selection.

## Reading rules

- “Native” means present in the cited paper or official code without changing the scientific prediction direction or architecture.
- “Adapted” means a new scientific capability is added, for example temperature conditioning to a fixed-temperature model. Such results must carry an `adapted` suffix.
- A shared Psat/VLE solver attached after a native activity-coefficient or excess-Gibbs-energy model is an evaluation adapter, not a change to the learned model. Its coverage must still be reported.
- Current official pretrained models (HANNA and TeNNet-SAC) are not same-split retraining baselines. They require explicit `official-pretrained` labels and overlap audits.
- Repository statements below refer to the immutable commits in the provenance table, not an unversioned default branch.

## Primary-source provenance snapshot

| Baseline | Primary paper | Official source inspected | Immutable revision | License visible at revision |
|---|---|---|---|---|
| Descriptor ANN | [Sun et al., *Chemical Engineering Science* 282 (2023) 119358](https://doi.org/10.1016/j.ces.2023.119358) | [`sungl123456/37lamdaA`](https://github.com/sungl123456/37lamdaA) | [`f103380`](https://github.com/sungl123456/37lamdaA/tree/f103380f3baa01d317ee3c904e46ff68b07bdece) | No LICENSE or source-header license found; reuse permission is therefore unresolved |
| SMILES-RNN | [Xue et al., *Chemical Engineering Science* 298 (2024) 120382](https://doi.org/10.1016/j.ces.2024.120382) | [`Xiyue17/SMILES-RNN`](https://github.com/Xiyue17/SMILES-RNN) | [`d41d533`](https://github.com/Xiyue17/SMILES-RNN/tree/d41d533586b43a8c6606f2e947a7e22398679c36) | No LICENSE or source-header license found; reuse permission is unresolved |
| UALF-GNN | [Sun et al., *AIChE Journal* 71 (2025) e18637](https://doi.org/10.1002/aic.18637) | No official public repository located or linked by the publisher article | N/A | Paper-based clean reimplementation; code license N/A |
| SolvGNN | [Qin et al., *Digital Discovery* 2 (2023) 138–151](https://doi.org/10.1039/D2DD00045H) | [`zavalab/ML`, branch `SolvGNN`](https://github.com/zavalab/ML/tree/SolvGNN) | [`66ec632`](https://github.com/zavalab/ML/tree/66ec632ca3499e379f29077c1fc5d195d9de12ca) | No repository-level LICENSE found on the branch; reuse permission is unresolved |
| GDI-GNN | [Rittig et al., *Digital Discovery* 2 (2023) 1752–1767](https://doi.org/10.1039/D3DD00103B) | [`avt-svt/public/GDI-NN`](https://git.rwth-aachen.de/avt-svt/public/GDI-NN) | [`6383142`](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/tree/6383142feb3b926fd279ae676a211fd8b3f1dac3) | [EPL-2.0](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/LICENSE) |
| GE-GNN | [Rittig & Mitsos, *Chemical Science* 15 (2024) 18504–18512](https://doi.org/10.1039/D4SC04554H) | Same official GDI-NN repository; it contains `gegnn_binary` | [`6383142`](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/tree/6383142feb3b926fd279ae676a211fd8b3f1dac3) | [EPL-2.0](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/LICENSE) |
| HANNA (current multicomponent) | [Hoffmann et al., *Nature Communications* 17 (2026) 3485](https://doi.org/10.1038/s41467-026-71430-y) | [`marco-hoffmann/HANNA`](https://github.com/marco-hoffmann/HANNA) | [`6fe873c`](https://github.com/marco-hoffmann/HANNA/tree/6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365) | [MIT](https://github.com/marco-hoffmann/HANNA/blob/6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365/LICENSE) |
| TeNNet-SAC | [Yang & Lin, *Journal of Chemical Information and Modeling* (2025)](https://doi.org/10.1021/acs.jcim.5c01804) | [`yueyue2299/TeNNet-SAC`](https://github.com/yueyue2299/TeNNet-SAC) | [`2367e89`](https://github.com/yueyue2299/TeNNet-SAC/tree/2367e89c87c335c4fd27ff7fa22c87f660f459ba) | [MIT](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/LICENSE) |

The GitHub default-branch HEAD of `zavalab/ML` is not the SolvGNN revision. The audited scientific branch is `SolvGNN` at `66ec632`; manifests must record both the branch and commit.

## Capability matrix

| Baseline | Learned output | Native components | Native state/task coverage | Explicit temperature conditioning? | Required ThermoFormer adapter |
|---|---|---:|---|---|---|
| Descriptor ANN | Direct scalar `T` or `y1`, with separate networks | Binary | Isobaric `T-x-y`; no native isothermal `P-x-y` | No: pressure is input and temperature is output | Feature reconstruction and split/training wrapper only |
| SMILES-RNN | Direct two-output state/composition predictions; target layout changes by experiment | Binary | Paper reports bubble/dew calculations under constant-P and constant-T settings | Direction-dependent: T is an input for P prediction and P is an input for T prediction | Faithful task-specific wrappers; never collapse to an unlabelled dual-task model |
| UALF-GNN | Mean and log variance for `T` and vapor composition | Binary | Isobaric `T-x-y` only, according to publisher abstract and specified architecture | No: P is the condition and T is output | Paper-based reimplementation; MC-dropout inference |
| SolvGNN | `ln(gamma_i)` | Binary and ternary, using distinct native models/checkpoints | Activity coefficients and `P-x-y` at 298 K | **No; native model is fixed at 298 K** | Shared Psat + isothermal VLE solver at 298 K only |
| GDI-GNN | Binary `ln(gamma_1), ln(gamma_2)` | Binary | Activity coefficients; paper demonstrates isothermal VLE at 298 K | **No; official input contains composition but no T** | Shared Psat + isothermal VLE solver at 298 K only |
| GE-GNN | Scalar `gE/RT`, differentiated to `ln(gamma_i)` | Binary | Activity coefficients; fixed-temperature source datasets/vle_reference/application | **No; official input contains composition but no T** | Shared Psat + isothermal VLE solver at 298 K only |
| HANNA current | `gE/RT`, autodiff `ln(gamma_i)` | Arbitrary N at inference | Temperature- and composition-dependent activity coefficients; phase equilibria require a solver | Yes | Shared Psat + isothermal/isobaric VLE solver |
| TeNNet-SAC | Molecular `ln(gamma_i)` via segment coefficients | Arbitrary N | Temperature- and composition-dependent activity coefficients; phase equilibria require a solver | Yes (`1/T` in segment network) | Shared Psat + isothermal/isobaric VLE solver |

`N/A` policy implied by this matrix:

- Descriptor ANN and UALF-GNN: ternary and isothermal P prediction are N/A.
- SMILES-RNN: ternary is N/A; each direct direction is reported separately.
- Native SolvGNN/GDI-GNN/GE-GNN: non-298-K cases and isobaric T prediction are N/A. Adding T to these networks creates a separately named `adapted` model.
- HANNA and TeNNet-SAC can supply binary and ternary gamma at variable T, subject to molecular-encoder and Psat/solver coverage.

## 1. Descriptor ANN (Sun et al., 2023)

### Verified native implementation

The publisher paper describes a data set of 210 binary mixtures and prediction of ambient-pressure `T-x-y` behavior using screened descriptors. The official repository identifies itself as an **alpha version**, provides the experimental spreadsheets, and trains separate ANN models for `T` and `Y` ([paper](https://doi.org/10.1016/j.ces.2023.119358); [README](https://github.com/sungl123456/37lamdaA/blob/f103380f3baa01d317ee3c904e46ff68b07bdece/README.md); [training script](https://github.com/sungl123456/37lamdaA/blob/f103380f3baa01d317ee3c904e46ff68b07bdece/cross_validation.py)).

The architecture is three ReLU hidden layers of width 64 and one scalar linear output ([`model.py`](https://github.com/sungl123456/37lamdaA/blob/f103380f3baa01d317ee3c904e46ff68b07bdece/model.py)). A critical dimensionality distinction must be preserved:

- `VLE_input.xlsx` contains 21 descriptor columns after excluding the spreadsheet index: three association flags and nine properties for each of two components.
- Pressure and liquid mole fraction `x1` are appended, so the executable source calls `NeuralNetwork(fnn=23)`.
- Therefore the faithful executable network is **23-64-64-64-1**, equivalently “21 descriptors + P + x -> 64-64-64 -> output.” Calling the complete input layer “21-wide” is inconsistent with the official source. Its parameter count is 9,921 per head and 19,842 for the separate T/y pair. A literal 21-64-64-64-1 variant would contain 9,793 parameters and must be labelled as an adapted specification, not official.

The 21 descriptor block includes association indicators, molecular weights, critical T/P, acentric factors, electronic-energy terms, normal boiling points and melting points. The repository does not provide a robust identity-to-property generation pipeline for new ThermoFormer molecules.

### Dependencies and assets

The README pins Python >=3.8, PyTorch 2.0.1, pandas 1.4.3, NumPy 1.24.3, scikit-learn 1.3.0, matplotlib 3.7.2, tensorboardX 2.6.2.2 and openpyxl 3.1.2. It provides data spreadsheets but no pretrained checkpoint and no explicit software license.

### Reproduction risks and required decisions

1. The official loader min-max normalizes the full input/output arrays before constructing train/validation/test subsets. Reusing that behavior would leak held-out statistics. The ThermoFormer wrapper must fit scalers on training data only and record this necessary protocol correction.
2. The original split is not the ThermoFormer split, is not seeded, and converts mixtures through a Python `set`; it must be replaced by committed system IDs without changing the model.
3. Several descriptors are not ordinary RDKit descriptors and may be unavailable for ThermoFormer compounds. Coverage, property source, missing-value policy and any reduced-descriptor variant must be explicit. Do not silently substitute the existing ThermoFormer RDKit vector.
4. The official source trains for 1000 epochs with SGD and does not checkpoint by validation optimum. Model selection must be validation-only in the unified runner.
5. No public license was found. Copying source verbatim into a distributable project needs permission review; a clean reimplementation from the paper/interface is safer but does not cure data licensing questions.

## 2. SMILES-RNN (Xue et al., 2024)

### Verified native implementation

The paper introduces a direct VLE model for 123 alcohol-containing binary systems. It uses molecular SMILES and thermodynamic properties and reports four experiment settings spanning constant-pressure, constant-temperature, bubble-point and dew-point calculations ([publisher paper](https://doi.org/10.1016/j.ces.2024.120382)). This is a direct predictor; it should not be converted into a gamma model.

The official code creates character-level SMILES encodings, concatenates the encodings for two compounds with per-component thermodynamic properties and the relevant state/composition condition, reshapes the entire feature vector as a sequence of length **one**, and applies `SimpleRNN(32, return_sequences=True) -> SimpleRNN(16, return_sequences=True) -> Dense(8, ReLU) -> Dense(2, linear)` ([`main.py`](https://github.com/Xiyue17/SMILES-RNN/blob/d41d533586b43a8c6606f2e947a7e22398679c36/main.py); [`smiles_onehot.py`](https://github.com/Xiyue17/SMILES-RNN/blob/d41d533586b43a8c6606f2e947a7e22398679c36/smiles_onehot.py)). Thus the checked-in RNN does not recurrently scan SMILES tokens; changing it to a token-by-token RNN is scientifically `adapted`, even if that design appears more natural. The source comment states that output targets change with the predicted direction. Consequently, task-specific direct heads/wrappers are faithful; a single newly designed four-output network would be adapted.

### Dependencies and assets

The README specifies Python 3.6, pandas 1.1.5, NumPy 1.19.5, TensorFlow 2.1.0, RDKit 2021.9.4, scikit-learn 0.24.2, Keras 2.3.1 and matplotlib 2.2.2 ([README](https://github.com/Xiyue17/SMILES-RNN/blob/d41d533586b43a8c6606f2e947a7e22398679c36/README.md)). No pretrained weights and no explicit license are present.

### Reproduction risks and required decisions

1. The checked-in `main.py` reads `data.txt`, but that file is absent. Only `compound.csv` is included. The official repository is therefore not end-to-end runnable on its publication data.
2. `smiles_onehot.py` serializes one-hot content into a tabular representation, while `main.py` treats those values as dummy-coded columns; the exact vocabulary, padding, maximum length and target-specific table construction must be reconstructed and frozen in a manifest.
3. At the audited revision, `main.py` constructs `X` with `comp1` and also sets one output column to `comp1`; this is direct target leakage in that checked-in example. The evaluated prediction directions therefore require explicit leakage-free task tables, whose adaptation must be documented rather than treating the snapshot as executable ground truth.
4. The code uses row-wise shuffled 10-fold `KFold` without a fixed random state, not system-disjoint ThermoFormer IDs, and subsequently retains only the final fold's model. Replace the split/training plumbing.
5. `keras==2.3.1` and `tensorflow==2.1.0` are an old and potentially incompatible stack on current Python/CUDA. A modern PyTorch/TensorFlow reimplementation should be numerically audited against a small deterministic fixture and marked `reimplemented-from-official`.
6. The public paper is alcohol-system-specific. Applying its architecture to the full ThermoFormer chemical domain is allowed as same-split retraining, but is a domain transfer and must be discussed as such.
7. Parameter count depends on the frozen character vocabulary and padded feature width, which the missing `data.txt` prevents deriving unambiguously. Report the instantiated count from the final manifest rather than guessing from the paper.

## 3. UALF-GNN (Sun et al., 2025)

### Verified paper boundary

The publisher abstract states that the model predicts **temperature and vapor-phase composition for binary mixtures** and combines a GNN with uncertainty-aware learning/inference ([publisher article](https://doi.org/10.1002/aic.18637)). No official code repository was found in the paper metadata or the authors' linked public materials during this audit.

The requested paper-based implementation specification is:

`RDKit graph -> Linear -> GraphConv -> ReLU -> GRU`, repeated for three graph-update blocks; concatenate initial and updated atom features; Set2Set pooling; concatenate the two molecule vectors with pressure `P` and liquid composition `x`; MLP emits a mean and log variance; train with heteroscedastic negative log likelihood and infer with MC dropout.

This specification is consistent with the paper's direct isobaric `P,x -> T,y` boundary. It is not evidence for native isothermal `T,x -> P,y`, and it provides no ternary construction.

### Reproduction risks and required decisions

1. This baseline is `reimplemented`, not `official`. Every paper-unspecified hyperparameter (atom feature list, hidden widths, GRU sharing, Set2Set iterations, dropout, MC sample count, variance clipping) must be frozen in config before test evaluation.
2. Select all such values using training/validation only. The test split may only be evaluated after the choice is fixed.
3. The uncertainty head predicts heteroscedastic aleatoric variance; MC dropout adds an epistemic approximation. Store both and their combination separately.
4. The two outputs have different physical units. Loss scaling/normalization must be trained from train-only statistics and recorded.
5. Parameter count is not uniquely recoverable from the publisher abstract; report the exact instantiated count after the configuration is frozen.

## 4. SolvGNN (Qin et al., 2023)

### Verified native implementation

SolvGNN is an activity-coefficient model, not a direct VLE state predictor. The paper and official README state that it combines atom-level graph convolution with molecule-level message passing through a molecular interaction graph, predicts composition-dependent activity coefficients for binary and ternary mixtures, and constructs `P-x-y` diagrams at **298 K** through thermodynamic calculations ([paper](https://doi.org/10.1039/D2DD00045H); [README](https://github.com/zavalab/ML/blob/66ec632ca3499e379f29077c1fc5d195d9de12ca/README.md)).

The official source has separate `solvgnn_binary` and `solvgnn_ternary` classes and separate training/checkpoint files. Each molecule is encoded with two shared `GraphConv` layers and mean node pooling. Molecule embeddings are concatenated with mole fractions; the molecular interaction network applies an edge-conditioned `NNConv` plus GRU and explicitly supplies intermolecular/intramolecular hydrogen-bond edge features. A shared MLP maps the updated molecular nodes to one logarithmic activity coefficient per component ([model source](https://github.com/zavalab/ML/blob/66ec632ca3499e379f29077c1fc5d195d9de12ca/solvgnn/model/model_GNN.py)).

The repository provides five binary and five ternary pretrained checkpoints, COSMO-RS-generated data tables, Antoine utilities, training scripts and notebooks. Those checkpoints were not trained on ThermoFormer splits, so the requested benchmark should retrain the native architecture on the committed ThermoFormer data if target gamma labels can be derived without test leakage.

### Dependencies, license and temperature boundary

The branch pins a legacy stack: Python 3.7, PyTorch 1.2.0, DGL CUDA 10.0 0.4.3, torchvision 0.4.0, RDKit 2019.03.2 and networkx 2.2 ([requirements](https://github.com/zavalab/ML/blob/66ec632ca3499e379f29077c1fc5d195d9de12ca/requirements.txt); [environment](https://github.com/zavalab/ML/blob/66ec632ca3499e379f29077c1fc5d195d9de12ca/environment.yml)). No explicit license was found on the audited branch.

Temperature is absent from the learned input. Native evaluation is therefore limited to 298 K. Simply calling the model at another temperature, or allowing an isobaric solver to vary T while holding gamma's learned representation fixed, is not a faithful native result. Adding T to node/global features is `SolvGNN-adapted-T`.

### Reproduction risks

1. Native binary and ternary models are separate; no weight sharing or automatic arbitrary-N extension should be invented.
2. The original training targets are COSMO-RS gamma values. ThermoFormer supplies VLE observations; converting them to gamma requires a documented low-pressure thermodynamic assumption, Psat and filtering. This changes target provenance and can reduce coverage.
3. H-bond indicators and atom feature construction should be ported from official utilities and regression-tested, not approximated by a generic GNN featurizer.
4. DGL/CUDA dependencies are obsolete. A current-DGL/PyG port is a compatibility reimplementation; verify layer semantics and parameter count against the official classes.
5. Repository licensing is unresolved.

## 5. GDI-GNN (Rittig et al., 2023)

### Verified native implementation

GDI-GNN predicts both binary log activity coefficients and adds the binary Gibbs-Duhem differential residual to the training objective using automatic differentiation. The paper also adds composition-only augmented samples so the differential constraint can be enforced where gamma labels are absent ([paper](https://doi.org/10.1039/D3DD00103B)).

The official code is derived from SolvGNN: two atom-level `GraphConv` layers, mean pooling, a molecule interaction `NNConv`/GRU using H-bond edge features, then composition concatenation and Softplus MLP prediction. The paper's selected `GDI-GNN_xMLP` variant concatenates composition after mixture message passing, predicts the two binary log-gamma values, and trains with prediction loss plus weighted Gibbs-Duhem loss ([repository README](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/README.md); [model source](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/model/model_GNN.py); [training source](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/train.py)).

It is binary-only and has no temperature input. The paper's VLE examples are isothermal at 298 K. Ternary and variable-temperature tasks are native N/A.

### Dependencies and risks

The official environment is Python 3.7.16, PyTorch 1.13.1/CUDA 11.6, DGL 1.0.1 CUDA 11.6, RDKit 2020.09.1, NumPy 1.21.5 and pandas 1.3.5 ([environment](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/env.yml)). Source code hard-codes `.cuda()` calls, so CPU/device-agnostic execution requires a mechanical compatibility patch.

The same gamma-label derivation and fixed-298-K restrictions as SolvGNN apply. The Gibbs-Duhem weighting, augmented-composition count, activation smoothness and onset epoch are hyperparameters and must be validation-selected. EPL-2.0 obligations must be preserved if source is copied or modified.

## 6. GE-GNN (Rittig & Mitsos, 2024)

### Verified native implementation

GE-GNN hard-codes thermodynamic consistency by predicting one scalar dimensionless excess Gibbs energy and obtaining both binary log activity coefficients by differentiation; unlike GDI-GNN, consistency is architectural rather than a soft loss ([paper](https://doi.org/10.1039/D4SC04554H)).

The official `gegnn_binary` uses the same molecular graph and interaction encoder as GDI-GNN. It transforms each updated molecule embedding concatenated with its mole fraction, averages the two transformed vectors to impose exchange symmetry, maps the pooled vector through a Softplus MLP to scalar `gE/RT`, and computes

`ln gamma1 = gE/RT + (1-x1) d(gE/RT)/dx1`

`ln gamma2 = gE/RT - x1 d(gE/RT)/dx1`

with autograd ([model source](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/blob/6383142feb3b926fd279ae676a211fd8b3f1dac3/model/model_GNN.py)). It is binary-only and the official implementation has no T input; non-298-K use is N/A unless a separately named `GE-GNN-adapted-T` is introduced.

### Reproduction risks

1. Evaluation must retain gradient tracking because gamma is a derivative; blanket `torch.no_grad()` breaks the model.
2. The source returns gamma derivatives and is hard-coded to CUDA. Device cleanup must not change the learned equations.
3. The scalar is dimensionless `gE/RT`; unit/normalization mistakes in the shared VLE decoder would silently corrupt results.
4. GDI-GNN and GE-GNN are in the same official repository and environment, but are scientifically distinct models and require distinct manifests and parameter counts.

## 7. HANNA current multicomponent version (Hoffmann et al., 2026)

### Verified native implementation

Current HANNA accepts all-component SMILES, liquid composition and temperature, predicts `gE/RT`, and derives thermodynamically consistent log activity coefficients by automatic differentiation. The official repository bundles ChemBERTa-2 and ten pretrained HANNA checkpoints and identifies the current implementation as arbitrary-component, unlike the older binary-only prototype ([paper](https://doi.org/10.1038/s41467-026-71430-y); [README](https://github.com/marco-hoffmann/HANNA/blob/6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365/README.md); [predictor](https://github.com/marco-hoffmann/HANNA/blob/6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365/utils/HANNA_predictor.py)).

The crucial multicomponent fact is explicit in the paper and code: **HANNA is trained only on binary-mixture data**. At inference it evaluates all unique binary component pairs and uses a Muggianu geometric projection to obtain multicomponent `gE`; the projection adds no learned ternary parameters ([Nature paper](https://doi.org/10.1038/s41467-026-71430-y); [model source](https://github.com/marco-hoffmann/HANNA/blob/6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365/models/HANNA/HANNA.py)). Hence ternary predictions are native current-HANNA predictions, but must be described as **binary-trained geometric projection**, not ternary-trained learning.

The implementation uses frozen 384-dimensional ChemBERTa embeddings, 96-wide Lipschitz/spectral-normalized interaction networks, symmetric pair aggregation and boundary corrections. One HANNA head has 65,190 trainable parameters; the ten-head ensemble has 651,900, excluding the bundled frozen ChemBERTa encoder.

### Dependencies, assets and risks

The MIT repository bundles the ChemBERTa tokenizer/model, temperature and embedding scalers, and ten HANNA parameter files. Tested versions include Python 3.10.19, PyTorch 2.10.0, NumPy 2.2.6, pandas 2.3.3, RDKit 2025.9.5, transformers 5.1.0 and scikit-learn 1.7.2.

The published ready-to-use ensemble was trained on more than 800,000 binary VLE, LLE, infinite-dilution-gamma and excess-enthalpy observations. It is therefore not a same-split trainable baseline. Before using it, audit overlap between its training systems and every ThermoFormer test system. Report it as `HANNA-official-pretrained`; do not combine it statistically with same-split seeds 0–4 as if the information conditions were equal. Evaluating gamma requires autograd, even at inference.

If same-split HANNA training code/data are unavailable in the current official inference repository, the valid initial comparison is pretrained inference plus overlap disclosure, not an invented retraining recipe.

## 8. TeNNet-SAC (Yang & Lin, 2025)

### Verified native implementation

TeNNet-SAC is a temperature-dependent, arbitrary-component activity-coefficient model inspired by COSMO-SAC. Its three learned modules are: (1) a SMILES-to-51-bin sigma-profile predictor, (2) a SMILES-to-area/volume geometry predictor, and (3) a segment activity-coefficient network. The physical decoder builds the mixture sigma profile, sums segment contributions and adds a Staverman-Guggenheim combinatorial term ([paper](https://doi.org/10.1021/acs.jcim.5c01804); [README](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/README.md); [decoder](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/utils/property.py)).

The molecular predictors combine a 768-dimensional SMI-TED-Light embedding with a 384-dimensional ChemBERTa-77M-MLM embedding. The segment network consumes normalized sigma profiles and an explicit `1/T` embedding, predicts a scalar free-energy-like quantity, and differentiates it with respect to profile bins to obtain segment coefficients ([profile model](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/models/Emb2Profile.py); [segment model](https://github.com/yueyue2299/TeNNet-SAC/blob/2367e89c87c335c4fd27ff7fa22c87f660f459ba/models/Prf2Gamma.py)). This mechanism is natively multicomponent and temperature-conditioned.

The repository bundles profile (`prf.ckpt`), geometry (`geo.ckpt`), base segment (`base.ckpt`) and ten experimentally fine-tuned checkpoints. The intended reproduction path is therefore to load author modules rather than repeat 39,745 quantum-solvation calculations and the one-million-sample COSMO-SAC pretraining. Source-dimension parameter counts are 1,122,515 for the profile model, 692,578 for geometry and 928,113 for one segment model; the base learned heads total 2,743,206 parameters excluding frozen language encoders.

### Dependencies, assets and risks

The MIT repository pins NumPy 1.26.4, PyTorch 2.1.2 and transformers 4.36.2. It vendors an adapted SMI-TED-Light source tree, but downloads external encoder weights at runtime. The README mentions `TeNNet-SAC.yml`, while that file is absent at the audited revision; use `requirements.txt` and record resolved packages.

The fine-tuning system list is included as `TrainingSystemsList.csv` (397 binary systems). Intersect it with ThermoFormer train/validation/test IDs and publish overlap counts. Label base and fine-tuned results separately (`TeNNet-SAC-base-official-pretrained`, `TeNNet-SAC-finetuned-official-pretrained`). Fine-tuned ensemble results are not same-split training. Evaluation requires network/cache access for missing foundation-model assets unless they are checksummed into the experiment cache.

## Unified-framework implications

### Scientific adapters that do not change the baseline identity

1. Canonical component ordering plus inverse permutation of outputs.
2. Train-only scaling and committed ThermoFormer system-ID datasets/splits/vle/seeds.
3. A shared Psat provider and low-pressure VLE solver for gamma/gE models.
4. Common metric calculation, valid-coverage accounting, seed-level CSV, summary and manifest writing.
5. Mechanical device/API ports that preserve graph features, equations, learned inputs and layer semantics.

### Changes that require an `adapted` name

1. Adding temperature to SolvGNN, GDI-GNN or GE-GNN.
2. Extending binary Descriptor ANN, SMILES-RNN, UALF-GNN, GDI-GNN or GE-GNN to ternary mixtures.
3. Adding a new prediction direction to a direct VLE model.
4. Replacing an author's molecular encoder/interaction graph with a materially different encoder.
5. Literal 21-wide Descriptor ANN input after dropping the official P/x or descriptor inputs.

### Fairness and dependency gates before formal seeds 0–4

- Freeze system IDs and verify zero train/val/test overlap for all same-split trainable baselines.
- Fit scalers and hyperparameters using train/validation only.
- Publish gamma-label derivation assumptions and coverage for VLE-derived training targets.
- Restrict fixed-temperature native models to matching 298 K records; report all other task rows as N/A.
- Audit official-pretrained HANNA and TeNNet-SAC training-system overlap; never describe them as same-split models.
- Record Psat availability independently from neural-network validity so solver coverage is diagnosable.
- Record immutable source revision, local patch hash, environment lock, model parameter count, checkpoint hash and input feature schema in each manifest.
- Keep license status visible: Descriptor ANN, SMILES-RNN and SolvGNN have no explicit repository license at the audited revisions.

## Recommended implementation-source labels

| Baseline | Label for the first faithful framework implementation |
|---|---|
| Descriptor ANN | `official-architecture/reimplemented-data-wrapper` (license-safe clean implementation preferred) |
| SMILES-RNN | `reimplemented-from-official` (official data pipeline incomplete) |
| UALF-GNN | `paper-reimplemented` |
| SolvGNN | `official-ported` |
| GDI-GNN | `official-ported` |
| GE-GNN | `official-ported` |
| HANNA | `official-pretrained` |
| TeNNet-SAC | `official-pretrained` and `official-finetuned` as separate rows |

These labels should appear in configuration files, seed CSVs, aggregate tables and manuscript text. They prevent an official checkpoint, a faithful port and a paper-based reimplementation from being presented as equivalent provenance.
