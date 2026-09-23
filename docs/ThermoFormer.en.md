# ThermoFormer: a thermodynamics-aware molecular interaction framework for vapour-liquid equilibrium prediction and autonomous separation design

> Source: `docs/ThermoFormer.pdf` (English). This file is the English version of the paper summary;
> it follows the PDF section by section and preserves the equations and data.

**Authors**: Jinlin Ye, Yang Li, Minghao Sun, Hang Yuan, Shuohan Wang, Yu Wang, Lang Jiang, Xinchen Kang\*, Wei Zhang\*
(School of Artificial Intelligence, Hebei University of Technology; School of Chemical Engineering, Hebei University of Technology; School of Chemical and Chemical Engineering, University of Chinese Academy of Sciences)

---

## Abstract

Vapour-liquid equilibrium (VLE) underpins the design and optimisation of distillation, absorption and
solvent-assisted separation processes. Accurate VLE data are essential for process development, yet
complete experimental measurement of every mixture is impractical across a rapidly expanding chemical
space. Comprehensive databases such as NIST ThermoML provide valuable experimental records, but cover
only a small fraction of the possible mixtures and thermodynamic states. Predictive thermodynamic
models are therefore needed to generalise these sparse experimental observations across composition,
temperature and chemical system.

Classical activity-coefficient models describe liquid-phase non-ideality through physically motivated
interaction parameters, including NRTL, Wilson, UNIQUAC and UNIFAC. NRTL, Wilson and UNIQUAC generally
require system-specific parameter regression, whereas UNIFAC extends prediction to unstudied systems
through predefined groups and group-interaction parameters. Their applicability nevertheless remains
limited by available experimental parameters or already-parameterised group interactions. Quantum
chemical methods such as COSMO-RS and COSMO-SAC reduce the reliance on empirical interaction tables
but introduce substantial molecular-level computation and sensitivity to computational settings.

Machine learning offers another route — learning phase-equilibrium behaviour directly from molecular
information and thermodynamic conditions. Early artificial neural network (ANN) studies demonstrated
the feasibility of data-driven VLE prediction for ternary salt-containing systems and ethanol-based
binary mixtures. Subsequent descriptor-based ANN and random forest models extended VLE prediction to a
wider binary mixture space, while SMILES-RNN architectures introduced molecular sequences together
with thermodynamic properties for joint dew-point and bubble-point prediction. In recent years,
graph neural network (GNN) models can learn directly from molecular graphs, reducing the reliance on
hand-designed molecular descriptors. Pretrained molecular models such as ChemBERTa-2, GROVER and
Uni-Mol provide increasingly transferable representations of molecular structure. For mixtures,
SolvGNN explicitly introduces molecular-level interaction learning and demonstrates
composition-dependent activity-coefficient prediction in binary and ternary systems. Despite this
progress, most direct VLE models remain binary-oriented, mapping molecular representations straight to
equilibrium observables and failing to resolve multicomponent interaction effects adequately. More
importantly, such end-to-end mapping provides no explicit thermodynamic link between the learned
molecular interactions and the resulting phase behaviour. This limitation has driven thermodynamic
machine learning beyond direct equilibrium-property regression towards physically meaningful
intermediate variables. SPT-NRTL combines molecular language representations with the NRTL framework,
GDI-GNN enforces Gibbs-Duhem consistency through regularisation, GE-GNN predicts the excess Gibbs
energy and obtains activity coefficients by automatic differentiation, HANNA embeds thermodynamic
consistency directly in the network architecture, and TeNNet-SAC introduces segment activity
coefficients as a physical intermediate variable to achieve thermodynamically consistent
multicomponent prediction. Nonetheless, multicomponent scalability is often achieved by projecting or
combining binary thermodynamic information rather than learning directly from multicomponent
measurements. Such pairwise construction is limited when additional components reorganise the
molecular environment — because the resulting multicomponent non-ideality cannot always be recovered
from binary behaviour alone.

This work develops a thermodynamics-aware molecular interaction framework for multicomponent VLE
prediction and autonomous separation design, with **ThermoFormer** as its core molecular
thermodynamic model. ThermoFormer encodes each component with pretrained molecular models and passes
the resulting molecular tokens through a multicomponent interaction Transformer, so that each
component updates its representation according to the identity and composition of the surrounding
species. Thermodynamic conditions then modulate these mixture-aware representations into
component-specific latent non-ideality states, which an activity-coefficient decoder converts into
physically meaningful activity coefficients, subsequently used in a differentiable VLE solver to
reconstruct equilibrium temperature, pressure and vapour composition. Model training is constrained
by composition conservation, Gibbs-Duhem consistency, pure-component limits, phase-diagram continuity
and component-permutation consistency, while the differentiable equilibrium solver imposes the
corresponding phase-equilibrium relations during VLE reconstruction. Finally, ThermoFormer is
deployed as a thermodynamic Skill inside an LLM-driven agent that coordinates phase-behaviour
analysis, solvent screening and process simulation for autonomous separation design.

---

## 2 Results

### 2.1 Dataset construction and coverage

Experimental VLE records were extracted from the NIST ThermoML archive with `thermoml-io` and
standardised into binary and ternary datasets. Records containing molecular identities, temperature,
pressure, and liquid and vapour compositions were retained, together with source provenance and any
available thermodynamic consistency annotations. Temperature and pressure were converted to K and
kPa respectively, and the relevant mole fractions were reconstructed subject to composition closure.
Molecular structures were normalised with RDKit, and system identity was defined by the unordered
component set. Chemical families were partitioned by priority SMARTS rules, with unresolved structures
grouped separately.

The resulting dataset contains **28,290 experimental VLE points**, of which **23,061 binary
observations come from 700 systems** and **5,229 ternary observations come from 126 systems**. The
binary and ternary subsets involve 333 and 125 molecular components respectively, covering a broad
and non-uniform temperature-pressure space. Chemical coverage comprises 79 unordered binary system
pairs and 49 ternary system groups, combining densely sampled classes with a long tail of
low-frequency chemical types. The composition space extends from near-pure-component limits to the
interior of the mixture domain. Among the ternary systems, 30 contain all three constituent binary
subsystems, 22 contain two, 69 contain one, and 5 contain none. These distributions support controlled
evaluation along three axes: interpolation within thermodynamic states, generalisation across
chemical space, and binary-to-ternary transfer.

### 2.2 Predictive performance and generalisation

ThermoFormer is evaluated on overall prediction, thermodynamic-state generalisation, unseen-component
extrapolation, and binary-to-ternary transfer. For each random seed, the final checkpoint is selected
by validation only, from among supervised and fugacity fine-tuning candidates. (Table 1 summarises the
coupled isothermal P-x-y and isobaric T-x-y results.)

**2.2.1 Overall predictive performance** — With binary-only training, the pressure and temperature
MAEs are 8.98±5.20 kPa and 2.45±0.15 K respectively, and the vapour-composition MAE is below 0.03 in
both prediction directions. Joint binary-ternary training maintains comparable binary performance
(pressure and temperature MAE 8.68±4.71 kPa and 2.45±0.31 K). The jointly trained model predicts
ternary pressure and temperature with MAEs of 1.85±0.27 kPa and 1.41±0.61 K. When binary and ternary
test records are evaluated jointly, the jointly trained model reaches pressure and temperature MAEs of
7.75±3.55 kPa and 2.38±0.33 K, with vapour-composition MAEs of 0.0251±0.0056 and 0.0299±0.0060 for
the isothermal and isobaric tasks respectively.

**2.2.2 Thermodynamic-state generalisation** — Composition interpolation is the most accurate case
(pressure and temperature MAE $1.61\pm0.20$ kPa and $0.58\pm0.04$ K). Composition-edge extrapolation
keeps pressure and temperature MAE below 2 kPa and 1 K respectively. High-temperature and
high-pressure extrapolation raise the isothermal pressure MAE to $5.83\pm0.42$ and $5.96\pm1.35$ kPa
respectively. The vapour-composition $R^2$ remains $\geq 0.995$ under high-temperature and
high-pressure extrapolation.

**2.2.3 Generalisation to unseen components** — Unseen components produce the largest performance
degradation across all evaluation protocols: pressure MAE rises to $32.39\pm11.31$ kPa and temperature
MAE to $36.45\pm1.49$ K, with corresponding $R^2$ of $0.494\pm0.242$ and $0.456\pm0.044$. The
vapour-composition MAE reaches $0.1057\pm0.0179$ and $0.1191\pm0.0062$ in the isothermal and isobaric
tasks. Molecular extrapolation beyond the chemical space represented in training remains the main
limitation of the current model.

**2.2.4 Binary-to-ternary transfer** — Without ternary supervision, ThermoFormer predicts ternary
pressure and temperature with MAEs of 1.62±0.38 kPa and 1.91±0.89 K. The zero-shot isothermal
vapour-composition MAE is 0.0129±0.0034, and the isobaric value is 0.0714±0.0167. Increasing the
ternary training fraction produces a non-monotonic performance trajectory: the minimum isothermal
pressure MAE occurs at 50% ternary supervision, reaching 1.30±0.50 kPa. The isobaric
vapour-composition error remains in a narrow range across all ternary training fractions. These
results establish direct binary-to-ternary transfer, while additional ternary supervision yields
task-dependent gains.

> **Table 1 (highlights)**: isothermal pressure MAE (overall $7.75\pm3.55$ kPa; composition
> interpolation $1.61\pm0.20$; unseen components $32.39\pm11.31$), isobaric temperature MAE (overall
> $2.38\pm0.33$ K; interpolation $0.58\pm0.04$; unseen components $36.45\pm1.49$). Each cell reports
> MAE, RMSE and $R^2$.

### 2.3 Comparison with machine-learning and thermodynamic models

Comparison against NRTL, Wilson and UNIQUAC on six binary and six ternary equilibrium curves.
ThermoFormer predictions come from the validation-selected seed-0 checkpoint, without
system-specific refitting. Classical models are fitted individually with Phasepy to all available
measurements for each displayed system, with vapour pressures from validated DIPPR/Antoine/Wagner
correlations. The comparison therefore contrasts system-specific recall-style fitting with
system-independent molecular prediction.

**Binary**: ThermoFormer achieves pressure MAE 1.17-1.92 kPa, temperature 0.57-1.52 K and
vapour composition 0.007-0.012 on four moderate cases. For 1-propanol-ethanol, ThermoFormer captures
the curve trend but the pressure MAE reaches 18.18 kPa, whereas the fitted classical models are around
2.42-2.45 kPa. The largest binary deviation occurs for isobutane-propionitrile, where ThermoFormer
reaches a pressure MAE of 50.38 kPa against 6.80-8.00 kPa for the classical models; its
vapour-composition MAE is only 0.0015, indicating that ThermoFormer preserves the liquid-vapour
composition mapping but has a pressure-scale offset.

**Ternary**: low-error cases show pressure MAE 1.19-1.52 kPa or temperature 0.67-1.00 K, with vapour
composition 0.005-0.016. A representative methylcyclohexane-cyclohexylamine-aniline case gives
pressure and vapour-composition MAEs of 1.24 kPa and 0.028. Of 21 unique molecular components, 18 have
validated vapour-pressure correlations, enabling classical calculations for 7 of the 10 unique
systems.

Across the 8 cases where the full classical model set is available, NRTL/Wilson/UNIQUAC achieve lower
state-variable MAEs after direct fitting. ThermoFormer achieves lower vapour-composition MAE in 2/1/2
cases. Classical models provide high-accuracy in-system representation when experimental records and
validated pure-component correlations are available; ThermoFormer provides system-independent
prediction without per-system interaction-parameter regression and maintains coverage when external
pure-property inputs are incomplete.

### 2.4 Ablation analysis

**2.4.1 Molecular representation** — Explicit RDKit descriptors provide the main representation gain,
reducing pressure and temperature MAE by 50.2% and 52.0% respectively relative to Uni-Mol v2 alone.
Functional-group features alone are insufficient to support stable VLE reconstruction (severe pressure
error, reduced solver coverage). Stacking Uni-Mol v2 on RDKit descriptors brings no consistent
improvement. Adding functional-group features reduces pressure MAE from 9.297±4.681 to 8.205±3.309 kPa
and temperature from 2.578±0.450 to 2.513±0.371 K. The full three-view representation is therefore
retained as the most balanced molecular encoding.

**2.4.2 Multicomponent interaction architecture** — Introducing chemical attention bias and contextual
conditioning pair potentials cannot consistently outperform the vanilla multicomponent Transformer.
Removing the attention bias while retaining contextual conditioning pair potentials improves
vapour-composition and isobaric temperature MAE, but raises pressure MAE from 8.205±3.309 to
8.741±4.253 kPa. Additional interaction mechanisms redistribute accuracy among prediction targets
rather than producing consistent gains. The vanilla multicomponent Transformer is retained for
stronger pressure prediction, comparable temperature and composition accuracy, and its simpler form.

**2.4.3 Fugacity-constrained fine-tuning** — Fugacity-constrained fine-tuning improves 10 of 12
averaged prediction metrics, reducing the teacher-forced fugacity residual from 0.009237±0.003908 to
0.008898±0.003440. The only aggregate regression is a small increase in temperature RMSE
(4.146 -> 4.202 K). Validation selection retains fine-tuned checkpoints for three seeds and reverts two
seeds to the supervised checkpoint, removing the aggregate temperature regression and improving all
12 averaged metrics.

### 2.5 Interpretability of the learned multicomponent thermodynamics

**2.5.1 Molecular-view dependence and cross-view interaction** — Grouped Shapley attribution reveals a
consistent hierarchy of information: RDKit descriptors account for 59-65% of the excess Gibbs energy,
activity coefficients, relative volatility and pair-potential strength, making them the dominant
molecular representation; functional-group features contribute a further 23-33%, with their largest
contribution to relative volatility; Uni-Mol v2 contributes 8-13% directly. The three views play
distinct roles: global physicochemical descriptors provide the main signal, while local chemical
motifs supplement it. Feature occlusion reveals that the most influential RDKit descriptors for
relative volatility are NH/OH counts, Balaban connectivity, polar surface area, logP, partial charges
and molar refractivity; aldehyde, ketone, halogenated, alcohol and ester motifs dominate the
functional-group attribution.

**2.5.2 Composition-dependent molecular interactions** — In pairwise analysis, alcohol/polyol-alkane
/cycloalkane pairs contribute most, followed by alcohol/polyol-ether/carbonyl pairs. Most dominant
pair classes show positive mean signed contributions. Explicitly held-out mixtures show that the model
can distinguish weak from strongly non-ideal binary systems: N,N-dimethylformamide-dimethyl sulfoxide
is close to ideal, whereas ethanol-isooctane exhibits a stronger and composition-dependent
activity-coefficient deviation. A ternary aniline-methylcyclohexane-cyclohexylamine case shows pair
contributions redistributing as the third component's share increases, indicating that the learned
ternary non-ideality reflects composition-dependent reorganisation of molecular interactions rather
than a fixed superposition of constituent binary responses.

**2.5.3 The role of fugacity-constrained fine-tuning** — Fine-tuning reduces the teacher-forced
fugacity RMSE for all five random seeds. The improvement concentrates in thermodynamic intermediate
variables rather than in the final vapour composition; saturated vapour-pressure prediction changes
most, followed by activity coefficients, while vapour-composition changes are markedly smaller.
Fugacity regularisation acts mainly by rebalancing pure-component volatility against liquid-phase
non-ideality while preserving observable equilibrium composition.

### 2.6 ThermoFormer-enabled autonomous separation design

(The body of this section is not yet expanded in the source paper; see the companion experiment
document `ThermoFormer萃取精馏过程.md`.)

---

## 3 Discussion

(This section is not expanded in the source paper; the body remains to be written.)

---

## 4 Methods

### 4.1 ThermoFormer formulation

ThermoFormer is constructed for bubble-point prediction in binary and ternary VLE systems. For an
$N$-component mixture ($N\in\{2,3\}$), the molecular structures are denoted $\{M_i\}$ and the liquid
composition $\mathbf{x}=(x_1,\dots,x_N)$ satisfies $\sum_{i} x_i=1$. Two complementary prediction
modes:

$$
\text{isothermal P-x-y}: \quad (\{M_i\}, T, \mathbf{x}) \rightarrow (P_b, \mathbf{y})
$$

$$
\text{isobaric T-x-y}: \quad (\{M_i\}, P, \mathbf{x}) \rightarrow (T_b, \mathbf{y}) \qquad \text{(Eq. 1)}
$$

where $P_b$ and $T_b$ are the bubble-point pressure and temperature respectively, and
$\mathbf{y}=(y_1,\dots,y_N)$ is the equilibrium vapour composition.

### 4.2 Multi-view molecular representation learning

Each component is represented by three complementary molecular views: RDKit descriptors ($d_i^R$),
Uni-Mol v2 embeddings ($d_i^U$), and SMARTS-derived functional-group features ($d_i^G$). The RDKit
view encodes explicit physicochemical and topological properties (molecular size, polarity,
lipophilicity, hydrogen-bonding capability, ring structure, atomic composition, molecular
flexibility), with descriptor values standardised using statistics fitted on the corresponding
training split. The Uni-Mol v2 view provides higher-order structural representations derived from
three-dimensional molecular conformations, precomputed with a frozen encoder and held fixed during VLE
model training. The functional-group view records the presence or occurrence counts of SMARTS motifs
(hydroxyl, carbonyl, carboxyl, ester, ether, amine, amide, aromatic, sulfur-containing,
halogen-containing groups). The three views pass through independent projection networks:

$$
h_i^R = f^R(d_i^R), \qquad h_i^U = f^U(d_i^U), \qquad h_i^G = f^G(d_i^G) \qquad \text{(Eq. 2)}
$$

The projected views are concatenated and transformed into a unified molecular token:

$$
h_i^0 = f^{\mathrm{fuse}}(h_i^R \,\|\, h_i^U \,\|\, h_i^G) \qquad \text{(Eq. 3)}
$$

The projection and fusion networks are shared across all component positions. The molecular tokens of
an $N$-component mixture are collected as $H^0=[h_1^0,\dots,h_N^0]$ (Eq. 4).

### 4.3 Permutation-equivariant multicomponent interaction encoding

The fused molecular tokens describe the constituent molecules independently, but do not account for
their mixing-dependent environment. A learnable mixture token $h_{\mathrm{mix}}^0$ is appended to the
component sequence: $H^{\mathrm{in}}=[h_{\mathrm{mix}}^0, h_1^0,\dots,h_N^0]$ (Eq. 5). The full
sequence is processed by a shared multicomponent Transformer encoder:

$$
[h_{\mathrm{mix}}, \bar{h}_1, \dots, \bar{h}_N] = T_\theta(H^{\mathrm{in}}; M) \qquad \text{(Eq. 6)}
$$

$M$ masks padded component positions, $\bar{h}_i$ is the mixture-conditioned representation of a
component, and $h_{\mathrm{mix}}$ is the global mixture representation. Each Transformer layer uses
standard multi-head self-attention:

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}} + M\right) V \qquad \text{(Eq. 7)}
$$

RDKit/Uni-Mol/functional-group information influences attention through the fused molecular tokens
rather than through an additional pairwise attention bias. Self-attention updates each component using
information from all coexisting species. For any component permutation $\Pi$, the encoder satisfies
$T_\theta([h_{\mathrm{mix}}^0, \Pi H^0])=[h_{\mathrm{mix}}, \Pi \bar{H}]$ (Eq. 8), so component
representations are permutation-equivariant and the mixture representation is permutation-invariant,
where $\bar{H}=[\bar{h}_1,\dots,\bar{h}_N]$.

### 4.4 State-conditioned molecular interaction learning

The multicomponent Transformer provides mixture-aware component representations $\bar{h}_i$ and a
global mixture representation $h_{\mathrm{mix}}$, which are further conditioned on thermodynamic state
before molecular interactions are evaluated. The conditioning vector of component $i$ combines the
global mixture context with temperature, pressure and liquid composition:

$$
c_i = \left[\, h_{\mathrm{mix}} \,\|\, \tilde{T} \,\|\, \tilde{P} \,\|\, x_i \,\right] \qquad \text{(Eq. 9)}
$$

where $\tilde{T}$ and $\tilde{P}$ are normalised temperature and pressure. A conditioning encoder maps
$c_i$ to component-specific scale and shift vectors:

$$
(s_i, t_i) = f^{\mathrm{cond}}(c_i) \qquad \text{(Eq. 10)}
$$

thereby modulating the mixture-aware representation:

$$
\tilde{h}_i = \bar{h}_i \odot \left[ 1 + \rho \tanh(s_i) \right] + t_i \qquad \text{(Eq. 11)}
$$

For each component pair $(i,j)$, the state-conditioned representations are combined by a symmetric
pair operator:

$$
z_{ij} = \left[\, \tilde{h}_i + \tilde{h}_j \,\|\, |\tilde{h}_i - \tilde{h}_j| \,\|\, h_{\mathrm{mix}} \,\right] \qquad \text{(Eq. 12)}
$$

A shared pair-potential network maps $z_{ij}$ to a scalar effective interaction:

$$
I_{ij} = \phi_\theta(z_{ij}) \qquad \text{(Eq. 13)}
$$

The symmetric construction gives $I_{ij}=I_{ji}$, allowing the same interaction network to be shared
across all component pairs. Because $\bar{h}_i$, $\bar{h}_j$ and $h_{\mathrm{mix}}$ are encoded
jointly from all components, $I_{ij}$ is conditioned on the full mixture environment.

### 4.5 Excess Gibbs energy and activity-coefficient decoding

ThermoFormer represents liquid-phase non-ideality through the dimensionless molar excess Gibbs energy:

$$
g^E \equiv \frac{G^E}{RT} = \sum_{1 \le i < j \le N} x_i x_j I_{ij} \qquad \text{(Eq. 14)}
$$

Binary: $g^E_{\mathrm{binary}} = x_1 x_2 I_{12}$ (Eq. 15); ternary:
$g^E_{\mathrm{ternary}} = x_1 x_2 I_{12} + x_1 x_3 I_{13} + x_2 x_3 I_{23}$ (Eq. 16). The composition
prefactor makes $g^E \to 0$ recovered at every pure-component vertex. Unlike fixed binary-parameter
constructions, the pair terms of Eq. (16) are conditioned on the full ternary environment. Activity
coefficients are obtained from the shared scalar potential by automatic differentiation:

$$
\ln\gamma_i = g^E + \frac{\partial g^E}{\partial x_i} - \sum_j x_j \frac{\partial g^E}{\partial x_j} \qquad \text{(Eq. 17)}
$$

All component activity coefficients are coupled through the same excess Gibbs energy representation,
rather than being predicted by independent output heads. This construction enforces Gibbs-Duhem
consistency through the shared differentiable potential.

### 4.6 Pure-component vapour-pressure branch

The saturated vapour pressure of component $i$ is predicted from its fused molecular token and
temperature:

$$
\ln P_i^{\mathrm{sat}}(T) = f^{\mathrm{sat}}(h_i^0, T) \qquad \text{(Eq. 18)}
$$

The branch depends only on the pure-component representation and temperature; mixture effects enter
through $\gamma_i$, thereby separating pure-component volatility from liquid-phase interaction effects.

### 4.7 Differentiable vapour-liquid equilibrium reconstruction

Activity coefficients and saturated vapour pressures are coupled through the modified Raoult relation:

$$
y_i P = x_i \gamma_i(T,P,\mathbf{x}) \, P_i^{\mathrm{sat}}(T) \qquad \text{(Eq. 19)}
$$

Under the low-pressure ideal-vapour approximation, the bubble-point pressure is computed as:

$$
P^{\mathrm{calc}}(T,P,\mathbf{x}) = \sum_i x_i \gamma_i(T,P,\mathbf{x}) \, P_i^{\mathrm{sat}}(T) \qquad \text{(Eq. 20)}
$$

The corresponding vapour composition:

$$
y_i = \frac{x_i \gamma_i P_i^{\mathrm{sat}}}{\sum_j x_j \gamma_j P_j^{\mathrm{sat}}} \qquad \text{(Eq. 21)}
$$

satisfies $\sum_i y_i = 1$. The isothermal task supplies $T$ and $\mathbf{x}$, and the bubble-point
pressure satisfies the implicit relation $P = P^{\mathrm{calc}}(T,P,\mathbf{x})$ (Eq. 22); because the
activity coefficients depend on pressure, it is solved by damped fixed-point iteration:

$$
P^{(k+1)} = (1-\eta)\,P^{(k)} + \eta\, P^{\mathrm{calc}}\!\left(T, P^{(k)}, \mathbf{x}\right) \qquad \text{(Eq. 23)}
$$

where $\eta$ is the damping coefficient. The isobaric task supplies $P^\ast$ and $\mathbf{x}$, and the
bubble-point temperature is determined by
$r(T) = P^{\mathrm{calc}}(T,P^\ast,\mathbf{x}) - P^\ast = 0$ (Eq. 24), with the residual derivative
estimated by central differences:

$$
\frac{\partial r}{\partial T} \approx \frac{r(T+\Delta T) - r(T-\Delta T)}{2\,\Delta T} \qquad \text{(Eq. 25)}
$$

Temperature is updated with a bracketed Newton step:

$$
T_{N}^{(k+1)} = T^{(k)} - \frac{r(T^{(k)})}{\left.\partial r/\partial T\right|_{T^{(k)}}} \qquad \text{(Eq. 26)}
$$

Out-of-range proposals are replaced by the bracket midpoint. All solver operations are implemented as
differentiable tensor computations, so error can propagate among the VLE solver, the vapour-pressure
branch, the excess Gibbs decoder, the pair-potential network, the Transformer and the molecular
representation modules.

### 4.8 Two-stage supervision and fugacity-constrained optimisation

The first stage minimises bubble-point pressure, bubble-point temperature and vapour-composition
prediction error:

$$
\mathcal{L}_{\mathrm{sup}} = \lambda_P L_P + \lambda_T L_T + \lambda_y L_y \qquad \text{(Eq. 27)}
$$

where $L_P$ and $L_T$ supervise the isothermal and isobaric bubble-point tasks respectively, and $L_y$
supervises the vapour composition in both modes. The best validation checkpoint initialises the
physics fine-tuning stage. At VLE, the liquid and vapour fugacities of each component are equal:

$$
f_i^{L} = f_i^{V} \qquad \text{(Eq. 28)}
$$

Under the low-pressure ideal-vapour approximation the component fugacities are:

$$
f_i^{L} = x_i \gamma_i P_i^{\mathrm{sat}}(T), \qquad f_i^{V} = y_i P \qquad \text{(Eq. 29, 30)}
$$

and the equilibrium condition is:

$$
x_i \gamma_i P_i^{\mathrm{sat}}(T) = y_i P \qquad \text{(Eq. 31)}
$$

For experimental sample $n$, the component-wise log fugacity residual is evaluated at the observed
state:

$$
r_{n,i}^{f} = \ln x_{n,i}^{\mathrm{obs}} + \ln\hat{\gamma}_{n,i} + \ln \hat{P}_{n,i}^{\mathrm{sat}}(T_n^{\mathrm{obs}}) - \ln y_{n,i}^{\mathrm{obs}} - \ln P_n^{\mathrm{obs}}
$$

$$
= \ln\!\left[ \frac{x_{n,i}^{\mathrm{obs}}\,\hat{\gamma}_{n,i}\,\hat{P}_{n,i}^{\mathrm{sat}}(T_n^{\mathrm{obs}})}{y_{n,i}^{\mathrm{obs}}\,P_n^{\mathrm{obs}}} \right] \qquad \text{(Eq. 32)}
$$

The residual uses the experimental $T,P,x,y$ rather than the VLE solver output (which satisfies
Eq. 31 by construction). Components near zero composition are excluded from the log residual:

$$
m_{n,i} = \mathbb{I}\big(x_{n,i}^{\mathrm{obs}} > x_{\min}\big)\, \mathbb{I}\big(y_{n,i}^{\mathrm{obs}} > y_{\min}\big) \qquad \text{(Eq. 33)}
$$

where $\mathbb{I}(\cdot)$ is the indicator function. The fugacity-equilibrium loss is computed with a
mass-weighted Huber penalty:

$$
\mathcal{L}_{\mathrm{fug}} = \frac{\sum_n \sum_i w_n m_{n,i}\, \rho_\delta\!\left(r_{n,i}^{f}\right)}{\sum_n \sum_i w_n m_{n,i}} \qquad \text{(Eq. 34)}
$$

where $w_n$ is the experimental quality weight and $\rho_\delta$ is the Huber function. The physics
fine-tuning objective is:

$$
\mathcal{L}_{\mathrm{fine}} = \mathcal{L}_{\mathrm{sup}} + \lambda_{\mathrm{fug}}\, \mathcal{L}_{\mathrm{fug}} \qquad \text{(Eq. 35)}
$$

$\mathcal{L}_{\mathrm{sup}}$ is retained to prevent fugacity regularisation from damaging directly
evaluated prediction. Because $\mathcal{L}_{\mathrm{fug}}$ constrains the product
$\gamma_i P_i^{\mathrm{sat}}$ rather than the two factors independently, the vapour-pressure branch is
frozen during fine-tuning; the molecular projectors, multi-view fusion and Transformer encoder are
also fixed, and only the pair-potential network, state-conditioning modules and mixture token are
fine-tuned.

---

## References (highlights; numbers correspond to the source)

1. Frenkel et al., ThermoML. J. Chem. Eng. Data 48, 2-13 (2003).
2. Renon & Prausnitz, NRTL. AIChE J. 14, 135-144 (1965).
3. Wilson, A new expression for the excess free energy of VLE. JACS 86, 127-130 (1964).
4. Abrams & Prausnitz, UNIQUAC. AIChE J. 21, 116-128 (1975).
5. Fredenslund et al., UNIFAC. AIChE J. 21, 1086-1099 (1975).
6. Weidlich & Gmehling, modified UNIFAC. I&EC Res. 26, 1372-1381 (1981).
7. Constantinescu & Gmehling, UNIFAC Dortmund version 6. J. Chem. Eng. Data 61, 2738-2748 (2016).
8. Klamt, COSMO-RS. JPCB 99, 2224-2235 (1995).
9. Lin & Sandler, COSMO-SAC. I&EC Res. 41, 899-913 (2002).
10-24. Machine-learning/thermodynamic VLE models: ANN, descriptor ANN/RF, SMILES-RNN, GNN,
ChemBERTa-2, GROVER, Uni-Mol, SolvGNN, SPT-NRTL, GDI-GNN, GE-GNN, HANNA, TeNNet-SAC and others (see
the source for details).
25. Boiko et al., Autonomous chemical research with LLMs. Nature 624, 570-578 (2023).
26. Bran et al., Augmenting LLMs with chemistry tools. Nat. Mach. Intell. 6, 525-535 (2024).
27. Schäfer et al., agentic process design in flowsheet simulation. AIChE J. 70488 (2026).
28. Tan et al., reasoning-agent distillation simulation. Commun. Eng. 5, 26 (2026).
29. RDKit. https://www.rdkit.org (2026).
30. Ji et al., Uni-Mol v2 (molecular pretrained model). NeurIPS 37 (2024).

---

*Translation note: sections 2.6 and 3 have no expanded body in the source PDF (headings only); this
version keeps the section framework and marks them accordingly. A companion experiment is in
`ThermoFormer萃取精馏过程.md`. Equation numbering follows the source.*
