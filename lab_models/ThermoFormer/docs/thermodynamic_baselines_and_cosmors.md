# Thermodynamic baselines and COSMO-RS comparison design

Research status: 2026-08-25. This note defines an evaluation plan; it does not
report experimental results.

## Recommendation

The paper should **not** compare ThermoFormer with NRTL, Wilson, and UNIQUAC
only on a few hand-picked phase diagrams. The defensible comparison has two
parts:

1. a quantitative evaluation on the complete held-out test partition; and
2. a small, pre-specified set of phase-diagram case studies that explains where
   the methods succeed or fail.

Case studies alone establish examples, not comparative predictive performance.
The quantitative table should use the same P/T/y metrics as ThermoFormer, but
the admissible protocol depends on where each classical model obtains its pair
parameters. Phase diagrams should be secondary evidence.

The existing `overall_binary_ternary` split is system-disjoint. NRTL, Wilson,
and UNIQUAC parameters therefore cannot be regressed from this split's training
rows for its held-out test systems. On this protocol, a classical-model result
is valid only when every required pair parameter comes from a frozen external
source selected without consulting the project test data; parameter coverage
must be reported. Train-only regression should instead be evaluated on the
within-system state-generalization protocols, or on a separately declared
within-system correlation split. COSMO-RS can be evaluated on the system-
disjoint protocol once compatible quantum inputs exist for every test component.

## Classical activity-coefficient baselines

NRTL, Wilson, and UNIQUAC are excess-Gibbs-energy models with pair-specific
parameters. Their original formulations are documented by
[Renon and Prausnitz for NRTL](https://trc.nist.gov/TDE/TDE_Help/NRTL-AC-Model.htm),
[Wilson](https://pubs.acs.org/doi/10.1021/ja01056a002), and
[Abrams and Prausnitz for UNIQUAC](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690210115).
The original UNIQUAC paper explicitly states that multicomponent extension uses
binary parameters without additional ternary parameters. NIST's ternary
evaluation likewise constructs ternary activity-coefficient models from the
three binary subsystems and does not fit the ternary observations
([NIST TDE paper, pp. 262 and 272](https://www.nist.gov/system/files/documents/mml/acmd/TDE-7-0-2012.pdf)).

### Recommended implementation

Use the open-source Python package
[`thermo`](https://github.com/CalebBell/thermo), pinned to an exact version and
commit. It provides NRTL, Wilson, and UNIQUAC equations, temperature-dependent
parameter forms, vapor-pressure correlations, and flash/phase-diagram tools;
its package metadata declares the MIT license
([official package metadata](https://github.com/CalebBell/thermo/blob/master/pyproject.toml)).
The official documentation covers
[NRTL](https://thermo.readthedocs.io/thermo.nrtl.html),
[Wilson](https://thermo.readthedocs.io/thermo.wilson.html), and
[UNIQUAC](https://thermo.readthedocs.io/thermo.uniquac.html). It also gives a
working example of generating T-x-y, P-x-y, and x-y diagrams through a
`GibbsExcessLiquid`/`FlashVL` workflow
([official example](https://thermo.readthedocs.io/Examples/Creating%20Txy,%20Pxy,%20and%20xy%20diagrams%20for%20the%20binary%20water%20ethanol%20system%20with%20Modified%20UNIFAC%20(Dortmund).html)).

The package equations can be wrapped by ThermoFormer's existing VLE solver.
This is preferable for the controlled comparison because the vapor-phase
assumption, pure-component vapor-pressure convention, numerical tolerances,
composition clipping, and failure definition then remain identical. Fit the
classical parameters directly against the training VLE residuals rather than
converting every observation to a noisy pointwise activity coefficient first.

### Fair fitting protocol

For every registered split:

1. Fit parameters using **training rows only**. Use validation rows only to
   choose temperature dependence, parameter bounds, regularization, NRTL
   non-randomness treatment, optimizer restart, and early stopping. Never use
   the test rows to initialize, fit, select, or reject a parameter set.
2. Use the same supervised targets and units as ThermoFormer. A practical
   weighted objective contains the P or T residual and every independent vapor
   composition residual. Determine the weights from the training set only and
   freeze them before evaluation.
3. Fit one shared, ordered parameterization per molecular pair. Apply those
   binary parameters unchanged in ternary mixtures. Store component order and
   both directional parameters explicitly.
4. Select parameter complexity on validation data. NIST warns that the
   temperature-dependence form must avoid both underfitting and unphysical
   overfitting; its supported common form includes constant, inverse-T,
   logarithmic-T, and higher terms
   ([NIST TDE activity-model documentation](https://trc.nist.gov/TDE/TDE_Help/ActivityCoefficientModels.htm)).
   With limited data, begin with the smallest identifiable form.
5. For UNIQUAC, obtain and record the pure-component `r` and `q` values and
   their provenance. Do not tune them on the test set.
6. Evaluate every test row, including failures. Report coverage, optimizer/solver
   failure rate, and nonphysical rate in addition to the existing four metric
   groups: isothermal P, isothermal y, isobaric T, and isobaric y, each with
   MAE, RMSE, and R². Retain the project's existing convention for aggregating
   independent composition components.
   An unavailable pair is **missing coverage**, not an ideal-mixture prediction
   and not a row that may silently be removed.
7. If seeds change the data split, refit each baseline on each seed's training
   partition and report mean plus standard deviation. If seeds change only
   neural initialization while the split is fixed, fit the deterministic
   classical baseline once and do not present duplicated values as independent
   runs.

NIST uses NRTL, UNIQUAC, Wilson, and related activity-coefficient models to fit
phase-equilibrium data and emphasizes model appropriateness, pure-property
consistency, and thermodynamic consistency tests
([NIST TDE overview](https://trc.nist.gov/tde.html),
[dynamic-evaluation paper](https://www.nist.gov/system/files/documents/mml/acmd/TDE-7-0-2012.pdf)).
Wilson should be flagged as structurally unsuitable for systems exhibiting
liquid-liquid splitting; NIST explicitly excludes it for LLE systems. This does
not prevent reporting its VLE result, but the limitation should not be hidden.

### Two comparisons that answer different questions

Keep these results separate:

| Comparison | Pure-property and solver treatment | Scientific question |
|---|---|---|
| Controlled interaction comparison | Same vapor-pressure inputs, vapor-phase convention, and VLE solver for all models | Is ThermoFormer's learned excess-Gibbs/activity-coefficient representation better? |
| Conventional end-to-end baseline | Each classical workflow uses a declared external pure-property source and standard implementation | Is the complete ThermoFormer workflow better than a conventional engineering workflow? |

Using literature or ChemSep binary parameters is a third, externally trained
condition. It is legitimate only if parameter provenance, coverage, and database
version are reported. It must not be mixed with train-only regression in one
row, because published parameters may already incorporate measurements closely
related to the test systems.

### Binary-to-ternary and unseen-component claims

For a clean binary-to-ternary study, regress each pair only on permitted binary
training data and transfer the fitted pair parameters unchanged to ternary test
mixtures. This directly tests the classical models' standard binary-parameter
extension against ThermoFormer.

Pair-specific NRTL, Wilson, and UNIQUAC cannot make a genuine unseen-pair
prediction when no parameter source is available. For `unseen_component`,
report parameter coverage and label uncovered rows as not applicable. A
literature-parameter database and COSMO-RS can be additional predictive
baselines, but neither should be described as trained on exactly the same data
as ThermoFormer.

## Phase-diagram case studies

Select cases before viewing test errors. A defensible set contains three to five
binary systems chosen from metadata available in training/validation data, for
example:

- one near-ideal system;
- one strongly non-ideal or azeotropic system;
- one hydrogen-bonding or associating system;
- one chemically dissimilar system;
- optionally one temperature/pressure extrapolation case.

If ternary diagrams are central to the paper, add one pre-specified ternary
system with complete pair coverage. For each case, show the same experimental
points and the full model-generated curve on a common composition grid:
isothermal P-x-y when T is fixed, or isobaric T-x-y when P is fixed. Report the
case-selection rule, dataset row identifiers, model version, parameter source,
and solver failures. Do not choose cases because ThermoFormer happens to win on
their test labels.

## Can COSMO-RS be implemented with Python?

**Yes, but there are two materially different routes.** Also, COSMO-RS is best
described as a quantum-chemistry-informed statistical-thermodynamic model: the
expensive quantum calculation is normally performed once per pure compound to
generate a screened surface, after which mixture calculations are fast.

### Open-source route: openCOSMO-RS

TU Hamburg maintains a
[fully Python openCOSMO-RS implementation](https://github.com/TUHH-TVT/openCOSMO-RS_py)
and a faster
[C++ implementation with Python bindings](https://github.com/TUHH-TVT/openCOSMO-RS_cpp).
The Python implementation returns activity-coefficient quantities that can be
coupled to ThermoFormer's existing bubble-point solver. Required quantum inputs
can be generated with the official
[RDKit/ORCA conformer pipeline](https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline),
which currently requires ORCA 6.1+, xTB, and the prescribed CPCM workflow.
ORCA's official documentation includes an openCOSMO-RS interface and warns that
changing the quantum-chemical level used for parameterization is strongly
discouraged; its documented ORCA 6.0 parameterization uses
BP86/def2-TZVPD
([ORCA tutorial](https://www.faccts.de/docs/orca/6.0/tutorials/prop/cpcm.html)).

This route is scriptable, but it is not a descriptor-only calculation: every
component needs a compatible `.orcacosmo`/surface file, conformer policy, atomic
radii, quantum level, and openCOSMO-RS parameter set. ORCA itself must be
downloaded under the applicable academic or commercial agreement
([official installation guidance](https://www.faccts.de/docs/orca/6.1/tutorials/first_steps/install.html)).

Licensing requires care. GitHub identifies the TUHH Python and conformer
repositories as LGPL-3.0 and their `LICENSE` files contain LGPL-3.0, while the
Python repository README currently says GPL-2.0. Record the exact commit and
follow the repository `LICENSE` file; resolve the upstream inconsistency before
redistributing a bundled copy.

### Commercial route: AMS COSMO-RS

The Amsterdam Modeling Suite provides an official Python workflow through PLAMS
and pyCRS. A `CRSJob` can be created and executed from Python, and the result is
returned as a Python object
([official scripting documentation](https://www.scm.com/doc/COSMO-RS/PLAMS_COSMO-RS_scripting.html)).
Each compound requires a reusable `.coskf` or `.compkf` file. SCM recommends the
DFT-derived `.coskf` for accuracy; `.compkf` is a faster group/QSPR estimate.
The `.coskf` generation itself can also be automated with the official PLAMS
recipe
([compound-generation API](https://www.scm.com/doc.2026/plams/interfaces/recipes.html)).

This is Python orchestration of the licensed AMS engines, not a standalone
open-source Python reimplementation. The COSMO-RS module and the ADF module used
to generate new DFT surfaces are separately licensed capabilities
([SCM licensing information](https://www.scm.com/pricing-and-licensing/),
[SCM module scope](https://www.scm.com/faq/)). Freeze the AMS release, COSMO-RS
parameterization, compound database, surface-generation level, and conformer
selection when reporting results.

[`BIOVIA COSMOtherm`](https://www.3ds.com/products/biovia/cosmo-rs/cosmotherm)
is another commercial COSMO-RS engine. A Python script can
write its input, invoke its command-line executable, and parse output, but that
does not make the engine or parameter files open-source. Treat COSMOtherm as a
licensed external executable and report its release, parameterization, COSMO
file source, and license-dependent reproducibility separately.

## Proposed order of work

1. Implement the three classical equations and shared VLE evaluation adapter.
   Run train-fitted NRTL/Wilson/UNIQUAC first on the existing within-system
   state-generalization protocols. Separately audit external pair-parameter
   coverage before deciding whether a system-disjoint
   `overall_binary_ternary` table is possible.
2. Validate equations against published `thermo` examples and synthetic
   limiting cases before fitting project data.
3. Fit and evaluate the full test set, then generate the pre-registered phase
   diagrams from the already fitted models.
4. Add openCOSMO-RS only after auditing compound coverage and the cost of
   generating compatible ORCA files. It is the more relevant baseline for
   unseen-component prediction, but substantially more expensive to prepare.
5. Add commercial AMS COSMO-RS or COSMOtherm only if a valid license and a
   redistributable/provenance-safe compound-file workflow are available.

The minimum publishable comparison is therefore full-test NRTL/Wilson/UNIQUAC
metrics on a protocol where their parameters are obtained without test leakage,
plus several pre-specified phase diagrams. COSMO-RS is feasible from Python,
but should be a separately scoped benchmark rather than a small add-on to the
classical-model script.
