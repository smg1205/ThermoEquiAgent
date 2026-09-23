# Machine-learning VLE baselines

This experiment reproduces ten machine-learning and learned-parameter baselines under the frozen
ThermoFormer overall predictive-performance assignments. Each baseline is assigned
to exactly one benchmark from its native component support:

1. binary training to binary test (`overall_binary`);
2. ternary training to ternary test for a ternary-only model (the ternary subset of
   `overall_binary_ternary`);
3. joint binary and ternary training to joint binary and ternary test
   (`overall_binary_ternary`).

Descriptor ANN, SMILES-RNN, UALF-GNN, GDI-GNN and GE-GNN therefore use only
binary training to binary test. SolvGNN uses the joint benchmark.
HANNA is a separately labelled official-pretrained-to-joint-test comparison because
its published weights were trained on an external binary corpus. TeNNet-SAC and
SPT-NRTL are separately labelled fixed-external-to-joint-test comparisons: TeNNet-SAC
uses the author's experimental tuned ensemble without retraining, whereas SPT-NRTL
performs exact canonical-SMILES lookup in the fixed public parameter database. No
currently registered baseline is ternary-only.

`SPT-NRTL adapted (ThermoFormer-train)` is a distinct same-data comparison. For each
registered seed, binary NRTL parameter labels are fitted from binary training systems
only. A character-level molecular-pair Transformer is trained on those labels and its
checkpoint is selected using validation-pair labels only. At test time it predicts
parameters from SMILES; ternary states combine the three predicted binary pairs in the
standard multicomponent NRTL equation. It does not use author weights, confidential
COSMO pretraining data, database parameters, or test-row fitting, and is not presented
as the official SPT-NRTL model.

`HANNA adapted (joint-trained residual head)` and `TeNNet-SAC adapted
(joint-trained residual head)` are separate adaptation experiments. Their official
pretrained molecular models remain frozen. A 1,345-parameter shared,
permutation-equivariant residual head corrects the official log activity coefficients;
it is trained jointly on registered binary and ternary training rows, selected only on
validation, and evaluated once on the joint test partition. These rows use the same
split but are not from-scratch same-data baselines because the frozen representations
retain external pretraining.
All trainable models use the registered train, validation and test sample IDs for
seeds 0--4. Validation is the only model-selection partition. Native task boundaries
are preserved: unsupported directions, component counts and temperatures are N/A.
Any temperature-conditioned or task-expanded derivative must carry an `adapted`
label and is not part of the native comparison.

The seed-0 diagnostic remains isolated. Formal execution trains on the registered
training partition, selects checkpoints with validation only, and evaluates the
selected checkpoint once on test. Unsupported directions, fixed-temperature rows
outside 298.15 +/- 0.5 K, and unavailable upstream assets remain explicit N/A or
blocked cells and contribute to valid-coverage reporting.

Detailed source and license auditing is recorded in
[`docs/ml_vle_baseline_source_audit.md`](../../../docs/ml_vle_baseline_source_audit.md).
The TeNNet-SAC and SPT-NRTL API, equation, asset and license audit is recorded in
[`docs/tennetsac_sptnrtl_source_audit.md`](../../../docs/tennetsac_sptnrtl_source_audit.md).

HANNA is evaluated with the unchanged official ten-model ensemble bundled from the
pinned MIT-licensed source revision. Its binary-trained Muggianu projection is used
for ternary states. Because the authors do not publish a training-system inventory,
HANNA is labelled as an external-pretrained comparison with unknown overlap rather
than a same-split confirmatory baseline. Execution uses the project's ggnn39
compatibility environment rather than the upstream version-pinned environment; exact
upstream-environment numerical parity is therefore not claimed. TeNNet-SAC is marked
with unknown external-training overlap because the released inventory is not sufficient
to establish disjointness. SPT-NRTL systems missing any constituent binary pair are
reported unavailable; no test-row fitting or parameter imputation is permitted.
The SPT-NRTL parameter repository does not declare a license at the frozen revision;
its results therefore carry an explicit publication-use license-review warning.
