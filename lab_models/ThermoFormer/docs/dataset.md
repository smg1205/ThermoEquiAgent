# Dataset and split protocol

The release dataset consists of one binary and one ternary English workbook in
`dataset/`. The loader records source files, row counts, unit conversions,
quality decisions, experiment-direction inference, deduplication, pressure
filtering, and pure-endpoint eligibility.

Quality codes follow the workbook definition: `1` passed, `0` failed, and `-1`
indeterminate. Failed records are excluded by default; indeterminate records
receive a configurable weight.

System-disjoint protocols use canonical unordered identities, so A--B and B--A
cannot cross partitions. Unseen-component protocols additionally exclude the
held-out molecular identities from training, and binary-to-ternary protocols
define transfer coverage only from binary systems present in the training split.

State-generalization protocols have a different scientific purpose: they retain
the same molecular system across partitions and hold out composition, temperature,
or pressure states within that system. Interpolation points are bracketed by
training states; extrapolation points lie beyond the corresponding training and
validation ranges. Pure-component reference systems remain on the training side
without violating these state constraints.

RDKit descriptor statistics are fitted only from molecules present in the
training partition. Uni-Mol, RDKit, functional-group, scaler, dataset, and split
digests are stored with formal results.
