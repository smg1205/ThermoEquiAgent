# Scientific provenance

Formal results bind the resolved configuration, dataset workbooks, split
assignment, molecular-feature definitions and cache subset, runtime environment,
source commit, checkpoint, predictions, and metrics through recorded SHA-256
digests. Text artifacts are hashed after UTF-8 and LF normalization; checkpoints
are hashed byte-for-byte and stored with Git LFS.

Seed manifests are completed only after all material artifacts are present.
Five-seed aggregation validates seed identity, protocol identity, environment,
and every recorded artifact before publishing aggregate tables. Smoke runs use
isolated diagnostic roots and are never accepted as formal aggregate inputs.

Validation selects checkpoints. Test metrics are generated only after selection
and are not used to tune configurations or loss weights.
