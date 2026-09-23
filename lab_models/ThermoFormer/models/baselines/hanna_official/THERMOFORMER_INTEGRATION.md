# HANNA official pretrained assets

This directory is a minimal, byte-preserving subset of the MIT-licensed
[`marco-hoffmann/HANNA`](https://github.com/marco-hoffmann/HANNA) repository at
commit `6fe873ca1a92c306eb9b5be3e9adb2ebbbb95365`.

It contains the official multicomponent model source, ten binary-trained ensemble
weights, the bundled ChemBERTa encoder, and the published temperature and embedding
scalers. ThermoFormer does not retrain or alter these files. The adapter supplies
the official activity coefficients to the same train-only vapor-pressure treatment
and differentiable isothermal/isobaric VLE solver used by the other thermodynamic
decoder baselines.

ThermoFormer executes these assets in the ggnn39 compatibility environment. This is
distinct from an exact replay of the upstream version-pinned environment; formal
manifests record the actual Python, PyTorch, Transformers, tokenizers, pandas and
related package versions.

The official training-system inventory is not published. Consequently, HANNA
results are labelled `external_pretrained_overlap_unknown` and are not interpreted
as a same-information comparison with models trained on the registered splits.
