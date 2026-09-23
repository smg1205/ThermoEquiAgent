# Models and checkpoints

ThermoFormer model implementations are located in `src/thermoformer/models/` for VLE and `src/thermoformer/lle_tp/` for LLE.

`registry.json` is the authoritative catalog for the validation-selected final checkpoints used by predictive-performance and generalization benchmarks. `checkpoint_catalog.csv` provides the same 105 records in tabular form:

- 50 VLE checkpoints: 10 training protocols × seeds 0--4.
- 55 LLE checkpoints: 11 training protocols × seeds 0--4.

Every entry records task, protocol, seed, repository-relative path, SHA-256 digest, file size, and availability. Mixed-training checkpoints are shared by their binary, ternary, and joint evaluation subsets.

Intermediate stages, optimizer states, and checkpoints dedicated to comparison, ablation, interpretability, or separation-design studies are not distributed. Their code, configurations, predictions, metrics, and figures remain available. External reference-model weights must be obtained from their cited upstream sources.

Validate every distributed checkpoint with:

```powershell
python scripts/validate_checkpoint_catalog.py --load
```

The command verifies the 10 VLE and 11 LLE protocol groups, seeds 0--4, hashes, checkpoint mappings, and deserialization on CPU.

## Reference baselines

External baseline implementations and encoder assets are organized under `models/baselines/`. The HANNA adapter uses `models/baselines/hanna_official/`; its optional pretrained ensemble weights are not distributed because the repository retains checkpoints only for ThermoFormer predictive-performance and generalization benchmarks. Obtain those weights from the cited upstream release before reproducing the HANNA comparison.
