# VLE interpretability

The registered analysis explains the validation-selected ThermoFormer models trained on the current VLE dataset. It reports exact grouped Shapley attribution across the RDKit, Uni-Mol v2, and SMARTS views; cross-view interactions; named-feature occlusion; learned pair contributions; and held-out composition paths.

Molecular structure heatmaps map local SMARTS-view occlusion effects back to matched atoms and bonds. They do not assign atom-level meaning to the pooled Uni-Mol embedding or global RDKit descriptors. All attribution values describe model dependence rather than causal molecular mechanisms or experimental interaction energies.

Run the complete five-seed analysis with:

```bash
python scripts/benchmarks/vle/generate_interpretability.py --device cuda
```

Use `--smoke --device cpu` for a short installation check. Formal outputs are written to `experiments/vle/interpretability/molecular_interactions/`, including CSV source tables, SVG/PDF/TIFF/PNG figures, English captions, and a hash manifest.
