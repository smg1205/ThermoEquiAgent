# Interaction-architecture ablation

These three variants correspond exactly to the multicomponent-interaction block
in manuscript Table 2:

- `vanilla_transformer/`
- `chemical_interaction_bias/`
- `context_pair_without_attention_bias/`

Every variant uses the same full three-view molecular representation, the fixed
`overall_binary_ternary` split, and seeds 0--4. The vanilla Transformer is the
architecture retained by the final model.
