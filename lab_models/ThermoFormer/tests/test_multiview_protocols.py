import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_multiview_suite import release_accelerator_memory
from src.config import load_experiment_config
from src.multiview_protocols import MULTIVIEW_VARIANTS, PREDICTIVE_PROTOCOLS


class MultiViewProtocolTests(unittest.TestCase):
    def test_accelerator_cleanup_is_safe_without_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            release_accelerator_memory()

    def test_retained_c1_view_ablations_are_concrete_and_distinct(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            tuple(MULTIVIEW_VARIANTS),
            ("v1_rdkit_only", "v3_functional_group_only", "v4_rdkit_unimol_naive"),
        )
        signatures = set()
        for variant in MULTIVIEW_VARIANTS.values():
            config = load_experiment_config(root / variant.config)
            signature = (
                config.encoder.representation,
                config.encoder.fusion_mode,
                config.encoder.use_rdkit_descriptors,
                config.encoder.use_unimol,
                config.encoder.use_functional_groups,
            )
            self.assertNotIn(signature, signatures)
            signatures.add(signature)
        self.assertEqual(PREDICTIVE_PROTOCOLS, ("overall_binary_ternary",))


if __name__ == "__main__":
    unittest.main()
