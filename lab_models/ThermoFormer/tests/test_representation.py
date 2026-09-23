import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import EncoderConfig
from src.representation import (
    FunctionalGroupEncoder,
    HybridMolecularEncoder,
    LegacyFixedScaleRDKit2DEncoder,
    RDKit2DEncoder,
    UniMolV2Encoder,
    build_molecular_encoder,
    encoder_cache_filename,
    functional_group_vocabulary_path,
)


class FakeUniMol:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def get_repr(self, smiles, return_atomic_reprs=False):
        self.calls.append((list(smiles), return_atomic_reprs))
        return {"cls_repr": np.arange(len(smiles) * 4, dtype=np.float32).reshape(len(smiles), 4)}


class UniMolV2EncoderTests(unittest.TestCase):
    def test_uses_cls_repr_dictionary_and_reuses_cache(self) -> None:
        backend = FakeUniMol()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "unimolv2.npz"
            encoder = UniMolV2Encoder(cache, batch_size=2, backend_factory=lambda **_: backend)

            first = encoder.encode(["O", "CCO", "O"])
            second = encoder.encode(["CCO", "O"])

        self.assertEqual(set(first), {"CCO", "O"})
        self.assertEqual(first["CCO"].shape, (4,))
        self.assertEqual(len(backend.calls), 1)
        self.assertFalse(backend.calls[0][1])
        np.testing.assert_array_equal(first["O"], second["O"])

    def test_rdkit_ablation_encoder_is_deterministic_finite_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "rdkit_2d.npz"
            encoder = RDKit2DEncoder(cache)

            first = encoder.encode(["O", "CCO", "O"])
            second = RDKit2DEncoder(cache).encode(["CCO", "O"])

        self.assertEqual(set(first), {"CCO", "O"})
        self.assertEqual(first["O"].shape, first["CCO"].shape)
        self.assertGreater(first["O"].shape[0], 10)
        self.assertTrue(np.isfinite(first["CCO"]).all())
        np.testing.assert_array_equal(first["O"], second["O"])

    def test_legacy_a1_fixed_scaling_remains_exactly_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = RDKit2DEncoder(root / "raw.npz").encode(["CCO"])["CCO"]
            legacy = LegacyFixedScaleRDKit2DEncoder(root / "legacy.npz")
            scaled = legacy.encode(["CCO"])["CCO"]
            expected = raw / np.asarray(
                [scale for _, scale in RDKit2DEncoder._DESCRIPTORS], dtype=np.float32
            )
            np.testing.assert_allclose(scaled, expected, rtol=0.0, atol=0.0)
            cached = LegacyFixedScaleRDKit2DEncoder(root / "legacy.npz").encode(["CCO"])
            np.testing.assert_array_equal(scaled, cached["CCO"])

        config = EncoderConfig(representation="rdkit_2d_legacy_fixed")
        self.assertEqual(encoder_cache_filename(config), "rdkit_2d_scaled24_v1.npz")

    def test_functional_group_encoder_exposes_chemical_fragment_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = FunctionalGroupEncoder(Path(directory) / "functional_groups.npz")
            features = encoder.encode(["CCO", "CC(=O)O", "c1ccccc1"])

        names = encoder.feature_names
        self.assertGreater(len(names), 20)
        self.assertGreater(features["CCO"][names.index("alcohol")], 0.0)
        self.assertGreater(features["CC(=O)O"][names.index("carboxylic_acid")], 0.0)
        self.assertGreater(features["c1ccccc1"][names.index("aromatic_ring")], 0.0)

    def test_audited_functional_group_assignment_and_empty_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = FunctionalGroupEncoder(
                Path(directory) / "functional_groups.npz",
                vocabulary_path=functional_group_vocabulary_path(),
            )
            features = encoder.encode(["CCO", "CC(=O)O", "c1ccccc1", "C"])
        names = encoder.feature_names
        self.assertGreater(features["CCO"][names.index("alcohol")], 0.0)
        self.assertGreater(features["CC(=O)O"][names.index("carboxylic_acid")], 0.0)
        self.assertGreater(features["c1ccccc1"][names.index("aromatic_ring")], 0.0)
        self.assertEqual(float(features["C"].sum()), 0.0)

    def test_hybrid_encoder_combines_all_three_branches_and_reuses_cache(self) -> None:
        backend = FakeUniMol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = HybridMolecularEncoder(
                root / "hybrid.npz",
                model_size="84m",
                batch_size=2,
                backend_factory=lambda **_: backend,
            )
            first = encoder.encode(["CCO", "O"])
            block_sizes = encoder.feature_block_sizes
            cached = HybridMolecularEncoder(
                root / "hybrid.npz",
                model_size="84m",
                backend_factory=lambda **_: self.fail("combined cache was not reused"),
            ).encode(["O", "CCO"])

        self.assertEqual(tuple(block_sizes), ("rdkit_2d", "unimol_v2", "functional_groups"))
        self.assertEqual(block_sizes["rdkit_2d"], len(RDKit2DEncoder._DESCRIPTORS))
        self.assertEqual(block_sizes["unimol_v2"], 4)
        self.assertGreater(block_sizes["functional_groups"], 20)
        self.assertEqual(first["CCO"].shape, (sum(block_sizes.values()),))
        np.testing.assert_array_equal(first["O"], cached["O"])

    def test_encoder_factory_uses_branch_specific_cache_identity(self) -> None:
        full = EncoderConfig(representation="multiview", fusion_mode="naive")
        without_unimol = replace(full, use_unimol=False)
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            encoder = build_molecular_encoder(
                full,
                cache_root / encoder_cache_filename(full),
                use_cuda=False,
                backend_factory=lambda **_: FakeUniMol(),
            )

        self.assertIsInstance(encoder, HybridMolecularEncoder)
        self.assertNotEqual(
            encoder_cache_filename(full),
            encoder_cache_filename(without_unimol),
        )
        self.assertIn("rdkit", encoder_cache_filename(full))
        self.assertIn("unimol", encoder_cache_filename(full))
        self.assertIn("functional", encoder_cache_filename(full))


if __name__ == "__main__":
    unittest.main()
