import unittest

from src.multiview_analysis import gate_statistics


class MultiViewAnalysisTests(unittest.TestCase):
    def test_gate_statistics_do_not_assume_a_dominant_view(self) -> None:
        records = [
            {
                "protocol": "unseen_component",
                "seed": 0,
                "component_count": 2,
                "generalization": "unseen_component",
                "chemical_family_pair": "alcohol <-> ketone",
                "composition_region": "interior",
                "rdkit": 0.2,
                "unimol": 0.3,
                "functional_group": 0.5,
            },
            {
                "protocol": "unseen_component",
                "seed": 0,
                "component_count": 2,
                "generalization": "unseen_component",
                "chemical_family_pair": "alcohol <-> ketone",
                "composition_region": "edge",
                "rdkit": 0.6,
                "unimol": 0.3,
                "functional_group": 0.1,
            },
        ]
        rows = gate_statistics(records)
        global_rows = {row["view"]: row for row in rows if row["scope"] == "global"}
        self.assertAlmostEqual(global_rows["rdkit"]["mean_weight"], 0.4)
        self.assertAlmostEqual(global_rows["functional_group"]["mean_weight"], 0.3)
        self.assertEqual(global_rows["unimol"]["pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
