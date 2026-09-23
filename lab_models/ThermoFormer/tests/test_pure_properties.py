import json
import tempfile
import unittest
from pathlib import Path

from src.pure_properties import (
    CORRELATION_PARAMETER_COUNT,
    CORRELATION_TYPE_DIPPR101,
    load_pure_property_catalog,
)


class PurePropertyCatalogTests(unittest.TestCase):
    def test_catalog_normalizes_legacy_antoine_entry_and_marks_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "antoine.json"
            path.write_text(
                json.dumps(
                    {
                        "O": {
                            "a": 8.07131,
                            "b": 1730.63,
                            "c": 233.426,
                            "minimum_temperature_k": 274.15,
                            "maximum_temperature_k": 373.15,
                        }
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_pure_property_catalog(path)
            parameters = catalog.parameters_for(("O", "CCO"))

        self.assertEqual(parameters.shape, (2, CORRELATION_PARAMETER_COUNT))
        self.assertEqual(parameters[0, -1], 1.0)
        self.assertEqual(parameters[1, -1], 0.0)
        self.assertEqual(catalog.covered_smiles, frozenset({"O"}))

    def test_catalog_supports_typed_dippr101_entry_with_declared_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pure_properties.json"
            path.write_text(
                json.dumps(
                    {
                        "O": {
                            "type": "dippr101",
                            "a": 11.512925464970229,
                            "b": 0.0,
                            "c": 0.0,
                            "d": 0.0,
                            "e": 1.0,
                            "pressure_unit": "Pa",
                            "minimum_temperature_k": 250.0,
                            "maximum_temperature_k": 450.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_pure_property_catalog(path)
            parameters = catalog.parameters_for(("O",))

        self.assertEqual(parameters[0, 0], CORRELATION_TYPE_DIPPR101)
        self.assertEqual(parameters[0, -1], 1.0)

    def test_catalog_rejects_invalid_temperature_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "antoine.json"
            path.write_text(
                json.dumps(
                    {
                        "O": {
                            "a": 8.0,
                            "b": 1700.0,
                            "c": 230.0,
                            "minimum_temperature_k": 400.0,
                            "maximum_temperature_k": 300.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "temperature range"):
                load_pure_property_catalog(path)


if __name__ == "__main__":
    unittest.main()
