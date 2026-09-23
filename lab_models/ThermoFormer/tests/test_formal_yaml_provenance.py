from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.thermoformer.protocols.runner import _config_source_paths


class FormalYamlProvenanceTests(unittest.TestCase):
    def test_yaml_inheritance_chain_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.yaml"
            child = root / "child.yaml"
            parent.write_text("name: parent\n", encoding="utf-8")
            child.write_text("extends: parent.yaml\nname: child\n", encoding="utf-8")

            paths = _config_source_paths(child)

        self.assertEqual(paths, (parent.resolve(), child.resolve()))


if __name__ == "__main__":
    unittest.main()
