"""Utilities for preserving historical ``src.*`` imports during migration."""

import sys
from types import ModuleType
from typing import Any


def export_module(namespace: dict[str, Any], implementation: ModuleType) -> None:
    """Alias a historical module name to its canonical implementation module."""
    namespace.update(
        {
            name: getattr(implementation, name)
            for name in dir(implementation)
            if not name.startswith("__")
        }
    )
    historical_name = str(namespace["__name__"])
    sys.modules[historical_name] = implementation
    parent_name, _, child_name = historical_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, implementation)
