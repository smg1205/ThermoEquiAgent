"""Compatibility adapter for interpretability primitives."""
from .. import _compat
from ..thermoformer.interpretability import core as _implementation
_compat.export_module(globals(), _implementation)
