"""Compatibility adapter for interpretability sample selection."""
from .. import _compat
from ..thermoformer.interpretability import selection as _implementation
_compat.export_module(globals(), _implementation)
