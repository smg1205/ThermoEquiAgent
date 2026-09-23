"""Compatibility adapter for the C1 generalization report builder."""
from . import _compat
from .thermoformer.reporting import c1_generalization as _implementation
_compat.export_module(globals(), _implementation)
