"""Compatibility adapter for thermodynamic-consistency diagnostics."""
from .. import _compat
from ..thermoformer.evaluation import thermodynamic_consistency as _implementation
_compat.export_module(globals(), _implementation)
