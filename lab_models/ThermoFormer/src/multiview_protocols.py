"""Compatibility adapter for molecular-representation protocols."""
from . import _compat
from .thermoformer.protocols import representations as _implementation
_compat.export_module(globals(), _implementation)
