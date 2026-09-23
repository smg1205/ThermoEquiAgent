"""Compatibility adapter for interaction-architecture protocols."""
from . import _compat
from .thermoformer.protocols import interactions as _implementation
_compat.export_module(globals(), _implementation)
