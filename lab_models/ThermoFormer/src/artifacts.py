"""Compatibility adapter; use :mod:`src.thermoformer.reporting.artifacts`."""
from . import _compat
from .thermoformer.reporting import artifacts as _implementation
_compat.export_module(globals(), _implementation)
