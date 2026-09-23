"""Compatibility adapter; use :mod:`src.thermoformer.features`."""
from . import _compat
from .thermoformer.features import fusion as _implementation
_compat.export_module(globals(), _implementation)
