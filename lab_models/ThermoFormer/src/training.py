"""Compatibility adapter; use :mod:`src.thermoformer.training`."""
from . import _compat
from .thermoformer.training import supervised as _implementation
_compat.export_module(globals(), _implementation)
