"""Compatibility adapter; use :mod:`src.thermoformer.data`."""
from . import _compat
from .thermoformer.data import loading as _implementation
_compat.export_module(globals(), _implementation)
