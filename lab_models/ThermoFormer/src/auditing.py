"""Compatibility adapter; use :mod:`src.thermoformer.data.auditing`."""
from . import _compat
from .thermoformer.data import auditing as _implementation
_compat.export_module(globals(), _implementation)
