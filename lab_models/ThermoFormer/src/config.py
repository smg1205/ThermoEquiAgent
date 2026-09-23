"""Compatibility adapter; use :mod:`src.thermoformer.configuration`."""
from . import _compat
from .thermoformer import configuration as _implementation
_compat.export_module(globals(), _implementation)
