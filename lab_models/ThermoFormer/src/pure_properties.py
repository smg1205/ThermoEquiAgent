"""Compatibility adapter; use :mod:`src.thermoformer.thermodynamics.vapor_pressure`."""
from . import _compat
from .thermoformer.thermodynamics import vapor_pressure as _implementation
_compat.export_module(globals(), _implementation)
