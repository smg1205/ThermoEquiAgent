"""Compatibility adapter; use :mod:`src.thermoformer.thermodynamics.vle_solver`."""
from . import _compat
from .thermoformer.thermodynamics import vle_solver as _implementation
_compat.export_module(globals(), _implementation)
