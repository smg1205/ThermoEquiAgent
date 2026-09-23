"""Compatibility adapter; use :mod:`src.thermoformer.training.losses`."""
from . import _compat
from .thermoformer.training import losses as _implementation
_compat.export_module(globals(), _implementation)
