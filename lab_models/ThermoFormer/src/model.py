"""Compatibility adapter; use :mod:`src.thermoformer.models`."""
from . import _compat
from .thermoformer.models import thermoformer as _implementation
_compat.export_module(globals(), _implementation)
