"""Compatibility adapter; use :mod:`src.thermoformer.protocols.generalization`."""
from . import _compat
from .thermoformer.protocols import generalization as _implementation
_compat.export_module(globals(), _implementation)
