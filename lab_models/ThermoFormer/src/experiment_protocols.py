"""Compatibility adapter; use :mod:`src.thermoformer.protocols.registry`."""
from . import _compat
from .thermoformer.protocols import registry as _implementation
_compat.export_module(globals(), _implementation)
