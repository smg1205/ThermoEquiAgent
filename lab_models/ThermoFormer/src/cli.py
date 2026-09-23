"""Compatibility command interface; use :mod:`src.thermoformer.cli`."""
from . import _compat
from .thermoformer import cli as _implementation
_compat.export_module(globals(), _implementation)
