"""Compatibility adapter; use :mod:`src.thermoformer.protocols.runner`."""
from . import _compat
from .thermoformer.protocols import runner as _implementation
_compat.export_module(globals(), _implementation)
