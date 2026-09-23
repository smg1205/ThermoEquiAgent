"""Compatibility adapter; use :mod:`src.thermoformer.reporting.aggregation`."""
from . import _compat
from .thermoformer.reporting import aggregation as _implementation
_compat.export_module(globals(), _implementation)
