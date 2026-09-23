"""Compatibility adapter; use :mod:`src.thermoformer.reporting`."""
from . import _compat
from .thermoformer.reporting import experiment_results as _implementation
_compat.export_module(globals(), _implementation)
