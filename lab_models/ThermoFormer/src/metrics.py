"""Compatibility adapter; use :mod:`src.thermoformer.evaluation.metrics`."""
from . import _compat
from .thermoformer.evaluation import metrics as _implementation
_compat.export_module(globals(), _implementation)
