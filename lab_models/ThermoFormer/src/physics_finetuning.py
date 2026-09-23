"""Compatibility adapter; use :mod:`src.thermoformer.training.fugacity_finetuning`."""
from . import _compat
from .thermoformer.training import fugacity_finetuning as _implementation
_compat.export_module(globals(), _implementation)
