"""Compatibility adapter; use :mod:`src.thermoformer.data.splitting`."""
from . import _compat
from .thermoformer.data import splitting as _implementation
_compat.export_module(globals(), _implementation)
