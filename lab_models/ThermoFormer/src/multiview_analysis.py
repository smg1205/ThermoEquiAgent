"""Compatibility adapter for molecular-view ablation analysis."""
from . import _compat
from .thermoformer.evaluation import ablation as _implementation
_compat.export_module(globals(), _implementation)
