"""Explicit checkpoint decoder dispatch; legacy checkpoints retain their model."""
from .model import TPLLE
from .binodal_model import BinodalLLE

def build_model(spec):
    spec=dict(spec);decoder=spec.pop('decoder',None)
    if decoder is None:return TPLLE(**spec)
    if decoder=='parallel_binodal':return BinodalLLE(**spec)
    raise ValueError('Unknown LLE decoder: '+str(decoder))
