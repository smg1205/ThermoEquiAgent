from .checkpoint_model import build_model as legacy_build
from .hybrid_binodal_model import HybridBinodalLLE

def build_model(spec):
    if spec.get('decoder')=='hybrid_binodal':
        return HybridBinodalLLE(**{k:v for k,v in spec.items() if k!='decoder'})
    return legacy_build(spec)
