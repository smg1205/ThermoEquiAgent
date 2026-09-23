"""Public inference entry point: molecular identities, kelvin and kPa only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .checkpoint_model_v2 import build_model
from .solver_certified import solve_many
from .campaign import seed_all
from ..configuration import EncoderConfig
from ..features import build_molecular_encoder

def predict(checkpoint,smiles,temperature_k,pressure_kpa,device='cpu',curve_points=24):
    if len(smiles) not in (2,3): raise ValueError('Provide two or three distinct components')
    if not np.isfinite(temperature_k) or temperature_k<=0 or not np.isfinite(pressure_kpa) or pressure_kpa<=0:
        raise ValueError('Temperature (K) and pressure (kPa) must be positive and finite')
    canon=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(s)
        if mol is None: raise ValueError('Invalid SMILES: '+s)
        canon.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
    if len(set(canon))!=len(canon): raise ValueError('Duplicate components')
    order=np.argsort(canon); canonical=[canon[i] for i in order]
    seed_all(0); checkpoint=Path(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=build_model(payload['spec']).to(device).double(); model.load_state_dict(payload['model']); model.eval()
    features={k:v.numpy() for k,v in payload['feature_map'].items()}
    missing=[s for s in canonical if s not in features]
    if missing:
        metadata=json.loads(checkpoint.with_name('features.json').read_text(encoding='utf8'))
        config=EncoderConfig(representation='multiview',fusion_mode='naive')
        encoder=build_molecular_encoder(config,checkpoint.parent/'inference_cache/hybrid_rdkit_unimol_84m_functional_v1.npz',use_cuda=device.startswith('cuda'))
        raw=encoder.encode(missing); scaler=metadata['rdkit_scaler']; d=len(scaler['mean'])
        for s,vector in raw.items():
            vector=vector.copy(); vector[:d]=(vector[:d]-np.array(scaler['mean']))/np.array(scaler['std']); features[s]=vector
    n=len(canonical); matrix=np.zeros((1,3,len(features[canonical[0]])),np.float32)
    matrix[0,:n]=np.stack([features[s] for s in canonical]); mask=np.zeros((1,3),np.float32); mask[:,:n]=1
    # Training states are float32. Canonicalize public scalar inputs through
    # that representation before deterministic float64 certification.
    tensor=lambda a:torch.as_tensor(np.asarray(a,dtype=np.float32),device=device).double()
    with torch.no_grad():
        ctx=model.context(tensor(matrix),tensor([[temperature_k]]),tensor([[pressure_kpa]]),tensor(mask))
        proposal=model.propose(ctx,curve_points)
    if n==2: proposal=proposal[:,:1]
    result=solve_many(model,ctx,proposal)[0]
    inverse=np.argsort(order)
    for p in result['pairs']: p['endpoints']=np.array(p['endpoints'])[:,inverse].tolist()
    result.update({'smiles':smiles,'temperature_k':temperature_k,'pressure_kpa':pressure_kpa,'composition_basis':'mole_fraction','input_component_order_preserved':True,'stability_check':'finite simplex-grid TPD screen'})
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--smiles',nargs='+',required=True)
    ap.add_argument('--temperature-k',type=float,required=True); ap.add_argument('--pressure-kpa',type=float,required=True); ap.add_argument('--device',default='cpu')
    args=ap.parse_args(); print(json.dumps(predict(args.checkpoint,args.smiles,args.temperature_k,args.pressure_kpa,args.device),indent=2))

if __name__=='__main__': main()
