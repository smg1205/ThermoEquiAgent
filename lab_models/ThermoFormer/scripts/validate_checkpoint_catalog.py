"""Build and verify the validation-selected prediction/generalization checkpoint catalog."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLE = "experiments/vle/training_records/v1/three/checkpoints/**/best_model.pt"
LLE = "experiments/lle/training_records/v1/formal/**/best.pt"

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def rows(load: bool=False):
    if load:
        import torch
    records=json.loads((ROOT/'models/registry.json').read_text(encoding='utf-8'))['checkpoints']
    for row in records:
        p=ROOT/row['path']
        if not p.is_file() or digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:
            raise RuntimeError(f"Checkpoint content mismatch: {row['path']}")
        if not row['path'].startswith('models/'):
            raise RuntimeError(f"Checkpoint outside models: {row['path']}")
        if load:
            payload=torch.load(p,map_location='cpu',weights_only=False)
            if not isinstance(payload,dict) or 'model' not in payload:
                raise RuntimeError(f"Invalid checkpoint: {row['path']}")
            if int(payload.get('seed',row['seed']))!=row['seed']:
                raise RuntimeError(f"Seed mismatch: {row['path']}")
    expected={'vle':50,'lle':55}
    counts={t:sum(r['task']==t for r in records) for t in expected}
    if counts != expected: raise RuntimeError(f'Expected {expected}, found {counts}')
    for task,n_protocols in [('vle',10),('lle',11)]:
        grouped={r['protocol'] for r in records if r['task']==task}
        if len(grouped)!=n_protocols: raise RuntimeError(f'{task}: expected {n_protocols} protocols, found {len(grouped)}')
        for protocol in grouped:
            seeds={r['seed'] for r in records if r['task']==task and r['protocol']==protocol}
            if seeds != set(range(5)): raise RuntimeError(f'{task}/{protocol}: seeds {seeds}')
    return records

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--load',action='store_true'); args=ap.parse_args()
    records=rows(args.load)
    payload={'schema_version':2,'scope':'Predictive performance and generalization','selection':'validation-selected final checkpoint','expected_seeds':[0,1,2,3,4],'checkpoints':records,'external_dependencies':[{'name':'HANNA reference implementation','status':'obtain_from_upstream_source','purpose':'VLE comparison only; weights are not distributed in this checkpoint catalog'}]}
    out=ROOT/'models'/'registry.json'; out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    csvp=ROOT/'models'/'checkpoint_catalog.csv'
    fields=['task','protocol','seed','path','sha256','bytes','status','checkpoint_seed','selection_epoch','validation_score']
    with csvp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    print(f'validated {len(records)} checkpoints ({sum(r["bytes"] for r in records)/1024**2:.1f} MiB)')

if __name__=='__main__': main()
