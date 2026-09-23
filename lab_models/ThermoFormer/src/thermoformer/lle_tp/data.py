"""Audited condition sets and deterministic, component-count-stratified splits."""
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..data.lle import load_lle_dataset, _identifier

@dataclass
class Condition:
    key: tuple
    rows: tuple
    @property
    def n(self): return len(self.key[0])
    @property
    def targets(self):
        return np.array([[r.phase_alpha, r.phase_beta] for r in self.rows], dtype=np.float64)

def load(root):
    root=Path(root)
    binary=root/'binary_lle.xlsx'
    ternary=root/'ternary_lle.xlsx'
    if not binary.is_file() or not ternary.is_file():
        raise ValueError('Expected binary_lle.xlsx and ternary_lle.xlsx')
    result=load_lle_dataset(root,binary_workbook=binary.name,ternary_workbook=ternary.name)
    groups=defaultdict(list)
    for r in result.samples:
        if r.temperature_k<=0: raise ValueError('Nonpositive absolute temperature')
        groups[(r.smiles,round(r.temperature_k,6),round(r.pressure_kpa,6))].append(r)
    conditions=[Condition(k,tuple(sorted(v,key=_identifier))) for k,v in sorted(groups.items())]
    audit=dict(result.audit)
    audit['quality_policy']='Inherited VLE policy: 0=failed/excluded, 1=passed, -1=unverified/weight 0.5'
    audit['inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (binary,ternary)}
    audit['by_components']={str(n):dict(rows=sum(len(c.rows) for c in conditions if c.n==n),conditions=sum(c.n==n for c in conditions),systems=len({c.key[0] for c in conditions if c.n==n}),multiple_tieline_conditions=sum(c.n==n and len(c.rows)>1 for c in conditions)) for n in (2,3)}
    return conditions,audit

def partition(conditions,mode,seed):
    if mode not in ('random','system'): raise ValueError(mode)
    out={k:[] for k in ('train','validation','test')}
    rng=np.random.default_rng(seed)
    for n in (2,3):
        buckets=defaultdict(list)
        for c in conditions:
            if c.n==n: buckets[c.key if mode=='random' else c.key[0]].append(c)
        keys=sorted(buckets)
        if not keys: continue
        if len(keys)<3: raise ValueError('At least three split groups required per component count')
        rng.shuffle(keys)
        nt=max(1,round(len(keys)*0.15)); nv=max(1,round(len(keys)*0.15))
        if nt+nv>=len(keys): raise ValueError('Empty training partition')
        for name,subset in [('test',keys[:nt]),('validation',keys[nt:nt+nv]),('train',keys[nt+nv:])]:
            out[name].extend(c for k in subset for c in buckets[k])
    for values in out.values(): values.sort(key=lambda c:c.key)
    sets=[{c.key if mode=='random' else c.key[0] for c in out[name]} for name in out]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert sum(map(len,out.values()))==len(conditions)
    return out

def split_manifest(parts):
    return {name:[{'smiles':c.key[0],'temperature_k':c.key[1],'pressure_kpa':c.key[2],'records':[_identifier(r) for r in c.rows]} for c in values] for name,values in parts.items()}

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    temporary.replace(path)
