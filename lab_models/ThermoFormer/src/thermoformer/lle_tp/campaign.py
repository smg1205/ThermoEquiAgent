"""Six protocols x five seeds, with train-only scaling and validation selection."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .data import load,partition,split_manifest,write_json
from .model import TPLLE,endpoint_loss,physics_loss,select_context
from .solver import solve
from .metrics import score
from ..configuration import EncoderConfig
from ..features import build_molecular_encoder,prepare_partition_features

ROOT=Path(__file__).resolve().parents[3]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(4)

def features_for(conditions,train,root):
    cache=root/'cache'; cache.mkdir(parents=True,exist_ok=True)
    source_cache=ROOT/'experiments/run_records/lle_tp_v1/cache'
    if not source_cache.exists(): source_cache=ROOT/'cache'
    for p in source_cache.glob('*.npz'):
        if not (cache/p.name).exists() and p.resolve()!=(cache/p.name).resolve(): shutil.copy2(p,cache/p.name)
    config=EncoderConfig(representation='multiview',fusion_mode='naive',use_unimol=True)
    encoder=build_molecular_encoder(config,cache/'hybrid_rdkit_unimol_84m_functional_v1.npz',use_cuda=torch.cuda.is_available())
    return prepare_partition_features(encoder,sorted({s for c in conditions for s in c.key[0]}),sorted({s for c in train for s in c.key[0]}))

def batch(conditions,features,device):
    b=len(conditions); d=len(next(iter(features.values())))
    mol=np.zeros((b,3,d),np.float32); mask=np.zeros((b,3),np.float32)
    length=max(len(c.rows) for c in conditions)
    target=np.zeros((b,length,2,3),np.float32); valid=np.zeros((b,length),bool)
    alpha=np.zeros((b,3),np.float32); beta=np.zeros_like(alpha); weight=[]
    for i,c in enumerate(conditions):
        mol[i,:c.n]=np.stack([features[s] for s in c.key[0]]); mask[i,:c.n]=1
        target[i,:len(c.rows),:,:c.n]=c.targets; valid[i,:len(c.rows)]=True
        row=c.rows[random.randrange(len(c.rows))]
        alpha[i,:c.n]=row.phase_alpha; beta[i,:c.n]=row.phase_beta
        weight.append(np.mean([r.quality_weight for r in c.rows]))
    tensor=lambda x:torch.as_tensor(x,device=device)
    state=[tensor(mol),tensor(np.array([[c.key[1]] for c in conditions],np.float32)),tensor(np.array([[c.key[2]] for c in conditions],np.float32)),tensor(mask)]
    return state,tensor(target),tensor(valid),tensor(alpha),tensor(beta),tensor(np.array(weight,np.float32))

def validation(model,conditions,features,device,batch_size):
    model.eval(); losses=[]
    for start in range(0,len(conditions),batch_size):
        cc=conditions[start:start+batch_size]
        state,target,valid,a,b,w=batch(cc,features,device)
        with torch.no_grad():
            ctx=model.context(*state); proposed=model.propose(ctx)
            loss=endpoint_loss(proposed,target,valid)
        phys=physics_loss(model,ctx,a,b).detach()
        losses.extend((loss+0.02*phys).cpu().tolist())
    return float(np.mean(losses))

def fit(parts,features,dimensions,out,args,seed):
    device=torch.device(args.device)
    model=TPLLE(dimensions,args.hidden,args.layers).to(device)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-5)
    best=float('inf'); best_state=None; history=[]; start_time=time.time(); stale=0
    for epoch in range(args.epochs):
        model.train(); order=np.random.permutation(len(parts['train'])); losses=[]
        # Endpoint proposal warmup is only an initializer, never a final LLE prediction.
        stage='proposal_warmup' if epoch<args.warmup else ('equilibrium_finetune' if epoch>=int(args.epochs*.8) else 'joint_supervision')
        strength=0 if stage=='proposal_warmup' else (0.02 if stage=='joint_supervision' else 0.05)
        for start in range(0,len(order),args.batch_size):
            cc=[parts['train'][i] for i in order[start:start+args.batch_size]]
            state,target,valid,a,b,w=batch(cc,features,device)
            opt.zero_grad(set_to_none=True)
            ctx=model.context(*state); proposed=model.propose(ctx)
            supervised=endpoint_loss(proposed,target,valid)
            physical=physics_loss(model,ctx,a,b) if strength else torch.zeros_like(supervised)
            loss=((supervised+strength*physical)*w).sum()/w.sum()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            losses.append(float(loss.detach()))
        if epoch%5==0 or epoch==args.epochs-1:
            random_state=random.getstate(); torch_state=torch.get_rng_state(); cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            random.seed(1729); torch.manual_seed(1729)
            value=validation(model,parts['validation'],features,device,args.batch_size)
            random.setstate(random_state); torch.set_rng_state(torch_state)
            if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
            if value<best and epoch>=args.warmup:
                best=value; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
                torch.save({'model':best_state,'spec':model.spec,'seed':seed,'epoch':epoch,'validation_score':best,'feature_map':{s:torch.tensor(v) for s,v in features.items()}},out/'best.pt')
            else: stale+=1
            history.append({'epoch':epoch,'stage':stage,'train_loss':float(np.mean(losses)),'validation_score':value,'seconds':time.time()-start_time})
            write_json(out/'history.json',history)
            print(json.dumps({'run':out.name,**history[-1]}),flush=True)
    if best_state is None: raise RuntimeError('No eligible validation checkpoint; epochs must exceed warmup')
    model.load_state_dict(best_state); model.eval()
    return model

def evaluate(model,conditions,features,out,args):
    from .solver_batch import solve_many
    records=[]; device=torch.device(args.device)
    model=copy.deepcopy(model).double().eval().requires_grad_(False)
    for n in (2,3):
        selected=[c for c in conditions if c.n==n]
        # Certification is deliberately condition-local. A batch of one makes
        # validation, test evaluation and public inference construct the exact
        # same double-precision thermodynamic context.
        for start in range(0,len(selected),1):
            cc=selected[start:start+1]
            state,_,_,_,_,_=batch(cc,features,device)
            with torch.no_grad():
                ctx=model.context(*(v.double() for v in state))
                proposed=model.propose(ctx,points=args.curve_points)
            if n==2: proposed=proposed[:,:1]
            solved=solve_many(model,ctx,proposed,iterations=args.solver_iterations,grid_resolution=args.grid_resolution)
            for c,pr,result in zip(cc,proposed,solved):
                records.append({'smiles':c.key[0],'temperature_k':c.key[1],'pressure_kpa':c.key[2],'observed':c.targets.tolist(),'proposal':pr[...,:c.n].cpu().tolist(),'predicted':[r['endpoints'] for r in result['pairs']],'solver':result})
            print(json.dumps({'evaluated':len(records),'total':len(conditions),'run':out.name}),flush=True)
            write_json(out/'predictions.partial.json',records)
    write_json(out/'predictions.json',records)
    metrics={}
    for label,n in [('all',None),('binary',2),('ternary',3)]:
        rr=[r for r in records if n is None or len(r['smiles'])==n]
        if rr: metrics[label]={'thermodynamic':score(rr),'proposal_only_diagnostic':score(rr,'proposal')}
    return metrics

def aggregate(root):
    grouped={}
    for p in root.glob('*/*/metrics.json'):
        protocol=p.parent.parent.name
        value=json.loads(p.read_text(encoding='utf8'))
        grouped.setdefault(protocol,[]).append(value)
    report={}
    lines=['# LLE TP experiment results','','R2 is phase-matched and uses independent mole fractions. Thermodynamic R2 on successful conditions alone is not an all-test-set result. The 0.95 target passes only with 100% solver coverage. Ternary primary metric is failure-penalized bidirectional set distance.','']
    for protocol,runs in sorted(grouped.items()):
        rr={}
        for subset in ('all','binary','ternary'):
            for source in ('thermodynamic','proposal_only_diagnostic'):
                for metric in ('failure_penalized_set_distance','observed_coverage_at_0.02','success_fraction','matched_endpoint_r2_successful_only','matched_endpoint_mae_successful_only'):
                    values=[r[subset][source][metric] for r in runs if subset in r and r[subset][source][metric] is not None]
                    if values: rr[f'{subset}.{source}.{metric}']={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else None,'n':len(values)}
        report[protocol]={'completed_seeds':len(runs),'statistics':rr}
        lines += [f'## {protocol} ({len(runs)}/5 seeds)','', '| Subset / output / metric | Mean | Sample std | n |','|---|---:|---:|---:|']
        for k,v in rr.items(): lines.append(f"| {k} | {v['mean']:.6g} | {v['std'] if v['std'] is not None else 'NA'} | {v['n']} |")
        lines.append('')
    write_json(root/'summary.json',report)
    (root/'report.md').write_text('\n'.join(lines),encoding='utf8')
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/run_records/lle_tp_v1')
    ap.add_argument('--protocols',nargs='+',default=[f'{scope}-{mode}' for scope in ('binary','ternary','mixed') for mode in ('random','system')])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--warmup',type=int,default=30)
    ap.add_argument('--hidden',type=int,default=96); ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--curve-points',type=int,default=24); ap.add_argument('--solver-iterations',type=int,default=28); ap.add_argument('--grid-resolution',type=int,default=18)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--validation-only',action='store_true'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--aggregate-only',action='store_true')
    args=ap.parse_args(); args.output=args.output.resolve()
    if args.aggregate_only: aggregate(args.output); return
    if args.smoke:
        args.output=args.output/'smoke'; args.epochs=3; args.warmup=1; args.solver_iterations=3; args.curve_points=4
    args.output.mkdir(parents=True,exist_ok=True)
    if args.epochs<=args.warmup: raise ValueError('epochs must exceed warmup')
    conditions,audit=load(ROOT/'datasets/vle_reference'); write_json(args.output/'data_audit.json',audit)
    seed_all(0)
    print(json.dumps(audit,ensure_ascii=True),flush=True)
    if args.prepare_only:
        features_for(conditions,conditions,args.output)
        print('FEATURE_CACHE_READY',flush=True); return
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('src','scripts','configs') for p in (ROOT/sub).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml') and 'lle_tp' not in str(p)}
    write_json(args.output/'vle_source_hashes.json',frozen)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    identity={k:v for k,v in config.items() if k not in ('protocols','seeds','aggregate_only','prepare_only')}
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    identity['code_sha256']=code_hashes
    identity['data_sha256']=audit['inputs']
    existing=args.output/'run_identity.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf8'))!=identity:
        raise ValueError('Output directory has a different code/config/data identity; choose a a separate --output')
    write_json(existing,identity)
    write_json(args.output/'campaign_config.json',config)
    for protocol in args.protocols:
        scope,mode=protocol.split('-'); n={'binary':2,'ternary':3,'mixed':None}[scope]
        selected=[c for c in conditions if n is None or c.n==n]
        for seed in args.seeds:
            out=args.output/protocol/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
            metric_file='validation_metrics.json' if args.validation_only else 'metrics.json'
            if (out/metric_file).exists(): continue
            seed_all(seed); parts=partition(selected,mode,seed)
            if args.smoke: parts={k:v[:(12 if k=='train' else 3)] for k,v in parts.items()}
            write_json(out/'split.json',split_manifest(parts))
            ff=features_for(selected,parts['train'],args.output); write_json(out/'features.json',ff.metadata)
            seed_all(seed)
            model=fit(parts,ff.values,ff.view_dimensions,out,args,seed)
            metrics=evaluate(model,parts['validation'] if args.validation_only else parts['test'],ff.values,out,args)
            write_json(out/metric_file,metrics)
            write_json(out/'manifest.json',{'status':'smoke' if args.smoke else ('validation_only' if args.validation_only else 'completed'),'protocol':protocol,'seed':seed,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),'code_sha256':code_hashes,'config':config,'data_sha256':audit['inputs'],'evaluation_partition':'validation' if args.validation_only else 'test','test_conditions':len(parts['test'])})
            aggregate(args.output)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    changed=[name for name,sha in frozen.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    write_json(args.output/'vle_preservation.json',{'changed_existing_files':changed})
    if changed: raise RuntimeError('Original source files changed during campaign: '+str(changed))

if __name__=='__main__': main()
