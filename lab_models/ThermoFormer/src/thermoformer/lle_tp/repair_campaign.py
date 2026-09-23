"""Independent repair campaign: held-out solver selection and unrolled supervision."""
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from . import campaign as base
from .data import load,partition,split_manifest,write_json
from .model import TPLLE,endpoint_loss,select_context
from .closed_loop import refine,surface_loss

ROOT=base.ROOT
_original_write=write_json
def write_json(path,value):
    for attempt in range(8):
        try:return _original_write(path,value)
        except PermissionError:
            if attempt==7:raise
            time.sleep(0.2*(attempt+1))
base.write_json=write_json

def final_validation(model,conditions,features,out,args):
    out.mkdir(parents=True,exist_ok=True)
    metrics=base.evaluate(model,conditions,features,out,args)
    write_json(out/'metrics.json',metrics)
    return metrics['all']['thermodynamic']['failure_penalized_set_distance']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--protocol',default='ternary-random')
    p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--physics-weight',type=float,default=0.1)
    p.add_argument('--validation-limit',type=int,default=0)
    p.add_argument('--validation-every',type=int,default=5)
    p.add_argument('--curve-points',type=int,default=24)
    p.add_argument('--solver-iterations',type=int,default=28)
    p.add_argument('--grid-resolution',type=int,default=18)
    p.add_argument('--device',default='cuda')
    p.add_argument('--test',action='store_true',help='Only enable after validation development is frozen.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    conditions,audit=load(ROOT/'datasets/vle_reference')
    scope,mode=args.protocol.split('-')
    if scope not in ('ternary','mixed'): raise ValueError('Use ternary or mixed; standalone binary is unchanged')
    conditions=[c for c in conditions if scope=='mixed' or c.n==3]
    identity={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='seeds'},'data':audit['inputs'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    old=args.output/'identity.json'
    if old.exists() and json.loads(old.read_text(encoding='utf8'))!=identity:raise ValueError('Use a separate output for changed code/config')
    write_json(old,identity)
    for seed in args.seeds:
        base.seed_all(seed);out=args.output/args.protocol/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
        parts=partition(conditions,mode,seed)
        write_json(out/'source_checkpoint.json',{'path':str(args.source/args.protocol/f'seed_{seed}'/'best.pt'),'sha256':hashlib.sha256((args.source/args.protocol/f'seed_{seed}'/'best.pt').read_bytes()).hexdigest()})
        saved=torch.load(args.source/args.protocol/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=True)
        old_split=json.loads((args.source/args.protocol/f'seed_{seed}'/'split.json').read_text())
        assert old_split==json.loads(json.dumps(split_manifest(parts))),'Warm-start partition mismatch'
        write_json(out/'split.json',split_manifest(parts))
        features={s:v.numpy() for s,v in saved['feature_map'].items()}
        model=TPLLE(**saved['spec']).to(args.device);model.load_state_dict(saved['model'])
        validation=parts['validation']
        if args.validation_limit: validation=[validation[i] for i in np.random.default_rng(1729).permutation(len(validation))[:args.validation_limit]]
        best=final_validation(model,validation,features,out/'validation_initial',args)
        def checkpoint(epoch,value):
            torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'spec':model.spec,'seed':seed,'epoch':epoch,'validation_score':value,'feature_map':saved['feature_map']},out/'best.pt')
        checkpoint(-1,best)
        opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=args.lr,weight_decay=1e-5)
        history=[];started=time.time()
        for epoch in range(args.epochs):
            model.train();order=np.random.permutation(len(parts['train']));stats=[]
            for start in range(0,len(order),args.batch_size):
                cc=[parts['train'][i] for i in order[start:start+args.batch_size]]
                state,target,valid,a,b,w=base.batch(cc,features,args.device)
                opt.zero_grad(set_to_none=True);ctx=model.context(*state)
                proposed=model.propose(ctx,points=args.curve_points)
                initial=endpoint_loss(proposed,target,valid)
                ternary=state[-1].sum(-1)==3
                endpoint=initial.clone();physical=torch.zeros_like(initial)
                if ternary.any():
                    tc=select_context(ctx,ternary)
                    refined=refine(model,tc,proposed[ternary],args.steps)
                    endpoint[ternary]=endpoint_loss(refined,target[ternary],valid[ternary])
                    physical[ternary]=surface_loss(model,tc,target[ternary],valid[ternary],proposed[ternary])
                loss=((0.25*initial+endpoint+args.physics_weight*physical)*w).sum()/w.sum()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite repair loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                stats.append([loss.item(),initial.mean().item(),endpoint.mean().item(),physical.mean().item()])
            row={'epoch':epoch,'losses':np.mean(stats,axis=0).tolist(),'seconds':time.time()-started}
            if (epoch+1)%args.validation_every==0 or epoch==args.epochs-1:
                value=final_validation(model,validation,features,out/f'validation_epoch_{epoch}',args)
                row['final_solver_validation_distance']=value
                if value<best:best=value;checkpoint(epoch,value)
            history.append(row);write_json(out/'history.json',history);print(json.dumps(row),flush=True)
        if args.test:
            model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True)['model'])
            metrics=base.evaluate(model,parts['test'],features,out,args);write_json(out/'metrics.json',metrics)
        write_json(out/'completion.json',{'status':'complete','best_validation_distance':best,'test_evaluated':args.test,'validation_conditions':len(validation)})

if __name__=='__main__':main()
