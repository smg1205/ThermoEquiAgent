"""Validation-selected constructive ternary decoder, kept separate from VLE."""
import argparse,hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np,torch
from . import campaign as base
from .data import load,partition,split_manifest
from .model import endpoint_loss
from .binodal_model import BinodalLLE
from .repair_campaign import write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
 p.add_argument('--protocol',default='ternary-random');p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--epochs',type=int,default=40)
 p.add_argument('--lr',type=float,default=.0003);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--validation-every',type=int,default=5);p.add_argument('--validation-limit',type=int,default=0)
 p.add_argument('--curve-points',type=int,default=24);p.add_argument('--solver-iterations',type=int,default=28);p.add_argument('--grid-resolution',type=int,default=18);p.add_argument('--device',default='cuda')
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 conditions,audit=load(base.ROOT/'datasets/vle_reference');conditions=[c for c in conditions if c.n==3]
 assert args.protocol in ('ternary-random','ternary-system')
 identity={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='seeds'},'data':audit['inputs'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'model_prior':'continuous parallel tie lines, LLE-only'}
 path=args.output/'identity.json'
 if path.exists():assert json.loads(path.read_text(encoding='utf8'))==identity,'Choose a new output for changed recipe'
 write_json(path,identity)
 for seed in args.seeds:
  base.seed_all(seed);parts=partition(conditions,args.protocol.split('-')[1],seed)
  source=args.source/args.protocol/f'seed_{seed}';out=args.output/args.protocol/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
  assert json.loads((source/'split.json').read_text(encoding='utf8'))==json.loads(json.dumps(split_manifest(parts)))
  write_json(out/'split.json',split_manifest(parts));shutil.copy2(source/'features.json',out/'features.json')
  payload=torch.load(source/'best.pt',weights_only=True,map_location='cpu');features={k:v.numpy() for k,v in payload['feature_map'].items()}
  model=BinodalLLE(**payload['spec']).to(args.device);model.load_state_dict(payload['model'])
  validation=parts['validation']
  if args.validation_limit:validation=[validation[i] for i in np.random.default_rng(1729).permutation(len(validation))[:args.validation_limit]]
  def validate(label):
   folder=out/label;folder.mkdir(parents=True,exist_ok=True);metrics=base.evaluate(model,validation,features,folder,args);write_json(folder/'metrics.json',metrics)
   return metrics['all']['thermodynamic']['failure_penalized_set_distance']
  best=validate('validation_initial');history=[];stale=0
  def save(epoch):torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'spec':model.spec,'seed':seed,'epoch':epoch,'validation_score':best,'feature_map':payload['feature_map']},out/'best.pt')
  save(-1);opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-5);started=time.time()
  for epoch in range(args.epochs):
   model.train();order=np.random.permutation(len(parts['train']));losses=[]
   for start in range(0,len(order),args.batch_size):
    cc=[parts['train'][i] for i in order[start:start+args.batch_size]];state,target,valid,_,_,w=base.batch(cc,features,args.device)
    opt.zero_grad(set_to_none=True);ctx=model.context(*state);ends=model.propose(ctx,args.curve_points)
    loss=(endpoint_loss(ends,target,valid)*w).sum()/w.sum()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite constructive endpoint loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
   row={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':time.time()-started}
   if (epoch+1)%args.validation_every==0 or epoch==args.epochs-1:
    value=validate(f'validation_epoch_{epoch}');row['final_solver_validation_distance']=value
    if value<best:best=value;stale=0;save(epoch)
    else:stale+=1
   history.append(row);write_json(out/'history.json',history);print(json.dumps({'seed':seed,**row}),flush=True)
   if epoch>=19 and stale>=2:break
  write_json(out/'completion.json',{'status':'complete','best_validation_distance':best,'test_evaluated':False,'validation_conditions':len(validation),'last_epoch':epoch,'early_stopping':'minimum 20 epochs, two full validation checks without improvement, maximum configured epochs'})
if __name__=='__main__':main()
