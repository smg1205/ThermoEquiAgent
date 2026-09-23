"""Training on the source-verified LLE dataset."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from . import campaign as base,solver_batch
from .solver_certified import solve_many
from .data import load,partition,split_manifest
from .generalization import partition_generalization
from .model import endpoint_loss
from .hybrid_binodal_model import HybridBinodalLLE
from .repair_campaign import write_json
solver_batch.solve_many=solve_many

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,default=base.ROOT/'datasets/lle')
 p.add_argument('--protocol',default='ternary-random');p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=160)
 p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--validation-every',type=int,default=10);p.add_argument('--min-epochs',type=int,default=80);p.add_argument('--patience',type=int,default=4)
 p.add_argument('--curve-points',type=int,default=24);p.add_argument('--solver-iterations',type=int,default=28);p.add_argument('--grid-resolution',type=int,default=18);p.add_argument('--device',default='cuda')
 p.add_argument('--check-splits-only',action='store_true')
 args=p.parse_args();args.dataset=args.dataset.resolve();args.output.mkdir(parents=True,exist_ok=True)
 conditions,audit=load(args.dataset)
 generalization=args.protocol in {'binary-temperature-low','binary-temperature-high','binary-pressure-low','binary-pressure-high','binary-unseen-component'}
 if generalization:n=2
 else:
  scope,mode=args.protocol.split('-');n={'binary':2,'ternary':3,'mixed':None}[scope]
 conditions=[c for c in conditions if n is None or c.n==n]
 identity={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='seeds'},'data':audit['inputs'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'model_prior':'binary RK plus continuous parallel ternary tie lines; LLE-only','solver_engine':'deterministic per-condition double precision certification; activity RMS <= 0.001; finite-grid TPD >= -0.0001','initialization':'newly initialized network; pretrained molecular features only'}
 path=args.output/'identity.json'
 if path.exists():assert json.loads(path.read_text(encoding='utf8'))==identity,'Choose a new output for changed recipe'
 write_json(path,identity);write_json(args.output/'data_audit.json',audit)
 for seed in args.seeds:
  base.seed_all(seed)
  if generalization:parts,split_metadata=partition_generalization(conditions,args.protocol,seed)
  else:parts=partition(conditions,mode,seed);split_metadata={'split_rule':mode}
  out=args.output/args.protocol/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
  if (out/'metrics.json').exists():continue
  write_json(out/'split.json',{'metadata':split_metadata,'partitions':split_manifest(parts)})
  write_json(out/'split_check.json',{'counts':{k:len(v) for k,v in parts.items()},'metadata':split_metadata,'condition_overlap':False})
  if args.check_splits_only:continue
  ff=base.features_for(conditions,parts['train'],args.output);features=ff.values;write_json(out/'features.json',ff.metadata)
  base.seed_all(seed);model=HybridBinodalLLE(ff.view_dimensions).to(args.device)
  best=float('inf');history=[];stale=0;opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-5);started=time.time()
  def save(epoch):torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'spec':model.spec,'seed':seed,'epoch':epoch,'validation_score':best,'feature_map':{k:torch.tensor(v) for k,v in features.items()}},out/'best.pt')
  for epoch in range(args.epochs):
   model.train();order=np.random.permutation(len(parts['train']));losses=[]
   for start in range(0,len(order),args.batch_size):
    cc=[parts['train'][i] for i in order[start:start+args.batch_size]];state,target,valid,_,_,w=base.batch(cc,features,args.device)
    opt.zero_grad(set_to_none=True);ctx=model.context(*state);ends=model.propose(ctx,args.curve_points)
    # These are analytically coexisting latent endpoints of the constructed gE,
    # not a separate unconstrained initial-guess head. Final inference rechecks
    # them through AD activities, TPD, and the unchanged acceptance tolerances.
    loss=(endpoint_loss(ends,target,valid)*w).sum()/w.sum()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite endpoint loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
   row={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':time.time()-started}
   if (epoch+1)%args.validation_every==0 or epoch==args.epochs-1:
    folder=out/f'validation_epoch_{epoch}';folder.mkdir(parents=True,exist_ok=True)
    metrics=base.evaluate(model,parts['validation'],features,folder,args);write_json(folder/'metrics.json',metrics)
    value=metrics['all']['thermodynamic']['failure_penalized_set_distance'];row['final_solver_validation_distance']=value
    if value<best:best=value;stale=0;save(epoch)
    else:stale+=1
   history.append(row);write_json(out/'history.json',history)
   if 'final_solver_validation_distance' in row:print(json.dumps({'protocol':args.protocol,'seed':seed,**row}),flush=True)
   if epoch+1>=args.min_epochs and stale>=args.patience:break
  model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True)['model'])
  metrics=base.evaluate(model,parts['test'],features,out,args);write_json(out/'metrics.json',metrics)
  write_json(out/'completion.json',{'status':'complete','best_validation_distance':best,'test_evaluated':True,'last_epoch':epoch,'fresh_initialization':True,'train_only_scaling':True,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()})
  print(json.dumps({'completed':args.protocol,'seed':seed,'test':metrics['all']['thermodynamic']}),flush=True)
  del model;torch.cuda.empty_cache()
if __name__=='__main__':main()
